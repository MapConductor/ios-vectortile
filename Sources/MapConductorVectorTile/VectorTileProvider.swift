import Foundation
import MapConductorCore
import MapConductorVectorStyle
import UIKit

/**
 Renders a MapLibre vector style to raster tiles, for map backends that cannot
 display a vector style themselves.

 Register it with `TileServerRegistry.get()` and point a raster layer at the
 resulting route; the backend only ever sees ordinary raster tiles, which is
 what makes this work on MapKit, Google Maps, HERE, ArcGIS and the rest.

 Network I/O happens here rather than in the native library, so the app's own
 URLSession behaviour keeps applying: `headers` covers auth tokens, and
 `fetchTile` hands over completely.

 # Two layers, not one

 A style is served as **two** stacked raster layers rather than one. `groundTiles`
 draws fills, lines and circles — the half a GPU can draw, and the half no font
 arriving can change. `labelTiles` draws the labels and icons on a transparent
 ground. The map shows one map; the halves are rendered in parallel, and a glyph
 range landing redraws only the transparent half, so a label appearing never
 blanks the map beneath it.

 `renderTile(request:)` still draws both into one tile, for callers that want a
 single layer.
 */
public final class VectorTileProvider: TileProvider {
    public static let defaultTileSize = 512

    /**
     The least grain the label layer is drawn at, as a multiple of the tile size.

     Labels are the one thing on a map read as *shapes*, so they show a tile's
     resolution the way nothing else does. On a 1x screen this is what keeps
     them from looking like a fax; on a Retina screen `renderScale` already
     asks for those pixels and this adds nothing.
     */
    public static let labelResolution = 2

    /**
     How many pixels to draw per point of screen.

     Neither MapKit nor MapLibre asks for `@2x` tiles — both request at
     `pixelRatio` 1 — so a tile drawn at its nominal size is stretched across
     twice as many pixels on a Retina screen and every edge in it goes soft.
     Small text is where that shows first, because a label is read as a shape
     and a blurred shape reads as a smaller one.

     This changes sharpness only. **How large the map looks is set by the tile
     size the raster layer declares**, not by this: apparent size works out to
     the style's own value: a tile is drawn for the size it is shown at. It
     was once `tileSize / 512` -- the renderer assumed every tile covered
     512pt -- which drew everything at half size wherever that was untrue.
     */
    private let renderScale: Int

    /// What a request should draw.
    public enum Content {
        /// Ground and labels in one tile: the single-layer behaviour.
        case full
        /// Fills, lines and circles. No labels, no icons, no glyph dependency.
        case ground
        /// Labels and icons on a transparent ground.
        case labels
    }

    private let renderer: VectorTileRenderer
    private let tileSize: Int
    private let fetchTile: (URL, FetchCancellation?) throws -> Data?
    /// The network for source tiles, with what the server said about caching
    /// them. Glyphs and the sprite keep using `fetchTile`.
    private let sourceFetch: (URL, FetchCancellation?) throws -> SourceResponse
    /// Bytes already on the device -- an offline package -- read at once.
    private let localTile: ((URL) -> Data?)?
    /// Asks the server whether a cached tile changed. Nil when the caller
    /// replaced the network and said nothing about checking: its transport
    /// is the only one allowed to go out.
    private let revalidator: ((URL, SourceValidators) throws -> SourceResponse)?
    private let sourceDiskCache: SourceTileDiskCache?

    /// Source tiles keyed by URL.
    ///
    /// Not an optimisation detail: neighbouring target tiles routinely need the
    /// same source tile — always, once overzoom kicks in, where one magnified
    /// ancestor serves 16 targets — and a map asks for a whole viewport at once.
    /// Budgeted by a strict byte limit and idle TTL. Counting
    /// entries is the easy mistake: a basemap tile is 150-300 KB, so a few
    /// hundred of them is tens of megabytes.
    private var cache = TileMemoryCache<Data>(byteLimit: 16 << 20)
    /// Present only when GPU rendering is in use.
    private let gpu: MetalTileRasterizer?
    private let gpuRenderCount = Counter()
    private let gpuFallbackCount = Counter()
    /// URLs known to hold nothing, so a missing tile is not re-requested.
    private let empties = TileMemoryCache<Bool>(byteLimit: 256 << 10, countLimit: 1024, ttl: 30)
    private var memoryWarningObserver: NSObjectProtocol?
    private let lock = NSLock()
    private var closed = false

    // MARK: Style assets

    private let glyphCache: StyleAssetCache?
    private let spriteCache: StyleAssetCache?

    /// Rendered PNGs, so a second launch does not redraw what the first drew.
    private let diskCache: TileDiskCache?
    /// The ground of every tile the GPU drew, straight RGBA, keyed like the
    /// disk cache. A glyph generation arriving redraws the tile, and with this
    /// the redraw is the labels and a PNG encode, not the geometry again.
    /// Strict LRU budget for raw RGBA; compressed source bytes have their own budget.
    private let groundRasters = TileMemoryCache<Data>(byteLimit: 32 << 20, countLimit: 64)

    /// Identifies the style these tiles were drawn from. A restyle changes it,
    /// so the old tiles simply stop being found rather than needing to be
    /// hunted down and deleted.
    private var styleKey: String

    /// Glyph ranges somebody has already claimed, so two tiles wanting the
    /// same range do not both fetch it.
    private var requestedGlyphs = Set<String>()
    private let glyphsInFlight = Counter()

    /// Bumped when glyphs arrive, so tiles drawn before them stop being served.
    private let glyphGeneration = Counter()
    private var notifyPending = false

    /**
     Whether any tile has been drawn short of glyphs since the last handover.

     Most glyph arrivals change nothing: the ranges landed before anything
     needed them. Handing over anyway makes the map refetch and redraw the whole
     viewport to arrive at the same pixels.
     */
    private var provisionalSinceHandover = false

    /// True when the style names no `glyphs` template, so labels can never
    /// need one and none of the fetching below has anything to do.
    private let glyphsUnavailable: Bool

    /// Fetches source tiles several at a time.
    ///
    /// Sized for the whole viewport rather than one tile: a plan is nine
    /// fetches and a screen is several tiles, and fetching them one after
    /// another spends a second waiting for what takes a fraction of it at once.
    /// The threads spend their lives blocked on sockets, so they are cheap.
    private let fetchSlots = DispatchSemaphore(value: 16)
    private let fetchQueue = DispatchQueue(
        label: "mapconductor.vectortile.fetch", attributes: .concurrent
    )

    /// Narrower than the tile pool on purpose: glyph ranges compete with the
    /// source tiles that are needed to draw anything at all.
    private let glyphQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "mapconductor.vectortile.glyphs"
        queue.qualityOfService = .utility
        queue.maxConcurrentOperationCount = 6
        return queue
    }()

    /// One fetch per URL. Two tiles asking for the same source at the same
    /// moment would otherwise both pay for it.
    private var inFlight: [String: SourceFetchFlight] = [:]

    /// Told when glyphs arrive and tiles drawn before them are now stale.
    ///
    /// The host must make the map refetch: the tiles already on screen were
    /// drawn without those labels, and nothing else will replace them.
    public var onGlyphsLoaded: (() -> Void)?

    /**
     Told when source tiles arrive that tiles already on screen were drawn
     without, or when a cached tile turned out to have changed upstream.

     A tile whose sources are slow is drawn after ``sourceGrace`` with what
     has arrived, rather than leaving the map blank for a network round trip
     (or a fifteen-second timeout on a Wi-Fi that answers nothing). Like
     ``onGlyphsLoaded``, the host must make the map refetch -- *both* halves,
     since the ground is missing geometry too.
     */
    public var onSourcesLoaded: (() -> Void)?

    /**
     How long a tile waits for sources that are not on the device before it
     is drawn without them.

     Long enough that a healthy connection still delivers whole tiles; short
     enough that a stalled one cannot hold the tile server's render lane. A
     tile drawn short is not kept, and redrawn once its sources land.
     */
    public static let sourceGrace: TimeInterval = 0.4

    /// ``sourceGrace``, adjustable by the tests that are about something else.
    var sourceGrace: TimeInterval = VectorTileProvider.sourceGrace

    /// Source URLs a tile was drawn without, so their arrival is worth a handover.
    private var drawnWithout = Set<String>()
    private var sourcesNotifyPending = false

    /**
     Set once a revalidation replaced bytes this provider may already have
     drawn from.

     The native renderer keeps decoded source tiles keyed by source and
     coordinates, not by bytes, and offers no way to forget one -- so until
     its bounded cache lets the old tile go, a redraw here can still show the
     old geometry. Nothing drawn from then on is kept on disk: the next
     provider, which decodes the new bytes, must not be served this one's
     pictures under the new revision.
     */
    private var replacedSources = false

    /// Revalidations under way, so a tile drawn by nine neighbours is checked once.
    private var revalidating = Set<String>()

    /// Background checks of stale cached tiles. Two at a time: they are never
    /// what the map is waiting for, and must not crowd out what it is.
    private let revalidationQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "mapconductor.vectortile.revalidate"
        queue.qualityOfService = .utility
        queue.maxConcurrentOperationCount = 2
        return queue
    }()

    /// How long to let a burst of ranges settle before telling the host.
    ///
    /// A viewport's worth arrives within a second or two of each other, and
    /// notifying per range redraws every visible tile dozens of times to reach
    /// the same picture.
    private let notifyQuiet: DispatchTimeInterval = .milliseconds(400)

    /// How long to keep holding the window open while ranges are in flight.
    ///
    /// The quiet window alone is not enough: ranges do not arrive in one burst,
    /// so the window closes between them again and again and every close is a
    /// handover. The cap is there because a range that never answers must not
    /// hold the labels back forever.
    private let notifyMaxWait: TimeInterval = 10

    /// How tiles are turned into pixels.
    public enum RenderMode {
        /// tiny-skia on the CPU. Always available, and the only path that
        /// draws every layer type the renderer supports.
        case cpu
        /// Metal. Roughly twice as fast per tile, and — the reason it exists —
        /// it spends the map's time on a processor the rest of the app is not
        /// competing for.
        case gpu
        /// GPU where it initialises, CPU otherwise.
        case auto
    }

    /// Which path this provider actually took, after `auto` resolved.
    public var renderMode: RenderMode { gpu != nil ? .gpu : .cpu }

    /// Tiles drawn on the GPU, and tiles that asked for the GPU and fell back.
    ///
    /// Worth reading in tests: a silent fallback looks exactly like success
    /// from the outside, and a suite once passed with the GPU path failing on
    /// every single tile.
    public var gpuRenders: Int { gpuRenderCount.value }
    public var gpuFallbacks: Int { gpuFallbackCount.value }

    /// - Throws: `VectorTileError.styleRejected` if the style cannot be parsed,
    ///   or `renderFailed` if `renderMode` is `.gpu` and Metal is unavailable.
    public init(
        styleJSON: String,
        /// Adjustments applied to the style before anything is drawn -- the
        /// same rules ``VectorStyle`` takes, so "roads are white" or "no
        /// buildings" reads the same wherever it is written. A rule that
        /// hides a role is the reason to reach for this rather than editing
        /// the document: those layers are switched off in what the renderer
        /// is given, so they cost nothing to not draw.
        rules: StyleRules = .none,
        tileSize: Int = VectorTileProvider.defaultTileSize,
        headers: [String: String] = [:],
        cacheBytes: Int = 16 * 1024 * 1024,
        renderMode: RenderMode = .auto,
        /// Where to keep glyph ranges, the sprite sheet, and rendered tiles
        /// between launches. Nil disables all three, at the cost of refetching
        /// and redrawing them on every cold start.
        assetCacheDirectory: URL? = nil,
        /// Budget for rendered tiles on disk. Rendered PNGs are 2-3x larger
        /// than a network tile would be — the encoder trades ratio for speed —
        /// so this is generous by design.
        renderedCacheBytes: Int = 48 * 1024 * 1024,
        /// Budget for fetched source MVT/PBF tiles kept across launches.
        sourceCacheBytes: Int = 128 * 1024 * 1024,
        /// Pixels drawn per point of screen. Defaults to the display's own
        /// scale, which is what stops a Retina screen stretching every tile.
        renderScale: Int? = nil,
        /// Replaces the network for source tiles, glyph ranges and the
        /// sprite. Nil means "no such resource" and is remembered; a thrown
        /// ``OfflineUnavailableError`` means "not now" and is not.
        fetchTile: ((URL) throws -> Data?)? = nil,
        /// Answers from local storage at once -- an offline package -- and is
        /// consulted before any network slot, so a tile on the device is never
        /// queued behind fetches waiting on a slow network. Nil means "not here".
        localTile: ((URL) -> Data?)? = nil,
        /// Asks the server whether a cached source tile changed, with the
        /// validators it was stored with. Defaults to plain HTTP with
        /// `headers` when `fetchTile` is nil; with a custom `fetchTile` and no
        /// `revalidate`, cached tiles are never checked. Throwing (offline,
        /// say) leaves the cached tile as it is.
        revalidate: ((URL, SourceValidators) throws -> SourceResponse)? = nil
    ) throws {
        // The renderer needs the points a tile covers on screen, not the
        // pixels a render asks for (the label pass draws at twice the
        // pixels). It sets the size the style draws at and the zoom its
        // expressions are read at.
        let drawn = try drawnStyleJSON(styleJSON, rules)
        self.renderer = try VectorTileRenderer(styleJSON: drawn, displayTileSize: tileSize)
        self.tileSize = tileSize
        if let fetchTile {
            // Public test/custom transports keep their existing synchronous
            // contract. The built-in URLSession path below is cancellable.
            self.fetchTile = { url, _ in try fetchTile(url) }
            self.sourceFetch = { url, _ in
                let bytes = try fetchTile(url)
                return SourceResponse(status: bytes == nil ? 404 : 200, body: bytes)
            }
            self.revalidator = revalidate
        } else {
            self.fetchTile = { url, cancellation in
                try VectorTileProvider.fetch(url, headers: headers, cancellation: cancellation)
            }
            // The built-in path keeps the response headers, so the tile can
            // be checked later instead of fetched again.
            self.sourceFetch = { url, cancellation in
                try VectorTileProvider.conditionalFetch(url, validators: .none, headers: headers, cancellation: cancellation)
            }
            self.revalidator = revalidate ?? { url, validators in
                try VectorTileProvider.conditionalFetch(url, validators: validators, headers: headers)
            }
        }
        self.localTile = localTile
        cache = TileMemoryCache<Data>(byteLimit: cacheBytes)
        self.glyphsUnavailable = (try? renderer.glyphsURLTemplate()) == nil
        // The *drawn* document, not the author's: rendered tiles are keyed by
        // it, and a style with its buildings hidden must not be served the
        // tiles drawn with them -- nor leave its own behind for the style
        // that comes back.
        self.styleKey = Digest.hex(drawn)
        self.renderScale = max(1, renderScale ?? Int(UIScreen.main.scale.rounded()))

        // Separate directories: a glyph range and a sprite sheet have nothing
        // to do with each other, and clearing one should not take the other.
        self.glyphCache = assetCacheDirectory.flatMap {
            StyleAssetCache(directory: $0.appendingPathComponent("glyphs"))
        }
        self.spriteCache = assetCacheDirectory.flatMap {
            StyleAssetCache(directory: $0.appendingPathComponent("sprites"))
        }
        self.diskCache = assetCacheDirectory.flatMap {
            TileDiskCache(
                directory: $0.appendingPathComponent("tiles"),
                budgetBytes: renderedCacheBytes
            )
        }
        self.sourceDiskCache = assetCacheDirectory.flatMap {
            SourceTileDiskCache(
                directory: $0.appendingPathComponent("sources"),
                budgetBytes: sourceCacheBytes
            )
        }

        // Built at the size tiles are actually drawn at: the rasteriser has one
        // fixed target, so a scale it does not know about would be a readback
        // of the wrong size.
        let drawnSize = tileSize * self.renderScale
        switch renderMode {
        case .cpu:
            self.gpu = nil
        case .auto:
            self.gpu = MetalTileRasterizer.createOrNull(tileSize: drawnSize)
        case .gpu:
            // Explicitly asking for the GPU and quietly getting the CPU is the
            // failure that hides worst, so this one throws.
            guard let rasterizer = MetalTileRasterizer.createOrNull(tileSize: drawnSize) else {
                throw VectorTileError.renderFailed(MvtStatus.renderFailed)
            }
            self.gpu = rasterizer
        }

        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.cache.removeAll()
            self?.groundRasters.removeAll()
            self?.empties.removeAll()
        }
        warmStyleAssets()
    }

    deinit {
        if let memoryWarningObserver { NotificationCenter.default.removeObserver(memoryWarningObserver) }
        close()
    }

    private func beginClose() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return false }
        closed = true
        onGlyphsLoaded = nil
        onSourcesLoaded = nil
        revalidationQueue.cancelAllOperations()
        empties.removeAll()
        return true
    }

    private func releaseResources() {
        renderer.close()
        // A successor may share these source bytes. The bounded cache releases them
        // when the last provider goes away; closing one must not evict them.
        groundRasters.removeAll()
        diskCache?.flush()
    }

    /// Called only before the successor is registered with the tile server.
    /// Styles in one rasterisation use the same transport and headers, so
    /// successful source bytes remain valid across paint changes.
    func reuseSourceTiles(from previous: VectorTileProvider) {
        cache = previous.cache
    }

    /// Stops accepting requests immediately; waits for active native calls off the UI thread.
    public func closeAsync() {
        guard beginClose() else { return }
        DispatchQueue.global(qos: .utility).async { self.releaseResources() }
    }

    /// Releases resources synchronously. UI owners should use `closeAsync()`.
    public func close() {
        guard beginClose() else { return }
        releaseResources()
    }

    private var isClosed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return closed
    }

    // MARK: - What the host must show

    /**
     The credits this style's sources ask to be shown.

     These are not optional. A style is data under someone's licence, and a
     basemap drawing OpenStreetMap requires the credit; reading it out of the
     style and then not showing it is the one way this provider can be used
     wrongly without anything looking wrong. May contain HTML — the credit is
     normally a link to the licence.
     */
    public func attributions() -> [String] {
        (try? renderer.attributions()) ?? []
    }

    /// Reasons the current style may not render as intended.
    public func diagnostics() -> [String] {
        (try? renderer.diagnostics()) ?? []
    }

    /// Replaces the style. Fetched vector tiles stay valid — the geometry is
    /// unchanged, only the paint applied to it — so recolouring costs a
    /// re-rasterise and no network traffic.
    ///
    /// The caller still has to make the map drop its *raster* tiles; pass a new
    /// `cacheKey` to `LocalTileServer.urlTemplate`.
    public func setStyle(_ styleJSON: String, rules: StyleRules = .none) throws {
        let drawn = try drawnStyleJSON(styleJSON, rules)
        try renderer.setStyle(drawn)
        // Rendered tiles are keyed by style, so the old ones simply stop being
        // found. Nothing has to be deleted.
        styleKey = Digest.hex(drawn)
        warmStyleAssets()
    }

    /// The generation to put in a tile URL, so tiles drawn short of their
    /// glyphs stop being served once the glyphs land.
    public var glyphGenerationValue: Int { glyphGeneration.value }

    // MARK: - The two halves

    /// The ground alone: fills, lines and circles, drawn on the GPU where there
    /// is one. Never invalidated by a font arriving.
    public private(set) lazy var groundTiles: TileProvider = Facade(self, .ground)

    /// The labels and icons alone, on a transparent ground.
    public private(set) lazy var labelTiles: TileProvider = Facade(self, .labels)

    /// One half of a split layer, as something a tile server can register.
    private final class Facade: TileProvider {
        private let content: Content
        private weak var owner: VectorTileProvider?

        init(_ owner: VectorTileProvider, _ content: Content) {
            self.owner = owner
            self.content = content
        }

        func renderTile(request: TileRequest) -> Data? {
            owner?.renderTile(request: request, content: content) { false }
        }

        func renderTile(request: TileRequest, isCancelled: () -> Bool) -> Data? {
            owner?.renderTile(request: request, content: content, isCancelled: isCancelled)
        }
    }

    public func renderTile(request: TileRequest) -> Data? {
        renderTile(request: request, content: .full) { false }
    }

    public func renderTile(request: TileRequest, isCancelled: () -> Bool) -> Data? {
        renderTile(request: request, content: .full, isCancelled: isCancelled)
    }

    /**
     Draws one tile.

     - Parameter isCancelled: asked repeatedly while the tile is being drawn.
       A map that has moved on is not owed this tile, and the render slots are
       shared, so a doomed tile is drawn *instead of* one still on screen.
     */
    public func renderTile(
        request: TileRequest,
        content: Content = .full,
        isCancelled: () -> Bool = { false }
    ) -> Data? {
        // Dispatch workers otherwise retain temporary Foundation/Metal objects
        // until their outer pool drains, potentially across many panned tiles.
        autoreleasepool {
            renderTileInPool(request: request, content: content, isCancelled: isCancelled)
        }
    }

    private func renderTileInPool(
        request: TileRequest, content: Content, isCancelled: () -> Bool
    ) -> Data? {
        if isClosed || isCancelled() { return nil }

        let sourceZoomOffset = request.sourceZoomOffset
        // Changes only when a revalidation found new bytes upstream; tiles
        // drawn from the old ones then stop being found.
        let sourceRevision = sourceDiskCache?.revision ?? 0
        let z = UInt8(clamping: request.z)
        let x = UInt32(clamping: request.x)
        let y = UInt32(clamping: request.y)

        /*
         Two keys, because a rendered tile goes stale for one reason only: it
         was drawn while some of its glyphs were still on the way. A tile that
         had them all is finished, and no later arrival can change it, so it is
         stored under a key with no generation in it and survives every
         handover. Only the ones drawn short are tied to the generation they
         were drawn at, and only they are redrawn.
         */
        func key(_ generation: String) -> String {
            Digest.hex(
                styleKey,
                "\(tileSize)",
                // A build that draws more than the one that filled this cache
                // must not serve its tiles.
                "v\(VectorTileRenderer.outputVersion)",
                generation,
                "scale\(renderScale)",
                "sourceOffset\(sourceZoomOffset)",
                "sources\(sourceRevision)",
                "\(request.z)/\(request.x)/\(request.y)"
            )
        }
        let completeKey: String? = diskCache.map { _ in
            switch content {
            // The ground has no glyphs to be short of; one key, forever.
            case .ground: return key("ground")
            case .labels: return key("labels-complete")
            case .full: return key("complete")
            }
        }
        let provisionalKey: String? = diskCache.flatMap { _ in
            switch content {
            case .ground: return nil
            case .labels: return key("labels-g\(glyphGeneration.value)")
            case .full: return key("g\(glyphGeneration.value)")
            }
        }
        for candidate in [completeKey, provisionalKey].compactMap({ $0 }) {
            if let hit = diskCache?.get(candidate) { return hit }
        }

        guard let plan = plan(z: z, x: x, y: y, sourceZoomOffset: sourceZoomOffset) else { return nil }

        // The ground never draws a label, so the ring of neighbours the label
        // pass asks for is nine fetches it has no use for. The entries stay in
        // place — the native side reads the plan positionally — but hold
        // nothing.
        let wanted = plan.map { entry in
            content == .ground && (entry["labelsOnly"] as? Bool ?? false)
                ? nil
                : (entry["url"] as? String).flatMap(VectorTileProvider.parse)
        }
        let priorities = plan.enumerated().map { index, entry in
            ((entry["labelsOnly"] as? Bool ?? false) ? 1_000 : 0) + index
        }
        let fetchStarted = DispatchTime.now()
        let fetchStats = SourceFetchStats()
        guard let tiles = fetchAll(wanted, priorities: priorities, stats: fetchStats, isCancelled: isCancelled) else {
            VectorTileProvider.debug(
                "tile \(request.z)/\(request.x)/\(request.y) content=\(content) cancelled "
                    + "mvt(mem=\(fetchStats.memoryHits) disk=\(fetchStats.diskHits) "
                    + "net=\(fetchStats.networkFetches) shared=\(fetchStats.sharedWaits) "
                    + "blocked=\(fetchStats.blocked) cancel=\(fetchStats.cancelled) "
                    + "queue=\(fetchStats.queueWaitMs)ms)"
            )
            return nil
        }
        let fetchMs = VectorTileProvider.millis(since: fetchStarted)
        if isClosed || isCancelled() { return nil }

        // Cached ranges belong in the store before the tile is judged to be
        // missing them; otherwise the first tile of a launch is drawn bare and
        // handed over a moment later for no reason. The ground has no glyphs,
        // so it never waits.
        let short = content != .ground && requestGlyphs(z: z, x: x, y: y, tiles: tiles, sourceZoomOffset: sourceZoomOffset)
        if short {
            lock.lock()
            provisionalSinceHandover = true
            lock.unlock()
        }
        if isClosed || isCancelled() { return nil }

        let renderStarted = DispatchTime.now()
        let png: Data?
        let groundKey = key("ground")
        // A tile drawn while the network was off and a source was not in the
        // package is missing ink, like one drawn short of glyphs -- but
        // nothing will arrive to redraw it, so it must not be kept at all.
        // So is one drawn before slow sources arrived; those are on their way
        // and `onSourcesLoaded` brings the finished tile.
        let drawnShortOfSources = fetchStats.blocked > 0 || fetchStats.pending > 0
        lock.lock()
        let keep = !drawnShortOfSources && !replacedSources
        lock.unlock()
        switch content {
        case .labels:
            png = renderLabelTile(z: z, x: x, y: y, tiles: tiles, sourceZoomOffset: sourceZoomOffset)
        case .full:
            if let ground = groundRasters.value(for: groundKey) {
                png = compositeLabels(onto: ground as Data, z: z, x: x, y: y, tiles: tiles, sourceZoomOffset: sourceZoomOffset)
            } else {
                png = renderGroundTile(
                    z: z, x: x, y: y, tiles: tiles, content: content,
                    cacheGroundAs: keep ? groundKey : nil, isCancelled: isCancelled,
                    sourceZoomOffset: sourceZoomOffset
                )
            }
        case .ground:
            png = renderGroundTile(z: z, x: x, y: y, tiles: tiles, content: content, cacheGroundAs: nil, isCancelled: isCancelled, sourceZoomOffset: sourceZoomOffset)
        }
        let renderMs = VectorTileProvider.millis(since: renderStarted)

        // Timed separately, and logged: when a tile is slow the answer is
        // almost always one or the other, and guessing which wastes an
        // afternoon. android-sdk's provider prints the same line; iOS had no
        // measurement at all, which is why "it feels slow" had nowhere to go.
        VectorTileProvider.debug(
            "tile \(request.z)/\(request.x)/\(request.y) content=\(content) "
                + "sources=\(wanted.compactMap { $0 }.count) planned=\(plan.count) "
                + "fetch=\(fetchMs)ms render=\(renderMs)ms bytes=\(png?.count ?? 0) "
                + "mvt(mem=\(fetchStats.memoryHits) disk=\(fetchStats.diskHits) "
                + "net=\(fetchStats.networkFetches) shared=\(fetchStats.sharedWaits) "
                + "blocked=\(fetchStats.blocked) pending=\(fetchStats.pending) "
                + "cancel=\(fetchStats.cancelled) queue=\(fetchStats.queueWaitMs)ms)"
        )

        if let png, keep, let store = short ? provisionalKey : completeKey {
            diskCache?.put(store, png)
        }
        return png
    }

    private static func millis(since start: DispatchTime) -> Int {
        Int((DispatchTime.now().uptimeNanoseconds &- start.uptimeNanoseconds) / 1_000_000)
    }

    /// On only when the app asks for it, so a release build prints nothing.
    private static let debugLogging =
        ProcessInfo.processInfo.environment["MAPCONDUCTOR_DEBUG_LOG"] == "1"

    private static func debug(_ message: @autoclosure () -> String) {
        guard debugLogging else { return }
        NSLog("[MapConductor][VectorTile] %@", message())
    }

    private func plan(z: UInt8, x: UInt32, y: UInt32, sourceZoomOffset: Int) -> [[String: Any]]? {
        guard let json = try? renderer.plan(z: z, x: x, y: y, sourceZoomOffset: sourceZoomOffset),
              let data = json.data(using: .utf8),
              let plan = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return nil }
        return plan
    }

    /// The ground, on the GPU when there is one and the style paints nothing
    /// it cannot draw. `.full` composites labels over the readback.
    private func renderGroundTile(
        z: UInt8, x: UInt32, y: UInt32, tiles: [Data?], content: Content, cacheGroundAs: String?,
        isCancelled: () -> Bool,
        sourceZoomOffset: Int
    ) -> Data? {
        // Patterns need the CPU; very complex polygons also use its scanline
        // renderer to avoid expensive GPU triangulation.
        let onGpu = gpu != nil && !((try? renderer.needsCPU(z: z, tiles: tiles, sourceZoomOffset: sourceZoomOffset)) ?? false)
        if onGpu, let png = renderOnGpu(
            z: z, x: x, y: y, tiles: tiles, withLabels: content == .full, cacheGroundAs: cacheGroundAs,
            isCancelled: isCancelled,
            sourceZoomOffset: sourceZoomOffset
        ) {
            gpuRenderCount.increment()
            return png
        }
        if isClosed || isCancelled() { return nil }
        if onGpu { gpuFallbackCount.increment() }

        // A GPU failure must not lose the tile; the CPU can always draw it.
        Self.debug("cpu tile \(z)/\(x)/\(y)")
        let resolution = UInt32(tileSize * renderScale)
        if content == .ground {
            return try? renderer.renderGeometry(
                z: z, x: x, y: y, tileSize: resolution, tiles: tiles,
                sourceZoomOffset: sourceZoomOffset
            )
        }
        return try? renderer.render(z: z, x: x, y: y, tileSize: resolution, tiles: tiles, sourceZoomOffset: sourceZoomOffset)
    }

    /**
     The labels and icons on a transparent ground.

     Pure CPU: no GPU pass, no readback — which is the point of serving them as
     their own layer. Several of these run while the GPU draws the ground.
     */
    private func renderLabelTile(z: UInt8, x: UInt32, y: UInt32, tiles: [Data?], sourceZoomOffset: Int) -> Data? {
        // The larger of the two, not the product: `labelResolution` exists to
        // reach device resolution on a 1x screen, and `renderScale` already
        // does that on a Retina one. Multiplying them would draw sixteen times
        // the pixels for no visible gain.
        let resolution = tileSize * max(renderScale, VectorTileProvider.labelResolution)
        guard let drawn = try? renderer.renderLabels(
            z: z, x: x, y: y, tileSize: UInt32(resolution), tiles: tiles,
            sourceZoomOffset: sourceZoomOffset
        ) else { return nil }

        // A tile with nothing on it is common — water, fields — and one shared
        // transparent PNG serves them all.
        guard drawn.placed > 0, !drawn.pixels.isEmpty else { return emptyLabelTile }

        var pixels = drawn.pixels
        return pixels.withUnsafeMutableBytes { buffer -> Data? in
            guard let base = buffer.baseAddress else { return nil }
            // Premultiplied: the label pass writes colour already scaled by
            // alpha, and encoding that as straight alpha turns every halo grey.
            return TilePngEncoder.encode(
                rgba: base, width: resolution, height: resolution, premultiplied: true
            )
        }
    }

    /// One transparent tile, encoded once and served for every empty one.
    private lazy var emptyLabelTile: Data? = {
        var pixels = [UInt8](repeating: 0, count: 4)
        return pixels.withUnsafeMutableBytes { buffer -> Data? in
            guard let base = buffer.baseAddress else { return nil }
            return TilePngEncoder.encode(rgba: base, width: 1, height: 1, premultiplied: true)
        }
    }()

    private func renderOnGpu(
        z: UInt8, x: UInt32, y: UInt32, tiles: [Data?], withLabels: Bool, cacheGroundAs: String?,
        isCancelled: () -> Bool,
        sourceZoomOffset: Int
    ) -> Data? {
        guard !isClosed, !isCancelled(), let gpu else { return nil }
        do {
            let started = DispatchTime.now()
            let tessellated = try renderer.tessellate(
                z: z, x: x, y: y, tileSize: UInt32(tileSize * renderScale), tiles: tiles,
                sourceZoomOffset: sourceZoomOffset
            )
            let tessellateMs = Self.millis(since: started)
            guard !isClosed, !isCancelled() else { return nil }
            let drawStarted = DispatchTime.now()
            // The tessellator draws fills and lines; labels come from a
            // distance field per glyph and are painted over the readback.
            // Without this the GPU path loses every label the style asked for.
            let png = gpu.renderPng(tessellated, isCancelled: isCancelled) { readback in
                if let cacheGroundAs, let base = readback.baseAddress {
                    self.groundRasters.insert(
                        Data(bytes: base, count: readback.count),
                        for: cacheGroundAs, cost: readback.count + cacheGroundAs.utf8.count + 128
                    )
                }
                guard withLabels else { return }
                _ = try? self.renderer.drawLabels(
                    z: z, x: x, y: y, tileSize: UInt32(self.tileSize * self.renderScale),
                    rgba: readback, tiles: tiles,
                    sourceZoomOffset: sourceZoomOffset
                )
            }
            Self.debug("gpu tile \(z)/\(x)/\(y) tessellate=\(tessellateMs)ms draw=\(Self.millis(since: drawStarted))ms")
            return png
        } catch {
            return nil
        }
    }

    /// Labels over a ground the GPU drew earlier: the whole cost of a glyph
    /// generation arriving, once the ground is in ``groundRasters``.
    private func compositeLabels(
        onto ground: Data, z: UInt8, x: UInt32, y: UInt32, tiles: [Data?],
        sourceZoomOffset: Int
    ) -> Data? {
        let resolution = tileSize * renderScale
        guard ground.count == resolution * resolution * 4 else { return nil }
        var rgba = ground
        guard (try? renderer.drawLabels(
            z: z, x: x, y: y, tileSize: UInt32(resolution), rgba: &rgba, tiles: tiles,
            sourceZoomOffset: sourceZoomOffset
        )) != nil else { return nil }
        return rgba.withUnsafeMutableBytes { buffer -> Data? in
            guard let base = buffer.baseAddress else { return nil }
            return TilePngEncoder.encode(
                rgba: base, width: resolution, height: resolution, premultiplied: false
            )
        }
    }

    // MARK: - Fetching

    /// Fetches a whole plan at once.
    ///
    /// A plan is nine tiles once the label pass asks for its neighbours, and
    /// nine round trips one after another is a second of waiting for what takes
    /// a fraction of it in parallel.
    private func fetchAll(
        _ urls: [URL?],
        priorities: [Int],
        stats: SourceFetchStats,
        isCancelled: () -> Bool
    ) -> [Data?]? {
        let results = FetchResults(count: urls.count)
        let group = DispatchGroup()
        let cancellation = FetchCancellation()
        // Left running when the tile is drawn without them: the bytes are
        // still wanted, by the redraw `onSourcesLoaded` asks for.
        var detached = false
        defer { if !detached { cancellation.cancel() } }

        // Sorted on this thread, before anything is dispatched: what is
        // already here, what somebody else is fetching, and what is ours to
        // fetch. Only the last kind goes to the queue and takes a slot.
        //
        // It used to take a slot for every entry, on this thread, before it
        // knew which kind it was -- and a tile whose source somebody else was
        // already fetching then sat in the queue holding a slot just to wait
        // for them. On a zoom out every source is new and sixteen children
        // want each one, so the sixteen slots filled with waiters and the
        // fetches that would have released them could not start. That is the
        // 1.3 s per request the simulator showed while nothing moved.
        var waitingOn: [(index: Int, key: String, latch: SourceFetchFlight)] = []

        let ordered = urls.indices.sorted { left, right in
            (priorities.indices.contains(left) ? priorities[left] : left)
                < (priorities.indices.contains(right) ? priorities[right] : right)
        }

        for index in ordered {
            guard let url = urls[index] else { continue }
            let key = url.absoluteString
            // What is on the device is read here, before anything can queue
            // behind the network. An offline package used to be read inside
            // the fetch slot, so on a Wi-Fi that answered nothing the tiles it
            // held waited out other tiles' fifteen-second timeouts and the
            // downloaded area stayed white.
            if let local = localBytes(url, key: key, stats: stats) {
                results.set(local, at: index)
                continue
            }
            lock.lock()
            if empties.value(for: key) == true {
                lock.unlock()
                results.set(nil, at: index)
                continue
            }
            if let latch = inFlight[key] {
                lock.unlock()
                stats.sharedWaits += 1
                waitingOn.append((index, key, latch))
                continue
            }
            let done = SourceFetchFlight()
            inFlight[key] = done
            lock.unlock()

            group.enter()
            fetchQueue.async {
                defer { group.leave() }
                results.set(
                    self.transfer(url, key: key, done: done, cancellation: cancellation, stats: stats),
                    at: index
                )
            }
        }

        // Everything below waits on the network, and no longer than this.
        let deadline = DispatchTime.now() + sourceGrace
        func expired() -> Bool { DispatchTime.now() >= deadline }

        while group.wait(timeout: .now() + .milliseconds(10)) != .success {
            if isClosed || isCancelled() {
                cancellation.cancel()
                stats.cancelled += 1
                return nil
            }
            if expired() { break }
        }

        // The ones somebody else was fetching ran alongside ours; waiting for
        // them now costs the slowest of them, not the sum, and holds nothing
        // but this thread.
        waiting: for (index, key, originalLatch) in waitingOn {
            var latch = originalLatch
            while true {
                while latch.wait(timeout: .now() + .milliseconds(10)) != .success {
                    if isClosed || isCancelled() { return nil }
                    if expired() { continue waiting }
                }
                latch.signal()
                // An oversized or evicted tile still belongs to all requests
                // already sharing this flight. Keep it only until they finish.
                let response = latch.response
                if response.received {
                    results.set(response.bytes, at: index)
                    break
                }
                if let bytes = cache.value(for: key) {
                    stats.memoryHits += 1
                    results.set(bytes as Data, at: index)
                    break
                }
                if let bytes = sourceDiskCache?.get(key) {
                    stats.diskHits += 1
                    cache.insert(bytes, for: key, cost: bytes.count + key.utf8.count + 128)
                    results.set(bytes, at: index)
                    break
                }
                if latch.isUnavailable {
                    stats.blocked += 1
                    results.set(nil, at: index)
                    break
                }
                // They were cancelled before the bytes landed, and a tile left
                // without its source is drawn blank and cached that way. Fetch it
                // ourselves rather than inherit their bad luck.
                lock.lock()
                if empties.value(for: key) == true {
                    lock.unlock()
                    results.set(nil, at: index)
                    break
                }
                let done = inFlight[key] ?? SourceFetchFlight()
                let ours = inFlight[key] == nil
                if ours { inFlight[key] = done }
                lock.unlock()
                if ours, let url = urls[index] {
                    // A cancelled owner may leave several waiters. The retry
                    // must stay cancellable too, rather than blocking this render
                    // thread in a synchronous 15-second transfer.
                    let retry = DispatchGroup()
                    retry.enter()
                    fetchQueue.async {
                        defer { retry.leave() }
                        results.set(self.transfer(
                            url, key: key, done: done, cancellation: cancellation, stats: stats
                        ), at: index)
                    }
                    while retry.wait(timeout: .now() + .milliseconds(10)) != .success {
                        if isClosed || isCancelled() { return nil }
                        if expired() { continue waiting }
                    }
                    break
                }
                // Another waiter claimed the retry. Wait for its actual result;
                // returning now would draw and cache a tile without that source.
                latch = done
            }
        }

        // Whatever has not answered by now is drawn without. The tile is not
        // kept, and its sources -- still coming -- bring the redraw.
        let snapshot = results.snapshot()
        let pending = urls.indices.compactMap { index -> String? in
            guard let url = urls[index], !snapshot.settled[index] else { return nil }
            return url.absoluteString
        }
        if !pending.isEmpty {
            detached = true
            stats.pending += pending.count
            lock.lock()
            drawnWithout.formUnion(pending)
            lock.unlock()
            // A source that landed between the deadline and the line above
            // found nobody waiting for it; it is owed the handover all the same.
            for key in pending where cache.value(for: key) != nil { sourceLanded(key) }
        }
        return snapshot.values
    }

    /**
     A source tile already on the device: in memory, in the offline package,
     or in the disk cache. Never waits for the network; a stale disk entry is
     served as it is and checked in the background.
     */
    private func localBytes(_ url: URL, key: String, stats: SourceFetchStats) -> Data? {
        if let cached = cache.value(for: key) {
            stats.memoryHits += 1
            return cached as Data
        }
        if let local = localTile?(url), !local.isEmpty {
            stats.localHits += 1
            cache.insert(local, for: key, cost: local.count + key.utf8.count + 128)
            return local
        }
        if let entry = sourceDiskCache?.entry(key) {
            stats.diskHits += 1
            cache.insert(entry.bytes, for: key, cost: entry.bytes.count + key.utf8.count + 128)
            if entry.meta?.isStale() ?? true { revalidate(url, key: key, cached: entry.bytes, meta: entry.meta) }
            return entry.bytes
        }
        return nil
    }

    /// The transfer itself, and nothing else under a fetch slot.
    ///
    /// Caller has already claimed `key` in `inFlight` with `done`; this
    /// releases both however it ends, so waiters always wake.
    private func transfer(
        _ url: URL,
        key: String,
        done: SourceFetchFlight,
        cancellation: FetchCancellation,
        stats: SourceFetchStats
    ) -> Data? {
        defer {
            lock.lock()
            inFlight.removeValue(forKey: key)
            lock.unlock()
            done.signal()
        }
        let queued = DispatchTime.now()
        while fetchSlots.wait(timeout: .now() + .milliseconds(10)) != .success {
            if cancellation.isCancelled || isClosed { return nil }
        }
        stats.queueWaitMs += VectorTileProvider.millis(since: queued)
        defer { fetchSlots.signal() }
        if cancellation.isCancelled || isClosed { return nil }

        // Somebody may have finished it while this one queued for a slot.
        if let cached = cache.value(for: key) {
            stats.memoryHits += 1
            done.publish(cached)
            return cached as Data
        }
        stats.networkFetches += 1
        let response: SourceResponse
        do {
            response = try sourceFetch(url, cancellation)
        } catch {
            if cancellation.isCancelled { return nil }
            // Not an answer: the network is off, or failed. Either way the
            // tile is drawn without this source and not kept, so it is
            // drawn again once the network is back -- and the URL is not
            // remembered as empty.
            stats.blocked += 1
            done.markUnavailable()
            lock.lock()
            drawnWithout.remove(key)
            lock.unlock()
            return nil
        }
        guard let bytes = response.body, !bytes.isEmpty else {
            if cancellation.isCancelled { return nil }
            // Only a genuine "no tile" answer is remembered. A transient
            // failure blacklisted here would leave a hole for the session.
            lock.lock()
            empties.insert(true, for: key, cost: key.utf8.count + 128)
            drawnWithout.remove(key)
            lock.unlock()
            done.publish(nil)
            return nil
        }
        cache.insert(bytes, for: key, cost: bytes.count + key.utf8.count + 128)
        sourceDiskCache?.put(key, bytes, meta: SourceTileDiskCache.Meta(
            etag: response.validators.etag,
            lastModified: response.validators.lastModified,
            fetchedAt: Date().timeIntervalSince1970,
            maxAge: response.maxAge
        ))
        done.publish(bytes)
        sourceLanded(key)
        return bytes
    }

    /// A source arrived. Worth a handover only if a tile was drawn without it.
    private func sourceLanded(_ key: String) {
        lock.lock()
        let wanted = drawnWithout.remove(key) != nil
        lock.unlock()
        if wanted { sourcesArrived() }
    }

    /**
     Checks a stale cached tile with the server, off every path a tile waits on.

     The cached bytes have already been drawn; this only decides whether they
     are still right. A 304 refreshes the sidecar, new bytes replace the old
     ones and redraw the map, and any failure -- offline included -- leaves the
     cache exactly as it was.
     */
    private func revalidate(_ url: URL, key: String, cached: Data, meta: SourceTileDiskCache.Meta?) {
        guard let revalidator, let disk = sourceDiskCache else { return }
        lock.lock()
        let mine = !closed && revalidating.insert(key).inserted
        lock.unlock()
        guard mine else { return }
        revalidationQueue.addOperation { [weak self] in
            defer {
                self?.lock.lock()
                self?.revalidating.remove(key)
                self?.lock.unlock()
            }
            guard let self, !self.isClosed,
                  let response = try? revalidator(url, meta?.validators ?? .none)
            else { return }
            let now = Date().timeIntervalSince1970
            if response.notModified {
                disk.confirm(key, meta: SourceTileDiskCache.Meta(
                    etag: response.validators.etag ?? meta?.etag,
                    lastModified: response.validators.lastModified ?? meta?.lastModified,
                    fetchedAt: now,
                    maxAge: response.maxAge ?? meta?.maxAge
                ))
                return
            }
            guard (200..<300).contains(response.status), let bytes = response.body, !bytes.isEmpty else { return }
            let fresh = SourceTileDiskCache.Meta(
                etag: response.validators.etag,
                lastModified: response.validators.lastModified,
                fetchedAt: now,
                maxAge: response.maxAge
            )
            // A server without validators answers 200 every time; the same
            // bytes again are a confirmation, not a change.
            guard bytes != cached else {
                disk.confirm(key, meta: fresh)
                return
            }
            disk.put(key, bytes, meta: fresh)
            self.cache.insert(bytes, for: key, cost: bytes.count + key.utf8.count + 128)
            self.lock.lock()
            self.replacedSources = true
            self.lock.unlock()
            disk.bumpRevision()
            self.sourcesArrived()
        }
    }

    /// Coalesced like the glyph handover: a viewport's sources land together.
    private func sourcesArrived() {
        lock.lock()
        if sourcesNotifyPending || closed {
            lock.unlock()
            return
        }
        sourcesNotifyPending = true
        lock.unlock()
        fetchQueue.asyncAfter(deadline: .now() + notifyQuiet) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            self.sourcesNotifyPending = false
            let notify = self.closed ? nil : self.onSourcesLoaded
            self.lock.unlock()
            notify?()
        }
    }

    // MARK: - Glyphs and sprites

    /// Loads whatever is already on disk, and fetches the sprite if it is not.
    private func warmStyleAssets() {
        guard !isClosed else { return }
        fetchQueue.async { [weak self] in
            guard let self, !self.isClosed else { return }
            if let cache = self.glyphCache, (try? self.renderer.glyphsURLTemplate()) != nil {
                // Nothing to enumerate a disk cache by — the ranges a style
                // wants are only known per tile — so warming happens as tiles
                // ask, through `requestGlyphs`, which checks the cache first.
                _ = cache
            }
            self.loadSprite()
        }
    }

    /// Fetches the sprite sheet, preferring what is already on disk.
    ///
    /// Icons are drawn only where the sheet has them, so until this lands a
    /// style's shields and pins are simply absent.
    private func loadSprite() {
        guard (try? renderer.needsSprite()) == true,
              let urls = try? renderer.spriteURLs(pixelRatio: 2)
        else { return }

        let index = spriteCache?.get(urls.json) ?? fetched(urls.json, into: spriteCache)
        let sheet = spriteCache?.get(urls.png) ?? fetched(urls.png, into: spriteCache)
        guard let index, let sheet,
              let text = String(data: index, encoding: .utf8), !isClosed
        else { return }
        _ = try? renderer.addSprite(json: text, png: sheet)
    }

    private func fetched(_ url: String, into store: StyleAssetCache?) -> Data? {
        guard let target = VectorTileProvider.parse(url) else { return nil }
        do {
            guard let bytes = try fetchTile(target, nil), !bytes.isEmpty else { return nil }
            store?.put(url, bytes)
            return bytes
        } catch is OfflineUnavailableError {
            // Unclaimed: the range is there to be had once the network is
            // back. Harmless for the sprite, which is not claimed.
            lock.lock()
            requestedGlyphs.remove(url)
            lock.unlock()
            return nil
        } catch {
            return nil
        }
    }

    /**
     Builds a URL from a string a style produced.

     `URL(string:)` returns nil for a space, and glyph URLs are full of them:
     the template's `{fontstack}` is a font *name*, so a MapLibre style asks for
     `.../font/Open Sans Semibold/0-255.pbf`. Rejecting those silently meant no
     glyph was ever fetched and no label was ever drawn, with nothing logged --
     the tiles came out bare and looked like a renderer that could not draw
     text.
     */
    static func parse(_ url: String) -> URL? {
        if let direct = URL(string: url) { return direct }
        return url
            .addingPercentEncoding(withAllowedCharacters: .urlFragmentAllowed)
            .flatMap(URL.init(string:))
    }

    /**
     Starts fetching the glyph ranges this tile's labels need, and returns.

     Waiting is the obvious thing and the wrong one: a range is a 0.4-1.3 second
     round trip, a tile at low zoom wants dozens, and waiting means the tile
     draws nothing until the last one lands. The tile is drawn with whatever is
     already in, exactly as MapLibre does it, and the ranges that arrive later
     bring their labels with them through `onGlyphsLoaded`.

     - Returns: whether this tile is about to be drawn without glyphs it wants,
       which is what decides whether a handover is worth making.
     */
    @discardableResult
    private func requestGlyphs(z: UInt8, x: UInt32, y: UInt32, tiles: [Data?], sourceZoomOffset: Int) -> Bool {
        if glyphsUnavailable || isClosed { return false }
        let needed = (try? renderer.neededGlyphs(z: z, x: x, y: y, tiles: tiles, sourceZoomOffset: sourceZoomOffset)) ?? []
        if needed.isEmpty { return false }

        lock.lock()
        let mine = needed.filter { requestedGlyphs.insert($0).inserted }
        lock.unlock()
        if mine.isEmpty { return true }

        for url in mine {
            glyphsInFlight.increment()
            // Admission belongs on the asset queue. Acquiring a seventh
            // permit on the render thread holds a server slot for an entire
            // network round trip (or timeout) before any pixels are returned.
            glyphQueue.addOperation {
                defer { self.glyphsInFlight.decrement() }
                guard !self.isClosed else { return }
                // Ranges never change, so a hit here is kept without expiry.
                let bytes = self.glyphCache?.get(url) ?? self.fetched(url, into: self.glyphCache)
                guard let bytes, !bytes.isEmpty else { return }
                // Left in the claimed set on failure: a range the server does
                // not have will not appear on a retry.
                if let added = try? self.renderer.addGlyphs(bytes), added > 0 {
                    self.glyphsArrived()
                }
            }
        }
        return true
    }

    /**
     Tells the host that tiles drawn before now are missing labels.

     Coalesced: a viewport's worth of ranges lands in a burst, and asking the
     map to refetch on each one would redraw everything dozens of times for the
     same result. The generation is what makes the already-rendered tiles stale
     — without it the map's own cache keeps serving the unlabelled ones.
     */
    private func glyphsArrived() {
        glyphGeneration.increment()

        lock.lock()
        if notifyPending {
            lock.unlock()
            return
        }
        notifyPending = true
        lock.unlock()

        let deadline = Date().addingTimeInterval(notifyMaxWait)
        scheduleNotify(deadline: deadline)
    }

    private func scheduleNotify(deadline: Date) {
        fetchQueue.asyncAfter(deadline: .now() + notifyQuiet) { [weak self] in
            guard let self, !self.isClosed else { return }
            // Hold the window open while ranges are still coming, or the whole
            // viewport hands over once per range instead of once.
            if self.glyphsInFlight.value > 0, Date() < deadline {
                self.scheduleNotify(deadline: deadline)
                return
            }

            self.lock.lock()
            self.notifyPending = false
            // Nothing to hand over to: every tile drawn since the last one had
            // all the glyphs it wanted, so what is on screen is already the
            // finished picture.
            let worthIt = self.provisionalSinceHandover
            self.provisionalSinceHandover = false
            self.lock.unlock()

            if worthIt { self.onGlyphsLoaded?() }
        }
    }

    /// Synchronous by design: `renderTile` is already called off the main
    /// thread by the tile server, and the plan/render contract is positional.
    /// Like ``get(_:headers:cancellation:)``, but a transport failure -- no
    /// network, a refused connection, a 5xx -- is thrown rather than turned
    /// into "no such tile", which the caller would remember for the session.
    static func fetch(
        _ url: URL,
        headers: [String: String],
        cancellation: FetchCancellation?
    ) throws -> Data? {
        var request = URLRequest(url: url, timeoutInterval: 15)
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        let done = DispatchSemaphore(value: 0)
        var payload: Data?
        var failure: Error?
        var status = 0
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            defer { done.signal() }
            failure = error
            guard let http = response as? HTTPURLResponse else { return }
            status = http.statusCode
            if (200..<300).contains(http.statusCode) { payload = data }
        }
        task.resume()
        while done.wait(timeout: .now() + .milliseconds(10)) != .success {
            if cancellation?.isCancelled == true {
                task.cancel()
                done.wait()
                return nil
            }
        }
        if let failure { throw failure }
        // A missing tile is normal at the edge of a source's coverage; a
        // server that is down is not.
        if status >= 500 { throw URLError(.badServerResponse) }
        return payload
    }

    static func get(
        _ url: URL,
        headers: [String: String],
        cancellation: FetchCancellation?
    ) -> Data? {
        var request = URLRequest(url: url, timeoutInterval: 15)
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }

        var payload: Data?
        let done = DispatchSemaphore(value: 0)
        let task = URLSession.shared.dataTask(with: request) { data, response, _ in
            defer { done.signal() }
            guard let http = response as? HTTPURLResponse else { return }
            // A missing tile is normal at the edge of a source's coverage.
            guard (200..<300).contains(http.statusCode) else { return }
            payload = data
        }
        task.resume()
        while done.wait(timeout: .now() + .milliseconds(10)) != .success {
            if cancellation?.isCancelled == true {
                task.cancel()
                // The completion handler signals `done`; waiting for it keeps
                // the captured payload alive until URLSession is finished.
                done.wait()
                return nil
            }
        }
        return payload
    }
}

/// Waiters share transport failures as well as bytes. Retrying a failed
/// owner once per waiter would multiply a network timeout across the viewport.
private final class SourceFetchFlight: @unchecked Sendable {
    private let done = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var unavailable = false
    private var received = false
    private var bytes: Data?

    func publish(_ bytes: Data?) {
        lock.lock(); defer { lock.unlock() }
        self.bytes = bytes
        received = true
    }

    var response: (received: Bool, bytes: Data?) {
        lock.lock(); defer { lock.unlock() }
        return (received, bytes)
    }

    func wait(timeout: DispatchTime) -> DispatchTimeoutResult { done.wait(timeout: timeout) }
    func signal() { done.signal() }

    func markUnavailable() {
        lock.lock(); unavailable = true; lock.unlock()
    }

    var isUnavailable: Bool {
        lock.lock(); defer { lock.unlock() }
        return unavailable
    }
}

final class FetchCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }
}

private final class FetchResults: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Data?]
    /// Answered, even if the answer was "nothing": what is not settled is
    /// still on its way.
    private var settled: [Bool]

    init(count: Int) {
        values = [Data?](repeating: nil, count: count)
        settled = [Bool](repeating: false, count: count)
    }

    func set(_ value: Data?, at index: Int) {
        lock.lock()
        values[index] = value
        settled[index] = true
        lock.unlock()
    }

    func snapshot() -> (values: [Data?], settled: [Bool]) {
        lock.lock()
        defer { lock.unlock() }
        return (values, settled)
    }
}

/// A counter that is safe to read from whichever thread the tile server used.
private final class Counter {
    private var count = 0
    private let lock = NSLock()

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func increment() {
        lock.lock()
        count += 1
        lock.unlock()
    }

    func decrement() {
        lock.lock()
        count -= 1
        lock.unlock()
    }
}

/**
 The document as it will be drawn: `styleJSON` with `rules` applied.

 Applied to the document rather than sent as per-layer deltas: both ways a
 style reaches a map from here -- rasterised tiles and the handover to a map
 that draws styles itself -- take a document, so applying it once means both
 see the same one. A rule that hides a role switches those layers off in the
 document, and a layer that is not there is not drawn.

 Without rules the author's own text comes back and the compiler never runs.
 */
func drawnStyleJSON(_ styleJSON: String, _ rules: StyleRules) throws -> String {
    guard rules != .none else { return styleJSON }
    return try VectorStyleRules.compile(styleJSON: styleJSON, rulesJSON: rules.json).styleJSON
}

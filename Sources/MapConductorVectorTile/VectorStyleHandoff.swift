import Foundation
import MapConductorCore
import MapConductorVectorStyle

/**
 Hands a style to a map that draws vector styles itself.

 The style document is served by the SDK's local tile server and the map is
 told, through its ``VectorStyleSupport``, to make it its basemap. Nothing is
 rasterised: on MapLibre, Mapbox and MapTiler this is the whole of what
 "vector tile layer as basemap" has to do, and it is the path an offline
 package will take. A map without the capability cannot use this; it gets
 ``VectorTileProvider`` and raster layers instead.

 The document id is the content and nothing else. The map only re-reads a
 style whose URL changed, so a changed style has to change the URL; but a
 host that rebuilds its overlay content on a design change hands the style
 over again from the rebuilt content, and if that minted a fresh URL it would
 be a fresh design, another rebuild, and so on. Same content, same URL, and
 the second handoff is a no-op.
 */
public final class VectorStyleHandoff {
    public let documentId: String
    public let url: String
    /// The credits the style's sources ask for, in style order, without duplicates.
    public let attributions: [String]
    public let support: VectorStyleSupport

    /**
     What the caller should know about the rules it passed: a rule that
     matched nothing, a rules document that could not be read, or an offline
     package, whose style is served from the package and so is handed over
     as it was packaged.

     Empty whenever there were no rules. Nothing here stops the map being
     drawn -- a bad rule costs the adjustment, never the style.
     */
    public let diagnostics: [String]

    private let server: LocalTileServer
    private var disposed = false

    /// The files route serving the package, when there is one.
    public let filesRoute: String?

    /**
     - Parameters:
       - offlinePackage: a package the map reads its tiles, glyphs and sprite
         from through the local server, which answers from the package and --
         while `online` -- fetches upstream for the rest. The route carries the
         mode: a map remembers a tile it was told does not exist, so going back
         online has to change the URLs to be noticed.
       - headers: sent with upstream fetches made for the package
       - sourceCacheDirectory: where tiles fetched around the package are
         kept, so the map is answered from the device first and the network
         is only asked -- in the background, with `If-None-Match` /
         `If-Modified-Since` -- whether they changed. Nil fetches every time.
     */
    public init(
        styleJSON: String,
        /// Adjustments applied before the map is given the document. The
        /// same ``StyleRules`` ``VectorTileProvider`` takes, so the style
        /// looks the same whether the map draws it or this rasterises it.
        /// A rules document the compiler rejects is reported through
        /// ``diagnostics`` and the author's style is handed over unchanged --
        /// a bad rule must not cost the map.
        rules: StyleRules = .none,
        support: VectorStyleSupport,
        server: LocalTileServer = TileServerRegistry.get(),
        offlinePackage: OfflinePackage? = nil,
        online: Bool = true,
        headers: [String: String] = [:],
        sourceCacheDirectory: URL? = nil,
        sourceCacheBytes: Int = 128 * 1024 * 1024
    ) {
        self.support = support
        self.server = server
        var notes: [String] = []
        // Applied to the document, as in `VectorTileProvider` -- a map that
        // draws the style itself is given the adjusted one, not the author's
        // with a list of corrections.
        var adjusted = styleJSON
        if rules != .none {
            do {
                adjusted = try VectorStyleRules.compile(styleJSON: styleJSON, rulesJSON: rules.json).styleJSON
            } catch {
                notes.append("the rules could not be applied: \(error)")
            }
        }
        let served: String
        if let offlinePackage {
            let route = "vectortile-package-\(offlinePackage.manifest.styleDigest.prefix(12))-\(online ? "online" : "offline")"
            filesRoute = route
            var fallback: (@Sendable (String) -> Data?)?
            if online {
                // The local server runs this on its concurrent queue, outside
                // the render lane, and answers package files on the
                // connection itself -- a stalled upstream here cannot hold
                // the package back. The cache keeps it from being asked twice.
                let upstreamCache = UpstreamCache(
                    disk: sourceCacheDirectory.flatMap { SourceTileDiskCache(directory: $0, budgetBytes: sourceCacheBytes) },
                    headers: headers
                )
                fallback = { relative in
                    guard
                        let upstream = offlinePackage.upstreamURL(forRelativePath: relative),
                        let target = VectorTileProvider.parse(upstream)
                    else { return nil }
                    return upstreamCache.bytes(for: target)
                }
            }
            server.registerFiles(routeId: route, directory: offlinePackage.directory, fallback: fallback)
            // The package serves its own style, rewritten to point at the
            // files beside it. Rules cannot reach it from here, and saying so
            // beats a silently unadjusted map.
            if rules != .none { notes.append("rules do not apply to a style served from an offline package") }
            served = offlinePackage.styleServed(by: server.filesUrl(routeId: route))
        } else {
            filesRoute = nil
            served = adjusted
        }
        documentId = "vectortile-style-\(Self.digest(served))"
        url = server.documentUrl(id: documentId)
        attributions = Self.attributions(ofStyle: served)
        diagnostics = notes
        server.registerDocument(id: documentId, contentType: "application/json", body: Data(served.utf8))
        support.showStyle(url: url, attributionRules: attributions.map { AttributionRule(attribution: $0) })
    }

    /// Restores the map's previous design and stops serving the document.
    public func dispose() {
        guard !disposed else { return }
        disposed = true
        support.clearStyle()
        server.unregisterDocument(id: documentId)
        if let filesRoute { server.unregisterFiles(routeId: filesRoute) }
    }

    /**
     The `attribution` of every source in a style. A style that will not
     parse credits nothing here; the map will refuse it anyway, more visibly.
     */
    public static func attributions(ofStyle json: String) -> [String] {
        guard
            let object = try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any],
            let sources = object["sources"] as? [String: Any]
        else { return [] }
        var seen = Set<String>()
        var result: [String] = []
        for key in sources.keys.sorted() {
            guard
                let source = sources[key] as? [String: Any],
                let attribution = source["attribution"] as? String,
                !attribution.trimmingCharacters(in: .whitespaces).isEmpty,
                seen.insert(attribution).inserted
            else { continue }
            result.append(attribution)
        }
        return result
    }

    /**
     What the map asks for and the package does not have: the cached copy at
     once when there is one -- checked with the server in the background once
     stale -- and the network otherwise.
     */
    private final class UpstreamCache: @unchecked Sendable {
        private let disk: SourceTileDiskCache?
        private let headers: [String: String]
        private let lock = NSLock()
        private var checking = Set<String>()
        private let queue: OperationQueue = {
            let queue = OperationQueue()
            queue.name = "mapconductor.vectorstyle.revalidate"
            queue.qualityOfService = .utility
            queue.maxConcurrentOperationCount = 2
            return queue
        }()

        init(disk: SourceTileDiskCache?, headers: [String: String]) {
            self.disk = disk
            self.headers = headers
        }

        func bytes(for url: URL) -> Data? {
            let key = url.absoluteString
            guard let disk else {
                return VectorTileProvider.get(url, headers: headers, cancellation: nil)
            }
            if let entry = disk.entry(key) {
                if entry.meta?.isStale() ?? true { check(url, key: key, entry: entry, disk: disk) }
                return entry.bytes
            }
            guard let response = try? VectorTileProvider.conditionalFetch(url, validators: .none, headers: headers),
                  let body = response.body, !body.isEmpty
            else { return nil }
            disk.put(key, body, meta: SourceTileDiskCache.Meta(
                etag: response.validators.etag, lastModified: response.validators.lastModified,
                fetchedAt: Date().timeIntervalSince1970, maxAge: response.maxAge
            ))
            return body
        }

        private func check(_ url: URL, key: String, entry: (bytes: Data, meta: SourceTileDiskCache.Meta?), disk: SourceTileDiskCache) {
            lock.lock()
            let mine = checking.insert(key).inserted
            lock.unlock()
            guard mine else { return }
            queue.addOperation { [headers] in
                defer {
                    self.lock.lock()
                    self.checking.remove(key)
                    self.lock.unlock()
                }
                guard let response = try? VectorTileProvider.conditionalFetch(
                    url, validators: entry.meta?.validators ?? .none, headers: headers
                ) else { return }
                let meta = SourceTileDiskCache.Meta(
                    etag: response.validators.etag ?? entry.meta?.etag,
                    lastModified: response.validators.lastModified ?? entry.meta?.lastModified,
                    fetchedAt: Date().timeIntervalSince1970,
                    maxAge: response.maxAge ?? entry.meta?.maxAge
                )
                if response.notModified {
                    disk.confirm(key, meta: meta)
                } else if let body = response.body, !body.isEmpty {
                    // The map was served the old bytes as `no-store`; it gets
                    // these the next time it asks.
                    disk.put(key, body, meta: meta)
                }
            }
        }
    }

    /// FNV-1a over the UTF-8 bytes: stable across launches, which `hashValue` is not.
    private static func digest(_ text: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return String(hash, radix: 16)
    }
}

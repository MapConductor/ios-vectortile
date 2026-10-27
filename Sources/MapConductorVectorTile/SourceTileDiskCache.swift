import Foundation

/**
 Persistent cache for fetched source MVT/PBF bytes.

 Each tile keeps what is needed to ask the server whether it changed -- its
 `ETag`, its `Last-Modified`, when it was fetched and how long the server
 said it stays fresh -- in a small sidecar beside the bytes. The bytes are
 always served at once; a stale entry is only *checked*, in the background,
 so a tile on disk never waits for the network again.
 */
final class SourceTileDiskCache {
    /// What a cached tile carries besides its bytes.
    struct Meta: Codable, Equatable {
        var etag: String?
        var lastModified: String?
        /// Seconds since 1970 at which the server last confirmed the bytes.
        var fetchedAt: TimeInterval
        /// The server's `max-age`, when it gave one.
        var maxAge: TimeInterval?

        var validators: SourceValidators { SourceValidators(etag: etag, lastModified: lastModified) }

        /// Fresh for the server's `max-age`, or a day when it said nothing:
        /// basemap tiles change rarely, and a day keeps a panning user from
        /// asking about every tile on every launch.
        func isStale(now: TimeInterval = Date().timeIntervalSince1970) -> Bool {
            now - fetchedAt >= (maxAge ?? SourceTileDiskCache.defaultFreshness)
        }
    }

    static let defaultFreshness: TimeInterval = 24 * 60 * 60

    private let directory: URL
    private let budgetBytes: Int
    private let queue = DispatchQueue(label: "mapconductor.vectortile.sources", attributes: .concurrent)
    private var writesSinceSweep = 0
    private static let sweepEvery = 32

    /**
     Bumped whenever revalidation replaced bytes that tiles had been drawn
     from. Rendered tiles are keyed by it, so those drawn from the old bytes
     stop being found. Kept on disk: a tile drawn from the old bytes in an
     earlier launch is just as stale in this one.
     */
    var revision: Int { queue.sync { storedRevision } }
    private var storedRevision: Int

    init?(directory: URL, budgetBytes: Int) {
        guard (try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )) != nil else { return nil }
        self.directory = directory
        self.budgetBytes = budgetBytes
        storedRevision = (try? String(contentsOf: directory.appendingPathComponent("revision"), encoding: .utf8))
            .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 0
    }

    private func file(_ key: String) -> URL {
        directory.appendingPathComponent("\(Digest.hex(key)).mvt")
    }

    private func metaFile(_ key: String) -> URL {
        directory.appendingPathComponent("\(Digest.hex(key)).meta")
    }

    func get(_ key: String) -> Data? {
        entry(key)?.bytes
    }

    /// The bytes and, when the tile was stored with them, its validators. A
    /// tile from before validators were kept has none, and counts as stale.
    func entry(_ key: String) -> (bytes: Data, meta: Meta?)? {
        queue.sync {
            let url = file(key)
            guard let bytes = try? Data(contentsOf: url), !bytes.isEmpty else { return nil }
            try? FileManager.default.setAttributes(
                [.modificationDate: Date()], ofItemAtPath: url.path
            )
            let meta = (try? Data(contentsOf: metaFile(key))).flatMap { try? JSONDecoder().decode(Meta.self, from: $0) }
            return (bytes, meta)
        }
    }

    func put(_ key: String, _ bytes: Data, meta: Meta? = nil) {
        let meta = meta ?? Meta(fetchedAt: Date().timeIntervalSince1970)
        queue.async(flags: .barrier) {
            guard (try? bytes.write(to: self.file(key), options: .atomic)) != nil else { return }
            self.writeMeta(key, meta)
            self.writesSinceSweep += 1
            if self.writesSinceSweep % SourceTileDiskCache.sweepEvery == 0 { self.sweep() }
        }
    }

    /// The server said the bytes are still good: only the sidecar changes.
    func confirm(_ key: String, meta: Meta) {
        queue.async(flags: .barrier) { self.writeMeta(key, meta) }
    }

    /// Bytes that tiles were drawn from changed upstream; see ``revision``.
    func bumpRevision() {
        queue.sync(flags: .barrier) {
            storedRevision += 1
            try? "\(storedRevision)".write(to: directory.appendingPathComponent("revision"), atomically: true, encoding: .utf8)
        }
    }

    /// Waits for writes queued so far. Tests read what was just put.
    func flush() {
        queue.sync(flags: .barrier) {}
    }

    private func writeMeta(_ key: String, _ meta: Meta) {
        guard let data = try? JSONEncoder().encode(meta) else { return }
        try? data.write(to: metaFile(key), options: .atomic)
    }

    private func sweep() {
        let keys: [URLResourceKey] = [.contentModificationDateKey, .fileSizeKey]
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: keys
        ) else { return }

        var described = entries.compactMap { url -> (URL, Date, Int)? in
            guard url.pathExtension == "mvt",
                  let values = try? url.resourceValues(forKeys: Set(keys)),
                  let modified = values.contentModificationDate,
                  let size = values.fileSize
            else { return nil }
            return (url, modified, size)
        }
        var total = described.reduce(0) { $0 + $1.2 }
        guard total > budgetBytes else { return }

        described.sort { $0.1 < $1.1 }
        for (url, _, size) in described {
            if total <= budgetBytes { break }
            if (try? FileManager.default.removeItem(at: url)) != nil {
                total -= size
                // The sidecar goes with its tile; one left behind would only
                // describe bytes that are no longer here.
                try? FileManager.default.removeItem(at: url.deletingPathExtension().appendingPathExtension("meta"))
            }
        }
    }
}

final class SourceFetchStats: @unchecked Sendable {
    private let lock = NSLock()
    private var values = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    var memoryHits: Int {
        get { get(0) }
        set { set(0, newValue) }
    }
    var diskHits: Int {
        get { get(1) }
        set { set(1, newValue) }
    }
    var networkFetches: Int {
        get { get(2) }
        set { set(2, newValue) }
    }
    var sharedWaits: Int {
        get { get(3) }
        set { set(3, newValue) }
    }
    var cancelled: Int {
        get { get(4) }
        set { set(4, newValue) }
    }
    var queueWaitMs: Int {
        get { get(5) }
        set { set(5, newValue) }
    }
    /// Fetches refused because the network was off and the package had no answer.
    var blocked: Int {
        get { get(6) }
        set { set(6, newValue) }
    }
    /// Sources still on their way when the tile was drawn without them.
    var pending: Int {
        get { get(7) }
        set { set(7, newValue) }
    }
    /// Answered by the caller's local storage (an offline package).
    var localHits: Int {
        get { get(8) }
        set { set(8, newValue) }
    }

    private func get(_ index: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return values[index]
    }

    private func set(_ index: Int, _ value: Int) {
        lock.lock()
        values[index] = value
        lock.unlock()
    }
}

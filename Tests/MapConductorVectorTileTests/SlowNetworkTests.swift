import UIKit
import MapConductorCore
import XCTest

@testable import MapConductorVectorTile

/**
 A Wi-Fi that is connected but answers nothing, and an offline package.

 Reported from a shopping mall: with an area downloaded and the network
 "on" but stalled, the map went white -- including the downloaded area --
 because the package was read inside the same fetch slots that every other
 tile's hung request was holding. Nothing here touches the network: sources
 come from the bundled tile, and "the network" is a closure that hangs.
 */
final class SlowNetworkTests: XCTestCase {
    private var styleJSON: String!
    private var tileData: Data!
    private var directories: [URL] = []
    private var cleanups: [() -> Void] = []

    private let sourceURL = "https://demotiles.maplibre.org/tiles/0/0/0.pbf"

    override func setUpWithError() throws {
        let bundle = Bundle.module
        styleJSON = try String(
            contentsOf: try XCTUnwrap(bundle.url(forResource: "demo-style", withExtension: "json")),
            encoding: .utf8
        )
        tileData = try Data(contentsOf: try XCTUnwrap(bundle.url(forResource: "tile-0-0-0", withExtension: "pbf")))
    }

    override func tearDown() {
        cleanups.reversed().forEach { $0() }
        cleanups = []
        directories.forEach { try? FileManager.default.removeItem(at: $0) }
        directories = []
    }

    private func directory() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        directories.append(url)
        return url
    }

    /// The tile drawn with its source in hand, to compare against.
    private func reference() throws -> Data {
        let provider = try VectorTileProvider(
            styleJSON: styleJSON, tileSize: 256, renderMode: .cpu, renderScale: 1
        ) { [tileData] _ in tileData }
        defer { provider.close() }
        return try XCTUnwrap(provider.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground))
    }

    /// A package holding the z0 source tile, and nothing else.
    private func package() throws -> OfflinePackage {
        let dir = directory()
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("tiles/maplibre/0/0"), withIntermediateDirectories: true
        )
        try tileData.write(to: dir.appendingPathComponent("tiles/maplibre/0/0/0.mvt"))
        try Data(styleJSON.utf8).write(to: dir.appendingPathComponent("style.json"))
        return try OfflinePackage.create(
            directory: dir,
            manifest: OfflinePackage.Manifest(
                bounds: OfflinePackage.Bounds(south: -10, west: -10, north: 10, east: 10),
                minZoom: 0, maxZoom: 0, styleDigest: "test", createdAt: 0,
                tiles: 1, glyphs: 0, sprites: 0, bytes: Int64(tileData.count),
                sources: ["maplibre": "https://demotiles.maplibre.org/tiles/{z}/{x}/{y}.pbf"],
                glyphsTemplate: nil, spriteBase: nil
            ),
            index: [sourceURL: "tiles/maplibre/0/0/0.mvt"]
        )
    }

    /// A network that answers nothing until released (or ten seconds pass).
    private func stalledNetwork() -> (fetch: (URL) -> Data?, release: () -> Void) {
        let gate = DispatchSemaphore(value: 0)
        let released = NSLock()
        var open = false
        let release = {
            released.lock()
            let first = !open
            open = true
            released.unlock()
            // Enough for every thread that could be parked on it.
            if first { for _ in 0..<256 { gate.signal() } }
        }
        cleanups.append(release)
        return ({ _ in _ = gate.wait(timeout: .now() + 10); return nil }, release)
    }

    // (a)
    func testPackagedTileIsDrawnAtOnceWhileOtherTilesHangOnTheNetwork() throws {
        let expected = try reference()
        let network = stalledNetwork()
        let fetcher = OfflinePackage.Fetcher(package: try package(), online: true, upstream: network.fetch)
        let provider = try VectorTileProvider(
            styleJSON: styleJSON, tileSize: 256, renderMode: .cpu, renderScale: 1,
            fetchTile: { try fetcher.fetch($0) },
            localTile: { fetcher.local($0) }
        )
        cleanups.append { provider.close() }

        // Sixty-four tiles outside the package: more hung fetches than slots.
        for x in 0..<8 {
            for y in 0..<8 {
                DispatchQueue.global().async {
                    _ = provider.renderTile(request: TileRequest(x: x, y: y, z: 3), content: .ground)
                }
            }
        }
        Thread.sleep(forTimeInterval: 0.3)

        let started = Date()
        let png = provider.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground)
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertLessThan(elapsed, 1.0, "the packaged tile waited behind the stalled network")
        XCTAssertEqual(png, expected, "the packaged tile was drawn without its source")
        XCTAssertGreaterThan(fetcher.stats.packageHits, 0)
    }

    // (b)
    func testTileDrawnShortIsNotKeptAndItsSourceBringsTheRedraw() throws {
        let expected = try reference()
        let gate = DispatchSemaphore(value: 0)
        cleanups.append { gate.signal() }
        let lock = NSLock()
        var fetches = 0
        let provider = try VectorTileProvider(
            styleJSON: styleJSON, tileSize: 256, renderMode: .cpu,
            assetCacheDirectory: directory(), renderScale: 1
        ) { [tileData] _ in
            lock.lock(); fetches += 1; lock.unlock()
            _ = gate.wait(timeout: .now() + 10)
            return tileData
        }
        cleanups.append { provider.close() }
        let loaded = expectation(description: "onSourcesLoaded")
        provider.onSourcesLoaded = { loaded.fulfill() }

        let started = Date()
        let short = provider.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground)
        XCTAssertLessThan(Date().timeIntervalSince(started), VectorTileProvider.sourceGrace + 1)
        XCTAssertNotNil(short, "a tile is drawn with what it has")
        XCTAssertNotEqual(short, expected)

        gate.signal()
        wait(for: [loaded], timeout: 5)
        let finished = provider.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground)
        XCTAssertEqual(finished, expected, "the short tile was kept and served again")
        XCTAssertEqual(fetches, 1, "the fetch kept running after the tile was drawn without it")
    }

    /// One empty layer: a valid tile that draws only the background.
    private let emptyTile = Data([0x1A, 0x08, 0x0A, 0x01, 0x78, 0x28, 0x80, 0x20, 0x78, 0x02])

    private func staleEntry(in assets: URL, bytes: Data) {
        let disk = SourceTileDiskCache(directory: assets.appendingPathComponent("sources"), budgetBytes: 1 << 20)!
        disk.put(sourceURL, bytes, meta: SourceTileDiskCache.Meta(
            etag: "\"v1\"", lastModified: "Mon, 26 Oct 2026 00:00:00 GMT", fetchedAt: 0, maxAge: nil
        ))
        disk.flush()
    }

    private func meta(in assets: URL) -> SourceTileDiskCache.Meta? {
        SourceTileDiskCache(directory: assets.appendingPathComponent("sources"), budgetBytes: 1 << 20)?
            .entry(sourceURL)?.meta
    }

    // (c) 304
    func testStaleCachedTileIsServedAtOnceAndKeptWhenTheServerSaysNotModified() throws {
        let expected = try reference()
        let assets = directory()
        staleEntry(in: assets, bytes: tileData)
        let asked = expectation(description: "revalidated")
        var sent: SourceValidators?
        let provider = try VectorTileProvider(
            styleJSON: styleJSON, tileSize: 256, renderMode: .cpu,
            assetCacheDirectory: assets, renderScale: 1,
            fetchTile: { url in
                XCTFail("a cached tile was fetched again: \(url)")
                return nil
            },
            revalidate: { _, validators in
                sent = validators
                defer { asked.fulfill() }
                return SourceResponse(status: 304, body: nil, maxAge: 3600)
            }
        )
        cleanups.append { provider.close() }
        provider.onSourcesLoaded = { XCTFail("nothing changed, nothing to redraw") }

        let started = Date()
        XCTAssertEqual(provider.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground), expected)
        XCTAssertLessThan(Date().timeIntervalSince(started), 1.0)
        wait(for: [asked], timeout: 5)
        XCTAssertEqual(sent, SourceValidators(etag: "\"v1\"", lastModified: "Mon, 26 Oct 2026 00:00:00 GMT"))

        let deadline = Date().addingTimeInterval(3)
        while Date() < deadline, (meta(in: assets)?.fetchedAt ?? 0) == 0 { Thread.sleep(forTimeInterval: 0.05) }
        let confirmed = try XCTUnwrap(meta(in: assets))
        XCTAssertGreaterThan(confirmed.fetchedAt, 0, "the 304 did not refresh the entry")
        XCTAssertEqual(confirmed.etag, "\"v1\"")
        XCTAssertEqual(confirmed.maxAge, 3600)
        XCTAssertFalse(confirmed.isStale())
    }

    // (c) 200
    func testStaleCachedTileIsReplacedWhenTheServerHasNewBytes() throws {
        let expected = try reference()
        let assets = directory()
        staleEntry(in: assets, bytes: emptyTile)
        let loaded = expectation(description: "onSourcesLoaded")
        let provider = try VectorTileProvider(
            styleJSON: styleJSON, tileSize: 256, renderMode: .cpu,
            assetCacheDirectory: assets, renderScale: 1,
            fetchTile: { _ in nil },
            revalidate: { [tileData] _, _ in
                SourceResponse(status: 200, body: tileData, validators: SourceValidators(etag: "\"v2\""))
            }
        )
        cleanups.append { provider.close() }
        provider.onSourcesLoaded = { loaded.fulfill() }

        let old = provider.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground)
        XCTAssertNotNil(old)
        XCTAssertNotEqual(old, expected)
        wait(for: [loaded], timeout: 5)
        XCTAssertEqual(meta(in: assets)?.etag, "\"v2\"")

        // The redraw the handover asks for. The native renderer may still hold
        // the old tile decoded (it is keyed by coordinates, not bytes), so
        // what this draws is not asserted -- only that it is not kept.
        _ = provider.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground)

        // The next provider -- next launch, or the next style -- reads the
        // new bytes, and is not served a picture drawn from the old ones.
        let next = try VectorTileProvider(
            styleJSON: styleJSON, tileSize: 256, renderMode: .cpu,
            assetCacheDirectory: assets, renderScale: 1,
            fetchTile: { url in
                XCTFail("the replaced tile was fetched again: \(url)")
                return nil
            }
        )
        cleanups.append { next.close() }
        XCTAssertEqual(
            next.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground), expected,
            "a tile drawn from the old bytes was served again"
        )
    }

    // (d)
    func testOfflineFetcherRefusesToRevalidate() throws {
        let assets = directory()
        staleEntry(in: assets, bytes: tileData)
        let fetcher = OfflinePackage.Fetcher(package: try package(), online: false)
        XCTAssertThrowsError(try fetcher.revalidate(URL(string: sourceURL)!, validators: .none)) { error in
            XCTAssertTrue(error is OfflineUnavailableError)
        }

        // Through a provider: the cached tile is drawn and left as it was.
        let provider = try VectorTileProvider(
            styleJSON: styleJSON, tileSize: 256, renderMode: .cpu,
            assetCacheDirectory: assets, renderScale: 1,
            fetchTile: { try fetcher.fetch($0) },
            revalidate: { try fetcher.revalidate($0, validators: $1) }
        )
        cleanups.append { provider.close() }
        // No `localTile` here, so z0 comes from the stale disk entry and is
        // checked -- through the Fetcher, which is offline.
        XCTAssertNotNil(provider.renderTile(request: TileRequest(x: 0, y: 0, z: 0), content: .ground))
        Thread.sleep(forTimeInterval: 0.5)
        XCTAssertEqual(fetcher.stats.networkFetches, 0)
        XCTAssertEqual(meta(in: assets)?.fetchedAt, 0, "a refused check must leave the entry as it was")
    }

    func testMaxAgeIsReadFromCacheControl() {
        XCTAssertEqual(VectorTileProvider.maxAge("public, max-age=600"), 600)
        XCTAssertEqual(VectorTileProvider.maxAge("no-cache"), 0)
        XCTAssertNil(VectorTileProvider.maxAge("public"))
        XCTAssertNil(VectorTileProvider.maxAge(nil))
    }
}

import Foundation

/**
 What a cached source tile can show the server to ask whether it changed.

 Both are opaque: they are sent back exactly as the server gave them, as
 `If-None-Match` and `If-Modified-Since`.
 */
public struct SourceValidators: Equatable, Sendable {
    public var etag: String?
    public var lastModified: String?

    public init(etag: String? = nil, lastModified: String? = nil) {
        self.etag = etag
        self.lastModified = lastModified
    }

    public static let none = SourceValidators()
}

/**
 What a request for a source tile came back with: the status, the body for a
 200, and what the server said about caching it.
 */
public struct SourceResponse: Sendable {
    /// 200 with new bytes, 304 when the validators still hold, 404 when there is no such tile.
    public let status: Int
    public let body: Data?
    public let validators: SourceValidators
    /// `Cache-Control: max-age`, when the server gave one.
    public let maxAge: TimeInterval?

    public init(status: Int, body: Data?, validators: SourceValidators = .none, maxAge: TimeInterval? = nil) {
        self.status = status
        self.body = body
        self.validators = validators
        self.maxAge = maxAge
    }

    public var notModified: Bool { status == 304 }
}

extension VectorTileProvider {
    /**
     One HTTP request for a source tile, conditional when `validators` hold
     anything.

     The response is read here rather than left to `URLSession`'s own cache:
     a 304 has to reach the caller as a 304, since "still the same" is what
     keeps the tile on disk, and the validators the server sends are what the
     next check will need. A transport failure or a 5xx is thrown -- "not
     now" -- and never turned into "no such tile".
     */
    public static func conditionalFetch(
        _ url: URL,
        validators: SourceValidators,
        headers: [String: String] = [:]
    ) throws -> SourceResponse {
        try conditionalFetch(url, validators: validators, headers: headers, cancellation: nil)
    }

    static func conditionalFetch(
        _ url: URL,
        validators: SourceValidators,
        headers: [String: String],
        cancellation: FetchCancellation?
    ) throws -> SourceResponse {
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15)
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        if let etag = validators.etag { request.setValue(etag, forHTTPHeaderField: "If-None-Match") }
        if let lastModified = validators.lastModified {
            request.setValue(lastModified, forHTTPHeaderField: "If-Modified-Since")
        }
        let done = DispatchSemaphore(value: 0)
        var payload: Data?
        var failure: Error?
        var response: HTTPURLResponse?
        let task = URLSession.shared.dataTask(with: request) { data, received, error in
            defer { done.signal() }
            failure = error
            response = received as? HTTPURLResponse
            payload = data
        }
        task.resume()
        while done.wait(timeout: .now() + .milliseconds(10)) != .success {
            if cancellation?.isCancelled == true {
                task.cancel()
                done.wait()
                throw CancellationError()
            }
        }
        if let failure { throw failure }
        guard let response else { throw URLError(.badServerResponse) }
        if response.statusCode >= 500 { throw URLError(.badServerResponse) }
        let ok = (200..<300).contains(response.statusCode)
        return SourceResponse(
            status: response.statusCode,
            body: ok ? payload : nil,
            validators: SourceValidators(
                etag: response.value(forHTTPHeaderField: "ETag"),
                lastModified: response.value(forHTTPHeaderField: "Last-Modified")
            ),
            maxAge: maxAge(response.value(forHTTPHeaderField: "Cache-Control"))
        )
    }

    /// `max-age=N` out of a `Cache-Control` value; `no-cache` and `no-store`
    /// count as zero, so the tile is checked every time it is drawn from disk.
    static func maxAge(_ cacheControl: String?) -> TimeInterval? {
        guard let cacheControl else { return nil }
        var result: TimeInterval?
        for directive in cacheControl.lowercased().split(separator: ",") {
            let part = directive.trimmingCharacters(in: .whitespaces)
            if part == "no-cache" || part == "no-store" { return 0 }
            if part.hasPrefix("max-age="), let seconds = TimeInterval(part.dropFirst("max-age=".count)) {
                result = seconds
            }
        }
        return result
    }
}

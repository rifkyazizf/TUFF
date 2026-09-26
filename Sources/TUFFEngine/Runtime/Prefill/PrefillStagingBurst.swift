import Foundation

/// Splits a layer's routed tiles into staging sub-bursts for overlapped
/// prefill fetch.
///
/// A burst holds whole tiles and stays at or under `burstExperts` experts, so
/// a burst read keeps the streamer's read pool busy (a single tile is only
/// eight experts) while the previous burst's tiles run on GPU. Tiles never
/// straddle a boundary, so a tile always commits against a fully staged burst.
enum PrefillStagingBurst {
    static func tileRanges(expertCounts: [Int], burstExperts: Int) -> [Range<Int>] {
        precondition(burstExperts > 0, "staging burst must hold at least one expert")
        guard !expertCounts.isEmpty else { return [] }
        var ranges: [Range<Int>] = []
        var start = 0
        var total = 0
        for (tileIndex, count) in expertCounts.enumerated() {
            // Close the burst before a tile that would push it past the cap;
            // a single oversized tile still becomes its own burst.
            if tileIndex > start, total + count > burstExperts {
                ranges.append(start..<tileIndex)
                start = tileIndex
                total = 0
            }
            total += count
        }
        ranges.append(start..<expertCounts.count)
        return ranges
    }
}

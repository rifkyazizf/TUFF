import Testing
@testable import TUFFEngine

/// Sub-burst planning for overlapped prefill staging. The bursts must cover
/// every tile exactly once, in order, split only at tile boundaries.
@Suite struct PrefillStagingBurstTests {
    @Test func emptyTilesYieldNoBursts() {
        #expect(PrefillStagingBurst.tileRanges(expertCounts: [], burstExperts: 32).isEmpty)
    }

    @Test func burstsSplitAtTileBoundariesUnderTheExpertCap() {
        // Eight-expert tiles, 32-expert cap: four tiles per burst.
        let counts = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        let ranges = PrefillStagingBurst.tileRanges(expertCounts: counts, burstExperts: 32)
        #expect(ranges == [0..<4, 4..<8, 8..<10])
        #expect(ranges.flatMap { $0 } == Array(0..<counts.count))
    }

    @Test func smallInputStaysOneBurst() {
        let ranges = PrefillStagingBurst.tileRanges(expertCounts: [8, 8], burstExperts: 32)
        #expect(ranges == [0..<2])
    }

    @Test func oversizedTileStillGetsItsOwnBurst() {
        let ranges = PrefillStagingBurst.tileRanges(expertCounts: [16, 16, 4], burstExperts: 16)
        #expect(ranges == [0..<1, 1..<2, 2..<3])
    }

    @Test func burstClosesBeforeExceedingTheCap() {
        let ranges = PrefillStagingBurst.tileRanges(expertCounts: [12, 12, 12], burstExperts: 32)
        // 12 + 12 fits (24); adding the third (36) would exceed 32.
        #expect(ranges == [0..<2, 2..<3])
    }
}

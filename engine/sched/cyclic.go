package sched

// The eviction policy of a cyclic scan, shared by the two tiers that page over
// that access pattern (the device tier and the host), whose mechanisms differ.
//
// A transformer visits blocks lo..hi-1 and then lo again, every token, so the
// distance to any block's next use is arithmetic, (li - cur) mod (hi - lo), and
// Bélády's optimal policy is implementable. Its answer is to give up the most
// recently used block (just behind the cursor): LRU evicts the block needed
// soonest and, when the period exceeds the budget, hits nothing at all, while
// MRU pins a prefix and hits (held-1)/blocks.

// Never is the distance reported for a block the scan will not reach again,
// which is any block outside [lo, hi). It is larger than any real distance, so
// an out-of-scan block is always the first to be given up.
const Never = 1 << 30

// NextUse is how many blocks a cyclic scan over [lo, hi) will visit, starting
// from cur, before it reaches li again.
//
// cur itself is 0 -- the block being computed is the one block that must not be
// given up, and callers exclude it rather than relying on a distance. hi <= lo
// is a scan with no blocks in it, where nothing has a next use.
func NextUse(li, cur, lo, hi int) int {
	n := hi - lo
	if n <= 0 || li < lo || li >= hi {
		return Never
	}
	d := (li - cur) % n
	if d < 0 {
		d += n
	}
	return d
}

// CyclicVictim is which of the blocks in [lo, hi) that held() reports resident
// should be given up to make room, with the scan at cur.
//
// It returns -1 when nothing is resident to take. cur is never chosen: it is
// the block being computed, and giving that up is correct (see model/ring.go --
// a clean page re-faults to the same bytes) and is a miss inflicted on the very
// next instruction.
//
// The scan is ordered rather than over a map, so the choice never depends on
// hash order. Inside [lo, hi) distances are distinct; blocks outside tie at
// Never, and jit_old/gpu/tier breaks those ties on the lowest index.
func CyclicVictim(cur, lo, hi int, held func(int) bool) int {
	best, bestD := -1, -1
	for li := lo; li < hi; li++ {
		if li == cur || !held(li) {
			continue
		}
		if d := NextUse(li, cur, lo, hi); d > bestD {
			best, bestD = li, d
		}
	}
	return best
}

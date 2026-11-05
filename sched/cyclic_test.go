package sched_test

import (
	"testing"

	"github.com/jitllm/jitllm/sched"
)

// The policy, simulated: the claim is that on a cyclic scan the obvious
// policies hit nothing and this one does not, so the actual access pattern is
// run through a fixed-size cache under two policies and the hits counted.

// simulate runs `tokens` passes over blocks [0, n) with room for `held` of
// them, choosing a victim with pick, and reports how many block visits found
// the block already resident.
func simulate(n, held, tokens int, pick func(res map[int]bool, cur int) int) int {
	res := map[int]bool{}
	hits := 0
	for tok := 0; tok < tokens; tok++ {
		for cur := 0; cur < n; cur++ {
			if res[cur] {
				hits++
				continue
			}
			for len(res) >= held {
				v := pick(res, cur)
				if v < 0 {
					break
				}
				delete(res, v)
			}
			res[cur] = true
		}
	}
	return hits
}

// TestCyclicVictimIsTheOnlyPolicyThatHitsAnything is the whole argument for
// sched/cyclic.go existing, run rather than asserted.
func TestCyclicVictimIsTheOnlyPolicyThatHitsAnything(t *testing.T) {
	// Llama-3.3-70B's shape, and the window `cap 26G` buys for it.
	const n, held, tokens = 80, 52, 4
	visits := n * tokens

	belady := simulate(n, held, tokens, func(res map[int]bool, cur int) int {
		return sched.CyclicVictim(cur, 0, n, func(li int) bool { return res[li] })
	})

	// Violation: LRU (and a sliding window). The least recently used block on
	// a cyclic scan is the one coming round soonest.
	lru := simulate(n, held, tokens, func(res map[int]bool, cur int) int {
		best, bestD := -1, -1
		for li := range res {
			if d := (cur - li + n) % n; d > bestD {
				best, bestD = li, d
			}
		}
		return best
	})

	if belady <= 0 {
		t.Fatalf("Bélády hit %d of %d visits", belady, visits)
	}
	// The steady state is a pinned prefix of held-1 blocks, hit on every token
	// after the first.
	if want := (held - 1) * (tokens - 1); belady < want {
		t.Fatalf("Bélády hit %d of %d visits, want at least %d -- the pinned prefix is "+
			"%d blocks and it should survive every token after the first",
			belady, visits, want, held-1)
	}
	if lru != 0 {
		t.Fatalf("LRU hit %d of %d visits; on a cyclic scan wider than the cache it is "+
			"supposed to hit exactly nothing, so this simulation is not running one",
			lru, visits)
	}
	t.Logf("%d blocks, room for %d, %d tokens: Bélády/MRU %d hits of %d visits, LRU %d",
		n, held, tokens, belady, visits, lru)
}

// TestNextUseIsTheDistanceToTheNextVisit pins the arithmetic itself, including
// the two answers that are easy to get wrong: the cursor is zero away from
// itself, and a block outside the scan is never reached.
func TestNextUseIsTheDistanceToTheNextVisit(t *testing.T) {
	const lo, hi = 4, 12 // an 8-block scan, as a device holding blocks 4..11
	for _, c := range []struct{ li, cur, want int }{
		{4, 4, 0},   // the cursor itself
		{5, 4, 1},   // the next block
		{11, 4, 7},  // the far end, still ahead
		{4, 5, 7},   // wrapped: the block just behind is furthest
		{10, 11, 7}, // and it is furthest wherever the cursor is
		{3, 4, sched.Never},
		{12, 4, sched.Never},
	} {
		if got := sched.NextUse(c.li, c.cur, lo, hi); got != c.want {
			t.Errorf("NextUse(%d, %d, %d, %d) = %d, want %d", c.li, c.cur, lo, hi, got, c.want)
		}
	}
	// An empty scan has no next use for anything, which is what the device
	// pager passes when it is trimming rather than running.
	if got := sched.NextUse(3, -1, 0, 0); got != sched.Never {
		t.Errorf("NextUse over an empty scan = %d, want Never", got)
	}
	// The furthest block is the one just behind the cursor, which is the whole
	// claim. Asserted by search rather than by the one case above.
	cur := 7
	best, bestD := -1, -1
	for li := lo; li < hi; li++ {
		if li == cur {
			continue
		}
		if d := sched.NextUse(li, cur, lo, hi); d > bestD {
			best, bestD = li, d
		}
	}
	if best != cur-1 {
		t.Errorf("the furthest next use is block %d with the cursor at %d; MRU says %d",
			best, cur, cur-1)
	}
}

// TestCyclicVictimAsksWhatIsHeld: the policy chooses among the blocks that are
// actually resident, and says so when none are.
//
// The block being computed (cur) is never given up: evicting it is correct but
// a self-inflicted miss on the very next instruction.
func TestCyclicVictimAsksWhatIsHeld(t *testing.T) {
	const lo, hi = 0, 16
	held := map[int]bool{3: true, 4: true, 5: true}
	// With the cursor at 6, the furthest next use among {3,4,5} is 5.
	if got := sched.CyclicVictim(6, lo, hi, func(li int) bool { return held[li] }); got != 5 {
		t.Errorf("victim %d, want 5 -- the block just behind the cursor", got)
	}
	// The cursor is never chosen, even when it is resident and furthest.
	held[6] = true
	if got := sched.CyclicVictim(6, lo, hi, func(li int) bool { return held[li] }); got != 5 {
		t.Errorf("victim %d with the cursor resident, want 5: the block being computed "+
			"must not be the one given up", got)
	}
	if got := sched.CyclicVictim(6, lo, hi, func(int) bool { return false }); got != -1 {
		t.Errorf("victim %d with nothing resident, want -1", got)
	}
	// Deterministic: the same question, the same answer, every time.
	for i := 0; i < 64; i++ {
		if got := sched.CyclicVictim(6, lo, hi, func(li int) bool { return held[li] }); got != 5 {
			t.Fatalf("victim %d on iteration %d", got, i)
		}
	}
}

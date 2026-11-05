package engine

import (
	"testing"
	"time"
)

// TestSpecChooseFollowsTheCosts holds the adaptive draft count to its prices
// with no device: a verification flat in its rows (a pass bound by its
// weights) drafts, one linear in them (a host stepping rows one by one, where
// even every draft accepted cannot beat decode) retires after retireAfter
// re-tests, and a flat one at an acceptance too low to pay keeps
// re-testing instead of retiring, since acceptance is what a later round can
// change.
func TestSpecChooseFollowsTheCosts(t *testing.T) {
	ms := time.Millisecond
	session := func(verify func(rows int) time.Duration, alpha float64) *Speculator {
		sp := &Speculator{}
		for r := 0; r <= specMaxDraft+1; r++ {
			sp.vN = append(sp.vN, 0)
			sp.vRecent = append(sp.vRecent, nil)
		}
		for r := 1; r <= 2; r++ { // what the first rounds measure
			sp.vN[r] = 3
			sp.vRecent[r] = []time.Duration{verify(r), verify(r), verify(r)}
		}
		sp.offered, sp.accepted = []int64{100}, []int64{int64(alpha * 100)}
		sp.tCatch, sp.nCatch, sp.tDraft, sp.nDraft = ms, 1, ms, 1
		return sp
	}
	// rounds runs n choices, each round's verification priced as Next would
	// count it, and returns the rounds drafted outside a re-check or re-test.
	rounds := func(sp *Speculator, verify func(rows int) time.Duration, n int) (drafted, retests int) {
		for range n {
			before, again := sp.retests, sp.againLeft > 0
			k := sp.choose()
			sp.vRecent[1+k] = append(sp.vRecent[1+k], verify(1+k))
			sp.vN[1+k]++
			if k > 0 && !again && sp.againLeft == 0 {
				drafted++
			}
			retests += sp.retests - before
		}
		return
	}

	flat := func(rows int) time.Duration { return 10*ms + time.Duration(rows)*ms/10 }
	if drafted, _ := rounds(session(flat, 0.8), flat, 64); drafted < 32 {
		t.Fatalf("a flat verification at 80%% acceptance drafted %d of 64 rounds", drafted)
	}

	linear := func(rows int) time.Duration { return 10 * ms * time.Duration(rows) }
	sp := session(linear, 0.95)
	drafted, retests := rounds(sp, linear, 64)
	if !sp.retired || !sp.stats.Retired || retests != retireAfter {
		t.Fatalf("a linear verification: retired %v after %d re-tests (want %d), %d rounds drafted",
			sp.retired, retests, retireAfter, drafted)
	}
	// An unmeasured count past the deepest measured is priced as that one, so
	// it is tried once; after that its own price stands.
	if drafted > specMaxDraft-2 {
		t.Fatalf("a linear verification drafted %d rounds outside its re-tests, more than one per unmeasured count", drafted)
	}

	sp = session(flat, 0.02)
	drafted, retests = rounds(sp, flat, 64)
	if sp.retired || retests <= retireAfter {
		t.Fatalf("a flat verification at 2%% acceptance: retired %v after %d re-tests -- it would win at a better one",
			sp.retired, retests)
	}
	if drafted != 0 {
		t.Fatalf("a flat verification at 2%% acceptance drafted %d rounds outside its re-tests", drafted)
	}
}

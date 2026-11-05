package engine

import "slices"

// Prompt lookup: speculation with no prediction block. A round drafts the
// tokens that followed the latest earlier occurrence of the sequence's last
// n-gram (the longest n in [ngMin, ngMax] that occurs), and the trunk
// verifies them exactly as it verifies the prediction block's drafts, with
// the same rollback of attention by position and of a recurrence by
// SpecRollback. Nothing runs to draft, so a round that finds no match is a
// decode step: one row, no wasted verification. This is llama.cpp's
// prompt lookup and vLLM's ngram proposer; the choice of n and of the latest
// occurrence is in docs/engineering-history/model-correctness.md,
// "engine/speclookup.go".
//
// RULE 8: the search compares token ids, integers the sampler has already
// decided, and decides only which rows the next verification carries. No
// value of any row's arithmetic depends on it -- a wrong proposal changes how
// many tokens a pass yields, never which -- so it is control, like the
// acceptance comparison beside it, and stays Go.

// specNGramMin and specNGramMax are the n-gram lengths matched by default:
// one token matches too often to be worth a verification row, and four is
// already rare outside a verbatim copy.
const (
	specNGramMin = 2
	specNGramMax = 4
)

// specLookupDraft is how many tokens a lookup round proposes when no count
// was fixed (WithSpecDraft): the verification is the only cost, and a match
// that held for n tokens tends to hold for several more.
const specLookupDraft = 5

// lookupInit readies a Speculator that drafts by prompt lookup.
func (sp *Speculator) lookupInit() {
	if sp.opt.ngMin == 0 {
		sp.opt.ngMin, sp.opt.ngMax = specNGramMin, specNGramMax
	}
	sp.hist = make([]int32, 0, sp.t.maxSeq+1)
	sp.rowTok = make([]int32, 0, specLookupDraft+specMaxDraft)
}

// lookupCount is how many tokens this round may propose. Sampling drafts
// none: the rejection rule needs the draft's distribution, and a lookup's is
// a point mass that this build does not verify against (the doc says so), so
// a sampled session through a lookup Speculator is plain sampled decode.
func (sp *Speculator) lookupCount(greedy bool) int {
	switch {
	case !greedy:
		return 0
	case sp.opt.sched != nil:
		return sp.opt.sched[int(sp.stats.Rounds)%len(sp.opt.sched)]
	case sp.opt.draft > 0:
		return sp.opt.draft
	}
	return specLookupDraft
}

// propose appends up to k drafts to dst: what followed the latest earlier
// occurrence of the history's last n tokens, the longest n that occurs.
// A gate's oracle replaces the search (Speculator.oracle).
func (sp *Speculator) propose(dst []int32, k int) []int32 {
	if k == 0 {
		return dst
	}
	if sp.oracle != nil {
		g := int(sp.stats.Emitted)
		for i := 0; i < k && g+i < len(sp.oracle); i++ {
			dst = append(dst, sp.oracleAt(g+i, 0))
		}
		return dst
	}
	return appendLookup(dst, sp.hist, sp.opt.ngMin, sp.opt.ngMax, k)
}

// appendLookup is the search itself: for n from hi down to lo, the latest
// j < len(h)-n with h[j:j+n] equal to h's last n, and then up to k tokens
// from h[j+n:].
func appendLookup(dst, h []int32, lo, hi, k int) []int32 {
	for n := hi; n >= lo; n-- {
		if len(h) <= n {
			continue
		}
		tail := h[len(h)-n:]
		for j := len(h) - n - 1; j >= 0; j-- {
			if h[j] == tail[0] && slices.Equal(h[j:j+n], tail) {
				return append(dst, h[j+n:min(j+n+k, len(h))]...)
			}
		}
	}
	return dst
}

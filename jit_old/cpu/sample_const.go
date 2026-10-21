package cpu

import "fmt"

// The sampler's kernels: the shape rules and the scratch layout all three tiers
// read. sample.go is the AVX2 tier, sse_sample.go the legacy-SSE one and
// sample_a64.go NEON.
//
// They read MoETopKConsts() rather than a block of their own: the index
// vector, the integer steps, INT_MAX, -Inf and the NaN filler are all there,
// and one copy of a layout cannot drift.

// sampleShapeErr is what every tier answers for a shape outside the contract.
func sampleShapeErr(what string, n int) error {
	return fmt.Errorf("jit: sampler %s: %d elements (want >= 1)", what, n)
}

// SampleSegLen is the segment length the ordering runs at, for a vocabulary of
// n logits.
//
// Taking the d largest of n by successive maximum costs d passes over n.
// Splitting the vocabulary into S segments of L and keeping the best of each
// turns one extraction into one rescan of the winner's segment (L) plus one
// sweep of the summary (S), since every other segment's best is undisturbed.
// L + n/L is minimised at L = sqrt(n): 256 for Llama-3.2's 128256 (757
// comparisons an extraction against 128256), 512 for gemma-2b's 256000. It is
// a power of two, so `id / L` is a shift, and never below 64, where the
// summary would dominate.
func SampleSegLen(n int) int {
	if n <= 64 {
		return 64
	}
	best, bestCost := 64, 64+(n+63)/64
	for l := 128; l <= 1<<20; l *= 2 {
		c := l + (n+l-1)/l
		if c <= bestCost {
			best, bestCost = l, c
		}
		if l >= n {
			break
		}
	}
	return best
}

// The sampler's draw kernel reads four parameters and writes four counters. The
// layout is named here rather than spelled out at each of the three emitters
// and the one caller.
//
// Parameters, AScale, float32:
//
//	0  minp   the min-p factor. 0 disables it by arithmetic rather than by a
//	          branch: the cut is minp*p[0] = 0 and a probability is never
//	          strictly below zero.
//	1  topp   the top-p mass. +Inf disables it: a cumulative never reaches it.
//	2  u      the uniform draw, already in [0,1).
//	3  sumAll the total probability mass of the whole distribution when the
//	          candidate array is a prefix of it, or a negative value when the
//	          array is the whole of it and the kernel should use its own
//	          accumulator (see sample.go).
//
// Counters, ASum, int32:
//
//	0  res       the index into the candidate array that the walk landed on
//	1  m         how many candidates survived the cuts
//	2  cut       1 if min-p or top-p fired, 0 if neither did
//	3  resolved  1 if the walk terminated inside the array rather than running
//	             off its end
//
// And Out, float32: the surviving mass, which is the divisor the walk used.
//
// RowStr, int64: nonzero asks for the mass alone and skips the walk, for the
// caller that needs a whole distribution's total so a prefix of it can be
// drawn from.
const (
	SampleParMinP = 0
	SampleParTopP = 1
	SampleParU    = 2
	SampleParSum  = 3
	SampleParN    = 4

	SampleOutRes      = 0
	SampleOutM        = 1
	SampleOutCut      = 2
	SampleOutResolved = 3
	SampleOutN        = 4
)

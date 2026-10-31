package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// The sampler's gates: repeat penalty, top-k, min-p, top-p, vocabulary scale,
// ordering and seed reproducibility. They were written against the Go
// implementation before the kernels replaced it, so they record behaviour.
//
// Every assertion is a set (which candidates survive a cut), a frequency (how
// often each is drawn) or the strict total order (higher logit first, lower id
// on a tie).

// refPenalize is llama.cpp's repeat-penalty rule, written out: a POSITIVE logit
// is divided and a NEGATIVE one multiplied, so the token always moves toward
// less likely rather than flipping its sign around zero. Applied to exactly the
// ids in hist and to nothing else.
func refPenalize(logits []float32, hist []int32, pen float64) []float32 {
	out := append([]float32(nil), logits...)
	if pen <= 1 {
		return out
	}
	for _, t := range hist {
		v := out[t]
		if v > 0 {
			v /= float32(pen)
		} else {
			v *= float32(pen)
		}
		out[t] = v
	}
	return out
}

// refArgmax is the strict total order's maximum: highest value, lowest id.
func refArgmax(v []float32) int32 {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return int32(best)
}

// The draw counts. smallN sees every survivor of a handful of candidates many
// times over; vocabN is enough for a support assertion at vocabulary scale;
// distN puts the standard error of a proportion well inside the frequency
// check's tolerance.
const (
	smallN = 60000
	vocabN = 600
	distN  = 8000
)

// support runs n draws and returns the set of ids that came out, with counts.
func support(s *Sampler, logits []float32, n int) map[int32]int {
	c := map[int32]int{}
	for i := 0; i < n; i++ {
		c[s.Sample(logits)]++
	}
	return c
}

func keys(m map[int32]int) []int32 {
	out := make([]int32, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

func sameSet(got map[int32]int, want []int32) bool {
	if len(got) != len(want) {
		return false
	}
	for _, w := range want {
		if got[w] == 0 {
			return false
		}
	}
	return true
}

// TestSamplerRepeatPenaltyFollowsTheSignRule pins the rule and its SCOPE with
// TopK = 1, which makes Sample return the argmax of the PENALIZED logits and
// nothing else: one draw is then a direct read of the penalty's effect rather
// than a distribution to estimate. It is run against three violations (sign
// rule flipped, a non-history id penalized, every id penalized).
func TestSamplerRepeatPenaltyFollowsTheSignRule(t *testing.T) {
	logits := []float32{2, -3, 0.5, -0.25, 4, -8, 1.5, 7}
	cases := []struct {
		hist []int32
		pen  float64
	}{
		{nil, 1.3},
		{[]int32{7}, 1.3},        // the leader is positive: divided, 7 -> 5.38
		{[]int32{7, 4}, 1.3},     // and then the runner-up too
		{[]int32{7, 4, 0}, 1.3},  // ... down to 2
		{[]int32{7}, 2},          // enough to put the leader BELOW token 4
		{[]int32{7, 4}, 2},       // and then the new leader too
		{[]int32{2}, 2},          // a penalty that does not touch the leader
		{[]int32{5}, 3},          // a NEGATIVE logit: multiplied, further down
		{[]int32{7, 4, 0, 6}, 4}, // everything positive knocked below 0.5
		{[]int32{7}, 1},          // pen == 1 is disabled: Observe stores nothing
		{[]int32{1, 3, 5}, 2},    // only negatives
		{[]int32{7, 7, 7, 7}, 2}, // a repeated id is still ONE penalty
	}
	for _, c := range cases {
		s := &Sampler{Temp: 1, TopK: 1, RepeatPen: c.pen, RepeatLastN: 64, Seed: 3}
		for _, h := range c.hist {
			s.Observe(h)
		}
		// A repeated id must not compound: the history is a SET for this rule.
		uniq := map[int32]bool{}
		var hs []int32
		for _, h := range c.hist {
			if !uniq[h] {
				uniq[h] = true
				hs = append(hs, h)
			}
		}
		want := refArgmax(refPenalize(logits, hs, c.pen))
		if got := s.Sample(logits); got != want {
			t.Errorf("hist %v pen %v: drew %d, want the penalized argmax %d (penalized %v)",
				c.hist, c.pen, got, want, refPenalize(logits, hs, c.pen))
		}
	}

	// The violations, run against the same cases: each must disagree somewhere.
	viol := []struct {
		name string
		f    func(logits []float32, hist []int32, pen float64) []float32
	}{
		{"sign rule flipped", func(l []float32, h []int32, pen float64) []float32 {
			out := append([]float32(nil), l...)
			if pen <= 1 {
				return out
			}
			for _, t := range h {
				if out[t] > 0 {
					out[t] *= float32(pen)
				} else {
					out[t] /= float32(pen)
				}
			}
			return out
		}},
		{"penalizes an id not in the history", func(l []float32, h []int32, pen float64) []float32 {
			out := refPenalize(l, h, pen)
			if pen > 1 {
				out[7] /= float32(pen)
			}
			return out
		}},
		{"penalizes every id", func(l []float32, h []int32, pen float64) []float32 {
			out := append([]float32(nil), l...)
			if pen <= 1 {
				return out
			}
			for i, v := range out {
				if v > 0 {
					out[i] = v / float32(pen)
				} else {
					out[i] = v * float32(pen)
				}
			}
			return out
		}},
	}
	for _, v := range viol {
		fired := false
		for _, c := range cases {
			s := &Sampler{Temp: 1, TopK: 1, RepeatPen: c.pen, RepeatLastN: 64, Seed: 3}
			for _, h := range c.hist {
				s.Observe(h)
			}
			if s.Sample(logits) != refArgmax(v.f(logits, c.hist, c.pen)) {
				fired = true
				break
			}
		}
		if !fired {
			t.Errorf("violation %q is invisible to this gate", v.name)
		}
	}
}

// TestSamplerTopKKeepsExactlyKCandidates: with distinct logits the top-k set is
// unambiguous, so the SUPPORT of many draws is exactly it. The boundary is k
// itself: k = 1, k = len, k > len and k = 0 (disabled).
func TestSamplerTopKKeepsExactlyKCandidates(t *testing.T) {
	logits := []float32{2, 1, 0.5, 3.5, -1, 3, 0.25, -4}
	order := []int32{3, 5, 0, 1, 2, 6, 4, 7} // descending, distinct
	for _, k := range []int{1, 2, 3, 5, 8, 12, 0} {
		want := order
		if k > 0 && k < len(order) {
			want = order[:k]
		}
		s := &Sampler{Temp: 2, TopK: k, Seed: int64(k) + 11}
		got := support(s, logits, smallN)
		if !sameSet(got, want) {
			t.Errorf("top-k %d drew %v, want exactly %v", k, keys(got), want)
		}
	}
	// The violation: an off-by-one cut keeps k+1, which shows up as an extra id.
	s := &Sampler{Temp: 2, TopK: 3, Seed: 5}
	got := support(s, logits, smallN)
	if sameSet(got, order[:4]) {
		t.Errorf("top-k 3 drew the top FOUR %v: the cut is off by one", keys(got))
	}
}

// TestSamplerMinPCutsAtItsThresholdInclusively: min-p keeps p >= MinP*p[0], so
// a candidate sitting EXACTLY on the threshold survives (`v < cut` breaks, and
// equality is not less-than). The boundary is hit exactly by a tie at the top
// with MinP = 1: the cut is p[0] bit for bit, so the tied leaders sit on it.
func TestSamplerMinPCutsAtItsThresholdInclusively(t *testing.T) {
	logits := []float32{5, 5, 3, 1, -2, 5}
	s := &Sampler{Temp: 1, MinP: 1, Seed: 17}
	got := support(s, logits, smallN)
	if !sameSet(got, []int32{0, 1, 5}) {
		t.Fatalf("min-p 1 drew %v, want exactly the three tied leaders [0 1 5]", keys(got))
	}
	// And a threshold strictly between two probabilities cuts exactly there.
	// p is softmax(logits) at T = 1: the three 5s share the mass, 3 is e^-2 of
	// one of them and 1 is e^-4.
	for _, tc := range []struct {
		minp float64
		want []int32
	}{
		{0.9, []int32{0, 1, 5}},           // below the leaders only
		{0.2, []int32{0, 1, 5}},           // p2/p0 is e^-2 = 0.135 < 0.2: out
		{0.1, []int32{0, 1, 5, 2}},        // and 0.135 >= 0.1 is in
		{0.01, []int32{0, 1, 5, 2, 3}},    // e^-4 = 0.018 >= 0.01
		{0, []int32{0, 1, 5, 2, 3, 4}},    // disabled
		{1e-6, []int32{0, 1, 5, 2, 3, 4}}, // e^-7 = 9.1e-4 is still in
		{0.002, []int32{0, 1, 5, 2, 3}},   // 9.1e-4 < 0.002 is out
	} {
		s := &Sampler{Temp: 1, MinP: tc.minp, Seed: 23}
		got := support(s, logits, smallN)
		if !sameSet(got, tc.want) {
			t.Errorf("min-p %v drew %v, want %v", tc.minp, keys(got), tc.want)
		}
	}
}

// TestSamplerTopPCutsAtItsThresholdInclusively: top-p keeps the shortest prefix
// whose cumulative REACHES TopP, so the candidate that takes the cumulative to
// exactly TopP is KEPT (`acc >= TopP` cuts at i+1). Three tied logits give
// bit-identical probabilities q, and q+q is exact, so TopP = q+q lands on the
// boundary with no rounding.
func TestSamplerTopPCutsAtItsThresholdInclusively(t *testing.T) {
	logits := []float32{5, 5, 5, 1, -2}
	// q is what the generated softmax produces for one of the three leaders.
	p := []float32{5, 5, 5, 1, -2}
	nn.Scale32JIT(p, 1)
	nn.Softmax32JIT(p, len(p))
	q := p[0]
	if p[1] != q || p[2] != q {
		t.Fatalf("the three tied logits gave %v, %v, %v -- the tie is the construction", p[0], p[1], p[2])
	}
	exact := float64(q + q)
	s := &Sampler{Temp: 1, TopP: exact, Seed: 29}
	got := support(s, logits, smallN)
	if len(got) != 2 {
		t.Fatalf("top-p at exactly the second cumulative (%v) drew %v ids, want 2 -- got %v",
			exact, len(got), keys(got))
	}
	// Every drawn id must be one of the three tied leaders, and the two of them
	// that survive are decided by the order, not by this gate.
	for _, id := range keys(got) {
		if id > 2 {
			t.Errorf("top-p drew %d, which is not one of the tied leaders", id)
		}
	}
	// One ulp above that cumulative needs a third candidate -- and it must be a
	// float32 ulp: the field is a float64 and the comparison is float32, so
	// math.Nextafter's step rounds straight back to the value it started from.
	up := float64(math.Float32frombits(math.Float32bits(q+q) + 1))
	if float32(up) == q+q {
		t.Fatal("the ulp step did not move the float32 the comparison uses")
	}
	s = &Sampler{Temp: 1, TopP: up, Seed: 31}
	if got := support(s, logits, smallN); len(got) != 3 {
		t.Errorf("top-p one ulp above the second cumulative drew %v, want three ids", keys(got))
	}
	// And the disabled values keep everything.
	for _, tp := range []float64{0, 1} {
		s := &Sampler{Temp: 1, TopP: tp, Seed: 37}
		if got := support(s, logits, smallN); len(got) != 5 {
			t.Errorf("top-p %v drew %v, want all five", tp, keys(got))
		}
	}
}

// TestSamplerTopKThenTopP pins the ORDER of the two filters, which is
// llama.cpp's and is not cosmetic: top-p over the top-k renormalized
// distribution keeps a different set than top-p over the whole vocabulary.
func TestSamplerTopKThenTopP(t *testing.T) {
	logits := []float32{5, 4.5, 4, 3, 2, 1, 0}
	// Over the top-3 alone the mass concentrates, so a TopP of 0.8 keeps two;
	// over all seven the same 0.8 reaches further down.
	s := &Sampler{Temp: 1, TopK: 3, TopP: 0.8, Seed: 41}
	withK := support(s, logits, smallN)
	s = &Sampler{Temp: 1, TopP: 0.8, Seed: 41}
	noK := support(s, logits, smallN)
	if len(withK) >= len(noK) {
		t.Errorf("top-k 3 then top-p 0.8 kept %d, top-p 0.8 alone kept %d: "+
			"the renormalization between them is not happening", len(withK), len(noK))
	}
	for _, id := range keys(withK) {
		if id > 2 {
			t.Errorf("top-k 3 then top-p drew %d, outside the top three", id)
		}
	}
}

// TestSamplerCutsBeforeTheTemperature pins llama.cpp's order of the cuts and
// the temperature: top-p and min-p read the softmax at temperature 1, and the
// temperature shapes only the draw over their survivors. At temperature 4 the
// tempered softmax is flat enough that the same top-p would keep four of
// these five and the same min-p all five; at temperature 1 each keeps two,
// drawn in the tempered proportion e^(3/4) : e^(2/4). Run with a top-k and
// over the whole row, the two paths that cut.
func TestSamplerCutsBeforeTheTemperature(t *testing.T) {
	logits := []float32{3, 2, 1, 0, -1}
	want0 := math.Exp(0.75) / (math.Exp(0.75) + math.Exp(0.5))
	for _, tc := range []struct {
		name       string
		topk       int
		topp, minp float64
	}{
		{"top-p over a top-k", 4, 0.8, 0},
		{"top-p over the row", 0, 0.8, 0},
		{"min-p over a top-k", 4, 0, 0.2},
		{"min-p over the row", 0, 0, 0.2},
	} {
		s := &Sampler{Temp: 4, TopK: tc.topk, TopP: tc.topp, MinP: tc.minp, Seed: 43}
		got := support(s, logits, smallN)
		if !sameSet(got, []int32{0, 1}) {
			t.Errorf("%s at temperature 4 drew %v, want [0 1]: the cut read the tempered softmax", tc.name, keys(got))
			continue
		}
		if f := float64(got[0]) / smallN; math.Abs(f-want0) > 0.01 {
			t.Errorf("%s drew token 0 at %.4f, want %.4f: the survivors are not drawn at the temperature", tc.name, f, want0)
		}
	}
}

// TestSamplerIsDeterministic: the same seed and the same logits give the same
// sequence, in one process, twice, and across two independent Samplers.
func TestSamplerIsDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	logits := make([]float32, 4096)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64()) * 3
	}
	run := func() []int32 {
		s := &Sampler{Temp: 0.8, TopK: 64, TopP: 0.95, MinP: 0.01,
			RepeatPen: 1.2, RepeatLastN: 16, Seed: 12345}
		out := make([]int32, 64)
		for i := range out {
			out[i] = s.Sample(logits)
			s.Observe(out[i])
		}
		return out
	}
	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("two Samplers with seed 12345 diverged at draw %d: %d vs %d", i, a[i], b[i])
		}
	}
	// A different seed must not give the same sequence, or the seed is ignored.
	s := &Sampler{Temp: 0.8, TopK: 64, TopP: 0.95, MinP: 0.01,
		RepeatPen: 1.2, RepeatLastN: 16, Seed: 54321}
	same := true
	for i := range a {
		v := s.Sample(logits)
		s.Observe(v)
		if v != a[i] {
			same = false
		}
	}
	if same {
		t.Fatal("seed 54321 produced seed 12345's sequence: the seed is not reaching the draw")
	}
}

// TestSamplerDrawsTheTemperedSoftmaxAtVocabScale is the distribution gate at a
// real vocabulary size, where a ragged vector, a segment boundary or an index
// above 32767 can go wrong.
func TestSamplerDrawsTheTemperedSoftmaxAtVocabScale(t *testing.T) {
	const nv, temp = 128256, 1.0
	logits := make([]float32, nv)
	rng := rand.New(rand.NewSource(7))
	for i := range logits {
		logits[i] = float32(rng.NormFloat64()) * 0.5
	}
	// Six deliberate peaks, spread so they land in different vector lanes and
	// different segments of any segmented ordering.
	peaks := []int32{0, 1, 255, 4097, 65536, nv - 1}
	for j, id := range peaks {
		logits[id] = float32(8 - j)
	}
	s := &Sampler{Temp: temp, TopK: 6, Seed: 101}
	counts := support(s, logits, distN)
	if !sameSet(counts, peaks) {
		t.Fatalf("top-6 over %d logits drew %v, want the six peaks %v", nv, keys(counts), peaks)
	}
	var z float64
	for _, id := range peaks {
		z += math.Exp(float64(logits[id]) / temp)
	}
	for _, id := range peaks {
		want := math.Exp(float64(logits[id])/temp) / z
		got := float64(counts[id]) / distN
		if math.Abs(got-want) > 0.015 {
			t.Errorf("token %d drawn %.4f of the time, want %.4f", id, got, want)
		}
	}
}

// TestSamplerOrdersTheWholeVocabularyByTheStrictTotalOrder holds the sampler to
// the strict total order the MoE router's top-k uses (higher logit first, lower
// id on a tie). The order is observable because the inverse-CDF walk decides
// which token a seed draws. With k tied leaders and TopK = k, exactly the k
// lowest tied ids may be drawn.
func TestSamplerOrdersTheWholeVocabularyByTheStrictTotalOrder(t *testing.T) {
	const nv = 128256
	logits := make([]float32, nv)
	rng := rand.New(rand.NewSource(13))
	for i := range logits {
		logits[i] = float32(rng.NormFloat64())*0.5 - 20
	}
	// Twelve tokens tied at the top, scattered; the top-4 of them by the total
	// order are the four lowest ids.
	tied := []int32{9, 77, 1024, 1025, 30011, 30012, 64000, 70001, 99999, 120000, 128254, 128255}
	for _, id := range tied {
		logits[id] = 12
	}
	for _, k := range []int{1, 2, 4, 7, 12} {
		s := &Sampler{Temp: 1, TopK: k, Seed: int64(k) * 7}
		got := support(s, logits, vocabN)
		if !sameSet(got, tied[:k]) {
			t.Errorf("top-%d over a 12-way tie drew %v, want the %d lowest tied ids %v",
				k, keys(got), k, tied[:k])
		}
	}
	// And a tie that is NOT at the top: the block below the leaders.
	for i := range logits {
		logits[i] = float32(rng.NormFloat64())*0.5 - 20
	}
	logits[500] = 12
	for _, id := range tied {
		logits[id] = 11
	}
	s := &Sampler{Temp: 1, TopK: 3, Seed: 3}
	got := support(s, logits, vocabN)
	if !sameSet(got, []int32{500, tied[0], tied[1]}) {
		t.Errorf("top-3 over one leader and a 12-way tie drew %v, want [500 %d %d]",
			keys(got), tied[0], tied[1])
	}
}

// TestSamplerZeroValueIsGreedy: the zero Sampler and any Temp <= 0 short-circuit
// to the argmax before any other field is read, which is what keeps
// TestGreedyMatchesLlamaCpp and `jitllm verify` comparisons deterministic.
func TestSamplerZeroValueIsGreedy(t *testing.T) {
	logits := []float32{2, 1, 0.5, 3.5, 3.5}
	for _, s := range []*Sampler{
		nil,
		{},
		{Temp: 0, TopK: 2, TopP: 0.5, MinP: 0.4, RepeatPen: 2, Seed: 9},
		{Temp: -1},
	} {
		if got := s.Sample(logits); got != 3 {
			t.Errorf("%+v sampled %d, want the argmax 3 (ties to the lower id)", s, got)
		}
	}
}

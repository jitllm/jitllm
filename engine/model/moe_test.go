package model

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestMoEFaultsAreVisible checks the router directly, since the llama.cpp
// oracles cannot: Qwen3-MOE-4x0.6B's experts are near-duplicates and
// tiny-qwen3moe's random weights make every token a near-tie. It computes the
// FFN with the shipped moe(), with a mirror of it, and with each way of getting
// it wrong. Every fault must move the output and the controls must not.
func TestMoEFaultsAreVisible(t *testing.T) {
	path := testmodels.Path("tiny-qwen3moe-f32.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %s (scripts/tiny-moe-gguf.py builds it) (set JITLLM_MODELS to the model directory)", path)
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.Cfg.MoE() {
		t.Fatalf("%s is not a mixture of experts", path)
	}
	s := m.NewState(8)
	defer s.Close()
	l := &m.layers[0]

	// A deterministic input scaled so the router logits spread; a constant
	// vector would make every expert equally likely.
	h := make([]float32, m.Cfg.NEmbd)
	for i := range h {
		h[i] = float32(math.Sin(float64(i)*1.7+0.3) * 2)
	}

	// Fault the block in first, as Forward does: moe() ensures only the
	// selected experts, and the router arrives through pageIn.
	if err := m.pageIn(0); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, m.Cfg.NEmbd)
	if err := s.moe(0, l, h, h, got); err != nil {
		t.Fatal(err)
	}

	// The faults read experts the router did not pick, and an expert's view is
	// bound only while its page is held, so hold every expert of the layer.
	all := make([]int32, m.Cfg.NExpert)
	for i := range all {
		all[i] = int32(i)
	}
	var hold expertHold
	err = m.ensureExperts(0, all, &hold)
	defer hold.release()
	if err != nil {
		t.Fatal(err)
	}

	// faultNone must reproduce the shipped path bit for bit, or the rest of
	// the table compares against something else.
	base := s.moeRef(l, h, faultNone)
	if d := maxAbs(got, base); d > 0 {
		t.Fatalf("the reference mirror disagrees with moe() by %.3e; the fault table below would be meaningless", d)
	}

	for _, f := range []struct {
		fault   moeFault
		name    string
		mustSee bool
		why     string
	}{
		// Not a fault: softmaxing all logits then renormalising the selected
		// k equals softmaxing the selected logits (softmax is monotonic). Kept
		// as a control because the order looks load-bearing.
		{faultSoftmaxAfterTopK, "softmax over the selected k (identity)", false,
			"CONTROL: algebraically the same as softmax-then-renormalise"},
		{faultDownFromWrongExpert, "down taken from the next expert", true,
			"gate and up are [n_embd, n_ff_exp] and down is [n_ff_exp, n_embd], so one " +
				"stride formula reused for all three lands inside the mapping and never faults"},
		{faultNoRenorm, "top-k weights not renormalised", true,
			"rescales every FFN output by the top-k probability mass"},
		{faultBottomK, "lowest-k experts instead of highest", true,
			"the single most obvious transcription slip in an argsort"},
		{faultOneExpert, "expert 0 for every token", true,
			"what a 3-D bank handed straight to a matvec computes"},
		{faultReverseOrder, "experts accumulated in reverse order", false,
			"CONTROL: float reassociation only, and it must stay green"},
	} {
		// The bar separates reassociation from routing: the FMA accumulate
		// makes reversed order differ by ~1e-9, while the smallest real fault
		// is ~9e-3.
		const bar = 1e-6
		d := maxAbs(s.moeRef(l, h, f.fault), base)
		switch {
		case f.mustSee && d < bar:
			t.Errorf("INVISIBLE: %s changed the FFN by %.3e -- nothing would catch it\n  (%s)", f.name, d, f.why)
		case !f.mustSee && d > bar:
			t.Errorf("the control moved by %.3e; a gate this sensitive is measuring reassociation, not routing\n  (%s)", d, f.why)
		default:
			t.Logf("%-45s max|d| %.3e", f.name, d)
		}
	}
}

type moeFault int

const (
	faultNone moeFault = iota
	faultSoftmaxAfterTopK
	faultNoRenorm
	faultBottomK
	faultOneExpert
	faultReverseOrder
	faultDownFromWrongExpert
)

// moeRef mirrors moe() over the same weights and the same matvec seam, with one
// deliberate defect. It is duplication on purpose: a fault injected into the
// shipping function would be shipping code that exists for a test, and a hook
// that can disable the renormalisation in production is worse than the bug.
func (s *State) moeRef(l *layer, h []float32, f moeFault) []float32 {
	c := s.c
	// Capacity for the generated softmax's padding, as State.moeProbs carries:
	// the mirror must call the same primitive.
	probs := make([]float32, c.NExpert, nn.SoftmaxPad(c.NExpert))
	if err := s.mv(probs, l.router, h); err != nil {
		panic(err)
	}
	if f != faultSoftmaxAfterTopK {
		s.softmax(probs, len(probs))
	}
	sel := make([]int, c.NExpertUsed)
	switch f {
	case faultBottomK:
		neg := make([]float32, len(probs))
		for i, v := range probs {
			neg[i] = -v
		}
		moeTopKRef(neg, sel)
	case faultOneExpert:
		for i := range sel {
			sel[i] = 0
		}
	default:
		moeTopKRef(probs, sel)
	}
	w := make([]float32, len(sel), nn.SoftmaxPad(len(sel)))
	for i, e := range sel {
		w[i] = probs[e]
	}
	if f == faultSoftmaxAfterTopK {
		s.softmax(w, len(w)) // softmaxing the SELECTED k, which also sums to 1
	} else if f != faultNoRenorm {
		var sum float32
		for _, v := range w {
			sum += v
		}
		for i := range w {
			w[i] /= sum
		}
	}
	order := make([]int, len(sel))
	for i := range order {
		order[i] = i
		if f == faultReverseOrder {
			order[i] = len(sel) - 1 - i
		}
	}
	gate := make([]float32, c.NFFNExp)
	up := make([]float32, c.NFFNExp)
	down := make([]float32, c.NEmbd)
	out := make([]float32, c.NEmbd)
	for _, i := range order {
		x := &l.experts[sel[i]]
		if f == faultDownFromWrongExpert {
			x = &l.experts[(sel[i]+1)%c.NExpert]
		}
		s.jit.NewInput()
		if err := s.mv(gate, x.gate, h); err != nil {
			panic(err)
		}
		if err := s.mv(up, x.up, h); err != nil {
			panic(err)
		}
		s.actmul(gate, up, c.Act)
		s.jit.NewInput()
		if err := s.mv(down, x.down, gate); err != nil {
			panic(err)
		}
		s.axpy(out, down, w[i])
	}
	return out
}

func maxAbs(a, b []float32) float64 {
	var d float64
	for i := range a {
		d = math.Max(d, math.Abs(float64(a[i]-b[i])))
	}
	return d
}

// moeTopKRef is the Go top-k selection engine/model/moe.go used before it became a
// generated kernel, kept here because the mirror must not call the kernel it is
// the reference for. Same tie rule: `p[e] > p[best]` scanning ascending, so the
// lowest index wins. jit_old/cpu/moetopkref_test.go carries the same
// transcription for the kernel's own gate; this copy lets the mirror select on a
// negated vector (faultBottomK).
func moeTopKRef(p []float32, dst []int) []int {
	for i := range dst {
		best := -1
		for e := range p {
			seen := false
			for _, v := range dst[:i] {
				if v == e {
					seen = true
					break
				}
			}
			if seen {
				continue
			}
			if best < 0 || p[e] > p[best] {
				best = e
			}
		}
		dst[i] = best
	}
	return dst
}

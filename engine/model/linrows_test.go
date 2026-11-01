package model

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// A recurrent block's projections as GEMMs over a prompt chunk (linrows.go)
// against the row-by-row arm (WithLinearRowByRow), on one model and two
// States: the projections, every recurrent state and convolution window at
// every prompt partition boundary, the logits, a teacher-forced continuation
// and a prefix-cache resume.
//
// Under nn.WithGEMMExact the GEMM keeps the matvec's float epilogue, so the
// two arms are the same arithmetic and the bound is equality (linExact): what
// was measured on every model below, on amd64 and the M4. The shipping GEMM
// may take a tuned epilogue whose rounding differs from the matvec's in the
// last bit, and a model whose next block quantizes its activations to int8
// turns that bit into a quantization step; linShipped is that envelope, set
// from the worst measured on any model and host with a margin of about two,
// as |ref - got| <= abs + rel*max|ref| over each vector: LFM2-350M Q4_K_M on
// the M4 measured proj 0.29, window 0.043, logits 0.079 and continuation
// 0.105, every other model 0 on both hosts (no model with a recurrent state
// measured a nonzero state error; its bound is the window's). The measured
// figures are logged on every run.
type linTol struct{ abs, rel float64 }

var linExact = map[string]linTol{}

var linShipped = map[string]linTol{
	"proj":   {0, 0.6},
	"state":  {0, 0.1},
	"conv":   {0, 0.1},
	"logits": {0, 0.2},
	"cont":   {0, 0.25},
}

// linEnv is the worst error seen per category, relative to the reference's
// largest magnitude in the same vector, and the bounds it holds them to.
type linEnv struct {
	worst map[string]float64
	tol   map[string]linTol
}

func newLinEnv(tol map[string]linTol) *linEnv {
	return &linEnv{worst: map[string]float64{}, tol: tol}
}

// cmp checks got against ref under the category's bound and records the
// worst relative error. It returns the absolute worst difference.
func (e *linEnv) cmp(t *testing.T, cat, what string, ref, got []float32) float64 {
	t.Helper()
	if len(ref) != len(got) {
		t.Fatalf("%s: %d values against %d", what, len(got), len(ref))
	}
	var worst, big float64
	for i := range ref {
		r, g := float64(ref[i]), float64(got[i])
		if math.IsNaN(r) || math.IsInf(r, 0) {
			t.Fatalf("%s: the reference is not finite at %d (%v): a degenerate oracle", what, i, r)
		}
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s: value %d is %v", what, i, g)
		}
		worst = max(worst, math.Abs(r-g))
		big = max(big, math.Abs(r))
	}
	tol := e.tol[cat]
	if worst > tol.abs+tol.rel*big {
		t.Errorf("%s: worst difference %.3e over max|ref| %.3e (bound %.0e + %.0e*max)", what, worst, big, tol.abs, tol.rel)
	}
	if big > 0 {
		e.worst[cat] = max(e.worst[cat], worst/big)
	}
	return worst
}

func (e *linEnv) String() string {
	var b strings.Builder
	for _, k := range []string{"proj", "state", "conv", "logits", "cont"} {
		if v, ok := e.worst[k]; ok {
			fmt.Fprintf(&b, " %s %.2e", k, v)
		}
	}
	return b.String()
}

// linPair is a State in each arm.
func linPair(m *Model, n int) (ref, bat *State) {
	ref, bat = m.NewState(n), m.NewState(n)
	ref.lin.ref, bat.lin.ref = true, false
	return ref, bat
}

// linStates compares every recurrent layer's state and window.
func linStates(t *testing.T, e *linEnv, what string, ref, bat *State) {
	t.Helper()
	linSnapCmp(t, e, what, linSnap(ref), bat)
}

// linSnapshot is a State's recurrent half, copied.
type linSnapshot struct{ state, conv [][]float32 }

func linSnap(s *State) linSnapshot {
	var o linSnapshot
	for li := range s.rstate {
		o.state = append(o.state, append([]float32(nil), s.rstate[li]...))
		o.conv = append(o.conv, append([]float32(nil), s.rconv[li]...))
	}
	return o
}

func linSnapCmp(t *testing.T, e *linEnv, what string, ref linSnapshot, bat *State) {
	t.Helper()
	n := 0
	for li := range ref.state {
		if bat.rstate[li] == nil {
			continue
		}
		n++
		e.cmp(t, "state", fmt.Sprintf("%s: layer %d state", what, li), ref.state[li], bat.rstate[li])
		e.cmp(t, "conv", fmt.Sprintf("%s: layer %d window", what, li), ref.conv[li], bat.rconv[li])
	}
	if n == 0 {
		t.Fatalf("%s: no recurrent layer: this gate is about nothing", what)
	}
}

// linPrompt is n tokens below vocab: a pseudo-random run, then one token
// repeated (a recurrence driven to its fixed point), then a cycle of three.
func linPrompt(n, vocab int) []int32 {
	p := make([]int32, n)
	x := uint32(12345)
	for i := range p {
		switch {
		case i < n/3:
			x = x*1664525 + 1013904223
			p[i] = int32(3 + int(x>>8)%(vocab-3))
		case i < 2*n/3:
			p[i] = int32(5 % vocab)
		default:
			p[i] = int32(3 + i%3)
		}
	}
	return p
}

// TestLinearRowsResumeFromThePrefixCache: a prompt resumed from a stored
// prefix, its recurrent summary included, in the batched arm against the
// row-by-row arm run whole.
func TestLinearRowsResumeFromThePrefixCache(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{ctx: 256})
	defer m.Close()
	const page = 8
	defer m.setKVPageForTest(page)()
	prompt := linPrompt(120, hyVocab)
	ref, _ := linPair(m, 200)
	defer ref.Close()
	want, err := ref.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	st := NewMemStore()
	run := func() (*State, []float32) {
		s := m.NewState(200)
		s.lin.ref = false
		s.SetKVStore(st)
		if err := s.SetCacheKey("lin/v1"); err != nil {
			t.Fatal(err)
		}
		lg, err := s.PrefillCached(prompt[:80])
		if err != nil {
			t.Fatal(err)
		}
		return s, lg
	}
	first, _ := run()
	first.Close()
	s, _ := run()
	defer s.Close()
	if s.KVRestored() == 0 {
		t.Fatal("nothing was restored: the resume this gate is about never ran")
	}
	got, err := s.Prefill(prompt[80:])
	if err != nil {
		t.Fatal(err)
	}
	if s.lin.chunks == 0 {
		t.Fatal("the batched arm ran no chunk")
	}
	e := newLinEnv(linExact)
	e.cmp(t, "logits", "resumed against whole", want, got)
	linStates(t, e, "resumed against whole", ref, s)
	t.Logf("restored %d positions; worst error relative to max|ref|:%v", s.KVRestored(), e)
}

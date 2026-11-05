package engine

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/nn"
)

// A hybrid's prompt resumed from the prefix cache, its recurrent summary
// included, against the same prompt run whole on the composed CPU: the logits
// and every recurrent state and convolution window, to equality (linExact).
type linTol struct{ abs, rel float64 }

func linExact() map[string]linTol { return map[string]linTol{} }

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

// linStates compares every recurrent layer's state and window.
func linStates(t *testing.T, e *linEnv, what string, ref, bat *Instance) {
	t.Helper()
	linSnapCmp(t, e, what, linSnap(ref), bat)
}

// linSnapshot is a session's recurrent half, copied.
type linSnapshot struct{ state, conv [][]float32 }

func linSnap(s *Instance) linSnapshot {
	var o linSnapshot
	for li := range s.view.Blocks() {
		o.state = append(o.state, append([]float32(nil), s.view.State(li)...))
		o.conv = append(o.conv, append([]float32(nil), s.view.Conv(li)...))
	}
	return o
}

func linSnapCmp(t *testing.T, e *linEnv, what string, ref linSnapshot, bat *Instance) {
	t.Helper()
	n := 0
	for li := range ref.state {
		if bat.view.State(li) == nil {
			continue
		}
		n++
		e.cmp(t, "state", fmt.Sprintf("%s: layer %d state", what, li), ref.state[li], bat.view.State(li))
		e.cmp(t, "conv", fmt.Sprintf("%s: layer %d window", what, li), ref.conv[li], bat.view.Conv(li))
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
// prefix, its recurrent summary included, against the prompt run whole.
func TestLinearRowsResumeFromThePrefixCache(t *testing.T) {
	m, openOpts := hybridModelOpt(t, hyOpt{ctx: 256})
	defer m.Close()
	const page = 8
	prompt := linPrompt(120, hyVocab)
	ref := testInstance(t, m, 200, append(append([]Option(nil), openOpts...), withKVPage(page))...)
	defer ref.Close()
	want, err := ref.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	st := NewMemStore()
	run := func() (*Instance, []float32) {
		s := testInstance(t, m, 200, append(append([]Option(nil), openOpts...), withKVPage(page))...)
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
	s0, d0 := s.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
	got, err := s.Prefill(prompt[80:])
	if err != nil {
		t.Fatal(err)
	}
	// The resumed rows ran on the composed CPU, each of their calls every
	// block, and nothing declined.
	if s1, d1 := s.Account(nn.CPU).Steps.Of(nn.ComposedBlock); s1 == s0 || (s1-s0)%m.Cfg.NLayer != 0 || d1 != d0 {
		t.Fatalf("the resumed rows: the composed CPU served %d block steps over %d blocks and declined %v", s1-s0, m.Cfg.NLayer, d1)
	}
	e := newLinEnv(linExact())
	e.cmp(t, "logits", "resumed against whole", want, got)
	linStates(t, e, "resumed against whole", ref, s)
	t.Logf("restored %d positions; worst error relative to max|ref|:%v", s.KVRestored(), e)
}

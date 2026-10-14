package model

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
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

// linCheck runs the whole comparison on m with a prompt of n tokens and a
// continuation of cont, and returns the envelope.
func linCheck(t *testing.T, m *Model, n, cont int, tol map[string]linTol) *linEnv {
	t.Helper()
	e := newLinEnv(tol)
	vocab := min(m.Cfg.NVocab, 1000)
	prompt := linPrompt(n, vocab)
	follow := linPrompt(cont+7, vocab)[7:]

	// The projections: every trace a linear block writes, row by row, on a
	// prompt short enough to hold.
	{
		short := prompt[:min(n, 48)]
		grab := func(s *State) map[string][][]float32 {
			got := map[string][][]float32{}
			m.Trace(func(l int, name string, v []float32) {
				switch name {
				case "qkv_mixed", "xBC", "x", "bx", "linear_attn_out", "ssm_out", "shortconv_out", "final_output", "ssm_norm":
					k := fmt.Sprintf("%d/%s", l, name)
					got[k] = append(got[k], append([]float32(nil), v...))
				}
			})
			defer m.Trace(nil)
			if _, err := s.Prefill(short); err != nil {
				t.Fatal(err)
			}
			return got
		}
		ref, bat := linPair(m, n+cont+1)
		rt, bt := grab(ref), grab(bat)
		if bat.lin.chunks == 0 || ref.lin.chunks != 0 {
			t.Fatalf("arms not selected: batched ran %d chunks, row-by-row %d", bat.lin.chunks, ref.lin.chunks)
		}
		if len(rt) == 0 {
			t.Fatal("no linear block traced: the projection comparison is about nothing")
		}
		for k, rows := range rt {
			if len(bt[k]) != len(rows) {
				t.Fatalf("%s: %d rows traced batched, %d row by row", k, len(bt[k]), len(rows))
			}
			for r := range rows {
				e.cmp(t, "proj", fmt.Sprintf("%s row %d", k, r), rows[r], bt[k][r])
			}
		}
		ref.Close()
		bat.Close()
	}

	// Partitions of the prompt: both arms take the same pieces, and the
	// batched one also against the reference's whole prompt.
	cw := m.NewState(1).chunkWidth()
	parts := [][]int{{n}, {1, n - 1}, {7, 13, n - 20}}
	if n > cw+3 {
		parts = append(parts, []int{cw - 1, 3, n - cw - 2})
	}
	var whole []float32
	var wholeSnap linSnapshot
	for pi, part := range parts {
		ref, bat := linPair(m, n+cont+1)
		at := 0
		var rl, bl []float32
		for _, k := range part {
			var err error
			if rl, err = ref.Prefill(prompt[at : at+k]); err != nil {
				t.Fatal(err)
			}
			if bl, err = bat.Prefill(prompt[at : at+k]); err != nil {
				t.Fatal(err)
			}
			at += k
			what := fmt.Sprintf("partition %v at %d", part, at)
			linStates(t, e, what, ref, bat)
			e.cmp(t, "logits", what+": logits", rl, bl)
		}
		if bat.lin.chunks == 0 || ref.lin.chunks != 0 {
			t.Fatalf("arms not selected: batched ran %d chunks, row-by-row %d", bat.lin.chunks, ref.lin.chunks)
		}
		if pi == 0 {
			whole, wholeSnap = append([]float32(nil), rl...), linSnap(ref)
		} else {
			e.cmp(t, "logits", fmt.Sprintf("partition %v against the whole prompt", part), whole, bl)
			linSnapCmp(t, e, fmt.Sprintf("partition %v against the whole prompt", part), wholeSnap, bat)
		}
		// The continuation, teacher-forced, from the first partition only.
		if pi == 0 {
			linFollow(t, e, ref, bat, rl, bl, follow)
		}
		bat.Close()
		ref.Close()
	}
	return e
}

// linFollow decodes follow in both arms and compares every step's logits. The
// greedy token is held equal only where the reference's top two are further
// apart than twice the error measured at that step.
func linFollow(t *testing.T, e *linEnv, ref, bat *State, rl, bl []float32, follow []int32) {
	t.Helper()
	held := 0
	for i, id := range follow {
		d := e.cmp(t, "cont", fmt.Sprintf("continuation step %d", i), rl, bl)
		a, b := linTop2(rl)
		if float64(rl[a]-rl[b]) > 2*d {
			held++
			if int(Greedy(bl)) != a {
				t.Errorf("continuation step %d: greedy %d, row by row %d with margin %.3e > 2 x %.3e",
					i, Greedy(bl), a, rl[a]-rl[b], d)
			}
		}
		var err error
		if rl, err = ref.Forward(id); err != nil {
			t.Fatal(err)
		}
		if bl, err = bat.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	// Under equality every step with a margin holds; under the shipped
	// envelope a model can measure no step whose margin clears twice its
	// error (LFM2 on the M4), which is reported rather than passed.
	if held == 0 && len(e.tol) == 0 {
		t.Errorf("no continuation step had a margin to hold the greedy token to")
	}
	t.Logf("greedy held at %d of %d continuation steps", held, len(follow))
}

func linTop2(v []float32) (int, int) {
	a, b := 0, 1
	if v[b] > v[a] {
		a, b = b, a
	}
	for i := 2; i < len(v); i++ {
		if v[i] > v[a] {
			a, b = i, a
		} else if v[i] > v[b] {
			b = i
		}
	}
	return a, b
}

// TestLinearRowsMatchRowByRow holds the batched projections to the
// row-by-row arm on every recurrent layer kind's fixture: the gated delta
// rule grouped and tiled (qwen3next, Qwen3.5) with and without a mixture,
// KDA with low-rank and full-rank gates (Kimi-Linear, Kimi-K3), Mamba-2
// alone and beside attention, Mamba-1, and the short convolution. The
// prompt runs to the fixture's context.
func TestLinearRowsMatchRowByRow(t *testing.T) {
	type fx struct {
		name string
		open func(t *testing.T) *Model
	}
	golden := func(dir, script, name string) func(t *testing.T) *Model {
		return func(t *testing.T) *Model {
			m, _ := openGolden(t, dir, script, name)
			return m
		}
	}
	file := func(name string) func(t *testing.T) *Model {
		return func(t *testing.T) *Model {
			path, ok := existingModel(testmodels.Path(name))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS) (RULE 11)", testmodels.Path(name))
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			return m
		}
	}
	fxs := []fx{
		{"hybrid", func(t *testing.T) *Model { return hybridModelOpt(t, hyOpt{ctx: 320}) }},
		{"hybrid-tiled-moe", func(t *testing.T) *Model { return hybridModelOpt(t, hyOpt{ctx: 320, tiled: true, moe: true}) }},
		{"synth-kimik3", golden("kimik3", k3GoldScript, "synth-kimik3")},
		{"synth-kimilinear", file("synth-kimilinear.jlm")},
		{"synth-qwen35-hybrid-mtp", file("synth-qwen35-hybrid-mtp.gguf")},
		{"synth-mamba2", golden("ssm", ssmGoldScript, "synth-mamba2")},
		{"synth-mamba1", file("synth-mamba1.gguf")},
		{"synth-jamba", file("synth-jamba.gguf")},
		{"synth-falconh1", file("synth-falconh1.gguf")},
		{"synth-granitehybrid", file("synth-granitehybrid.gguf")},
		{"synth-nemotronh", file("synth-nemotronh.gguf")},
		{"synth-lfm2", file("synth-lfm2.gguf")},
	}
	for _, f := range fxs {
		t.Run(f.name, func(t *testing.T) {
			m := f.open(t)
			defer m.Close()
			ctx := min(m.Cfg.NCtx, 320)
			cont := 24
			e := linCheck(t, m, ctx-cont-1, cont, linExact)
			t.Logf("%d-token prompt, %d continued: worst error relative to max|ref|:%v", ctx-cont-1, cont, e)
		})
	}
}

// TestLinearRowsMatchRowByRowReal is the same comparison on trained
// checkpoints, whose activations the random fixtures do not resemble: held to
// equality with the GEMM on the matvec's epilogue, and to the shipped envelope
// with the GEMM as it ships.
func TestLinearRowsMatchRowByRowReal(t *testing.T) {
	for _, c := range []struct {
		name string
		n    int
	}{
		{"qwen35/Qwen3.5-0.8B-Q4_K_M.jlm", 512},
		{"Qwen3.5-0.8B-f16.jlm", 256},
		{"lev/lev-Q8_0.jlm", 160},
		{"lfm2/LFM2-350M-Q4_K_M.jlm", 512},
	} {
		for _, arm := range []struct {
			name  string
			exact bool
			tol   map[string]linTol
		}{{"exact", true, linExact}, {"shipped", false, linShipped}} {
			t.Run(c.name+"/"+arm.name, func(t *testing.T) {
				path, ok := existingModel(testmodels.Path(c.name))
				if !ok {
					testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS) (RULE 11)", testmodels.Path(c.name))
				}
				m, err := Open(path, noTune, WithJITOptions(nn.WithGEMMExact(arm.exact)))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				e := linCheck(t, m, c.n, 32, arm.tol)
				t.Logf("%d-token prompt, 32 continued: worst error relative to max|ref|:%v", c.n, e)
			})
		}
	}
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

// TestLinearRowsCounts logs what a prompt costs in each arm, per prompt
// token: matrix passes (a matvec or a GEMM each reads its weight once), GEMM
// calls and pool regions. JITLLM_LINROWS_MODEL names a container; the
// default is the 0.8B Qwen3.5. The only assertions are that each arm is the
// one selected and that the batched arm ran fewer weight passes.
func TestLinearRowsCounts(t *testing.T) {
	name := testmodels.Resolve(envOr("JITLLM_LINROWS_MODEL", "qwen35/Qwen3.5-0.8B-Q4_K_M.jlm"))
	path, ok := existingModel(testmodels.Path(name))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (RULE 11)", name)
	}
	m, err := Open(path, noTune, WithJITOptions(nn.WithMatMulProfile(true)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const n = 152
	prompt := linPrompt(n, min(m.Cfg.NVocab, 1000))
	passes := map[bool]int64{}
	for _, ref := range []bool{true, false} {
		s := m.NewState(n + 1)
		s.lin.ref = ref
		s.jit.ResetRegions()
		nn.MatMulProfile()
		g0 := s.jit.PackedBatchCalls()
		mv0 := s.mvCalls
		if _, err := s.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		par, ser := s.jit.Regions()
		_, _, _, _, calls, _ := nn.MatMulProfile()
		gemm := s.jit.PackedBatchCalls() - g0 + calls
		mv := s.mvCalls - mv0
		passes[ref] = gemm + mv
		arm := "batched"
		if ref {
			arm = "row-by-row"
		}
		t.Logf("%s: %d tokens: %d GEMM calls, %d matvecs, %d weight passes (%.2f/token), %d regions (%.1f/token), %d linear chunks batched",
			arm, n, gemm, mv, gemm+mv, float64(gemm+mv)/n, par+ser, float64(par+ser)/n, s.lin.chunks)
		if (s.lin.chunks == 0) != ref {
			t.Errorf("%s arm ran %d batched chunks", arm, s.lin.chunks)
		}
		s.Close()
	}
	if passes[false] >= passes[true] {
		t.Errorf("the batched arm read the weights %d times, row by row %d", passes[false], passes[true])
	}
}

// TestLinearRowsRideABatchStep holds a batch session's recurrent blocks --
// prompt chunks of several sequences and decoding rows in one step -- with
// their projections as GEMMs over the step to the same session row by row.
// The batched arm must have run (lin.chunks), the row-by-row arm not.
func TestLinearRowsRideABatchStep(t *testing.T) {
	for _, f := range []struct {
		name string
		open func(t *testing.T) *Model
	}{
		{"hybrid", func(t *testing.T) *Model { return hybridModelOpt(t, hyOpt{ctx: 320}) }},
		{"synth-lfm2", func(t *testing.T) *Model {
			path, ok := existingModel(testmodels.Path("synth-lfm2.gguf"))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS) (RULE 11)", testmodels.Path("synth-lfm2.gguf"))
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			return m
		}},
	} {
		t.Run(f.name, func(t *testing.T) {
			m := f.open(t)
			defer m.Close()
			vocab := min(m.Cfg.NVocab, 1000)
			run := func(ref bool) ([][][]float32, int64) {
				sc := NewScheduler(m, 3, 200, WithPromptBudget(40))
				defer sc.Close()
				sc.st.lin.ref = ref
				lg := make([][][]float32, 3)
				seqs := make([]Seq, 3)
				for i := range seqs {
					seqs[i] = Seq{Prompt: linPrompt(30+25*i, vocab)[i:], MaxTokens: 4}
					seqs[i].Sample = func(row []float32) int32 {
						lg[i] = append(lg[i], append([]float32(nil), row...))
						return int32(3 + len(lg[i])%5)
					}
					sc.Submit(&seqs[i])
				}
				for sc.Pending() > 0 || sc.Live() > 0 {
					done, err := sc.Step()
					if err != nil {
						t.Fatal(err)
					}
					for _, s := range done {
						if s.Err != nil {
							t.Fatal(s.Err)
						}
					}
				}
				if sc.Stats().MixedRows == 0 {
					t.Fatal("no prompt rows rode a decoding step: the mixed step is untested")
				}
				return lg, sc.st.lin.chunks
			}
			want, refChunks := run(true)
			got, chunks := run(false)
			if chunks == 0 || refChunks != 0 {
				t.Fatalf("arms not selected: batched ran %d chunks, row-by-row %d", chunks, refChunks)
			}
			e := newLinEnv(linExact)
			for i := range want {
				if len(got[i]) != len(want[i]) || len(want[i]) != 4 {
					t.Fatalf("sequence %d: %d logit rows against %d", i, len(got[i]), len(want[i]))
				}
				for r := range want[i] {
					e.cmp(t, "logits", fmt.Sprintf("sequence %d row %d", i, r), want[i][r], got[i][r])
				}
			}
			t.Logf("%d step-chunks batched; worst error relative to max|ref|:%v", chunks, e)
		})
	}
}

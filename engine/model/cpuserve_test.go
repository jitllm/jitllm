package model

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// cpuServedModels are the models whose every block the composer serves on
// the CPU at decode: Ministral-3-3B among them, the attention temperature
// read at each row's position, HunyuanOCR's XD-RoPE, and Gemma 4 E2B's per-layer
// inputs and blocks reading an earlier block's history, and latent
// attention: DeepSeek's fixtures (dense, V3's grouped router, V2's),
// Kimi-Linear's beside its delta rule and Kimi-K3-0.40B. Gemma 3n E2B is
// not here: its clean arm is in the band (median 0.634, worst 9.1e-4 where
// main's control reaches 1.5e-3), the page fault and the new key written a
// row early leave it, but the key-row violation stays in it even dropping
// the step's own row (median 0.655, worst 1.25e-3): over 80 steps its
// attention reads nearly the same value from any of its rows (attending to
// its own key alone in every block moves the logits 6.5e-4).
var cpuServedModels = []string{"stories15M-q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf",
	"newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", "hunyuanvl/HunyuanOCR-Q8_0.gguf",
	"mamba/mamba-130m-hf.Q8_0.gguf", "lfm2/LFM2-350M-Q4_K_M.gguf", "granite/granite-4.0-h-350m-Q8_0.gguf",
	"falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf", "gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf",
	"synth-deepseek-dense" + q8Suffix, "synth-deepseek" + q8Suffix, "synth-deepseek-lite" + q8Suffix,
	"synth-kimilinear" + q8Suffix, "kimik3/Kimi-K3-0.40B.Q8_0.gguf"}

// TestComposedCPUBlocksServe runs each model's blocks on the CPU three ways,
// teacher-forced on one token sequence: on main's kernels, on what the
// composer forms for the CPU (its int8 form: activations quantized in
// windows of 32, integer dots), and on main's kernels at another activation
// window, main's own
// spread (the control), the history in the State's KV pages. Every block
// step must be served (served = steps x blocks, nothing declined, each block
// composed once), and the composed logits held to main's within main's own
// spread, by the band the card arms are held to (bandRatio, bandWorst): at
// the median step no further from main than bandRatio times the control, at
// the worst no further than bandWorst times the control's worst. Every
// step's head is the CPU's composed head. Each violation -- the first
// history page pointed at another slot, the first and the newest key rows
// moved by one, the new key written a row early, the head reading another
// row than the residual -- must leave the band.
func TestComposedCPUBlocksServe(t *testing.T) {
	const steps, page = 80, 32
	for _, file := range cpuServedModels {
		t.Run(file, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(file))
			ref, err := Open(path, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			ref.setKVPageForTest(page)
			toks := cpuTokens(t, ref, steps)
			want := cpuLogits(t, ref, toks, cpuFaultNone, false)
			cm, err := Open(path, noTune, WithJITOptions(nn.WithActWindow(cpuControlWindow)))
			if err != nil {
				t.Fatal(err)
			}
			defer cm.Close()
			cm.setKVPageForTest(page)
			ctl := cpuLogits(t, cm, toks, cpuFaultNone, false)
			m, err := Open(path, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			m.setKVPageForTest(page)
			got := cpuLogits(t, m, toks, cpuFaultNone, true)
			med, worst, widest := cpuBand(t, want, got, ctl)
			_, agree := cpuCompare(want, got)
			t.Logf("over %d steps the composed CPU is %.3f of main's spread from main at the median, worst NMSE %.3g where main's control reaches %.3g; argmax agrees with main %d/%d",
				steps, med, worst, widest, agree, steps)
			if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
				t.Fatalf("the composed CPU blocks leave main's band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
					med, bandRatio, worst, bandWorst, widest)
			}
			for _, f := range append(cpuFaultsOf(t, m), cpuFaultHead) {
				bad := cpuLogits(t, m, toks, f, true)
				bm, bw, _ := cpuBand(t, want, bad, ctl)
				t.Logf("violation %d: median ratio %.3f, worst %.3g", f, bm, bw)
				if bm <= bandRatio && bw <= bandWorst*widest {
					t.Errorf("violation %d stays in the band (median ratio %.3f, worst %.3g): the gate cannot see it", f, bm, bw)
				}
			}
			for _, rw := range cpuLatentRewrites(t, m, false) {
				var undo []func()
				for li := range m.Cfg.NLayer {
					if u := rewriteBlock(t, m, li, rw.rewrite); u != nil {
						undo = append(undo, u)
					}
				}
				bad := cpuLogits(t, m, toks, cpuFaultNone, true)
				for _, u := range undo {
					u()
				}
				bm, bw, _ := cpuBand(t, want, bad, ctl)
				t.Logf("%s in %d blocks: median ratio %.3f, worst %.3g", rw.what, len(undo), bm, bw)
				if bm <= bandRatio && bw <= bandWorst*widest {
					t.Errorf("%s stays in the band (median ratio %.3f, worst %.3g): the gate cannot see it", rw.what, bm, bw)
				}
			}
		})
	}
}

// cpuLatentRewrite turns the key's rotary part of a latent block's cached
// row into the latent's first NRot: the row's rotary tail not the one the
// projection wrote.
var cpuLatentRewrite = latentRewrite{"the cached row's rotary tail read from the latent's start", func(g *graph.Graph) bool {
	for i, o := range g.Ops {
		if o.Kind == graph.OpSlice && o.Attr.HeadDim == 0 && o.Attr.Offset > 0 && strings.HasPrefix(g.Values[o.Out[0]].Name, "k.pe.") {
			g.Ops[i].Attr.Offset = 0
			return true
		}
	}
	return false
}}

// cpuLatentRewrites is the latent violations the CPU arm runs on m, none for
// a model of no latent block: latentRewrites and cpuLatentRewrite, each
// required to reach a block but the banks swapped where their shapes
// differ and the rotation a pair short where no block turns by the rotary.
// The rotation a pair short runs only on a float arm: on the int8 fixtures
// it moves the logits 1.9e-3 and 3.8e-3, under main's own int8 spread there
// (6.1e-3, 1e-2), where a float arm reads it orders past its clean NMSE.
func cpuLatentRewrites(t *testing.T, m *Model, float bool) []latentRewrite {
	t.Helper()
	if !m.Cfg.MLA() {
		return nil
	}
	reaches := func(rw latentRewrite) bool {
		for li := range m.Cfg.NLayer {
			if g := m.Composing().BlockGraph(li); g != nil && rw.rewrite(cloneGraph(g)) {
				return true
			}
		}
		return false
	}
	short := latentRewrites[1].what
	all := append(slices.Clone(latentRewrites), cpuLatentRewrite)
	// The rotation a pair short turns the latent query's and key's rotary
	// parts alone: an indexer beside them turns its own.
	all[1] = latentRewrite{short, func(g *graph.Graph) bool {
		q, k := false, false
		for i, o := range g.Ops {
			if o.Kind != graph.OpRope {
				continue
			}
			switch n := g.Values[o.Out[0]].Name; {
			case strings.HasPrefix(n, "q.pe.rope."):
				q = true
			case strings.HasPrefix(n, "k.pe.rope."):
				k = true
			default:
				continue
			}
			g.Ops[i].Attr.NRot -= 2
		}
		return q && k
	}}
	var rws []latentRewrite
	for _, rw := range all {
		switch {
		case rw.what == short && !float:
		case reaches(rw):
			rws = append(rws, rw)
		case rw.what == short && m.Cfg.NoPosEnc, rw.what == latentRewrites[2].what:
		default:
			t.Fatalf("%s reaches no block of a latent model: the gate cannot see it", rw.what)
		}
	}
	return rws
}

// cpuFaultsOf is the violations a composed CPU gate runs on m: the
// history's page and key row where a block attends, every recurrent block
// bound to the first's state where it has several.
func cpuFaultsOf(t *testing.T, m *Model) []cpuFault {
	t.Helper()
	var fs []cpuFault
	attends, rec := false, 0
	for li := range m.Cfg.NLayer {
		attends = attends || m.Cfg.LayerKind(li).Attends() || !m.Cfg.LayerKind(li).Recurrent()
		if m.Cfg.LayerKind(li).Recurrent() {
			rec++
		}
	}
	if attends {
		fs = append(fs, cpuFaultPage, cpuFaultRow, cpuFaultWrite)
	}
	if rec > 1 {
		fs = append(fs, cpuFaultRec)
	}
	if m.Cfg.predictsStreams() && m.Cfg.Streams() > 3 {
		fs = append(fs, cpuFaultStreams)
	}
	if len(fs) == 0 {
		t.Fatal("no violation reaches the model: the gate cannot discriminate")
	}
	return fs
}

// cpuControlWindow is the activation window main's control arm quantizes
// over, in place of the one its weight types decide (32 or 256): its distance from main
// is main's own int8 spread.
const cpuControlWindow = 64

// cpuBand is an arm's distance from main against the control's: the median
// over the steps of their ratio, the arm's worst and the control's worst.
func cpuBand(t *testing.T, want, got, ctl [][]float32) (med, worst, widest float64) {
	t.Helper()
	var ratios []float64
	for i := range want {
		ec, eg := logitNMSE(ctl[i], want[i]), logitNMSE(got[i], want[i])
		if math.IsNaN(ec) || math.IsNaN(eg) || !(ec > 0) {
			t.Fatalf("step %d: the control is %.3g from main and the arm %.3g: a degenerate control proves nothing", i, ec, eg)
		}
		ratios = append(ratios, eg/ec)
		worst, widest = max(worst, eg), max(widest, ec)
	}
	slices.Sort(ratios)
	return ratios[len(ratios)/2], worst, widest
}

// cpuTokens is the greedy sequence of n tokens from the model's BOS, or
// token 1 where it states none: a token the model never trained on may
// carry a row of zeros (Falcon-H1's >>ABSTRACT<<, token 1), whose logits
// are zero on every arm and every distance from them 0/0.
func cpuTokens(t *testing.T, m *Model, n int) []int32 {
	t.Helper()
	s := m.NewState(n + 1)
	defer s.Close()
	toks := []int32{1}
	if m.Vocab != nil && m.Vocab.BOS >= 0 {
		toks[0] = int32(m.Vocab.BOS)
	}
	for len(toks) < n {
		lg, err := s.Forward(toks[len(toks)-1])
		if err != nil {
			t.Fatal(err)
		}
		toks = append(toks, int32(argmax(lg)))
	}
	return toks
}

func cpuLogits(t *testing.T, m *Model, toks []int32, f cpuFault, composed bool) [][]float32 {
	t.Helper()
	s := m.NewState(len(toks) + 1)
	defer s.Close()
	s.cpu.fault = f
	var out [][]float32
	for _, tk := range toks {
		lg, err := s.Forward(tk)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	served, declined := s.CPUBlockSteps()
	if composed {
		for i, n := range declined {
			if n > 0 {
				t.Errorf("%d block steps declined: %s (%v)", n, jit.DeclineName(i), s.CPUFormErr())
			}
		}
		if want := len(toks) * m.Cfg.NLayer; served != want {
			t.Fatalf("%d block steps served, want %d (steps x blocks)", served, want)
		}
		// Composed once, at placement: no step composes.
		for li, n := range s.CPUCompositions() {
			if n != 1 {
				t.Fatalf("block %d was composed for the CPU %d times, where it was placed there once", li, n)
			}
		}
		// Every step's head on the CPU's composed head.
		heads, hdecl := s.CPUHeadSteps()
		for i, n := range hdecl {
			if n > 0 {
				t.Errorf("%d head steps declined: %s (%v)", n, jit.DeclineName(i), s.CPUFormErr())
			}
		}
		if heads != len(toks) {
			t.Fatalf("%d head steps served, want %d", heads, len(toks))
		}
		// A model of several stated streams expands every step's row where
		// it has an expansion and collapses it for the head on the composed
		// CPU.
		if m.Cfg.statesStreams() {
			fg, graphs := m.frontGraphs(), 0
			for _, g := range []*graph.Graph{fg.expand, fg.collapse} {
				if g != nil {
					graphs++
				}
			}
			rows, sdecl := s.CPUStreamSteps()
			for i, n := range sdecl {
				if n > 0 {
					t.Errorf("%d rows of the streams' expansion or collapse declined: %s (%v)", n, jit.DeclineName(i), s.CPUFormErr())
				}
			}
			if rows != graphs*len(toks) {
				t.Fatalf("the streams' expansion and collapse served %d rows, want %d", rows, graphs*len(toks))
			}
		}
	} else if served != 0 {
		t.Fatalf("main's arm served %d composed block steps", served)
	}
	return out
}

func cpuCompare(want, got [][]float32) (worst float64, agree int) {
	for i := range want {
		var num, den float64
		for j := range want[i] {
			d := float64(got[i][j] - want[i][j])
			num += d * d
			den += float64(want[i][j]) * float64(want[i][j])
		}
		e := num / den
		if math.IsNaN(e) || e > worst || math.IsNaN(worst) {
			worst = e
		}
		if argmax(want[i]) == argmax(got[i]) {
			agree++
		}
	}
	return worst, agree
}

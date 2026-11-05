package engine

import (
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
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
func cpuServedModels() []string {
	return []string{"stories15M-q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf",
		"newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", "hunyuanvl/HunyuanOCR-Q8_0.gguf",
		"mamba/mamba-130m-hf.Q8_0.gguf", "lfm2/LFM2-350M-Q4_K_M.gguf", "granite/granite-4.0-h-350m-Q8_0.gguf",
		"falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf", "gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf",
		"synth-deepseek-dense" + q8Suffix, "synth-deepseek" + q8Suffix, "synth-deepseek-lite" + q8Suffix,
		"synth-kimilinear" + q8Suffix, "kimik3/Kimi-K3-0.40B.Q8_0.gguf"}
}

// TestComposedCPUBlocksServe runs each model's blocks on the CPU as the
// composer forms them (its int8 form: activations quantized in windows of
// 32, integer dots), teacher-forced on one token sequence, the history in the
// session's KV pages, and holds them to the CPU's float reference by
// llama.cpp's spread from it (cpuReference, the user's decision): at the
// median step no further than bandRatio times the band, at the worst no
// further than bandWorst times its widest. Every block step must be served
// (served = steps x blocks, nothing declined, each block composed once), and
// every step's head is the CPU's composed head. Each violation -- the first
// history page pointed at another slot, the first and the newest key rows
// moved by one, the new key written a row early, the head reading another
// row than the residual -- must leave the band.
func TestComposedCPUBlocksServe(t *testing.T) {
	const steps, page = 80, 32
	for _, file := range cpuServedModels() {
		t.Run(file, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(file))
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			toks := cpuTokens(t, m, steps, withKVPage(page))
			want, band := cpuReference(t, path, fixtureSource(testmodels.Path(file)), page, toks)
			got := cpuLogits(t, m, toks, cpuFaultNone, true, withKVPage(page))
			med, worst, widest := cpuBandTo(t, want, got, band)
			_, agree := cpuCompare(want, got)
			t.Logf("over %d steps the composed CPU is %.3f of the band from the float reference at the median, worst NMSE %.3g where the band's widest is %.3g; argmax agrees with the reference %d/%d",
				steps, med, worst, widest, agree, steps)
			if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
				t.Fatalf("the composed CPU blocks leave the band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
					med, bandRatio, worst, bandWorst, widest)
			}
			for _, f := range append(cpuFaultsOf(t, m), cpuFaultHead) {
				bad := cpuLogits(t, m, toks, f, true, withKVPage(page))
				bm, bw, _ := cpuBandTo(t, want, bad, band)
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
				bad := cpuLogits(t, m, toks, cpuFaultNone, true, withKVPage(page))
				for _, u := range undo {
					u()
				}
				bm, bw, _ := cpuBandTo(t, want, bad, band)
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
func cpuLatentRewrite() latentRewrite {
	return latentRewrite{"the cached row's rotary tail read from the latent's start", func(g *graph.Graph) bool {
		for i, o := range g.Ops {
			if o.Kind == graph.OpSlice && o.Attr.HeadDim == 0 && o.Attr.Offset > 0 && strings.HasPrefix(g.Values[o.Out[0]].Name, "k.pe.") {
				g.Ops[i].Attr.Offset = 0
				return true
			}
		}
		return false
	}}
}

// cpuLatentRewrites is the latent violations the CPU arm runs on m, none for
// a model of no latent block: latentRewrites and cpuLatentRewrite, each
// required to reach a block but the banks swapped where their shapes
// differ and the rotation a pair short where no block turns by the rotary.
// The rotation a pair short runs only on a float arm: on the int8 fixtures
// it moves the logits 1.9e-3 and 3.8e-3, under main's own int8 spread there
// (6.1e-3, 1e-2), where a float arm reads it orders past its clean NMSE.
func cpuLatentRewrites(t *testing.T, m *model.Model, float bool) []latentRewrite {
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
	short := latentRewrites()[1].what
	all := append(slices.Clone(latentRewrites()), cpuLatentRewrite())
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
		case rw.what == short && m.Cfg.NoPosEnc, rw.what == latentRewrites()[2].what:
		default:
			t.Fatalf("%s reaches no block of a latent model: the gate cannot see it", rw.what)
		}
	}
	return rws
}

// cpuFaultsOf is the violations a composed CPU gate runs on m: the
// history's page and key row where a block attends, every recurrent block
// bound to the first's state where it has several.
func cpuFaultsOf(t *testing.T, m *model.Model) []cpuFault {
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
	if m.Cfg.PredictsStreams() && m.Cfg.Streams() > 3 {
		fs = append(fs, cpuFaultStreams)
	}
	if len(fs) == 0 {
		t.Fatal("no violation reaches the model: the gate cannot discriminate")
	}
	return fs
}

// withCPUFloat opens a model whose CPU forms are the float reference
// (modelOpts.cpuFloat).
func withCPUFloat() Option { return func(l *loadOpts) { l.opt.cpuFloat = true } }

// cpuFloat opens the model at path with the CPU's float forms over an f32
// history, pages of page positions (withCPUFloat): a gate's reference arm.
func cpuFloat(t *testing.T, path string, page int) (*model.Model, []Option) {
	t.Helper()
	instOpts := []Option{WithKVF16(false), withCPUFloat(), withKVPage(page)}
	fm, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return fm, instOpts
}

// cpuReference is what a composed CPU arm of the model at path, its GGUF
// source src, is held to on toks teacher-forced: the logits of the CPU's
// float forms (cpuFloat) and each step's band (lcppBand).
func cpuReference(t *testing.T, path, src string, page int, toks []int32) ([][]float32, []float64) {
	t.Helper()
	fm, openOpts := cpuFloat(t, path, page)
	defer fm.Close()
	ref := cpuLogits(t, fm, toks, cpuFaultNone, true, openOpts...)
	return ref, lcppBand(t, src, fm.Cfg.NLayer, ref, lcppSeq{toks[:1], toks[1:], nil})
}

// lcppSeq is one sequence llama.cpp is teacher-forced over: ids, then toks
// a step each; rows picks which of its rows -- the prompt's last, then each
// step's -- an arm's logits are, nil for every one.
type lcppSeq struct {
	ids, toks []int32
	rows      []int
}

// lcppBand is each row's band around ref, the reference's logits for seqs'
// rows in order: the wider of llama.cpp's CUDA and CPU arms' distances from
// them (JITLLM_LCPP), no narrower than bandFloor of layers, or bandFloor
// where llama.cpp does not read src (the user's decision).
func lcppBand(t *testing.T, src string, layers int, ref [][]float32, seqs ...lcppSeq) []float64 {
	t.Helper()
	floor := bandFloor(layers)
	band := make([]float64, len(ref))
	for i := range band {
		band[i] = floor
	}
	dir := os.Getenv("JITLLM_LCPP")
	if dir == "" {
		t.Fatal("JITLLM_LCPP is unset: it names a CUDA build of llama.cpp holding scripts/lcpplogits, the band's reference")
	}
	for _, ngl := range []int{99, 0} {
		at := 0
		for _, sq := range seqs {
			lg, ok := lcppSteps(t, dir, src, sq.ids, sq.toks, ngl)
			if !ok {
				t.Logf("llama.cpp does not read %s: the band is the floor, %.3g", src, floor)
				for i := range band {
					band[i] = floor
				}
				return band
			}
			rows := sq.rows
			if rows == nil {
				rows = make([]int, len(lg))
				for i := range rows {
					rows[i] = i
				}
			}
			for _, r := range rows {
				if at >= len(ref) || r >= len(lg) || len(lg[r]) != len(ref[at]) {
					t.Fatalf("llama.cpp's rows do not line up with the reference's: row %d of %d at %d of %d", r, len(lg), at, len(ref))
				}
				band[at] = max(band[at], logitNMSE(lg[r], ref[at]))
				at++
			}
		}
		if at != len(ref) {
			t.Fatalf("llama.cpp gave %d rows where the reference has %d", at, len(ref))
		}
	}
	return band
}

// cpuBandTo is an arm's distance from the reference against each step's
// band: the median over the steps of their ratio, the arm's worst and the
// band's widest.
func cpuBandTo(t *testing.T, want, got [][]float32, band []float64) (med, worst, widest float64) {
	t.Helper()
	ratios := make([]float64, 0, len(want))
	for i := range want {
		eg := logitNMSE(got[i], want[i])
		if math.IsNaN(eg) || !(band[i] > 0) {
			t.Fatalf("step %d: the arm is %.3g from the reference, the band %.3g: a degenerate band proves nothing", i, eg, band[i])
		}
		ratios = append(ratios, eg/band[i])
		worst, widest = max(worst, eg), max(widest, band[i])
	}
	slices.Sort(ratios)
	return ratios[len(ratios)/2], worst, widest
}

// cpuTokens is the greedy sequence of n tokens from the model's BOS, or
// token 1 where it states none: a token the model never trained on may
// carry a row of zeros (Falcon-H1's >>ABSTRACT<<, token 1), whose logits
// are zero on every arm and every distance from them 0/0.
func cpuTokens(t *testing.T, m *model.Model, n int, io ...Option) []int32 {
	t.Helper()
	s := testInstance(t, m, n+1, io...)
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

func cpuLogits(t *testing.T, m *model.Model, toks []int32, f cpuFault, composed bool, io ...Option) [][]float32 {
	t.Helper()
	s := testInstance(t, m, len(toks)+1, io...)
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
				t.Errorf("%d block steps declined: %s (%v)", n, composer.DeclineName(i), s.CPUFormErr())
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
				t.Errorf("%d head steps declined: %s (%v)", n, composer.DeclineName(i), s.CPUFormErr())
			}
		}
		if heads != len(toks) {
			t.Fatalf("%d head steps served, want %d", heads, len(toks))
		}
		// A model of several stated streams expands every step's row where
		// it has an expansion and collapses it for the head on the composed
		// CPU.
		if m.Cfg.StatesStreams() {
			fg, graphs := m.FrontGraphs(), 0
			for _, g := range []*graph.Graph{fg.Expand, fg.Collapse} {
				if g != nil {
					graphs++
				}
			}
			rows, sdecl := s.CPUStreamSteps()
			for i, n := range sdecl {
				if n > 0 {
					t.Errorf("%d rows of the streams' expansion or collapse declined: %s (%v)", n, composer.DeclineName(i), s.CPUFormErr())
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

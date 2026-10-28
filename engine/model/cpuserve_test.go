package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// cpuServedModels are the models whose every block the composer serves on
// the CPU at decode: Ministral-3-3B among them, the attention temperature
// read at each row's position, HunyuanOCR's XD-RoPE, and Gemma 4 E2B's per-layer
// inputs and blocks reading an earlier block's history.
var cpuServedModels = []string{"stories15M-q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf",
	"newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", "hunyuanvl/HunyuanOCR-Q8_0.gguf",
	"mamba/mamba-130m-hf.Q8_0.gguf", "lfm2/LFM2-350M-Q4_K_M.gguf", "granite/granite-4.0-h-350m-Q8_0.gguf",
	"falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf", "gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"}

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
// history page pointed at another slot, the first key row moved by one, the
// head reading another row than the residual -- must leave the band.
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
			m, err := Open(path, noTune, WithComposedCPU(true))
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
		})
	}
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
		fs = append(fs, cpuFaultPage, cpuFaultRow)
	}
	if rec > 1 {
		fs = append(fs, cpuFaultRec)
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

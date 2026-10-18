package model

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestStepNMSEBand is the instrument denseStepBand was set from, not a gate:
// it measures how far a step across sessions sits from each session alone, and
// how far the references the step gate could be held to sit from each other,
// on the same model, device, prompts and teacher-forced tokens. Every arm is
// compared with the device decoding each session alone (the step gate's
// reference), per session and per step:
//
//	host        each session decoded alone on the host -- the device's own
//	            reduction-order band against the host (RULE 11c)
//	split=N     each session decoded alone on the device at a pinned split:
//	            the same arithmetic summed in another order, and nothing else
//	together=n  the first n sessions through Step (n <= 4 runs decode's matvec
//	            per token, groupMV; past that the tiled twins)
//	nofuse      four sessions through Step with decode's fusions apart
//	            (tier.Config.NoRagFuse)
//
// It reports each arm's worst, median and per-step-range worst, so a band that
// grows with the position, the row count or the model's depth shows as one.
//
//	JITLLM_STEP_BAND=Llama-3.2-1B-Instruct-Q4_K_M.gguf go test ./engine/model -run TestStepNMSEBand -v
func TestStepNMSEBand(t *testing.T) {
	v := os.Getenv("JITLLM_STEP_BAND")
	if v == "" {
		t.Skip("JITLLM_STEP_BAND names the models to measure -- a probe, not a gate")
	}
	for _, name := range strings.Split(v, ",") {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m, err := Open(stepArchPath(t, name), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) { stepBand(t, m, dev) })
			}
		})
	}
}

func stepBand(t *testing.T, m *Model, dev string) {
	prompts := []string{"The capital of France is", "Water boils at a temperature of",
		"The clerk counted barrels of salt on the upper floor, and", "Once upon a time",
		"In 1905, Albert Einstein published", "The recipe calls for two cups of",
		"def fibonacci(n):", "The three primary colors are"}
	const steps = 32
	open := func(opts ...tier.Option) *tier.GPU {
		g, err := openTier(t, append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff)},
			append(opts, testTierOpts(t)...)...)...)
		if err != nil || g == nil {
			noDevice(t, dev, err)
		}
		return g
	}
	// session opens prompt i's State on g (the host where nil) and prefills it.
	session := func(g *tier.GPU, i int) (*State, []float32) {
		ids := m.Vocab.Encode(prompts[i], true)
		st := m.NewState(len(ids) + steps + 2)
		if g != nil {
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
				t.Skipf("CARD TOO SMALL: %d of %d blocks (%s)", st.GPULayers(), m.Cfg.NLayer, g.Err())
			}
		}
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		return st, lg
	}
	// alone decodes every session on g by itself, fed toks, or its own greedy
	// chain where toks is nil, and returns [session][step] logits and the
	// tokens fed.
	alone := func(g *tier.GPU, toks [][]int32) ([][][]float32, [][]int32) {
		out := make([][][]float32, len(prompts))
		fed := make([][]int32, len(prompts))
		for i := range prompts {
			st, lg := session(g, i)
			next := Greedy(lg)
			for s := range steps {
				tok := next
				if toks != nil {
					tok = toks[i][s]
				}
				fed[i] = append(fed[i], tok)
				lg, err := st.Forward(tok)
				if err != nil {
					t.Fatal(err)
				}
				out[i] = append(out[i], slices.Clone(lg))
				next = Greedy(lg)
			}
			st.Close()
		}
		return out, fed
	}
	together := func(g *tier.GPU, n int, toks [][]int32) [][][]float32 {
		sts := make([]*State, n)
		for i := range sts {
			sts[i], _ = session(g, i)
			defer sts[i].Close()
		}
		out := make([][][]float32, n)
		in := make([]int32, n)
		s0 := g.Stats()
		for s := range steps {
			for i := range in {
				in[i] = toks[i][s]
			}
			lgs, err := Step(sts, in)
			if err != nil {
				t.Fatal(err)
			}
			for i, lg := range lgs {
				out[i] = append(out[i], slices.Clone(lg))
			}
		}
		if rows := g.Stats().SessionRows - s0.SessionRows; rows != n*steps {
			t.Fatalf("together=%d: %d rows across sessions, want %d (%v)", n, rows, n*steps, sts[0].StepRefusal())
		}
		return out
	}
	report := func(arm string, got, want [][][]float32) {
		var all []float64
		var rng [3]float64
		for i := range got {
			for s := range got[i] {
				e := logitNMSE(got[i][s], want[i][s])
				all = append(all, e)
				r := 0
				switch {
				case s >= 16:
					r = 2
				case s >= 8:
					r = 1
				}
				rng[r] = max(rng[r], e)
			}
		}
		slices.Sort(all)
		t.Logf("BAND %-12s n=%3d  worst %.3e  median %.3e  p90 %.3e  | steps 0-7 %.3e  8-15 %.3e  16-31 %.3e",
			arm, len(all), all[len(all)-1], all[len(all)/2], all[len(all)*9/10], rng[0], rng[1], rng[2])
	}
	g := open()
	ref, toks := alone(g, nil)
	g.Close()
	host, _ := alone(nil, toks)
	report("host", host, ref)
	for _, sp := range []int{1, 4} {
		g := open(tier.WithSplit(sp))
		got, _ := alone(g, toks)
		g.Close()
		report(fmt.Sprintf("split=%d", sp), got, ref)
	}
	for _, n := range []int{2, 3, 4, 8} {
		g := open()
		got := together(g, n, toks)
		t.Logf("together=%d: %d few-sequence decode matvecs, %d tensor-core matvecs", n, g.Stats().RagGroupMV,
			g.Stats().VoltaMV)
		g.Close()
		report(fmt.Sprintf("together=%d", n), got, ref)
	}
	g = open(tier.WithConfig(func(c *tier.Config) { c.NoRagFuse = true }))
	got := together(g, 4, toks)
	g.Close()
	report("nofuse=4", got, ref)
}

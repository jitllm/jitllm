package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// cpuRecModels are the linear families whose prompts the composed CPU runs
// whole: the gated delta rule beside attention (Qwen3.5), Mamba-1's
// selective scan alone (mamba-130m), the gated short convolution beside
// attention (LFM2), Mamba-2's mixer beside attention and an MLP
// (granite-4.0-h) and in one block with attention (Falcon-H1).
func cpuRecModels() []string {
	return []string{"qwen35/Qwen3.5-0.8B-Q4_K_M.gguf", "mamba/mamba-130m-hf.Q8_0.gguf",
		"lfm2/LFM2-350M-Q4_K_M.gguf", "granite/granite-4.0-h-350m-Q8_0.gguf", "falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf"}
}

// TestComposedCPURecurrentPromptsServe holds a linear model's prompts on
// the composed CPU -- every linear block's rows of a chunk one call of its
// rows' form, the scan stepping the state in memory -- to main's kernels
// within main's own spread, as TestComposedCPURowsServe holds an attention
// model's: main, main at another activation window (the control) and the
// composed CPU, teacher-forced on one text at several prompt lengths and
// the decode after each. Every block step is served and none declined, and
// every linear block's step of several rows ran whole (CPURecSteps), none a
// row at a time. With the state not carried from one chunk to the next, or
// every linear block bound to the first's state, the prompts leave the
// band.
func TestComposedCPURecurrentPromptsServe(t *testing.T) {
	const page, decode = 32, 4
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 40)
	// The longest crosses a chunk boundary (MaxPrefillChunk).
	lengths := []int{5, 33, MaxPrefillChunk + 90}
	for _, file := range cpuRecModels() {
		t.Run(file, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(file))
			open := func(opts ...Option) (*model.Model, []Option) {
				t.Helper()
				instOpts := append(append([]Option(nil), opts...), withKVPage(page))
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				return m, instOpts
			}
			fm, fOpts := cpuFloat(t, path, page)
			cm, cOpts := open()
			defer fm.Close()
			defer cm.Close()
			src := fixtureSource(testmodels.Path(file))
			linear := 0
			for li := range cm.Cfg.NLayer {
				if cm.Cfg.LayerKind(li).Recurrent() {
					linear++
				}
			}
			if linear == 0 {
				t.Fatalf("%s has no linear block", file)
			}
			ids := cm.Vocab.Encode(text, true)
			run := func(m *model.Model, n int, f cpuFault, io ...Option) [][]float32 {
				t.Helper()
				if len(ids) < n+decode {
					t.Fatalf("%d tokens, %d wanted", len(ids), n+decode)
				}
				s := testInstance(t, m, n+decode+1, io...)
				defer s.Close()
				setFault(s, f)
				lg, err := s.Prefill(ids[:n])
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), lg...)}
				for i := range decode {
					if lg, err = s.Forward(ids[n+i]); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				cpuServed(t, s, "a prompt", (1+decode)*m.Cfg.NLayer, true)
				whole, stepped := s.Account(nn.CPU).Steps.Whole, s.Account(nn.CPU).Steps.Stepped
				if n > 1 && (whole < linear || stepped != 0) {
					t.Fatalf("a prompt of %d: %d linear block steps ran whole and %d a row at a time, want at least %d whole and none a row at a time (%v)",
						n, whole, stepped, linear, s.Account(nn.CPU).FormErr)
				}
				return out
			}
			var want, got [][]float32
			var band []float64
			for _, n := range lengths {
				w, g := run(fm, n, cpuFaultNone, fOpts...), run(cm, n, cpuFaultNone, cOpts...)
				b := lcppBand(t, src, fm.Cfg.NLayer, w, lcppSeq{ids[:n], ids[n : n+decode], nil})
				med, worst, widest := cpuBandTo(t, w, g, b)
				t.Logf("a prompt of %d: %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g", n, med, worst, widest)
				want, band, got = append(want, w...), append(band, b...), append(got, g...)
			}
			med, worst, widest := cpuBandTo(t, want, got, band)
			if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
				t.Fatalf("the composed CPU's linear prompts leave the band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)",
					med, bandRatio, worst, bandWorst, widest)
			}
			faults := []cpuFault{cpuFaultChunk}
			if linear > 1 {
				faults = append(faults, cpuFaultRec)
			}
			for _, f := range faults {
				var bad [][]float32
				for _, n := range lengths {
					bad = append(bad, run(cm, n, f, cOpts...)...)
				}
				bm, bw, _ := cpuBandTo(t, want, bad, band)
				t.Logf("violation %d: median ratio %.3f, worst %.3g", f, bm, bw)
				if bm <= bandRatio && bw <= bandWorst*widest {
					t.Errorf("violation %d stays in the band (median ratio %.3f, worst %.3g): the gate cannot see it", f, bm, bw)
				}
			}
		})
	}
}

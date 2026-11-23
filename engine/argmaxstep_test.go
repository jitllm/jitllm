package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestAStepsArgmaxIsTheLogitsLargest: rows of a joint step that take their
// greedy token alone (Run.Argmax) get the largest of the logits the same
// step reads back for them without it, step after step, and the step reads
// no row's logits back: the card's read bytes across it stay under one
// row's vocabulary. Both arms are fed the tokens the logits arm chose, so a
// near-tie never parts them. The violation: one row of the step wanting its
// logits reads every row's back, and the count catches it.
func TestAStepsArgmaxIsTheLogitsLargest(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) { argmaxStep(t, m, spec) })
	}
}

func argmaxStep(t *testing.T, m *model.Model, spec string) {
	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff)},
		testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	const steps = 8
	vocab := uint64(m.Cfg.NVocab)
	arm := func() ([]*Instance, []int32) {
		var sts []*Instance
		var next []int32
		for _, pr := range stepPrompts() {
			ids := m.Vocab.Encode(pr, true)
			st := testInstance(t, m, len(ids)+steps+2, WithKVF16(false))
			t.Cleanup(func() { st.Close() })
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
				t.Skipf("CARD TOO SMALL: %d of %d blocks, head %v -- this gate proved nothing here (%s)",
					st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
			}
			lg, err := st.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			sts, next = append(sts, st), append(next, Greedy(lg))
		}
		return sts, next
	}
	logits, toks := arm()
	argmax, _ := arm()
	n := len(toks)
	step := func(sts []*Instance, fed []int32, am bool, wantLogits int) ([][]float32, uint64) {
		runs := make([]Run, n)
		for i, st := range sts {
			runs[i] = Run{Instance: st, Tokens: []int32{fed[i]}, Logits: true, Argmax: am && i >= wantLogits}
		}
		_, b0 := g.Counters().HostReads()
		out, err := StepRuns(runs)
		if err != nil {
			t.Fatal(err)
		}
		_, b1 := g.Counters().HostReads()
		return out, b1 - b0
	}
	rows0 := g.Stats().SessionRows
	for s := range steps {
		lg, _ := step(logits, toks, false, 0)
		out, read := step(argmax, toks, true, 0)
		if read >= 4*vocab {
			t.Fatalf("step %d: an argmax step read %d bytes back, a row's logits are %d", s, read, 4*vocab)
		}
		next := make([]int32, n)
		for i := range n {
			next[i] = Greedy(lg[i])
			tok, ok := argmax[i].Session.StepArgmax()
			switch {
			case out[i] != nil || !ok:
				t.Fatalf("step %d row %d: logits %v and a token %v -- the argmax path did not run", s, i, out[i] != nil, ok)
			case tok != next[i]:
				t.Fatalf("step %d row %d: the step's argmax is %d, the largest of its logits %d", s, i, tok, next[i])
			}
		}
		toks = next
	}
	if got := g.Stats().SessionRows - rows0; got < 2*steps*n {
		t.Fatalf("%d rows ran as rows of joint steps, want %d: a step fell back to one session after another", got, 2*steps*n)
	}
	// The violation: the first row wants its logits, so every row's come back.
	if _, read := step(argmax, toks, true, 1); read < uint64(n)*4*vocab {
		t.Fatalf("a step with one row wanting its logits read %d bytes, %d rows of logits are %d: the count does not discriminate",
			read, n, uint64(n)*4*vocab)
	}
}

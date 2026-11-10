package engine

import (
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestBatchedMixturePrefillMatchesRowByRow prefills a mixture on the device in
// batched chunks and one row at a time (Config.NoBatch), and demands the same
// logits and the same continuation. The selection check is the composer's
// prompt part at the two widths (ServedPart(ComposedPrompt)): without it a
// refused batch runs one row at a time and the comparison is of one path
// against itself.
//
// JITLLM_MOE_MODEL names another container (the GPU test host carries olmoe,
// the 30B and DeepSeek-V2-Lite).
func TestBatchedMixturePrefillMatchesRowByRow(t *testing.T) {
	name := os.Getenv("JITLLM_MOE_MODEL")
	if name == "" {
		name = "Qwen3-MOE-4x0.6B-Q4_K_M.jlm"
	}
	batchedMixtureCase(t, name, false, false)
}

// floatMoEModels are mixtures whose expert banks are float: the Qwen3.5 hybrid
// mixture with a prediction block (scripts/mtpgold.py) in F32, F16 and BF16,
// every weight of each in that format, on every card.
func floatMoEModels() []string {
	names := []string{"synth-qwen35moe-hybrid-mtp.gguf", "synth-qwen35moe-hybrid-mtp-f16.gguf",
		"synth-qwen35moe-hybrid-mtp-bf16.gguf", "synth-glm4moe.gguf",
		"synth-qwen2moe.gguf", "synth-ernie45moe.gguf", "synth-hunyuanmoe.gguf", "synth-minimaxm2.gguf",
		"synth-gemma4-moe.gguf", "synth-minimaxm3.gguf", "synth-kimik3.gguf"}
	for _, fx := range moe2Fixtures() {
		names = append(names, fx.name+".gguf")
	}
	return names
}

// TestBatchedMixturePrefillFloatBanks is the batched-mixture gate on float
// expert banks, which every card ran one row at a time until the grouped
// matvec's float form was wired: the selection check is that the float form
// ran at all. Its violation is in groupedfault_test.go.
func TestBatchedMixturePrefillFloatBanks(t *testing.T) {
	for _, name := range floatMoEModels() {
		t.Run(name, func(t *testing.T) { batchedMixtureCase(t, name, true, false) })
	}
}

// batchedMixtureCase is the gate on one container (a GGUF is converted
// first); float demands the float form ran. It returns the last-row logit
// NMSE between the arms; violate (a fault armed) returns it before the bounds,
// for the caller to demand it broke them.
func batchedMixtureCase(t *testing.T, name string, float, violate bool) float64 {
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	if strings.HasSuffix(p, ".gguf") {
		p = jlmOf(t, p)
	}
	m, err := model.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	plen := 300
	if v, err := strconv.Atoi(os.Getenv("JITLLM_MOE_PROMPT")); err == nil && v > 0 {
		plen = v
	}
	var prompt []int32
	for len(prompt) < plen {
		prompt = append(prompt, m.Vocab.Encode("The capital of France is Paris. Water boils at 100 degrees, "+
			"and the quick brown fox jumps over the lazy dog. ", false)...)
	}
	prompt = prompt[:plen]
	const gen = 12
	run := func(noBatch bool) ([]float32, []int32, gpu.Stats, string) {
		opts := []gpu.Option{gpu.WithDeviceTune(gpu.TuneOff), gpu.WithPoisonScratch(true)}
		if testdevices.Selected() {
			opts = append(opts, gpu.WithDevices(testdevices.Spec(""))) // a model bigger than one card
		}
		g, err := openGPU(t, append(opts,
			gpu.WithConfig(func(c *gpu.Config) {
				c.NoBatch = noBatch
				// The A/B for the matrix unit: its forms read binary16
				// activations where every other path reads int8.
				c.NoMMA = os.Getenv("JITLLM_MOE_NOMATRIX") != ""
			}))...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		st := testInstance(t, m, len(prompt)+gen+1)
		defer st.Close()
		st.SetDevice(g)
		if st.GPULayers() != m.Cfg.NLayer {
			t.Skipf("the device took %d of %d blocks (%s) -- this gate needs them all", st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		lg = append([]float32(nil), lg...)
		out := []int32{Greedy(lg)}
		for len(out) < gen {
			next, err := st.ForwardGreedy(out[len(out)-1])
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, next)
		}
		if st.DeviceDemotions() != 0 {
			t.Fatalf("the device failed: %s", g.Err())
		}
		return lg, out, g.Stats(), g.Err()
	}
	rowLg, rowIDs, rowSt, _ := run(true)
	batLg, batIDs, batSt, batErr := run(false)
	nmseOf := func(a, b []float32) float64 {
		var num, den float64
		for i := range b {
			d := float64(a[i] - b[i])
			num, den = num+d*d, den+float64(b[i])*float64(b[i])
		}
		return num / den
	}
	if os.Getenv("JITLLM_MOE_HOST") != "" {
		// Which arm is further from the host, measured rather than guessed.
		st := testInstance(t, m, len(prompt)+1)
		hl, err := st.Prefill(prompt)
		st.Close()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("against the host: batched NMSE %.3e, rows NMSE %.3e", nmseOf(batLg, hl), nmseOf(rowLg, hl))
	}
	var num, den float64
	for i := range rowLg {
		d := float64(batLg[i] - rowLg[i])
		num, den = num+d*d, den+float64(rowLg[i])*float64(rowLg[i])
	}
	nmse := num / den
	t.Logf("%s: last-row logit NMSE %.3e\n  batched %v\n  rows    %v", name, nmse, batIDs, rowIDs)
	// The arms are the composer's prompt part at two widths: a chunk of one
	// row at a time, and the prompt's rows batched in fewer chunks.
	rp, _ := rowSt.ServedPart(nn.ComposedPrompt)
	bp, _ := batSt.ServedPart(nn.ComposedPrompt)
	if bp == 0 || bp >= rp {
		t.Fatalf("composed prompt steps: %d one row at a time, %d batched -- the arms are not the two widths (%s)",
			rp, bp, batErr)
	}
	if violate {
		return nmse
	}
	// The bound is the batched path's own: the arms differ in attention too,
	// already ~1e-3 on dense models, while a swapped slot or wrong expert reads
	// ~2e-2.
	if !(nmse < 1e-2) {
		t.Fatalf("last-row logits NMSE %.3e between batched and row-by-row", nmse)
	}
	// A flip past the tie band is not by itself a mixture bug: on a Volta-class
	// card batched prefill differs from per-token by a few logits on dense
	// models too, so a flip is judged by its gap on the host.
	for j := range rowIDs {
		if batIDs[j] != rowIDs[j] {
			if gap := hostGap(t, m, prompt, rowIDs[:j], rowIDs[j], batIDs[j]); gap > batchTie {
				t.Fatalf("token %d: batched %d, rows %d, %.4f apart on the host, which picks %d",
					j, batIDs[j], rowIDs[j], gap, hostArgmax(t, m, prompt, rowIDs[:j]))
			}
			break
		}
	}
	return nmse
}

// hostArgmax is the host's greedy choice after the prompt and prefix.
func hostArgmax(t *testing.T, m *model.Model, prompt, prefix []int32, io ...Option) int32 {
	t.Helper()
	st := testInstance(t, m, len(prompt)+len(prefix)+1, io...)
	defer st.Close()
	lg, err := st.Prefill(append(append([]int32{}, prompt...), prefix...))
	if err != nil {
		t.Fatal(err)
	}
	return Greedy(lg)
}

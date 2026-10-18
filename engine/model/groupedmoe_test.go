package model

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestBatchedMixturePrefillMatchesRowByRow prefills a mixture on the device in
// batched chunks -- the router run for the whole chunk, the (token, slot) pairs
// sorted by expert, the experts run grouped -- and one row at a time
// (Config.NoBatch), and demands the same logits and the same continuation.
// Stats.GroupedMoE is the selection check: without it a refused batch falls
// back to rows and the comparison is of one path against itself. On a latent
// (MLA) model -- the DeepSeek-V2 family -- Stats.MLABatched is the second one:
// its attention is batched too (tier's mlabatch.go), and a chunk that refused
// it went one row at a time through every block, mixture included.
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
// every weight of each in that format. Their banks run through the grouped
// matvec's float form (Stats.GroupedFloat), on every card.
var floatMoEModels = []string{"synth-qwen35moe-hybrid-mtp.gguf", "synth-qwen35moe-hybrid-mtp-f16.gguf",
	"synth-qwen35moe-hybrid-mtp-bf16.gguf", "synth-glm4moe.gguf",
	"synth-qwen2moe.gguf", "synth-ernie45moe.gguf", "synth-hunyuanmoe.gguf", "synth-minimaxm2.gguf",
	"synth-gemma4-moe.gguf", "synth-minimaxm3.gguf", "synth-deepseek4.gguf", "synth-kimik3.gguf"}

// lastGroupedVolta is the batched arm's expert matvecs on the f16 GEMM in the
// last batchedMixtureCase, for a caller that armed a violation.
var lastGroupedVolta int

// TestFewRowFormIsTheMixtureRewrite holds the few-row mixture form (the dp4a
// grouped matvec where the staged f16 GEMM would pad each expert to a
// 32-column block) to graph.MoEFewRow: with the narrow class forced, the
// experts leave the tensor cores (batchedMixtureCase demands it); with the
// rewrite's guard refusing every target, they take them again, so the class
// is the rewrite's decision and not the call site's. Only a card whose
// batched experts take the f16 GEMM (sm_70, or the 8x8 tiles) has the two
// forms.
func TestFewRowFormIsTheMixtureRewrite(t *testing.T) {
	name := os.Getenv("JITLLM_MOE_MODEL")
	if name == "" {
		name = "Qwen3-MOE-4x0.6B-Q4_K_M.jlm"
	}
	batchedMixtureCase(t, name, false, false)
	if lastGroupedVolta == 0 {
		t.Skip("this card's batched experts take no f16 GEMM: it has one expert form -- this gate proved nothing")
	}
	wide := lastGroupedVolta
	t.Setenv("JITLLM_GPU_VOLTA_MOE_MIN_PAIRS", "1000000")
	batchedMixtureCase(t, name, false, false)
	pays := graph.MoEFewRow.Pays
	defer func() { graph.MoEFewRow.Pays = pays }()
	graph.MoEFewRow.Pays = func(*graph.Graph, graph.Region, graph.Target) error { return errors.New("refused for the gate") }
	batchedMixtureCase(t, name, false, true)
	if lastGroupedVolta == 0 {
		t.Fatal("with the few-row rewrite refused the experts still left the tensor cores: the class is decided elsewhere")
	}
	t.Logf("wide: %d expert matvecs on the f16 GEMM; narrow: none; narrow with the rewrite refused: %d", wide, lastGroupedVolta)
}

// TestBatchedMixturePrefillFloatBanks is the batched-mixture gate on float
// expert banks, which every card ran one row at a time until the grouped
// matvec's float form was wired: the selection check is that the float form
// ran at all. Its violation is in groupedfault_test.go.
func TestBatchedMixturePrefillFloatBanks(t *testing.T) {
	for _, name := range floatMoEModels {
		t.Run(name, func(t *testing.T) { batchedMixtureCase(t, name, true, false) })
	}
}

// batchedMixtureCase is the gate on one container (a GGUF is converted
// first); float demands the float form ran. It returns the last-row logit
// NMSE between the arms; violate (a fault armed) returns it before the bounds,
// for the caller to demand it broke them.
func batchedMixtureCase(t *testing.T, name string, float, violate bool, extra ...tier.Option) float64 {
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	if strings.HasSuffix(p, ".gguf") {
		p = jlmOf(t, p)
	}
	m, err := Open(p)
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
	run := func(noBatch bool) ([]float32, []int32, tier.Stats, string, int) {
		opts := []tier.Option{tier.WithDeviceTune(tier.TuneOff), tier.WithPoisonScratch(true)}
		if testdevices.Selected() {
			opts = append(opts, tier.WithDevices(testdevices.Spec(""))) // a model bigger than one card
		}
		opts = append(opts, extra...)
		g, err := openTier(t, append(opts,
			tier.WithConfig(func(c *tier.Config) {
				c.NoBatch = noBatch
				// The A/B for sm_70: its batched matvecs read f16 activations
				// where every other path reads int8.
				c.NoVolta = os.Getenv("JITLLM_MOE_NOVOLTA") != ""
				// And the A/B for the experts alone: the dp4a grouped matvec
				// where sm_70's grouped GemmVolta would run them.
				c.NoVoltaMoE = os.Getenv("JITLLM_MOE_NOVOLTAMOE") != ""
				// The width class: a scratch below this many routed pairs
				// builds the dp4a half on sm_70 (a speculation's few-row
				// verification); a prompt chunk is never that narrow, so a
				// large value is the narrow class's arm.
				if v, err := strconv.Atoi(os.Getenv("JITLLM_GPU_VOLTA_MOE_MIN_PAIRS")); err == nil {
					c.VoltaMoEMinPairs = v
				}
			}))...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		st := m.NewState(len(prompt) + gen + 1)
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
		return lg, out, g.Stats(), g.Err(), st.DeviceRowChunks()
	}
	rowLg, rowIDs, rowSt, _, _ := run(true)
	batLg, batIDs, batSt, batErr, batRetry := run(false)
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
		st := m.NewState(len(prompt) + 1)
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
	t.Logf("%s: %d grouped blocks (%d expert matvecs on sm_70 tensor cores), %d batched latent blocks, "+
		"%d paged prefill attentions (%d on sm_70's m8n8k4), last-row logit NMSE %.3e\n  batched %v\n  rows    %v",
		name, batSt.GroupedMoE, batSt.GroupedVolta, batSt.MLABatched, batSt.PagedPrefillLaunches, batSt.PagedPrefill70,
		nmse, batIDs, rowIDs)
	// A chunk the device ran grouped and then refused (its head, say) goes
	// round again a row at a time, and the batched arm's logits are then the
	// row-by-row arm's: GroupedMoE counted a run whose answer was thrown away.
	// MiniMax-M3's float head was refused so on every chunk, and this gate
	// read NMSE 0 with any violation armed.
	if batRetry != 0 {
		t.Fatalf("the batched arm ran %d prompt chunks again a row at a time (%s): it is the row-by-row arm",
			batRetry, batErr)
	}
	if rowSt.GroupedMoE != 0 || batSt.GroupedMoE == 0 {
		t.Fatalf("grouped mixture blocks: %d row by row, %d batched -- the arms are not the two paths (%s)",
			rowSt.GroupedMoE, batSt.GroupedMoE, batErr)
	}
	if float && batSt.GroupedFloat == 0 {
		t.Fatalf("%d grouped blocks and no float-bank matvec among them: the float form never ran (%s)",
			batSt.GroupedMoE, batErr)
	}
	// On sm_70 the experts must have taken the tensor cores, or a device whose
	// dense batched matvecs did is testing the dp4a grouped matvec while
	// believing it tests GemmVolta.
	narrow := os.Getenv("JITLLM_GPU_VOLTA_MOE_MIN_PAIRS") != ""
	lastGroupedVolta = batSt.GroupedVolta
	if narrow && !violate && batSt.GroupedVolta != 0 {
		t.Fatalf("the narrow class was forced and %d expert matvecs still ran on sm_70 tensor cores (%s)", batSt.GroupedVolta, batErr)
	}
	if batSt.VoltaMV > 0 && os.Getenv("JITLLM_MOE_NOVOLTAMOE") == "" && !narrow && batSt.GroupedVolta == 0 {
		t.Fatalf("%d sm_70 tensor-core matvecs and 0 of the experts on them (%s)", batSt.VoltaMV, batErr)
	}
	// And the same for a latent block's attention: on sm_70 it runs on
	// m8n8k4 over the paged latent rows (PagedAttnScoresMMA70 and its
	// accumulate, mlabatch.go), and a chunk that fell to the FMA tile would
	// pass this gate testing the other kernel.
	if m.Cfg.MLA() && batSt.VoltaMV > 0 && os.Getenv("JITLLM_MOE_NOVOLTA") == "" && batSt.PagedPrefill70 == 0 {
		t.Fatalf("%d batched latent blocks and no prefill attention on sm_70's tensor cores (%s)",
			batSt.MLABatched, batErr)
	}
	if m.Cfg.MLA() && (rowSt.MLABatched != 0 || batSt.MLABatched == 0) {
		t.Fatalf("batched latent blocks: %d row by row, %d batched -- the arms are not the two paths (%s)",
			rowSt.MLABatched, batSt.MLABatched, batErr)
	}
	if violate {
		return nmse
	}
	// The bound is the batched path's own: the arms differ in attention too,
	// already ~1e-3 on dense models, while a swapped slot or wrong expert reads
	// ~2e-2. backend.TestGroupedMatVecMatchesPerSlot holds the per-column
	// arithmetic bit-exact; this gates the orchestration.
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
func hostArgmax(t *testing.T, m *Model, prompt, prefix []int32) int32 {
	t.Helper()
	st := m.NewState(len(prompt) + len(prefix) + 1)
	defer st.Close()
	lg, err := st.Prefill(append(append([]int32{}, prompt...), prefix...))
	if err != nil {
		t.Fatal(err)
	}
	return Greedy(lg)
}

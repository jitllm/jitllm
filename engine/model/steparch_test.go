package model

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestStepAcrossSessionsEveryArchitecture is the step gate (stepGate) over one
// model of every architecture the model directory carries: three sessions
// decode together through Step, teacher-forced, against each session alone. A
// model that does not step as rows is a failure that names the refusal
// (State.StepRefusal), never a quiet fallback to one session after another --
// which gives the same answer at a pass over the weights per session, and so is
// invisible to every gate that compares logits. A model the card cannot hold
// whole skips by name, and so does one whose block the device declines.
//
// JITLLM_STEP_ARCHS names the models, comma-separated: a GGUF or a container
// by file name, or a safetensors fixture by directory. The default is the
// smallest model of each architecture on the development box.
func TestStepAcrossSessionsEveryArchitecture(t *testing.T) {
	for _, name := range stepArchModels() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m, err := Open(stepArchPath(t, name), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					ctl := stepControl(t, m, name, dev)
					g := stepTier(t, m, dev, false)
					defer g.Close()
					r := stepGate(t, m, g, false)
					checkLinearRowsForm(t, m, g)
					bound := stepBound(m, name, r, ctl)
					t.Logf("STEP %s %s %s: worst logit NMSE %.3e (bound %.3e)", m.Cfg.Arch, filepath.Base(name),
						g.Name(), r.worst, bound)
					if !(r.worst < bound) {
						t.Fatalf("%s's step across sessions disagrees with each session alone: NMSE %.3e at row %d",
							m.Cfg.Arch, r.worst, r.at)
					}
				})
			}
		})
	}
}

// stepBound is the step gate's bound for model name, opened as m, whose step
// measured r: a synthetic fixture's (fixtureStepNMSE) unless its grouped
// experts ran in binary16 (sm_70's tensor cores, Metal's tiles); a mixture's,
// whose router turns the step's grouped arithmetic into a wider band, or
// denseStepBand times its order-only control where that is wider still; or
// denseStepBand times ctl, the order-only control measured on the same device
// in the same run (stepControl). A mixture's own control is what widens it:
// granite-3.1-1b-a400m's rows sit near router ties, an order change alone moves
// its worst row by 1.2e-02..2.0e-02 on the M4 where Qwen3-MoE-4x0.6B's whole
// step reads 4.6e-03..7.3e-03, and its step there reads up to 8.4e-02 over six
// prompt sets with the grouped experts in binary16 and 6.8e-02 with them int8.
func stepBound(m *Model, name string, r stepResult, ctl float64) float64 {
	switch {
	case m.Cfg.MoE() && r.groupedVolta > 0:
		return max(mixtureStepNMSE, denseStepBand*ctl)
	case fixture(name):
		return fixtureStepNMSE
	case m.Cfg.MoE():
		return max(mixtureStepNMSE, denseStepBand*ctl)
	}
	return denseStepBand * ctl
}

// stepControl is the order-only control stepBound sizes a model's band from,
// measured on dev before the gate opens its tier, or 0 for a synthetic
// fixture, whose bound is a constant.
func stepControl(t *testing.T, m *Model, name, dev string) float64 {
	if fixture(name) {
		return 0
	}
	return orderControl(t, m, dev)
}

// fixture reports a synthetic fixture: random weights of a few blocks.
func fixture(name string) bool { return strings.HasPrefix(filepath.Base(name), "synth-") }

// fixtureStepNMSE is the step gate's bound on a synthetic fixture (synth-*):
// random weights of a few blocks, f32 or Q8_0, on which a step and each
// session alone read 0 to 2.9e-12 on CUDA, SPIR-V and Metal -- where a
// published model's band is set by its quantized activations near a tie. A
// bound of the published models' width would pass a missing Llama 4
// temperature, which reads 6.4e-04 on the fixture. A mixture whose experts
// ran in binary16 is not held to it: synth-gptoss's MXFP4 banks on Metal's
// tiles read 8.6e-04.
const fixtureStepNMSE = 1e-9

// stepArchModels is the models TestStepAcrossSessionsEveryArchitecture runs.
func stepArchModels() []string {
	if v := os.Getenv("JITLLM_STEP_ARCHS"); v != "" {
		return strings.Split(v, ",")
	}
	return []string{
		"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf",
		"Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf",
		"gemma-2b.gguf", "gemma-2-2b-it-Q4_K_M.gguf", "gemma-3-1b-it-Q4_K_M.gguf",
		"Phi-3.5-mini-instruct-Q4_K_M.gguf", "granite/granite-3.3-2b-instruct-Q4_K_M.gguf",
		"granite/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf",
		"synth-gptoss.gguf", "synth-llama4.gguf", "synth-phi2.gguf", "synth-starcoder.gguf",
		"synth-starcoder2.gguf", "synth-commandr.gguf", "synth-cohere2.gguf", "synth-stablelm.gguf",
		"synth-stablelm-par.gguf", "synth-falcon.gguf", "synth-nemotron.gguf", "synth-dbrx.gguf",
		"synth-deepseek", "synth-deepseek-dense", "synth-kimilinear", "synth-glm4moe.gguf", "synth-qwen2moe.gguf",
		"synth-ernie45moe.gguf", "synth-ernie45.gguf", "synth-hunyuanmoe.gguf", "synth-hunyuan.gguf",
		"synth-smollm3.gguf", "synth-arcee.gguf", "synth-seedoss.gguf", "synth-olmo2.gguf",
		"synth-exaone4.gguf", "synth-mistral3.gguf", "synth-olmo2-gqa.gguf", "synth-olmo3.gguf",
		"synth-minimaxm2.gguf", "synth-gemma4.gguf", "synth-gemma4-kv.gguf", "synth-gemma4-moe.gguf", "synth-gemma4-e.gguf",
		"synth-deepseek32.gguf", "synth-gemma3n.gguf", "synth-minimaxm3.gguf",
		"synth-bailingmoe2.gguf", "synth-dots1.gguf", "synth-phimoe.gguf", "synth-apertus.gguf", "synth-deepseek4.gguf",
		"synth-kimik3.gguf", "synth-bitnet-f16.gguf", "synth-bitnet-i2s.gguf",
	}
}

// stepArchPath is name's container: a GGUF converted (jlmOf), a container as
// it is, a safetensors fixture's directory converted (hfContainer).
func stepArchPath(t *testing.T, name string) string {
	t.Helper()
	if strings.HasSuffix(name, ".gguf") || strings.HasSuffix(name, ".jlm") {
		return jlmOf(t, testmodels.Path(name))
	}
	return hfContainer(t, name)
}

// TestStepAcrossSessionsAppliesTheFinalSoftcap is the step gate on a final
// logit softcap (gemma2's), which a step's per-row head applies on the device
// as decode's head does. The cap is perturbed rather than waited for, as
// TestFinalSoftcapReachesTheDeviceHead does it: Llama-3.2-1B fits any card
// here where gemma-2-2b does not, and a cap of an eighth of the largest logit
// is far from linear, so an uncapped row parts from its session alone by more
// than its own norm. The violation is the jitllmfault "nocap"
// (TestStepAcrossSessionsSoftcapGateDiscriminates).
func TestStepAcrossSessionsAppliesTheFinalSoftcap(t *testing.T) {
	m := openSoftcapStep(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			ctl := orderControl(t, m, dev)
			g := stepTier(t, m, dev, false)
			defer g.Close()
			r := stepGate(t, m, g, false)
			if bound := denseStepBand * ctl; !(r.worst < bound) {
				t.Fatalf("a capped model's step across sessions disagrees with each session alone: NMSE %.3e "+
					"at row %d, bound %.3e", r.worst, r.at, bound)
			}
		})
	}
}

// openSoftcapStep opens Llama-3.2-1B with a final softcap of an eighth of its largest
// logit on the step gate's first prompt.
func openSoftcapStep(t *testing.T) *Model {
	t.Helper()
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	amax := 0.0
	for _, l := range teacherForce(t, m, m.Vocab.Encode("The capital of France is", true)) {
		for _, v := range l {
			amax = max(amax, math.Abs(float64(v)))
		}
	}
	if amax == 0 {
		m.Close()
		t.Fatal("the host's logits are all zero -- a cap would compare nothing")
	}
	m.Cfg.FinalSoftcap = float32(amax / 8)
	return m
}

// TestStepRowsRefuseAShortLocalRopeTable is the guard on a model whose local
// layers train their own rotary base (gemma3): a rows step handed no local
// table must be refused by name. It once was handed none on every rows path,
// and the device rotated those layers by whatever its table buffer last held
// -- the prompt's positions -- reading 2.953e-01 against each session alone on
// gemma-3-1b, every row fluent.
func TestStepRowsRefuseAShortLocalRopeTable(t *testing.T) {
	m, err := Open(stepArchPath(t, "gemma-3-1b-it-Q4_K_M.gguf"), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.ropeSWA == nil {
		t.Fatal("gemma-3-1b carries one rotary base -- this gate would prove nothing")
	}
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g := stepTier(t, m, dev, false)
			defer g.Close()
			st := m.NewState(16)
			defer st.Close()
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if !st.Steppable() {
				t.Skipf("CARD TOO SMALL or not steppable: %v -- this arm proved nothing", st.StepRefusal())
			}
			if _, err := st.Prefill(m.Vocab.Encode("The capital of France is", true)); err != nil {
				t.Fatal(err)
			}
			c := m.Cfg
			st.growBatch(1)
			if err := m.embedRow(st.bx[:c.NEmbd], 3); err != nil {
				t.Fatal(err)
			}
			st.ropeRow(0, st.pos)
			if st.ldRows.LayersRows(st.lo, st.hi, []int{st.pos}, []int{st.pos}, st.maxSeq, st.bx[:c.NEmbd],
				st.bcs[:c.NRot], nil, nil) {
				t.Fatal("a rows step with no local rotary table ran: the local layers rotated by a stale table")
			}
			if e := g.Err(); !strings.Contains(e, "local layers' rotary table") {
				t.Fatalf("refused, but not by name: %q", e)
			}
			t.Logf("refused: %s", g.Err())
		})
	}
}

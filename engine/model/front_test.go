package model

import (
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// uncoveredReasons is every reason the front end names.
var uncoveredReasons = []string{UncoveredKind, UncoveredMixture, UncoveredLatent, UncoveredValueWidth, UncoveredGeometry,
	UncoveredSelect, UncoveredStreams, UncoveredRopeAxes, UncoveredPartialRope, UncoveredNonCausal, UncoveredClassic,
	UncoveredQKNormPlace, UncoveredVNorm, UncoveredKVShared, UncoveredVFromK, UncoveredOutGate, UncoveredClamp,
	UncoveredTemp, UncoveredResidScale, UncoveredPLE, UncoveredCompressed, UncoveredNGram, UncoveredSparseAct,
	UncoveredAct, UncoveredPostNorm, UncoveredSubNorm, UncoveredPooling, UncoveredEmbedRows}

// TestFrontStatesTheModelAsOps builds every block's, the head's and the
// input rows' graph from the container of plain dense models (biases, q/k
// norms, GQA, a GELU and a SiLU MLP, tied and untied heads), verifies each,
// and has the composer form the whole model from them for a CPU and a GPU
// device, every part: the model as ops, formed without running anything.
// Models with features the front end does not write yet are refused block by
// block by a reason it names, never by a graph that fails to verify.
func TestFrontStatesTheModelAsOps(t *testing.T) {
	step := jit.Step{Rows: 1, Page: 64, Keys: 16, ActWindow: 32}
	for _, file := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "tinyllama-1.1b-q3_K_M.gguf",
		"gemma-2b.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf"} {
		t.Run(file, func(t *testing.T) {
			m := frontModel(t, file)
			f := m.Composing()
			for li := range m.Cfg.NLayer {
				if f.BlockGraph(li) == nil {
					t.Fatalf("block %d has no graph: %s", li, m.Uncovered(li))
				}
			}
			if f.HeadGraph() == nil || f.EmbedGraph() == nil {
				t.Fatalf("no head (%s) or input rows (%s)", m.Uncovered(-1), m.Uncovered(-2))
			}
			parts, err := jit.Front(f, false)
			if err != nil {
				t.Fatal(err)
			}
			for _, d := range []jit.Device{{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: jit.DeviceCPU, Slots: 6, Lanes: 8}} {
				var run []jit.PartPlan
				for _, p := range parts {
					run = append(run, jit.PartPlan{Part: p, Device: d})
				}
				ks, err := jit.Default().Kernels(f, run, 0, step)
				if err != nil {
					t.Fatalf("%v: %v", d.Kind, err)
				}
				t.Logf("%v: %d parts formed into %d kernels", d.Kind, len(parts), len(ks))
			}
		})
	}
	for _, c := range []struct{ file, want string }{
		{"olmoe-1b-7b-0924-instruct-Q4_K_M.gguf", UncoveredMixture},
		{"gemma-3-1b-it-Q4_K_M.gguf", UncoveredPostNorm},
		{"hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf", UncoveredQKNormPlace},
	} {
		t.Run(c.file, func(t *testing.T) {
			// Every block of these models has the feature: every block is
			// refused by its name, so a refusal that disappears fails here.
			m := frontModel(t, c.file)
			for li := range m.Cfg.NLayer {
				why := m.Uncovered(li)
				if why != c.want {
					t.Fatalf("block %d is refused by %q, want %q", li, why, c.want)
				}
				if !slices.Contains(uncoveredReasons, why) {
					t.Fatalf("block %d is refused by %q, which is not a named reason", li, why)
				}
			}
		})
	}
}

// TestFrontNamesNoFamily holds the front end to reading numbers: its source
// names no model family.
func TestFrontNamesNoFamily(t *testing.T) {
	src, err := os.ReadFile("front.go")
	if err != nil {
		t.Fatal(err)
	}
	family := regexp.MustCompile(`(?i)gemma|llama|qwen|deepseek|kimi|mistral|mixtral|granite|hunyuan|minimax|nemotron|jamba|mamba|olmo|ernie|glm|bitnet|phi[0-9]|phimoe|gpt-?oss|falcon|starcoder|command-?r|cohere|stablelm|dbrx|exaone|smollm|arcee|seed_?oss|apertus|bailing|dots1|bert|lfm|clip|altup|laurel|jlm\.Arch|\.Arch\b`)
	if loc := family.FindIndex(src); loc != nil {
		t.Fatalf("front.go names a model family or reads the architecture: %q", src[loc[0]:loc[1]])
	}
}

func frontModel(t *testing.T, file string) *Model {
	t.Helper()
	path := testmodels.Path(file)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
	}
	m, err := Open(jlmOf(t, path), noTune)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// TestFrontRefusesWhatItDropsFromABlock is the gate against a violation: a
// front end that leaves out part of a block (Qwen3-0.6B read as if it had
// no q/k norm) leaves stored tensors unbound, and the block is refused naming
// them rather than given a graph that computes something else.
func TestFrontRefusesWhatItDropsFromABlock(t *testing.T) {
	m := frontModel(t, "Qwen3-0.6B-Q8_0.gguf")
	m.Cfg.QKNorm = false
	why := m.Uncovered(0)
	if !strings.Contains(why, "does not bind") || !strings.Contains(why, jlm.RoleAttnQNorm.String()) {
		t.Fatalf("a block built without its stored q norm is refused by %q, want the unbound q norm named", why)
	}
	t.Logf("block 0 without its q/k norm: %s", why)
}

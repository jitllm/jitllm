package model

import (
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/ir"
)

// namedReason reports whether u is a feature the front end names as not yet
// written: not Covered, and not a graph it built wrong.
func namedReason(u Uncovered) bool { return u > Covered && u < UncoveredUnbound }

// TestFrontStatesTheModelAsOps builds every block's, the head's and the
// input rows' graph from the container of plain dense models (biases, q/k
// norms, GQA, a GELU and a SiLU MLP, tied and untied heads), verifies each,
// and has the composer form the whole model from them for a CPU and a GPU
// device, every part, and lower the GPU's at a subgroup of 32, 8 and 1 lanes,
// the narrower with no op that depends on one: the model as ops, formed
// without running anything.
// Models with features the front end does not write yet are refused block by
// block by a reason it names, never by a graph that fails to verify.
func TestFrontStatesTheModelAsOps(t *testing.T) {
	step := jit.Step{Rows: 1, Page: 64, Keys: 16, ActWindow: 32}
	for _, file := range append(append(slices.Clone(servedModels), formedModels...), mixtureModels...) {
		t.Run(file, func(t *testing.T) {
			m := frontModel(t, file)
			f := m.Composing()
			for li := range m.Cfg.NLayer {
				if f.BlockGraph(li) == nil {
					t.Fatalf("block %d has no graph: %s %v", li, m.Uncovered(li), m.UncoveredErr(li))
				}
			}
			if f.EmbedGraph() == nil {
				t.Fatalf("no input rows (%s %v)", m.Uncovered(-2), m.UncoveredErr(-2))
			}
			// The head is stated, and its first op is a LayerNorm exactly where
			// the model's norms are.
			hg := f.HeadGraph()
			if hg == nil {
				t.Fatalf("no head (%s %v)", m.Uncovered(-1), m.UncoveredErr(-1))
			}
			if m.Cfg.LayerNorm != (hg.Ops[0].Kind == graph.OpLayerNorm) {
				t.Fatalf("the head's first op is %v, where the model's norms are LayerNorm: %v", hg.Ops[0].Kind, m.Cfg.LayerNorm)
			}
			parts, err := jit.Front(f, false)
			if err != nil {
				t.Fatal(err)
			}
			// Every rotary the model states is a table part: no part kind is
			// left out of what is formed.
			wantTables := 0
			for _, r := range f.Rotaries() {
				if r.NRot != 0 {
					wantTables++
				}
			}
			if len(f.Rotaries()) == 0 && f.Config().NRot != 0 {
				wantTables = 1
			}
			if got := rotaryTables(parts); got != wantTables {
				t.Fatalf("%d rotary table part(s) of %d parts, where the model states %d rotar(ies)", got, len(parts), wantTables)
			}
			for _, d := range []jit.Device{{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 8},
				{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 1}, composedCPUDevice(6)} {
				var run []jit.PartPlan
				for _, p := range parts {
					run = append(run, jit.PartPlan{Part: p, Device: d})
				}
				reg := jit.Default()
				ks, err := reg.Kernels(f, run, 0, step)
				if err != nil {
					t.Fatalf("%v: %v", d.Kind, err)
				}
				// Lowering runs nothing: a kernel the device's lowering
				// cannot emit fails here, not first on a card.
				// A device that guarantees no subgroup is given no op whose
				// result depends on one.
				if d.Kind == jit.DeviceGPU {
					for _, k := range ks {
						l, err := reg.Lowerings[d.Kind].Lower(k, d)
						if err != nil {
							t.Fatalf("%d lanes: lowering %s: %v", d.Lanes, k.Name, err)
						}
						if d.Lanes < ir.SubgroupLanes && l.GPU.Lanes != 0 {
							t.Fatalf("%d lanes: %s depends on a subgroup of %d", d.Lanes, k.Name, l.GPU.Lanes)
						}
					}
				}
				t.Logf("%v at %d lanes: %d parts formed into %d kernels", d.Kind, d.Lanes, len(parts), len(ks))
			}
		})
	}
	for _, c := range []struct {
		file string
		want Uncovered
	}{
		{"synth-phimoe.gguf", UncoveredRouterScore},
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
				if !namedReason(why) {
					t.Fatalf("block %d is refused by %q, which is not a named reason", li, why)
				}
			}
		})
	}
}

// TestFrontRefusesSeveralStreams holds every block of each model whose
// residual is several streams (AltUp, hyper-connections, residual attention)
// to a named refusal and no graph: every way to Covered passes a Streams
// check, so the tier never meets a block output wider than a row, which its
// copy over x (servedCopy) cannot hold. Where the block reaches that check
// the refusal is UncoveredStreams by name.
func TestFrontRefusesSeveralStreams(t *testing.T) {
	byStreams := 0
	for _, file := range []string{"synth-gemma3n.gguf", "synth-deepseek4.gguf", "synth-kimik3.gguf", "synth-qwen4exp.gguf"} {
		t.Run(file, func(t *testing.T) {
			m := frontModel(t, file)
			if m.Cfg.Streams() < 2 {
				t.Fatalf("%d residual streams: the fixture does not select the shape", m.Cfg.Streams())
			}
			streams := 0
			for li := range m.Cfg.NLayer {
				why := m.Uncovered(li)
				if why == Covered || m.Composing().BlockGraph(li) != nil {
					t.Fatalf("block %d of %d streams has a graph", li, m.Cfg.Streams())
				}
				if !namedReason(why) {
					t.Fatalf("block %d is refused by %q, which is not a named reason", li, why)
				}
				if why == UncoveredStreams {
					streams++
				}
			}
			t.Logf("%d streams: %d of %d blocks refused for them, the rest earlier", m.Cfg.Streams(), streams, m.Cfg.NLayer)
			byStreams += streams
		})
	}
	if byStreams == 0 {
		t.Fatal("no block is refused for its streams: the gate selects nothing")
	}
}

// rotaryTables is how many of parts are a rotary's table.
func rotaryTables(parts []jit.Part) int {
	n := 0
	for _, p := range parts {
		if p.Kind == jit.PartTable {
			n++
		}
	}
	return n
}

// TestFrontNamesNoFamily holds the front end to reading numbers: its source
// names no model family.
func TestFrontNamesNoFamily(t *testing.T) {
	family := regexp.MustCompile(`(?i)gemma|llama|qwen|deepseek|kimi|mistral|mixtral|granite|hunyuan|minimax|nemotron|jamba|mamba|olmo|ernie|glm|bitnet|phi[0-9]|phimoe|gpt-?oss|falcon|starcoder|command-?r|cohere|stablelm|dbrx|exaone|smollm|arcee|seed_?oss|apertus|bailing|dots1|bert|lfm|clip|altup|laurel|jlm\.Arch|\.Arch\b`)
	for _, file := range []string{"front.go", "frontdelta.go", "frontssm.go"} {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		if loc := family.FindIndex(src); loc != nil {
			t.Fatalf("%s names a model family or reads the architecture: %q", file, src[loc[0]:loc[1]])
		}
	}
}

func frontModel(t *testing.T, file string) *Model {
	t.Helper()
	path := testmodels.Path(file)
	if _, err := os.Stat(fixtureSource(path)); err != nil {
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
	why, err := m.Uncovered(0), m.UncoveredErr(0)
	if why != UncoveredUnbound || err == nil || !strings.Contains(err.Error(), jlm.RoleAttnQNorm.String()) {
		t.Fatalf("a block built without its stored q norm is refused by %q (%v), want the unbound q norm named", why, err)
	}
	t.Logf("block 0 without its q/k norm: %s: %v", why, err)
}

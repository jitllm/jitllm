package engine

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpudev"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// namedReason reports whether u is a feature the front end names as not yet
// written: not Covered, and not a graph it built wrong.
func namedReason(u model.Uncovered) bool { return u > model.Covered && u < model.UncoveredUnbound }

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
	step := composer.Step{Rows: 1, Page: 64, Keys: 16, ActWindow: 32}
	for _, file := range append(append(slices.Clone(servedModels()), formedModels()...), mixtureModels()...) {
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
			parts, err := composer.Front(f, false)
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
			for _, d := range []composer.Device{{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 8},
				{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 1}, cpudev.Target(cpu.Probe().ISA(), 6)} {
				var run []composer.PartPlan
				for _, p := range parts {
					run = append(run, composer.PartPlan{Part: p, Device: d})
				}
				reg := composer.Default()
				ks, err := reg.Kernels(f, run, 0, step)
				if err != nil {
					t.Fatalf("%v: %v", d.Kind, err)
				}
				// Lowering runs nothing: a kernel the device's lowering
				// cannot emit fails here, not first on a card.
				// A device that guarantees no subgroup is given no op whose
				// result depends on one.
				if d.Kind == composer.DeviceGPU {
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
}

// rotaryTables is how many of parts are a rotary's table.
func rotaryTables(parts []composer.Part) int {
	n := 0
	for _, p := range parts {
		if p.Kind == composer.PartTable {
			n++
		}
	}
	return n
}

// TestFrontNamesNoFamily holds the front end to reading numbers: its source
// names no model family.
func TestFrontNamesNoFamily(t *testing.T) {
	testmodels.SourceTree(t)
	family := regexp.MustCompile(`(?i)gemma|llama|qwen|deepseek|kimi|mistral|mixtral|granite|hunyuan|minimax|nemotron|jamba|mamba|olmo|ernie|glm|bitnet|phi[0-9]|phimoe|gpt-?oss|falcon|starcoder|command-?r|cohere|stablelm|dbrx|exaone|smollm|arcee|seed_?oss|apertus|bailing|dots1|bert|lfm|clip|altup|laurel|jlm\.Arch|\.Arch\b`)
	files, err := filepath.Glob(filepath.Join("..", "model", "front*.go"))
	if err != nil {
		t.Fatal(err)
	}
	read := 0
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		read++
		// llama.cpp is the reference a comment cites, not a family.
		src = bytes.ReplaceAll(src, []byte("llama.cpp"), nil)
		if loc := family.FindIndex(src); loc != nil {
			t.Fatalf("%s names a model family or reads the architecture: %q", file, src[loc[0]:loc[1]])
		}
	}
	if read == 0 {
		t.Fatal("read no front-end file: the gate read nothing")
	}
}

func frontModel(t *testing.T, file string) *model.Model {
	t.Helper()
	path := testmodels.Path(file)
	if _, err := os.Stat(fixtureSource(path)); err != nil {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// TestFrontRefusesWhatItDropsFromABlock is the gate against a stated graph
// that does not describe its block: a tensor the block stores and the graph
// does not bind is refused by name, and so is one the graph binds and the
// block does not store, rather than either given a graph that computes
// something else.
func TestFrontRefusesWhatItDropsFromABlock(t *testing.T) {
	m := frontModel(t, "Qwen3-0.6B-Q8_0.gguf")
	if len(m.Container.Config().Blocks.Of) == 0 {
		t.Fatal("Qwen3-0.6B states no block graphs: the stated arm would not be selected")
	}
	for _, c := range []struct {
		what string
		inv  func(inv map[jlm.Role]graph.Shape)
		why  model.Uncovered
		role jlm.Role
	}{
		{"a stored tensor it does not bind", func(inv map[jlm.Role]graph.Shape) { inv[jlm.RolePostAttnNorm] = graph.Shape{K: m.Cfg.NEmbd} },
			model.UncoveredUnbound, jlm.RolePostAttnNorm},
		{"a tensor it binds and the block does not store", func(inv map[jlm.Role]graph.Shape) { delete(inv, jlm.RoleAttnK) },
			model.UncoveredUnverified, jlm.RoleAttnK},
	} {
		undo := restateInventory(m, c.inv)
		why, err := m.Uncovered(0), m.UncoveredErr(0)
		undo()
		if why != c.why || err == nil || !strings.Contains(err.Error(), c.role.String()) {
			t.Fatalf("block 0's stated graph beside %s is refused by %q (%v), want %q naming %s", c.what, why, err, c.why, c.role)
		}
		t.Logf("block 0's stated graph beside %s: %s: %v", c.what, why, err)
	}
}

package engine

import (
	"bytes"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// statedGraphModels are fixtures of each class whose model.json states its
// blocks' graph.
func statedGraphModels() []string {
	return []string{
		"qwen3", "synth-hf-qwen3-yarn",
		"qwen2.5-tiny-random", "synth-qwen2",
		"synth-hf-llama3", "tiny-random-LlamaForCausalLM",
		"synth-hf-smollm3", "synth-hf-seedoss", "synth-ernie45-hf",
		"synth-gemma", "tiny-random-GemmaForCausalLM",
		"synth-phi3", "tiny-random-Phi3ForCausalLM", "tiny-random-phi3",
		"synth-hf-arcee", "synth-apertus-hf", "synth-hf-olmo2", "synth-hf-exaone4",
		"tiny-qwen3moe-hf", "synth-qwen2moe-hf", "synth-mixtral", "synth-hf-mixtral-swa", "tiny-random-OlmoeForCausalLM",
		"synth-dots1-hf", "synth-ernie45moe-hf",
		// GGUFs of the architectures the library's graphs state.
		"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "tinyllama-1.1b-q3_K_M.gguf", "SmolLM2-360M-Instruct-Q8_0.gguf",
		"Qwen3-0.6B-Q8_0.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "tiny-qwen3moe-f32.gguf",
		"olmoe-1b-7b-0924-instruct-Q4_K_M.gguf", "gemma-2b.gguf", "synth-qwen2moe.gguf",
		"synth-apertus.gguf", "synth-arcee.gguf", "synth-dots1.gguf", "synth-ernie45.gguf", "synth-ernie45moe.gguf",
		"synth-exaone4.gguf", "synth-olmo2.gguf", "synth-olmo2-gqa.gguf", "synth-olmo3.gguf", "synth-seedoss.gguf",
		"synth-smollm3.gguf",
	}
}

// TestStatedBlockGraphsAreTheBuiltOnes converts each model whose model.json
// states its blocks' graph, and holds every block's stated graph to the one
// the front end builds from the config: the same ops, operands, weights,
// states and attributes, byte for byte as a container stores them.
func TestStatedBlockGraphsAreTheBuiltOnes(t *testing.T) {
	for _, name := range statedGraphModels() {
		t.Run(name, func(t *testing.T) {
			path := hfContainer
			if strings.HasSuffix(name, ".gguf") {
				path = func(t *testing.T, name string) string { return jlmOf(t, testmodels.Path(name)) }
			}
			m, err := model.Open(path(t, name))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			c := m.Container.Config()
			if n := int(c.NLayer + c.NMTP); len(c.Blocks.Of) != n || n == 0 {
				t.Fatalf("the container states graphs for %d blocks of %d", len(c.Blocks.Of), n)
			}
			stated := slices.Clone(m.FrontGraphs().Blocks)
			kept := c.Blocks
			c.Blocks = jlm.BlockGraphs{}
			m.Front.Restated++
			built := slices.Clone(m.FrontGraphs().Blocks)
			c.Blocks = kept
			m.Front.Restated++
			for li := range stated {
				if stated[li] == nil || built[li] == nil {
					why, err := m.UncoveredAt(li)
					t.Fatalf("block %d: stated %v, built %v (%v: %v)", li, stated[li] != nil, built[li] != nil, why, err)
				}
				if d := graphDiff(stated[li], built[li]); d != "" {
					t.Errorf("block %d: the stated graph is not the built one: %s\nstated\n%s\nbuilt\n%s", li, d, stated[li], built[li])
				}
			}
		})
	}
}

// builtGraphs has the front end build every block's graph from the config
// until the returned undo, as it does for a container that states none: a
// violation of the config then reaches the blocks.
func builtGraphs(m *model.Model) func() {
	c := m.Container.Config()
	fr := &m.Front
	fr.Mu.Lock()
	kept := c.Blocks
	c.Blocks = jlm.BlockGraphs{}
	fr.Restated++
	fr.Mu.Unlock()
	return func() {
		fr.Mu.Lock()
		c.Blocks = kept
		fr.Restated++
		fr.Mu.Unlock()
	}
}

// graphDiff is the first difference between two graphs, "" for none.
func graphDiff(a, b *graph.Graph) string {
	ea, err := graph.Encode(a)
	if err != nil {
		return err.Error()
	}
	eb, err := graph.Encode(b)
	if err != nil {
		return err.Error()
	}
	if bytes.Equal(ea, eb) {
		return ""
	}
	switch {
	case len(a.Ops) != len(b.Ops):
		return fmt.Sprintf("%d ops against %d", len(a.Ops), len(b.Ops))
	case len(a.Values) != len(b.Values):
		return fmt.Sprintf("%d values against %d", len(a.Values), len(b.Values))
	}
	for i := range a.Values {
		if a.Values[i] != b.Values[i] {
			return fmt.Sprintf("value %d: %+v against %+v", i, a.Values[i], b.Values[i])
		}
	}
	for i := range a.Ops {
		if x, y := fmt.Sprintf("%+v", a.Ops[i]), fmt.Sprintf("%+v", b.Ops[i]); x != y {
			return fmt.Sprintf("op %d: %s against %s", i, x, y)
		}
	}
	return fmt.Sprintf("inputs %v against %v, outputs %v against %v, routed %v against %v", a.In, b.In, a.Out, b.Out, a.Routed, b.Routed)
}

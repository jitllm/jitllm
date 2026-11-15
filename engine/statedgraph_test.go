package engine

import (
	"bytes"
	"fmt"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// statedGraphModels are fixtures of each class whose model.json states its
// blocks' graph.
var statedGraphModels = []string{
	"qwen3", "synth-hf-qwen3-yarn",
	"qwen2.5-tiny-random", "synth-qwen2",
	"synth-hf-llama3", "tiny-random-LlamaForCausalLM",
}

// TestStatedBlockGraphsAreTheBuiltOnes converts each model whose model.json
// states its blocks' graph, and holds every block's stated graph to the one
// the front end builds from the config: the same ops, operands, weights,
// states and attributes, byte for byte as a container stores them.
func TestStatedBlockGraphsAreTheBuiltOnes(t *testing.T) {
	for _, name := range statedGraphModels {
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(hfContainer(t, name))
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

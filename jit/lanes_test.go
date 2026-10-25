package jit_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestNarrowSubgroupsShuffleNothing forms a norm before a projection and
// split attentions for a GPU at a subgroup of 32, 8 and 1 lanes: every kernel
// lowers, an attention's positions are split at every lane count, and below
// the IR's subgroup no kernel depends on one -- a device that guarantees
// fewer lanes runs no shuffle.
func TestNarrowSubgroupsShuffleNothing(t *testing.T) {
	type form struct {
		name    string
		graph   func() *graph.Graph
		formats map[jlm.Role]jlm.Type
		split   bool
	}
	norm := func() *graph.Graph {
		h := graph.New()
		h.G.Block = -1
		h.Out(h.MatVec(h.Norm(h.In("x", 2048), jlm.RoleOutputNorm, 1e-5), jlm.RoleOutput, 256))
		return h.G
	}
	attend := func(heads, kv, dim int) func() *graph.Graph {
		return func() *graph.Graph {
			b := graph.New()
			n := b.KVAppend(b.In("k", kv*dim), b.In("v", kv*dim), 0)
			b.Out(b.Attend(b.In("q", heads*dim), n, graph.Attrs{NHead: heads, NKVHead: kv, HeadDim: dim, Scale: 0.1}))
			return b.G
		}
	}
	forms := []form{
		{"norm and projection", norm, map[jlm.Role]jlm.Type{jlm.RoleOutput: jlm.TypeQ8}, false},
		{"attention 6x48", attend(6, 6, 48), nil, true},
		{"attention 16:8x128", attend(16, 8, 128), nil, true},
		{"attention 8:1x256", attend(8, 1, 256), nil, true},
	}
	for _, lanes := range []int{32, 8, 1} {
		for _, f := range forms {
			t.Run(fmt.Sprintf("%d lanes/%s", lanes, f.name), func(t *testing.T) {
				_, codes, err := jit.Default().Form(jit.Request{Graph: f.graph(), Formats: f.formats,
					Step:    jit.Step{Rows: 1, Page: 128, Keys: 128, ActWindow: 32},
					Device:  jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: lanes, Units: 80},
					History: jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: true}}})
				if err != nil {
					t.Fatal(err)
				}
				split, shuffling := false, 0
				for _, c := range codes {
					split = split || slices.Contains(c.Passes, "splitrange")
					if c.GPU.Lanes != 0 {
						shuffling++
						if lanes < ir.SubgroupLanes {
							t.Errorf("%s depends on a subgroup of %d lanes, where the device guarantees %d", c.Name, c.GPU.Lanes, lanes)
						}
					}
				}
				if f.split && !split {
					t.Fatalf("no kernel of %d was split: the form does not reach the split attention", len(codes))
				}
				t.Logf("%d kernels, %d of them shuffling", len(codes), shuffling)
			})
		}
	}
}

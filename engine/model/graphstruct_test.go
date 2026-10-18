package model

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
)

// TestCoverageIsStructural: whether a block runs through the graph is decided
// from its structure, never from the architecture's name. Every covered
// fixture is reopened with its architecture renamed to one no table knows;
// its blocks' reasons are unchanged, every block still runs the graph, and
// the logits are bit for bit the named model's. The violation: a name check
// put back into the decision would make the renamed model hand-written.
func TestCoverageIsStructural(t *testing.T) {
	for _, name := range []string{"Qwen3-0.6B-Q8_0", "Qwen3-MOE-4x0.6B-Q4_K_M", "synth-qwen2moe", "synth-glm4moe",
		"synth-ernie45moe", "synth-bailingmoe2", "synth-dots1", "synth-minimaxm2",
		"synth-phi2", "synth-starcoder", "synth-starcoder2", "synth-commandr", "synth-cohere2", "synth-stablelm",
		"synth-stablelm-par", "synth-falcon", "synth-falcon40", "synth-nemotron", "synth-dbrx", "synth-apertus", "synth-glm4",
		"synth-olmo2", "synth-exaone4", "synth-arcee", "gemma-2-2b-it-Q4_K_M", "gemma-3-1b-it-Q4_K_M", "synth-gptoss", "synth-deepseek", "synth-deepseek-lite", "synth-deepseek-dense", "synth-deepseek-yarn", "synth-deepseek-yarn-af",
		"synth-glm4moelite", "synth-kimilinear", "synth-gemma4", "synth-hunyuan", "synth-hunyuanmoe", "synth-llama4", "synth-llama4-q8", "synth-qwen2vl", "synth-qwen25vl", "synth-qwen3vl", "synth-glm4vmoe", "synth-granitehybrid", "synth-granitehybridmoe", "synth-lfm2moe", "synth-qwen35-hybrid-mtp", "synth-qwen35moe-hybrid-mtp", "Qwen3.5-0.8B-Q4_K_M"} {
		t.Run(name, func(t *testing.T) {
			p := graphModel(t, name)
			named, err := Open(p, noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer named.Close()
			renamed, err := Open(p, noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer renamed.Close()
			renamed.Cfg.Arch = "renamed-" + named.Cfg.Arch
			if err := renamed.buildGraphs(); err != nil {
				t.Fatal(err)
			}
			for li := range named.graphs {
				if a, b := named.BlockGraph(li), renamed.BlockGraph(li); (a == nil) != (b == nil) || a != nil && len(a.Ops) != len(b.Ops) {
					t.Fatalf("renaming the architecture moved block %d's graph", li)
				}
			}
			const n = 6
			want, ranA, _ := graphDecode(t, named, n, 0)
			got, ranB, _ := graphDecode(t, renamed, n, 0)
			if ranA == 0 || ranB != ranA {
				t.Fatalf("graph blocks: named %d, renamed %d", ranA, ranB)
			}
			for i := range want {
				if !slices.Equal(want[i], got[i]) {
					t.Fatalf("position %d: the renamed model's logits differ", i)
				}
			}
			t.Logf("%s as %q: %d graph blocks, logits identical", named.Cfg.Arch, renamed.Cfg.Arch, ranB)
		})
	}
}

// TestEveryGraphOpHasALowering: every op of every covered block names its
// host lowering, and the mixture's ops are recorded as the monolithic routine
// they still are (the composer's work list), so the manifest cannot show a
// mixture as composed before it is. The violation: an op kind removed from
// the table refuses the model at build.
func TestEveryGraphOpHasALowering(t *testing.T) {
	m, err := Open(graphModel(t, "Qwen3-MOE-4x0.6B-Q4_K_M"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	mono := 0
	for li := range m.Cfg.NLayer {
		for _, l := range m.BlockLowering(li) {
			if l.Host == "" {
				t.Fatalf("block %d %s: no lowering", li, l)
			}
			// The route is its own routine of generated kernels; the
			// experts still run as one grouped region.
			if l.Kind == graph.OpRoute && !l.Composed {
				t.Fatalf("block %d: the route is recorded as monolithic", li)
			}
			if (l.Kind == graph.OpExpertMatVec || l.Kind == graph.OpCombine) && l.Composed {
				t.Fatalf("block %d: %s is recorded as composed", li, l)
			}
			if l.Kind == graph.OpActMul && strings.HasPrefix(l.Host, "moeExperts") == l.Composed {
				t.Fatalf("block %d: %s", li, l)
			}
			if !l.Composed {
				mono++
			}
		}
	}
	if mono == 0 {
		t.Fatal("no op recorded as monolithic on a mixture model")
	}
	// A latent attention's projections run inside mlaProject.
	dm, err := Open(graphModel(t, "synth-deepseek"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	for _, l := range dm.BlockLowering(dm.Cfg.NLayer - 1) {
		if l.Kind == graph.OpMatVec && l.Composed != !strings.HasPrefix(l.Host, "mlaProject") {
			t.Fatalf("%s", l)
		}
		if l.Kind == graph.OpSlice && l.Composed {
			t.Fatalf("%s is recorded as composed", l)
		}
	}
	t.Logf("block 0:\n%v", m.BlockLowering(0))
	saved := hostLowering[graph.OpRoute]
	delete(hostLowering, graph.OpRoute)
	err = m.buildGraphs()
	hostLowering[graph.OpRoute] = saved
	if err == nil || !strings.Contains(err.Error(), "no host lowering") {
		t.Fatalf("a route with no lowering built: %v", err)
	}
	if err := m.buildGraphs(); err != nil {
		t.Fatal(err)
	}
}

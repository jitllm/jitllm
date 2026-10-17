package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// mixGraphModels are mixtures whose every mixture block the op language
// states (graph.Mixture): softmax and sigmoid routers, renormalised or not, a
// routed scale and expert groups, an ungated and a sigmoid-gated shared
// expert, the router's and the banks' biases absent and present.
var mixGraphModels = []string{"Qwen3-MOE-4x0.6B-Q4_K_M.jlm", "olmoe-1b-7b-0924-instruct-Q4_K_M.jlm",
	"DeepSeek-V2-Lite.Q4_K_M.jlm", "synth-glm4moe.gguf", "synth-qwen2moe.gguf", "synth-ernie45moe.gguf",
	"synth-minimaxm2.gguf", "synth-bailingmoe2.gguf", "synth-dots1.gguf", "synth-phimoe.gguf"}

// TestMixtureGraphsOfTheModelsVerify: every mixture block of each model
// builds a sublayer graph that verifies against the container, and both
// mixture rewrites find its routed experts by their edges; a dense lead
// (DeepSeek's) has none. The selection is counted (RULE 10): the graphs
// equal the blocks Config.MoEAt names. The violation: a block's graph built
// with another top-k does not verify.
func TestMixtureGraphsOfTheModelsVerify(t *testing.T) {
	ran := 0
	for _, name := range mixGraphModels {
		t.Run(name, func(t *testing.T) {
			p, ok := existingModel(testmodels.Path(name))
			if !ok {
				t.Skipf("MODEL MISSING: %s -- this gate proved nothing", name)
			}
			m, err := Open(jlmOf(t, p))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if errs := m.MixGraphErrors(); len(errs) != 0 {
				t.Fatalf("a mixture graph did not verify: %v", errs)
			}
			moe, graphs := 0, 0
			for li := 0; li < m.Cfg.NLayer; li++ {
				if m.Cfg.MoEAt(li) {
					moe++
				}
				g := m.MixGraph(li)
				if g == nil {
					continue
				}
				if !m.Cfg.MoEAt(li) {
					t.Fatalf("block %d is no mixture and has a mixture graph", li)
				}
				graphs++
				for _, rw := range []*graph.Rewrite{graph.MoEFewRow, graph.MoEWaves} {
					if ds := graph.Apply(g, rw); len(ds) != 1 || (rw == graph.MoEFewRow && ds[0].Err != nil) {
						t.Fatalf("block %d: %s %v", li, rw.Name, ds)
					}
				}
			}
			if moe == 0 || graphs != moe {
				t.Fatalf("%d mixture blocks, %d mixture graphs", moe, graphs)
			}
			// The violation: the facts say another top-k than the graph.
			for li := 0; li < m.Cfg.NLayer; li++ {
				f, ok := m.mixtureFacts(li)
				if !ok {
					continue
				}
				g := f.Mixture.Build()
				f.Mixture.K++
				if graph.Verify(g, f) == nil {
					t.Fatalf("block %d: a graph of top %d verified against facts of top %d", li, f.Mixture.K-1, f.Mixture.K)
				}
				break
			}
			t.Logf("%d mixture blocks, each with a verified graph", graphs)
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no model ran")
	}
}

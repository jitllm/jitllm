package model

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// pathsRun runs a prompt, a few decoded tokens and a ragged step across two
// sessions on m (on the device g when it is not nil, every block placed) and
// returns the first State's path record, the ragged step's merged in. On a
// card too small for the model the blocks past the seam run on the host.
func pathsRun(t *testing.T, m *Model, g *tier.GPU) []BlockPath {
	t.Helper()
	states := []*State{m.NewState(96), m.NewState(96)}
	for _, s := range states {
		defer s.Close()
		if g != nil {
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if s.GPULayers() == 0 {
				t.Skipf("no block placed on the device: the device arm proved nothing")
			}
		}
	}
	ids := m.Vocab.Encode("The capital of France is Paris, and the capital of Spain is", true)
	for _, s := range states {
		lg, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			if lg, err = s.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err := Step(states, []int32{ids[1], ids[2]}); err != nil {
		t.Fatal(err)
	}
	out := states[0].BlockPaths()
	for li, p := range states[1].BlockPaths() {
		out[li].Graph += p.Graph
		out[li].Device += p.Device
		// The second session may place blocks the first ran at home (a card
		// with room for one session's history): the merged record's device
		// runs are graphed by the same rule BlockPaths applies to one.
		out[li].DeviceGraphed = out[li].Device > 0 && m.BlockGraph(li) != nil
	}
	return out
}

// legacyBlocks is the gate's check: an error naming every text block that
// ran on neither the graph's host lowering nor a device with a graphed plan;
// a device arm must have run some block on the device.
func legacyBlocks(paths []BlockPath, device bool) error {
	var bad []string
	onDev, graphed := 0, 0
	for li, p := range paths {
		switch {
		case p.Device > 0 && !p.DeviceGraphed:
			bad = append(bad, fmt.Sprintf("block %d ran on the device with a plan not lowered from the graph", li))
		case p.Device == 0 && p.Graph == 0:
			bad = append(bad, fmt.Sprintf("block %d never ran through the graph", li))
		}
		if p.Device > 0 {
			onDev++
		}
		graphed++
	}
	if graphed == 0 {
		bad = append(bad, "no block of the model is on the graph")
	}
	if device && onDev == 0 {
		bad = append(bad, "no block ran on the device")
	}
	if bad != nil {
		return fmt.Errorf("%d blocks off the graph path: %v", len(bad), bad)
	}
	return nil
}

// TestCoveredBlocksTakeTheGraphPath: on every model the block graph covers,
// by default, every text block runs through the graph on the host (prompt,
// decode, a ragged step across sessions) and with a graph-lowered plan on
// every device arm -- zero blocks through the hand-written body. Forcing one
// block off the graph fires the check. TestEveryTextBlockIsCovered holds that
// no model is left that the graph does not cover.
func TestCoveredBlocksTakeTheGraphPath(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0", "Qwen3-MOE-4x0.6B-Q4_K_M",
		"olmoe-1b-7b-0924-instruct-Q4_K_M", "synth-qwen2moe", "synth-glm4moe", "synth-ernie45moe", "synth-bailingmoe2",
		"synth-dots1", "synth-minimaxm2", "synth-seedoss", "synth-smollm3", "synth-lfm2",
		"synth-phi2", "synth-starcoder", "synth-starcoder2", "synth-commandr", "synth-cohere2", "synth-stablelm",
		"synth-stablelm-par", "synth-falcon", "synth-falcon40", "synth-nemotron", "synth-dbrx", "synth-apertus", "synth-glm4",
		"synth-olmo2", "synth-exaone4", "synth-arcee", "gemma-2-2b-it-Q4_K_M", "gemma-3-1b-it-Q4_K_M", "synth-gptoss", "synth-gemma", "synth-phimoe", "synth-mistral3-af", "synth-deepseek", "synth-deepseek-lite", "synth-deepseek-dense", "synth-deepseek-yarn", "synth-deepseek-yarn-af", "synth-glm4moelite", "synth-kimilinear", "synth-gemma4", "synth-gemma4-kv", "synth-gemma4-e", "synth-gemma4-moe", "synth-hunyuan", "synth-hunyuanmoe", "synth-llama4", "synth-llama4-q8", "synth-qwen2vl", "synth-qwen25vl", "synth-qwen3vl", "synth-glm4vmoe", "synth-granitehybrid", "synth-granitehybridmoe", "synth-lfm2moe", "synth-nemotronh", "synth-nemotronhmoe", "synth-falconh1", "synth-deepseek32", "synth-minimaxm3", "synth-qwen4exp", "synth-hunyuanvl", "synth-gemma3n", "synth-deepseek4", "synth-kimik3", "synth-mamba2", "synth-falconmamba", "synth-mamba1", "synth-jamba", "synth-lfm2moe", "synth-qwen35-hybrid-mtp", "synth-qwen35moe-hybrid-mtp", "Qwen3.5-0.8B-Q4_K_M"} {
		t.Run(name, func(t *testing.T) {
			p := graphModel(t, name)
			m, err := Open(p, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for li := 0; li < m.Cfg.NLayer; li++ {
				if m.BlockGraph(li) == nil {
					t.Fatalf("block %d opened with no graph", li)
				}
			}
			paths := pathsRun(t, m, nil)
			if err := legacyBlocks(paths, false); err != nil {
				t.Fatal("host: ", err)
			}
			t.Logf("host: block 0 %+v", paths[0])
			for _, spec := range testdevices.Specs("cuda", "vulkan", "metal") {
				t.Run(spec, func(t *testing.T) {
					g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					paths := pathsRun(t, m, g)
					if err := legacyBlocks(paths, true); err != nil {
						t.Fatal(spec, ": ", err)
					}
					t.Logf("%s: block 0 %+v", spec, paths[0])
				})
			}
			// The violation: the graph is every block's only body, so a
			// block that ran nowhere is the record the check must refuse.
			paths[len(paths)-1] = BlockPath{}
			err = legacyBlocks(paths, false)
			if err == nil {
				t.Fatal("a block that ran through no graph passed the check")
			}
			t.Logf("violation fires: %v", err)
		})
	}
}

// TestEveryTextBlockIsCovered: every model the graph's gates run opens, every
// text block (the trunk's and the prediction blocks') with its graph. A block
// the op language does not state refuses the model at Open by its reason on
// the named list (UncoveredReasons), so an open is the coverage.
func TestEveryTextBlockIsCovered(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0", "Qwen3-MOE-4x0.6B-Q4_K_M", "olmoe-1b-7b-0924-instruct-Q4_K_M", "synth-qwen2moe", "synth-glm4moe", "synth-ernie45moe", "synth-bailingmoe2", "synth-dots1", "synth-minimaxm2", "synth-seedoss", "synth-smollm3", "synth-lfm2", "synth-phi2", "synth-starcoder", "synth-starcoder2", "synth-commandr", "synth-cohere2", "synth-stablelm", "synth-stablelm-par", "synth-falcon", "synth-falcon40", "synth-nemotron", "synth-dbrx", "synth-apertus", "synth-glm4", "synth-olmo2", "synth-exaone4", "synth-arcee", "gemma-2-2b-it-Q4_K_M", "gemma-3-1b-it-Q4_K_M", "synth-gptoss", "synth-gemma", "synth-phimoe", "synth-mistral3-af", "synth-deepseek", "synth-deepseek-lite", "synth-deepseek-dense", "synth-deepseek-yarn", "synth-deepseek-yarn-af", "synth-glm4moelite", "synth-kimilinear", "synth-gemma4", "synth-gemma4-kv", "synth-gemma4-e", "synth-gemma4-moe", "synth-hunyuan", "synth-hunyuanmoe", "synth-llama4", "synth-llama4-q8", "synth-qwen2vl", "synth-qwen25vl", "synth-qwen3vl", "synth-glm4vmoe", "synth-granitehybrid", "synth-granitehybridmoe", "synth-lfm2moe", "synth-nemotronh", "synth-nemotronhmoe", "synth-falconh1", "synth-deepseek32", "synth-minimaxm3", "synth-qwen4exp", "synth-hunyuanvl", "synth-gemma3n", "synth-deepseek4", "synth-mamba2", "synth-falconmamba", "synth-mamba1", "synth-jamba", "synth-qwen35-hybrid-mtp", "synth-qwen35moe-hybrid-mtp", "Qwen3.5-0.8B-Q4_K_M", "synth-qwen35vl", "synth-kimik3", "synth-mixtral", "synth-mistral3", "synth-phi3", "synth-qwen2", "synth-ernie45", "synth-olmo3", "synth-kimivl", "synth-pixtral"} {
		t.Run(name, func(t *testing.T) {
			m, err := Open(graphModel(t, name), noTune)
			if err != nil {
				t.Fatal(err)
			}
			for li := range m.Cfg.NLayer + m.Cfg.NMTP {
				if m.BlockGraph(li) == nil {
					t.Errorf("block %d opened with no graph", li)
				}
			}
			m.Close()
		})
	}
}

// TestRunTimeAlternativeIsNeverTakenUnasked holds the decision-time plumbing
// to its one run-time alternative, graph.MLPPartition: every covered block
// lists it as an open alternative with its cost and its chooser, a device
// tier never takes it unasked (no unmet choice, no refusal recorded), and a
// caller's Take on a tier with no split executor is counted as unmet with
// the reason, never dropped silently. The violation: the Take's count is
// what fires.
func TestRunTimeAlternativeIsNeverTakenUnasked(t *testing.T) {
	m, err := Open(graphModel(t, "Llama-3.2-1B-Instruct-Q4_K_M"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for li := 0; li < m.Cfg.NLayer; li++ {
		open := 0
		for _, d := range m.BlockRewrites(li) {
			if d.Rewrite == graph.MLPPartition.Name {
				s := d.String()
				if d.Err != nil || d.When != graph.WhenRun || d.Cost == nil || !strings.Contains(s, "open alternative") ||
					!strings.Contains(s, "capacity") || !strings.Contains(s, "tier.WithRewrite") {
					t.Fatalf("block %d: %s", li, s)
				}
				open++
			}
		}
		if open != 1 {
			t.Fatalf("block %d lists %d MLP partitions", li, open)
		}
	}
	t.Logf("block 0: %s", m.BlockRewrites(0)[len(m.BlockRewrites(0))-1])
	ran := 0
	for _, spec := range testdevices.Specs("cuda", "vulkan", "metal") {
		t.Run(spec, func(t *testing.T) {
			auto, err := openTier(t, append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
			if err != nil || auto == nil {
				noDevice(t, spec, err)
			}
			pathsRun(t, m, auto)
			sa := auto.Stats()
			auto.Close()
			if sa.RewritesUnmet != 0 {
				t.Fatalf("unasked: %d unmet (%s)", sa.RewritesUnmet, sa.RewritesUnmetWhy)
			}
			take, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
				tier.WithRewrite(graph.MLPPartition.Name, graph.ChoiceTake)}, testTierOpts(t)...)...)
			if err != nil || take == nil {
				t.Fatalf("%s opened once and not twice: %v", spec, err)
			}
			pathsRun(t, m, take)
			st := take.Stats()
			take.Close()
			if st.RewritesUnmet < m.Cfg.NLayer || !strings.Contains(st.RewritesUnmetWhy, "split executor") {
				t.Fatalf("a Take with no split executor: %d unmet over %d blocks (%q)", st.RewritesUnmet, m.Cfg.NLayer, st.RewritesUnmetWhy)
			}
			t.Logf("%s: the Take is unmet on %d block preparations: %s", spec, st.RewritesUnmet, st.RewritesUnmetWhy)
			ran++
		})
	}
	if _, _, err := tier.ParseRewriteChoice("mlp-partitoin=take"); err == nil {
		t.Fatal("a misspelt rewrite choice was accepted")
	}
	if ran == 0 {
		t.Skip("no device on this host: only the plan's listing was checked")
	}
}

// TestDeviceRecordsTheFormsItRan: a block decoded on a device records, per op
// span, the lowering its step took (tier.GPU.BlockForms) -- the device's half
// of a block's per-op lowering, read back rather than assumed.
func TestDeviceRecordsTheFormsItRan(t *testing.T) {
	m, err := Open(graphModel(t, "Qwen3-0.6B-Q8_0"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := openTier(t, append([]tier.Option{tier.WithDevices("cuda")}, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, "cuda", err)
	}
	defer g.Close()
	if f := g.BlockForms(0); f != nil {
		t.Fatalf("forms before any block was placed: %v", f)
	}
	s := m.NewState(32)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() == 0 {
		t.Skip("CARD TOO SMALL: no block placed -- this gate proved nothing")
	}
	lg, err := s.Prefill(m.Vocab.Encode("The capital of France is", true))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if lg, err = s.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
	}
	f := g.BlockForms(0)
	for _, span := range []string{"qkv", "gate_up", "down"} {
		if f[span] == "" {
			t.Fatalf("block 0 recorded no %s form: %v", span, f)
		}
	}
	t.Logf("block 0 on cuda: %v", f)
}

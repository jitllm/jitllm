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
		"synth-dots1-hf", "synth-ernie45moe-hf", "synth-glm4moe-hf",
		// GGUFs of the architectures the library's graphs state.
		"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "tinyllama-1.1b-q3_K_M.gguf", "SmolLM2-360M-Instruct-Q8_0.gguf",
		"Qwen3-0.6B-Q8_0.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "tiny-qwen3moe-f32.gguf",
		"olmoe-1b-7b-0924-instruct-Q4_K_M.gguf", "gemma-2b.gguf", "synth-qwen2moe.gguf",
		"synth-apertus.gguf", "synth-arcee.gguf", "synth-dots1.gguf", "synth-ernie45.gguf", "synth-ernie45moe.gguf",
		"synth-exaone4.gguf", "synth-olmo2.gguf", "synth-olmo2-gqa.gguf", "synth-olmo3.gguf", "synth-seedoss.gguf",
		"synth-smollm3.gguf", "synth-glm4moe.gguf",
		"gemma-2-2b-it-Q4_K_M.gguf", "gemma-3-1b-it-Q4_K_M.gguf", "gemma-3-4b-it-Q4_K_M.gguf", "synth-glm4.gguf",
		"granite/granite-3.3-2b-instruct-Q4_K_M.gguf", "granite/granite-4.0-micro-Q4_K_M.gguf",
		"granite/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf",
		"Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "synth-qwen2vl.gguf", "synth-qwen25vl.gguf",
		// With their towers, whose taps the first blocks add.
		"synth-qwen2vl.gguf+mmproj-synth-qwen2vl.gguf", "synth-qwen3vl.gguf+mmproj-synth-qwen3vl.gguf",
		"synth-qwen3vlmoe.gguf+mmproj-synth-qwen3vlmoe.gguf",
		"Qwen3-VL-2B-Instruct-f16.gguf+mmproj-Qwen3-VL-2B-Instruct-f16.gguf",
		// The gated delta rule beside gated attention, with prediction blocks.
		"synth-hf-qwen35", "synth-qwen35-hybrid-mtp.gguf", "synth-qwen35moe-hybrid-mtp.gguf",
		"synth-qwen35moe-hybrid-mtp-bf16.gguf", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf", "qwen35/Qwen3.5-0.8B-MTP-Q8_0.gguf",
		"synth-qwen35vl.gguf+mmproj-synth-qwen35vl.gguf",
		// Vision-language models whose text model is a claimed architecture's.
		"SmolVLM-256M-Instruct-Q8_0.gguf", "minicpmv/Janus-Pro-1B-Q4_K_M.gguf", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
		"newarch/EXAONE-4.0-1.2B-Q4_K_M.gguf", "newarch/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf",
		"library/ERNIE-4.5-0.3B-PT-Q4_K_M.gguf", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "stories260K.gguf",
		"embed/qwen3-embedding-0.6b.gguf",
		// The selective state-space mixers.
		"synth-mamba2.gguf", "synth-mamba1.gguf", "synth-mamba1-ragged.gguf", "synth-falconmamba.gguf",
		"mamba/mamba-130m-hf.Q8_0.gguf",
		// The gated short convolution beside attention.
		"synth-lfm2.gguf", "lfm2/LFM2-350M-Q4_K_M.gguf", "lfm2/LFM2-1.2B-Q4_K_M.gguf",
		"vlmb/LFM2-VL-450M-F16.gguf+vlmb/mmproj-LFM2-VL-450M-F16.gguf",
		// The classic transformers' blocks.
		"synth-phi2.gguf", "synth-starcoder.gguf", "synth-starcoder2.gguf", "synth-stablelm.gguf", "synth-stablelm-par.gguf",
		"synth-falcon.gguf", "synth-falcon40.gguf", "synth-nemotron.gguf", "synth-commandr.gguf", "synth-cohere2.gguf",
		"synth-dbrx.gguf",
		// The state-space mixer beside attention, the feed-forward where a block stores one.
		"synth-granitehybrid.gguf", "synth-granitehybridmoe.gguf", "synth-nemotronh.gguf", "synth-nemotronhmoe.gguf",
		"granite/granite-4.0-h-350m-Q8_0.gguf", "granite/granite-4.0-h-1b-Q4_K_M.gguf", "granite/granite-4.0-h-tiny-Q4_K_M.gguf",
		"nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
		"synth-jamba.gguf", "jamba/jamba-reasoning-3b-Q4_K_M.gguf",
		"synth-falconh1.gguf", "synth-falconh1-nonorm.gguf", "falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf",
		// Llama's block with the attention temperature, the L2 head norms and
		// the router weighting the experts' input.
		"synth-llama4.gguf", "synth-llama4-q8.gguf", "synth-mistral3.gguf", "synth-mistral3-af.gguf",
		"newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf+vlmb/mmproj-Ministral-3-3B-BF16.gguf",
		// The q/k norm over the whole projection or after the rotary, the
		// router's selection bias, the shared expert, sinks.
		"synth-minimaxm2.gguf", "synth-hunyuan.gguf", "synth-hunyuanmoe.gguf",
		"hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf", "synth-gptoss.gguf", "gpt-oss-20b-MXFP4.gguf",
		"lfm2/LFM2-8B-A1B-Q4_K_M.gguf",
		// The classic block with a router whose picks' weights are stated.
		"synth-phimoe.gguf",
		// DeepSeek-V3's router beside shared experts on per-head-normed attention.
		"synth-bailingmoe2.gguf",
		// Llama's block over int8 inputs, with the inner norms.
		"synth-bitnet-f16.gguf", "synth-bitnet-i2s.gguf", "bitnet-b1.58-2B-4T-i2_s.gguf",
		// Attention through a latent beside DeepSeek-V3's mixture.
		"synth-deepseek-yarn.gguf", "synth-deepseek-yarn-af.gguf", "DeepSeek-V2-Lite.Q4_K_M.gguf",
		"Moonlight-16B-A3B-Instruct-Q4_K_M.gguf", "Kimi-VL-A3B-Thinking-2506-Q4_K_M.gguf+mmproj-Kimi-VL-A3B-Instruct-f16.gguf",
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
			switch text, proj, pair := strings.Cut(name, "+"); {
			case pair:
				path = func(t *testing.T, _ string) string { return jlmOfPair(t, testmodels.Path(text), testmodels.Path(proj)) }
			case strings.HasSuffix(name, ".gguf"):
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

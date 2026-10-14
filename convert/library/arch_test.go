package library

import (
	"testing"

	"github.com/jitllm/jitllm/convert"
)

// obtainableExempt is an architecture the converter implements that the
// download catalogue deliberately does not offer, and why.
//
// Adding to it is a decision, and the gate checks it in both directions so an
// entry that stops being true fails rather than lingering.
var obtainableExempt = map[string]string{
	"clip": "the vision tower, never a model on its own: it converts as the second " +
		"file of a VLM pair and the catalogue lists the pair",
	"gemma": "gemma 1, superseded by gemma2 and gemma3 which are both listed; a " +
		"gemma 1 GGUF still converts, it is simply not offered as a download",
	"dbrx": "132B and gated at the source: built and gated on a synthetic fixture " +
		"from transformers' own DbrxForCausalLM, and no published quant has been " +
		"converted and verified here, which is what a catalogue entry promises",
	"glm4moe": "GLM-4.5-Air is 106B, 73 GB at Q4_K_M in two GGUF parts, and a catalogue " +
		"entry names one file; it is verified (20 of 20 teacher-forced against llama.cpp " +
		"on a V100 box) and converts from the first part",
	"hunyuan-moe": "built and gated on a synthetic fixture from transformers' own " +
		"HunYuanMoEV1ForCausalLM; Hunyuan-A13B (80B) is not yet verified here",
	"deepseek32": "DeepSeek-V3.2 is 685B, over 400 GB at Q4_K_M in many GGUF parts; built " +
		"and gated on a synthetic fixture from transformers' own DeepseekV32ForCausalLM",
	"minimax-m2": "MiniMax-M2 is 230B, over 130 GB at Q4_K_M in several GGUF parts, and a " +
		"catalogue entry names one file; built and gated on a synthetic fixture from " +
		"transformers' own MiniMaxM2ForCausalLM",
	"minimax-m3": "MiniMax-M3 is 428B, about 200 GB at Q3_K_M in five GGUF parts, and a " +
		"catalogue entry names one file; built and gated on a synthetic fixture from " +
		"transformers' own MiniMaxM3VLForCausalLM",
	"deepseek4": "every published DeepSeek V4 is hundreds of GB in many GGUF parts, and a " +
		"catalogue entry names one file; built and gated on a synthetic fixture from " +
		"transformers' own DeepseekV4ForCausalLM",
	"kimi-k3": "Kimi-K3 is 2.8T and its pruned derivatives hundreds of GB in many GGUF parts, " +
		"and a catalogue entry names one file; the one small trained K3, Kimi-K3-0.40B, is a test " +
		"model fitted to a single copypasta, verified against Moonshot's own code at every " +
		"published quantization but not a model to offer",
	"gemma4": "built and gated on synthetic fixtures from transformers' own Gemma4ForCausalLM; " +
		"the 31B is the only published size whose graph is complete here (E2B/E4B carry " +
		"per-layer embeddings, the 26B a mixture), and it is not yet verified against llama.cpp",
	"qwen4exp": "Qwen3.8-Flash-Next is 125B, 111 GB at UD-Q4_K_XL in four GGUF parts, and a " +
		"catalogue entry names one file; built and gated on a synthetic fixture from " +
		"transformers' own Qwen4ExpForCausalLM, and its UD-Q4_K_XL converts from the first part",
	"dots1": "dots.llm1 is 142B, 96 GB at Q4_K_M in three GGUF parts, and a catalogue entry " +
		"names one file; built and gated on a synthetic fixture from transformers' own " +
		"Dots1ForCausalLM, and 18 of 20 teacher-forced against llama.cpp (V100 box)",
}

// TestEveryArchitectureIsObtainable asserts that every architecture the
// converter implements can actually be got from the app. It is the other
// direction of TestEveryEntryIsWellFormed, which catches a typo in the
// catalogue but never an absence from it.
func TestEveryArchitectureIsObtainable(t *testing.T) {
	archs := convert.Architectures()
	if len(archs) < 5 {
		t.Fatalf("the converter reports %d architectures -- this gate proved nothing", len(archs))
	}
	have := map[string]string{}
	for _, m := range Models {
		if _, seen := have[m.Arch]; !seen {
			have[m.Arch] = m.Name
		}
	}
	for _, a := range archs {
		if name, ok := have[a]; ok {
			if why, exempt := obtainableExempt[a]; exempt {
				t.Errorf("%q is exempt from the catalogue (%q) and %q offers it -- "+
					"drop the exemption", a, why, name)
			}
			continue
		}
		if _, exempt := obtainableExempt[a]; exempt {
			continue
		}
		t.Errorf("the converter implements %q and nothing in the catalogue offers it, "+
			"so the only way to obtain one is to already have the file. Add an entry, "+
			"or add it to obtainableExempt with the reason", a)
	}
	// An exemption for an architecture the converter no longer implements is
	// stale in the other direction, and a stale exemption is how the next one
	// gets waved through.
	known := map[string]bool{}
	for _, a := range archs {
		known[a] = true
	}
	for a := range obtainableExempt {
		if !known[a] {
			t.Errorf("%q is exempt from the catalogue and the converter does not "+
				"implement it -- drop the exemption", a)
		}
	}
}

package convert

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// This file is the only place the source format's key vocabulary lives. A
// container carries a typed, complete jlm.Config written once here, so the
// reader never parses another project's key names. The architecture switch
// below is this engine's knowledge of what each graph does, mostly read off a
// reference implementation's source; an unimplemented architecture fails here,
// at conversion, rather than at load.

var archOf = map[string]jlm.Arch{
	"llama":     jlm.ArchLlama, // llama2 and llama3 are this arch too
	"qwen2":     jlm.ArchQwen2,
	"qwen3":     jlm.ArchQwen3,
	"qwen3moe":  jlm.ArchQwen3MoE,
	"olmoe":     jlm.ArchOLMoE,
	"gemma":     jlm.ArchGemma,
	"phi3":      jlm.ArchPhi3, // phi4 uses this arch string too
	"clip":      jlm.ArchCLIP,
	"qwen3next": jlm.ArchQwen3Next,
	// Qwen3.5/3.6 are qwen3next's block re-cut, and every cut is in the file:
	// beta and alpha arrive as two tensors (fused back by fuseBetaAlpha), the
	// value heads are tiled (jlm.FlagDeltaKeyTiled), qwen35's FFN is dense, and
	// the interleaved M-RoPE is exactly NEOX for a text prompt. See qwen35Of.
	"qwen35":    jlm.ArchQwen3Next,
	"qwen35moe": jlm.ArchQwen3Next,
	// Clef-Flash: Qwen3.5-9B's text model under another name, with a joint
	// schema head after it that the decision readout reads (decisionOf,
	// jlm.DecisionClef).
	"clef":    jlm.ArchQwen3Next,
	"qwen2vl": jlm.ArchQwen2VL,
	"gemma2":  jlm.ArchGemma2,
	"gemma3":  jlm.ArchGemma3,
	"gpt-oss": jlm.ArchGPTOSS,
	// DeepSeek-V2/V3/R1, Kimi-K2 and Moonlight-16B-A3B. Not GLM: see
	// jlm.ArchDeepseek2 for which GLM is this graph and which is not.
	"deepseek2": jlm.ArchDeepseek2,
	// The BERT family of embedding models: all-minilm, mxbai-embed-large,
	// bge-large and the bert-sized snowflake-arctic-embed tags declare "bert";
	// nomic-embed-text declares "nomic-bert". See jlm.ArchBERT.
	"bert":       jlm.ArchBERT,
	"nomic-bert": jlm.ArchNomicBERT,
	// ModernBERT, and the decision encoders built on it (Laya). See
	// jlm.ArchModernBERT.
	"modern-bert": jlm.ArchModernBERT,
	// Llama 4 Scout (16 experts) and Maverick (128), text only. See
	// jlm.ArchLlama4 for the five things that make it not llama.
	"llama4": jlm.ArchLlama4,
	// IBM Granite 3.x/4.0, dense and mixture: llama's block with four scaling
	// constants. See jlm.ArchGranite. (granitehybrid, the Mamba-2 hybrid, is a
	// different graph and is not here.)
	"granite":    jlm.ArchGranite,
	"granitemoe": jlm.ArchGranite,
	// The classic-transformer blocks (C6): LayerNorm with biases, ungated
	// FFNs, parallel residuals. See jlm.ArchPhi2.
	"phi2":      jlm.ArchPhi2,
	"starcoder": jlm.ArchStarcoder,
	// Cohere: Command-R, Aya and Aya Expanse declare "command-r"; Command-R7B
	// and Command-A declare "cohere2". See jlm.ArchCommandR.
	"command-r":  jlm.ArchCommandR,
	"cohere2":    jlm.ArchCommandR,
	"stablelm":   jlm.ArchStableLM,
	"starcoder2": jlm.ArchStarcoder2,
	"falcon":     jlm.ArchFalcon,
	"nemotron":   jlm.ArchNemotron,
	// Databricks' DBRX: the C6 kit's bias-free LayerNorm on a mixture, with
	// q, k and v clamped. See jlm.ArchDBRX.
	"dbrx": jlm.ArchDBRX,
	// GLM-4.5, GLM-4.6 and GLM-4.5-Air: DeepSeek-V3's router on GQA attention.
	// See jlm.ArchGLM4MoE.
	"glm4moe": jlm.ArchGLM4MoE,
	// Qwen1.5-MoE and Qwen2-57B-A14B. See jlm.ArchQwen2MoE.
	"qwen2moe": jlm.ArchQwen2MoE,
	// ERNIE 4.5: the dense models are llama's graph exactly (interleaved
	// rotary, tied head, biases only as tensors), as llama3 is; the mixtures
	// are their own. See jlm.ArchErnie45MoE.
	"ernie4_5":     jlm.ArchLlama,
	"ernie4_5-moe": jlm.ArchErnie45MoE,
	// The dense llama-family group: each is llama's block with switches the
	// graph already runs. See convert/dense.go and jlm.ArchSmolLM3.
	"smollm3":  jlm.ArchSmolLM3,
	"arcee":    jlm.ArchArcee,
	"seed_oss": jlm.ArchSeedOSS,
	"olmo2":    jlm.ArchOLMo2, // OLMo 3 declares olmo2 too
	"exaone4":  jlm.ArchExaone4,
	"mistral3": jlm.ArchMistral3,
	// Tencent's Hunyuan, mixture and dense: one graph, q/k norm after the
	// rotary. See jlm.ArchHunyuan.
	"hunyuan-moe":   jlm.ArchHunyuan,
	"hunyuan-dense": jlm.ArchHunyuan,
	// MiniMax-M2/M2.1/M2.5: OLMoE's whole-projection q/k norm on GQA with
	// partial NEOX rotary, and DeepSeek-V3's sigmoid router without a shared
	// expert. See jlm.ArchMiniMaxM2.
	"minimax-m2": jlm.ArchMiniMaxM2,
	// Gemma 4 (text): gemma3's block with its global layers at another
	// attention geometry. See jlm.ArchGemma4.
	"gemma4": jlm.ArchGemma4,
	// DeepSeek V3.2: deepseek2's graph with DeepSeek Sparse Attention's
	// lightning indexer. See jlm.ArchDeepseek32.
	"deepseek32": jlm.ArchDeepseek32,
	// Gemma 3n (text): gemma3's attention with Gemma 4's E-model pieces,
	// inside AltUp's parallel residual streams. See jlm.ArchGemma3n.
	"gemma3n": jlm.ArchGemma3n,
	// The state-space hybrids: Mamba-2 blocks (jlm.LayerSSD), alone or beside
	// attention. See convert/ssm.go and jlm.ArchMamba2.
	"mamba2":        jlm.ArchMamba2,
	"granitehybrid": jlm.ArchGraniteHybrid,
	// Nemotron-H and Nemotron Nano 2: one mixer a source layer, merged into
	// blocks at conversion (convert/nemotronh.go).
	"nemotron_h": jlm.ArchNemotronH,
	// Nemotron 3 (Nano): the same blocks with the MLP a mixture of ungated
	// experts. See nemotronConfig.
	"nemotron_h_moe": jlm.ArchNemotronH,
	// Mamba-1: plain Mamba and FalconMamba (one arch, the latter's dt/B/C
	// norms written as ones), and Jamba beside attention. See convert/mamba1.go.
	"mamba": jlm.ArchMamba,
	"jamba": jlm.ArchJamba,
	// Falcon-H1: attention and Mamba-2 side by side in every block.
	"falcon-h1": jlm.ArchFalconH1,
	// LFM2: gated short convolutions beside attention (convert/lfm2.go).
	"lfm2": jlm.ArchLFM2,
	// LFM2's mixture (LFM2-8B-A1B, LFM2-24B-A2B): the same blocks with a
	// sigmoid top-k after a dense lead, read off the keys as for any mixture.
	"lfm2moe": jlm.ArchLFM2,
	// MiniMax-M3 (text): MiniMax-M2's attention with DeepSeek-V3's mixture
	// and MiniMax Sparse Attention. See jlm.ArchMiniMaxM3.
	"minimax-m3": jlm.ArchMiniMaxM3,
	// The modern text group: Ling 2.0, dots.llm1, Phi-3.5-MoE and Apertus.
	// See convert/moefamily2.go, convert/apertus.go and jlm.ArchBailingMoE2.
	"bailingmoe2": jlm.ArchBailingMoE2,
	"dots1":       jlm.ArchDots1,
	"phimoe":      jlm.ArchPhiMoE,
	"apertus":     jlm.ArchApertus,
	// BitNet b1.58 2B-4T as bitnet.cpp's converter names it (convert/bitnet.go,
	// jlm.ArchBitNet). llama.cpp's own "bitnet" is 1bitLLM's reproduction, a
	// SiLU gate, and its converter cannot write transformers' BitNetForCausalLM
	// at all (it maps none of its inner-norm names); it is not this graph.
	"bitnet-b1.58": jlm.ArchBitNet,
	// DeepSeek V4 (text): hyper-connected residual streams, single-key MQA
	// over a sliding window and a compressed history. See jlm.ArchDeepseek4.
	"deepseek4": jlm.ArchDeepseek4,

	// Qwen3-VL's text models: qwen3 and qwen3moe under an interleaved M-RoPE.
	// See jlm.ArchQwen3VL.
	"qwen3vl":    jlm.ArchQwen3VL,
	"qwen3vlmoe": jlm.ArchQwen3VLMoE,
	// GLM-4-0414 and the text model of GLM-4.1V / 4.6V-Flash. See jlm.ArchGLM4.
	// GLM-4.5V's and 4.6V's is glm4moe with M-RoPE sections, which configOf
	// carries as jlm.ArchGLM4VMoE.
	"glm4": jlm.ArchGLM4,
	// HunyuanVL's text model (HunyuanOCR): hunyuan-dense's block under
	// XD-RoPE. See jlm.ArchHunyuanVL.
	"hunyuan_vl": jlm.ArchHunyuanVL,
	// Kimi-K3 (text): Kimi-Linear's KDA + MLA hybrid with residual attention,
	// a latent mixture and the situ activation. See jlm.ArchKimiK3.
	"kimi-k3": jlm.ArchKimiK3,
	// Qwen3.8-Flash-Next (text): qwen3next's hybrid inside low-rank gated
	// hyper-connections, with QSA block selection and an n-gram embedding.
	// See jlm.ArchQwen4Exp.
	"qwen4exp": jlm.ArchQwen4Exp,
}

// Supported reports whether the converter implements a GGUF architecture
// name -- the list above, which is the list (RULE 7).
func Supported(arch string) bool {
	_, ok := archOf[arch]
	return ok
}

// Architectures is every name in that list, sorted.
//
// It exists so the list can be checked against, not just consulted:
// library.TestEveryArchitectureIsObtainable reads it to confirm everything the
// converter implements is reachable from the download catalogue.
func Architectures() []string {
	out := make([]string, 0, len(archOf))
	for name := range archOf {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// ArchitectureGraph is the graph a GGUF architecture name converts to, and
// whether the name is in the list. Several names share one graph. The file can
// refine it further (olmo2 with a sliding window is jlm.ArchOLMo3, glm4moe
// with M-RoPE sections jlm.ArchGLM4VMoE); this is the name's own entry.
func ArchitectureGraph(name string) (jlm.Arch, bool) {
	a, ok := archOf[name]
	return a, ok
}

func configOf(f *meta.File) (*jlm.Config, error) {
	name := f.Arch()
	arch, ok := archOf[name]
	if !ok {
		return nil, fmt.Errorf("convert: architecture %q is not implemented (%s): %w",
			name, strings.Join(Architectures(), ", "), ErrNotImplemented)
	}
	// A state-space hybrid writes its head counts per layer with zeros on the
	// recurrent layers, and a model with no attention at all writes zero:
	// layerCount reads the attention layers' count and lets the all-recurrent
	// one through (ssdConfig fills a placeholder geometry).
	count := perLayerCount
	if ssdArch(arch) || arch == jlm.ArchLFM2 || mamba1Arch(arch) || arch == jlm.ArchKimiK3 {
		count = attnLayerCount
	}
	nhead, err := count(f, "attention.head_count", 0)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	c := &jlm.Config{
		Arch:       arch,
		NLayer:     uint32(f.UintKey("block_count", 0)),
		NEmbd:      uint32(f.UintKey("embedding_length", 0)),
		NHead:      nhead,
		NFFN:       uint32(f.UintKey("feed_forward_length", 0)),
		NCtx:       uint32(f.UintKey("context_length", 2048)),
		RMSEps:     float32(f.FloatKey("attention.layer_norm_rms_epsilon", 1e-5)),
		RopeBase:   float32(f.FloatKey("rope.freq_base", 10000)),
		EmbdScale:  1,
		AttnFactor: float32(f.FloatKey("rope.scaling.attn_factor", 1)),
	}
	if c.NHead == 0 && (ssdArch(arch) || mamba1Arch(arch)) {
		// No layer attends (plain Mamba-2): one placeholder head as wide as
		// the embedding keeps every attention-shaped derivation finite, and
		// no layer reads it.
		c.NHead = 1
	}
	if c.NLayer == 0 || c.NEmbd == 0 || c.NHead == 0 {
		return nil, fmt.Errorf("convert: %s: missing block_count/embedding_length/head_count", name)
	}
	// Absent head_count_kv means MHA, not zero heads. Gemma 4 writes one per
	// layer that differs between its sliding and global layers, which
	// gemma4Config reads against the sliding pattern.
	if arch != jlm.ArchGemma4 {
		if c.NKVHead, err = count(f, "attention.head_count_kv", c.NHead); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if c.NKVHead == 0 && arch != jlm.ArchGemma4 {
		c.NKVHead = c.NHead
	}
	// key_length when present, else embd/heads. For gemma these coincide at 256,
	// so reading it from the wrong place goes unnoticed there.
	c.HeadDim = uint32(f.UintKey("attention.key_length", uint64(c.NEmbd/c.NHead)))
	c.HeadDimV = uint32(f.UintKey("attention.value_length", uint64(c.HeadDim)))
	c.NRot = uint32(f.UintKey("rope.dimension_count", uint64(c.HeadDim)))

	// On an MLA model those two keys mean something else. The source writes four
	// head widths for DeepSeek:
	//
	//	attention.key_length        576 = kv_lora_rank + qk_rope_head_dim
	//	attention.value_length      512 = kv_lora_rank
	//	attention.key_length_mla    192 = qk_nope_head_dim + qk_rope_head_dim
	//	attention.value_length_mla  128 = v_head_dim
	//
	// The first two describe the absorbed cache row; the last two are the head
	// geometry and what the softmax is scaled by. Taking key_length would load
	// and run fluently with every score scaled by sqrt(576) instead of sqrt(192).
	// The container stores the natural pair and derives the absorbed one.
	if kl, ok := f.Key("attention.key_length_mla"); ok {
		if v, ok := kl.Uint(); ok {
			c.HeadDim = uint32(v)
		}
		c.HeadDimV = uint32(f.UintKey("attention.value_length_mla", uint64(c.HeadDim)))
	}
	// Stored only when it says something HeadDim does not.
	if c.HeadDimV == c.HeadDim {
		c.HeadDimV = 0
	}

	switch arch {
	case jlm.ArchDeepseek2, jlm.ArchDeepseek32:
		// The two latent ranks. The explicit key is the authority; value_length is
		// the fallback because on this arch it is kv_lora_rank (the absorbed row,
		// see above). The safetensors path reads these from config.json directly, so
		// only a real GGUF exercises this.
		c.KVLoraRank = uint32(f.UintKey("attention.kv_lora_rank",
			f.UintKey("attention.value_length", 0)))
		// Absent on V2-Lite, whose q_lora_rank is null: it projects the query
		// straight to NHead*HeadDim, and model.mlaProject asks the config.
		c.QLoraRank = uint32(f.UintKey("attention.q_lora_rank", 0))
		if arch == jlm.ArchDeepseek32 {
			if err := deepseek32Config(f, c); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
		}
	case jlm.ArchLlama:
		// llama3 is this arch too: it differs only in values the header carries
		// (rope base, GQA, vocabulary size) and in its BPE tokenizer.
	case jlm.ArchBitNet:
		bitnetConfig(c)
	case jlm.ArchQwen2:
		// llama's graph with biases on q, k and v, which are tensors.
		c.Flags |= jlm.FlagRopeNeox
	case jlm.ArchQwen3:
		// Each head of q and k is RMSNormed with a shared head_dim-wide weight,
		// after the projection and before RoPE (after RoPE also runs, and is wrong).
		c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm
	case jlm.ArchQwen3MoE:
		// qwen3's attention with a mixture FFN; being a mixture is read from the
		// file below, not from the architecture.
		c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm
	case jlm.ArchOLMoE:
		// qwen3moe's shape with two differences, read off the reference source:
		//  1. the q/k RMSNorm spans the whole projection ({n_embd}, before the
		//     reshape into heads), where qwen3's is {head_dim} and per head;
		//  2. the mixture weights are not renormalised.
		c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm | jlm.FlagQKNormWide | jlm.FlagNoExpertNorm
	case jlm.ArchPhi3:
		// phi3, and phi4 which uses this arch string too. No fusion flags: the
		// converter unfuses phi3's attn_qkv and fused ffn_up (see unfuse), because
		// a packed row range has no byte range and the device tier cannot slice one.
		c.Flags |= jlm.FlagRopeNeox
		// Every phi3 layer slides (HuggingFace and llama.cpp both mask every layer
		// to sliding_window). The key when the file has it; otherwise llama.cpp's
		// defaults for files written before the key existed (its PR 8931), and a
		// refusal for anything else, as llama.cpp refuses. A key that is
		// present and zero says no window at all: llama.cpp's converter writes
		// 0 for Phi-4-reasoning-vision, whose config's sliding_window is null,
		// and reads it as full attention. Only an ABSENT key falls to the
		// defaults.
		win := uint32(f.UintKey("attention.sliding_window", 0))
		if _, stated := f.Key("attention.sliding_window"); win == 0 && !stated {
			switch nkv := c.NKVHead; {
			case (c.NLayer == 32 || c.NLayer == 40) && c.NCtx == 4096:
				win = 2047
			case c.NLayer == 32 && nkv == 32 && c.NCtx == 131072:
				win = 262144
			case c.NLayer == 40 && c.NCtx == 131072:
				win = 131072
			case c.NLayer == 40 && c.NCtx == 16384 && nkv == 10:
				// phi-4 (14B): its config's sliding_window is null -- full
				// attention on every layer -- so the absent key means none.
			default:
				return nil, fmt.Errorf("convert: %s: no attention.sliding_window and no "+
					"known default for %d layers at context %d", name, c.NLayer, c.NCtx)
			}
		}
		if win > 0 {
			c.SWAWindow, c.SWAPeriod = win, allLocal(c.NLayer)
		}
	case jlm.ArchGemma:
		// The embedding is scaled by sqrt(n_embd) and the FFN activation is GELU.
		// The RMSNorm +1 is already baked into the weights by the HF converter, so
		// it is not applied again. Gemma's rotary is NEOX, not NORM: NORM runs
		// fluently and is wrong, and every tier agrees with itself either way.
		c.EmbdScale = float32(math.Sqrt(float64(c.NEmbd)))
		c.Flags |= jlm.FlagGELU | jlm.FlagRopeNeox
	case jlm.ArchGemma2:
		// gemma1's block plus three load-bearing differences.
		c.EmbdScale = float32(math.Sqrt(float64(c.NEmbd)))
		c.Flags |= jlm.FlagGELU | jlm.FlagRopeNeox
		// Alternating local and global layers: SWAPeriod=2 makes Config.SWA(il)
		// (il%period < period-1) true on even layers. An absent window is gemma2's
		// 4096, not none, as llama.cpp's loader defaults it, and the local layers
		// rotate at the global base unless the file says otherwise.
		c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 4096))
		if err := swaPatternOf(f, c, 2); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		c.RopeBaseSWA = float32(f.FloatKey("rope.freq_base_swa", float64(c.RopeBase)))
		gemmaAttnScale(c)
		// Two softcaps with different values (50 on scores, 30 on logits), hence
		// two fields rather than a flag.
		c.AttnSoftcap = float32(f.FloatKey("attn_logit_softcapping", 0))
		c.FinalSoftcap = float32(f.FloatKey("final_logit_softcapping", 0))
	case jlm.ArchGemma3:
		c.EmbdScale = float32(math.Sqrt(float64(c.NEmbd)))
		c.Flags |= jlm.FlagGELU | jlm.FlagRopeNeox | jlm.FlagQKNorm
		// Five local then one global: SWAPeriod 6. No window key means no sliding
		// layers at all (llama.cpp's LLAMA_SWA_TYPE_NONE), not a default window;
		// some gemma3-arch models ship every layer global.
		if c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 0)); c.SWAWindow > 0 {
			if err := swaPatternOf(f, c, 6); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			// The local base is the architecture's (10000, llama.cpp's default) unless
			// the file says; gemma-3-1b carries a 1e6 global base and no local key.
			c.RopeBaseSWA = float32(f.FloatKey("rope.freq_base_swa", 10000))
		}
		// gemma3 removed gemma2's attention softcap; its final softcap is optional.
		c.FinalSoftcap = float32(f.FloatKey("final_logit_softcapping", 0))
		gemmaAttnScale(c)
	case jlm.ArchQwen3Next, jlm.ArchQwen4Exp:
		// A hybrid: most layers run a gated delta rule over a recurrent state
		// and every fourth runs ordinary attention. NEOX rope on the attention
		// layers, per-head q/k norms, an output gate folded into the q
		// projection, and a shared expert beside the routed ones.
		c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm | jlm.FlagAttnOutGate
		c.SSM = jlm.SSMConfig{
			ConvKernel: uint32(f.UintKey("ssm.conv_kernel", 0)),
			Groups:     uint32(f.UintKey("ssm.group_count", 0)),
			Inner:      uint32(f.UintKey("ssm.inner_size", 0)),
			StateSize:  uint32(f.UintKey("ssm.state_size", 0)),
			NHeadV:     uint32(f.UintKey("ssm.time_step_rank", 0)),
		}
		if c.SSM.ConvKernel == 0 || c.SSM.Inner == 0 || c.SSM.StateSize == 0 || c.SSM.NHeadV == 0 {
			return nil, fmt.Errorf("convert: %s: incomplete recurrent geometry %+v", name, c.SSM)
		}
		if err := qwen35Of(f, name, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		if arch == jlm.ArchQwen4Exp {
			if err := qwen4expConfig(f, c); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
		}
	case jlm.ArchQwen2VL:
		// qwen2's graph with multimodal rope: the sections split each head's rotary
		// pairs between the (t, h, w) components of a position (Qwen2-VL-2B is
		// [16, 24, 24, 0], summing to NRot/2 = 64). See mropeSectionsOf.
		c.Flags |= jlm.FlagRopeNeox
		if err := mropeSectionsOf(f, name, c); err != nil {
			return nil, err
		}
	case jlm.ArchQwen3VL, jlm.ArchQwen3VLMoE:
		// qwen3's and qwen3moe's blocks under an INTERLEAVED M-RoPE (Qwen3-VL-2B
		// is [24, 20, 20, 0] of 64 pairs). See jlm.ArchQwen3VL.
		c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm
		if err := mropeSectionsOf(f, name, c); err != nil {
			return nil, err
		}
	case jlm.ArchGPTOSS:
		// NEOX rope (llama.cpp's rope-type table groups it with qwen3), the
		// clamped swiglu-oai activation, and a sliding window on alternate
		// layers starting at layer 0 -- set_swa_pattern(2), the same rule
		// gemma2 uses. The sinks and every bias are tensors, read off the file.
		c.Flags |= jlm.FlagRopeNeox | jlm.FlagSwiGLUOAI
		c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 0))
		if c.SWAWindow == 0 {
			return nil, fmt.Errorf("convert: %s: no attention.sliding_window", name)
		}
		c.SWAPeriod = 2
		// The YaRN range is not rounded: HF's gpt-oss config carries
		// "truncate": false, where llama.cpp floors and ceils it for every model.
		// This matches the model's own definition (RULE 7m).
		c.Flags |= jlm.FlagYarnExact
	case jlm.ArchLlama4:
		if err := llama4Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchGranite:
		if err := graniteConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchPhi2:
		if err := phi2Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchStarcoder:
		if err := starcoderConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchCommandR:
		if err := commandRConfig(f, c, name == "cohere2"); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchStarcoder2:
		// llama.cpp's starcoder2.cpp: the classic sequential block, every
		// matrix biased, NEOX rotary over the whole head.
		layerNormEps(f, c)
		c.Flags |= jlm.FlagLayerNorm | jlm.FlagRopeNeox | jlm.FlagGELU
	case jlm.ArchStableLM:
		if err := stableLMConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchFalcon:
		// llama.cpp's falcon.cpp: the parallel block, NEOX rotary over the
		// whole head, an ungated GELU FFN, LayerNorm with biases and no
		// matrix biases. The two-norm form is resolved at the tensor names
		// (falconNorms).
		layerNormEps(f, c)
		c.Flags |= jlm.FlagLayerNorm | jlm.FlagParallel | jlm.FlagRopeNeox | jlm.FlagGELU
	case jlm.ArchNemotron:
		// llama.cpp's nemotron.cpp: the sequential block, NEOX rotary on
		// rope.dimension_count of head_dim, up -> relu^2 -> down.
		layerNormEps(f, c)
		c.Flags |= jlm.FlagLayerNorm | jlm.FlagRopeNeox | jlm.FlagReLU2
		if err := partialRotary(c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchDBRX:
		if err := dbrxConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchGLM4MoE:
		if err := glm4moeConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		// GLM-4.5V / 4.6V: the same block under Qwen2-VL's M-RoPE, a code of
		// its own (jlm.ArchGLM4VMoE).
		if _, ok := f.Key("rope.dimension_sections"); ok {
			c.Arch = jlm.ArchGLM4VMoE
			if err := mropeSectionsOf(f, name, c); err != nil {
				return nil, err
			}
		}
	case jlm.ArchGLM4:
		// llama's block with biased q/k/v (tensors), a norm after the
		// attention and after the MLP (RolePostAttnNorm, RolePostFFNNorm), a
		// fused gate|up (unfuse), and a rotary on rope.dimension_count of each
		// head: NORM pairs, or with sections the NEOX M-RoPE llama.cpp's
		// converter permutes q and k into.
		if err := partialRotary(c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		if _, ok := f.Key("rope.dimension_sections"); ok {
			c.Flags |= jlm.FlagRopeNeox
			if err := mropeSectionsOf(f, name, c); err != nil {
				return nil, err
			}
		}
		if f.UintKey("nextn_predict_layers", 0) != 0 {
			return nil, fmt.Errorf("convert: %s: a glm4 file with prediction layers (GLM-OCR): %w",
				name, ErrNotImplemented)
		}
	case jlm.ArchQwen2MoE:
		if err := qwen2moeConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchErnie45MoE:
		if err := ernie45moeConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchSmolLM3, jlm.ArchArcee, jlm.ArchSeedOSS, jlm.ArchOLMo2, jlm.ArchExaone4, jlm.ArchMistral3, jlm.ArchOLMo3:
		if err := denseConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchHunyuan:
		if err := hunyuanConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchHunyuanVL:
		if err := hunyuanVLConfig(f, name, c); err != nil {
			return nil, err
		}
	case jlm.ArchMiniMaxM2:
		if err := minimaxM2Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchGemma4:
		if err := gemma4Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchGemma3n:
		if err := gemma3nConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchMamba2, jlm.ArchGraniteHybrid, jlm.ArchNemotronH, jlm.ArchFalconH1:
		if err := ssdConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchMamba, jlm.ArchJamba:
		if err := mamba1Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		if c.LayerKinds, err = mamba1Kinds(f, c.NLayer); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchLFM2:
		if err := lfm2Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchMiniMaxM3:
		if err := minimaxM3Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchBailingMoE2:
		if err := bailingmoe2Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchDots1:
		if err := dots1Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchPhiMoE:
		if err := phimoeConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchApertus:
		if err := apertusConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchDeepseek4:
		if err := deepseek4Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchKimiK3:
		if err := kimiK3Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case jlm.ArchCLIP:
		return nil, fmt.Errorf("convert: %q is a vision tower, not a language model", name)
	case jlm.ArchBERT, jlm.ArchNomicBERT:
		// The encoder's norms are LayerNorms under attention.layer_norm_epsilon
		// (all-minilm's is 1e-12); layer_norm_rms_epsilon read above is absent.
		c.RMSEps = float32(f.FloatKey("attention.layer_norm_epsilon", 1e-12))
		// An encoder has no causal mask: every token attends to every token.
		c.Flags |= jlm.FlagNonCausal
		if arch == jlm.ArchNomicBERT {
			// Rotary on q and k in place of the position table, NEOX-paired
			// (llama.cpp's rope-type table groups nomic-bert with qwen2). The
			// base comes off rope.freq_base -- 1000 on nomic-embed-text-v1.5.
			c.Flags |= jlm.FlagRopeNeox
			if n := f.UintKey("moe_every_n_layers", 0); n > 0 {
				return nil, fmt.Errorf("convert: %s: a mixture every %d layers "+
					"(nomic-bert-moe) is not implemented: %w", name, n, ErrNotImplemented)
			}
		} else {
			c.Flags |= jlm.FlagGELU
		}
	case jlm.ArchModernBERT:
		if err := modernBERTConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}

	// A clamp on any other architecture is refused, not dropped: this converter
	// writes it only for ArchDBRX, whose code an older reader refuses.
	if arch != jlm.ArchDBRX {
		if v := f.FloatKey("attention.clamp_kqv", 0); v != 0 {
			return nil, fmt.Errorf("convert: %s: attention.clamp_kqv %g outside dbrx: %w",
				name, v, ErrNotImplemented)
		}
	}

	// An embedding model is flagged by two keys, read for every architecture:
	// pooling_type (1 mean, 2 first position, 3 last) and attention.causal=false.
	// A decoder can carry either (qwen3-embedding, EmbeddingGemma), so neither
	// makes a new architecture.
	if kv, ok := f.Key("pooling_type"); ok {
		switch v, _ := kv.Uint(); v {
		case 0:
		case 1:
			c.Flags |= jlm.FlagPoolMean
		case 2:
			c.Flags |= jlm.FlagPoolCLS
		case 3:
			c.Flags |= jlm.FlagPoolLast
		default:
			// 4 is RANK, a reranker's classification head; the vector would be
			// the wrong object entirely.
			return nil, fmt.Errorf("convert: %s: pooling_type %d is not implemented: %w",
				name, v, ErrNotImplemented)
		}
	}
	if kv, ok := f.Key("attention.causal"); ok {
		if v, ok := kv.Uint(); ok && v == 0 {
			c.Flags |= jlm.FlagNonCausal
		}
	}

	// block_count includes any multi-token-prediction blocks after the trunk;
	// from here on NLayer is the trunk alone. Before the layer kinds below,
	// which are the trunk's.
	if err := nextnOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}

	// Rotary scaling is read for every architecture; a type this converter does
	// not implement is refused rather than dropped.
	if err := ropeScalingOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}

	// The shared expert is read off a key: its presence is the fact, not the
	// architecture.
	c.NFFNShExp = uint32(f.UintKey("expert_shared_feed_forward_length", 0))

	// The per-layer kinds are read off the tensors (a layer with a recurrent
	// block is linear), not off an "every fourth layer" rule keyed on the name.
	if (c.Arch == jlm.ArchQwen3Next || c.Arch == jlm.ArchQwen4Exp || ssdArch(c.Arch) || c.Arch == jlm.ArchLFM2) && c.LayerKinds == nil {
		rec := jlm.LayerLinearAttn
		if ssdArch(c.Arch) {
			rec = jlm.LayerSSD
		}
		kinds := make([]jlm.LayerKind, c.NLayer)
		linear := 0
		for i := range kinds {
			kinds[i] = jlm.LayerFullAttn
			if _, ok := f.Get("blk." + itoa(i) + ".shortconv.conv.weight"); ok {
				kinds[i], linear = jlm.LayerShortConv, linear+1
			}
			if _, ok := f.Get("blk." + itoa(i) + ".ssm_conv1d.weight"); ok {
				kinds[i], linear = rec, linear+1
				// A block that also carries attention runs both (Falcon-H1).
				if _, att := f.Get("blk." + itoa(i) + ".attn_q.weight"); att && rec == jlm.LayerSSD {
					kinds[i] = jlm.LayerSSDAttn
				}
			}
		}
		if linear == 0 {
			return nil, fmt.Errorf("convert: %s: no layer carries a recurrent block", name)
		}
		c.LayerKinds = kinds
	}

	// A whole-projection q/k norm is each projection's width: {n_embd} for q
	// and {n_embd_gqa} for k, which is how llama.cpp creates OLMo 2's
	// attn_k_norm. A tensor of another width is refused here, by name, rather
	// than read past or left short at run time.
	if c.Flags.Has(jlm.FlagQKNormWide) {
		for _, nw := range [2]struct {
			n string
			w uint64
		}{{"attn_q_norm", uint64(c.NHead) * uint64(c.HeadDim)}, {"attn_k_norm", uint64(c.NKVHead) * uint64(c.HeadDim)}} {
			n, w := nw.n, nw.w
			if t, ok := f.Get("blk.0." + n + ".weight"); ok && (len(t.Dims) != 1 || t.Dims[0] != w) {
				return nil, fmt.Errorf("convert: %s: blk.0.%s.weight is %v, a whole-projection norm "+
					"of %d q heads and %d kv heads wants [%d]", name, n, t.Dims, c.NHead, c.NKVHead, w)
			}
		}
	}

	if t, ok := f.Get("token_embd.weight"); ok && len(t.Dims) > 1 {
		c.NVocab = uint32(t.Dims[1])
	}
	// Tied embeddings are read off the file: qwen3-1.7B ties and qwen3-8B does
	// not, under the same arch string.
	if _, ok := f.Get("output.weight"); !ok {
		c.Flags |= jlm.FlagTiedEmbd
	}

	// A mixture is a property of the file, not the arch: Mixtral ships as arch
	// "llama" with expert_count 8.
	c.NExpert = uint32(f.UintKey("expert_count", 0))
	if c.NExpert > 0 {
		c.NExpertUsed = uint32(f.UintKey("expert_used_count", 0))
		// Not feed_forward_length: that is the dense width, which a mixture never
		// uses.
		c.NFFNExp = uint32(f.UintKey("expert_feed_forward_length", 0))
		// When the key is absent, read the bank's shape rather than guess: olmoe's
		// banks are {n_embd, n_ff, n_expert}, and a Mixtral-era file ships
		// per-expert tensors (sourceOf stacks them) where n_ff/n_used is off by 2x.
		// The first block that has a bank: Jamba's block 0 is dense.
		for i := 0; c.NFFNExp == 0 && i < int(c.NLayer); i++ {
			if t, ok := f.Get("blk." + itoa(i) + ".ffn_gate_exps.weight"); ok && len(t.Dims) > 1 {
				c.NFFNExp = uint32(t.Dims[1])
			} else if t, ok := f.Get("blk." + itoa(i) + ".ffn_gate.0.weight"); ok && len(t.Dims) > 1 {
				c.NFFNExp = uint32(t.Dims[1])
			}
		}
		if c.NFFNExp == 0 && c.NExpertUsed > 0 {
			c.NFFNExp = c.NFFN / c.NExpertUsed
		}
		if c.NExpertUsed == 0 || c.NExpertUsed > c.NExpert || c.NFFNExp == 0 {
			return nil, fmt.Errorf("convert: %s: expert_count=%d expert_used_count=%d "+
				"expert_feed_forward_length=%d", name, c.NExpert, c.NExpertUsed, c.NFFNExp)
		}
		// DeepSeek carries no shared-expert width: llama.cpp writes
		// expert_shared_count, and the width is that times expert_feed_forward_length.
		// A zero here is not refused downstream; it surfaces as a misleading kernel
		// decline, since the shared expert is bound off the container's tensors.
		if c.NFFNShExp == 0 {
			if n := uint32(f.UintKey("expert_shared_count", 0)); n > 0 {
				c.NFFNShExp = n * c.NFFNExp
			}
		}
		// expert_weights_scale and expert_weights_norm are read by presence, never
		// with a default: qwen3moe carries neither, renormalises unconditionally and
		// applies no scale, so defaulting norm to false or scale to 0 would silently
		// skip the division or zero the FFN.
		if kv, ok := f.Key("expert_weights_scale"); ok {
			if v, ok := kv.Float(); ok && v != 0 {
				c.ExpertScale = float32(v)
			}
		}
		// meta.Value.Int decodes Bool, so Uint is the accessor for a
		// GGUF bool and 0 is false.
		if kv, ok := f.Key("expert_weights_norm"); ok {
			if v, ok := kv.Uint(); ok && v == 0 {
				c.Flags |= jlm.FlagNoExpertNorm
			}
		}
		// An absent expert_weights_norm is not one answer for every arch: llama.cpp's
		// qwen3moe builder passes true as a literal, its deepseek2 builder reads the
		// key with a default of false, and DeepSeek-V2-Lite's config.json says
		// norm_topk_prob: false. Renormalising it generates fluent nonsense from the
		// first token. See docs/engineering-history/model-correctness.md.
		if arch == jlm.ArchDeepseek2 || arch == jlm.ArchDeepseek32 || arch == jlm.ArchQwen2MoE {
			if _, ok := f.Key("expert_weights_norm"); !ok {
				c.Flags |= jlm.FlagNoExpertNorm
			}
		}
		// 1 is softmax, 2 is sigmoid; anything else is refused rather than
		// approximated, because a wrong gate is a fluent model with another mixture.
		if kv, ok := f.Key("expert_gating_func"); ok {
			switch v, _ := kv.Uint(); v {
			case 0, 1:
			case 2:
				c.Flags |= jlm.FlagExpertSigmoid
			case 4:
				// sqrt(softplus): DeepSeek V4's alone (deepseek4Config).
				if arch != jlm.ArchDeepseek4 {
					return nil, fmt.Errorf("convert: %s: expert_gating_func 4 (sqrt-softplus) outside "+
						"deepseek4: %w", name, ErrNotImplemented)
				}
			default:
				return nil, fmt.Errorf("convert: %s: expert_gating_func %d is not implemented: %w",
					name, v, ErrNotImplemented)
			}
		}
		// DeepseekV32TopkRouter hardcodes the sigmoid and its config carries no
		// scoring_func, so llama.cpp's converter writes no gating key for a
		// checkpoint that leaves it out; a key saying softmax contradicts the
		// class and is refused.
		if arch == jlm.ArchDeepseek32 {
			if kv, ok := f.Key("expert_gating_func"); ok {
				if v, _ := kv.Uint(); v != 2 {
					return nil, fmt.Errorf("convert: %s: expert_gating_func %d, and "+
						"DeepseekV32TopkRouter is sigmoid: %w", name, v, ErrNotImplemented)
				}
			}
			c.Flags |= jlm.FlagExpertSigmoid
		}
		// Grouped selection. Both keys or neither; one alone cannot be acted on.
		c.NExpertGroup = uint32(f.UintKey("expert_group_count", 0))
		c.NExpertGroupUsed = uint32(f.UintKey("expert_group_used_count", 0))
		// llama.cpp's kimi-k3 converter writes topk_group (1) as the used
		// count and no group count beside it: one group, all of it used, which
		// is the ungrouped selection.
		if arch == jlm.ArchKimiK3 && c.NExpertGroup == 0 && c.NExpertGroupUsed == 1 {
			c.NExpertGroupUsed = 0
		}
		if (c.NExpertGroup == 0) != (c.NExpertGroupUsed == 0) {
			return nil, fmt.Errorf("convert: %s: expert_group_count=%d with expert_group_used_count=%d",
				name, c.NExpertGroup, c.NExpertGroupUsed)
		}
	}

	// The leading dense run is read for every architecture, as a property of the
	// file. Llama 4's shared expert is as wide as a routed one and the file does
	// not say so (llama.cpp sets n_ff_shexp = n_ff_exp), which needs NFFNExp.
	if arch == jlm.ArchLlama4 && c.NFFNShExp == 0 {
		c.NFFNShExp = c.NFFNExp
	}
	// A dense model has no shared expert, whatever a key says: granite-4.0-micro
	// carries expert_shared_feed_forward_length beside expert_count 0.
	if c.NExpert == 0 {
		c.NFFNShExp = 0
	}
	c.NDenseLead = uint32(f.UintKey("leading_dense_block_count", 0))
	if c.NDenseLead > c.NLayer {
		return nil, fmt.Errorf("convert: %s: leading_dense_block_count %d in a %d-block model",
			name, c.NDenseLead, c.NLayer)
	}
	if arch == jlm.ArchNemotronH {
		if err := nemotronMixture(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if arch == jlm.ArchErnie45MoE || arch == jlm.ArchHunyuan || arch == jlm.ArchBailingMoE2 || arch == jlm.ArchDots1 {
		if err := ernieSharedWidth(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if arch == jlm.ArchGemma4 {
		if err := gemma4MoEConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if arch == jlm.ArchMiniMaxM3 {
		if err := minimaxM3Blocks(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if err := decisionOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	return c, nil
}

// llama4Config reads Llama 4's graph into c. llama.cpp hardcodes the chunk
// (8192), the temperature's scale (0.1), floor (8192) and offset (1), and a
// NoPE period of 4; each is read here from a key a file may carry and falls
// back to that constant.
func llama4Config(f *meta.File, c *jlm.Config) error {
	// Normal rotary over adjacent pairs, as llama.cpp's rope-type table and
	// transformers do; the checkpoint is not permuted the way llama's is.
	//
	// A sliding_window of zero is MobileLLM's "every layer full attention", with
	// no experts, which llama.cpp itself refuses at load; refused here by name.
	if kv, ok := f.Key("attention.sliding_window"); ok {
		if v, ok := kv.Uint(); ok && v == 0 {
			return fmt.Errorf("attention.sliding_window 0 (every layer full attention, "+
				"MobileLLM) is not implemented: %w", ErrNotImplemented)
		}
	}
	c.SWAWindow = uint32(f.UintKey("attention.chunk_size", 8192))
	period := uint64(4)
	if kv, ok := f.Key("attention.sliding_window_pattern"); ok {
		v, ok := kv.Uint()
		if !ok {
			// llama.cpp reads it with get_key_or_arr; a per-layer array is a
			// pattern this container can only say through the period.
			return fmt.Errorf("attention.sliding_window_pattern is not a scalar: %w", ErrNotImplemented)
		}
		period = v
	}
	if period < 2 || c.SWAWindow == 0 {
		return fmt.Errorf("chunk %d with a NoPE period of %d", c.SWAWindow, period)
	}
	c.SWAPeriod = uint32(period)
	c.Flags |= jlm.FlagSWAChunked | jlm.FlagNoPEGlobal
	c.AttnTempScale = float32(f.FloatKey("attention.temperature_scale", 0.1))
	c.AttnTempFloor = uint32(f.UintKey("attention.temperature_length", 8192))
	c.AttnTempOffset = 1
	// The qk norm is keyed on the expert count, which is llama.cpp's rule: the
	// GGUF has no key for use_qk_norm (Scout true, Maverick false). A checkpoint
	// that breaks that correlation needs a key.
	if f.UintKey("expert_count", 0) != 128 {
		c.Flags |= jlm.FlagQKL2Norm
	}
	// The router: top-1 sigmoid, not renormalised, weighting the expert's input
	// (literals in llama.cpp's build_moe_ffn call for this arch).
	c.Flags |= jlm.FlagExpertSigmoid | jlm.FlagNoExpertNorm | jlm.FlagExpertWeightIn
	c.MoEStep = uint32(f.UintKey("interleave_moe_layer_step", 1))
	return nil
}

// ropeScalingOf reads the rotary scaling keys into c.
//
// YaRN's magnitude correction is folded into AttnFactor, which already scales
// cos and sin: 0.1*ln(factor)+1, times the file's own attn_factor -- what both
// llama.cpp (whose context setup cancels and re-applies it) and transformers
// (attention_factor, when the config gives none) arrive at.
//
// A config that states attention_factor reaches the file as
// rope.scaling.yarn_attn_factor (llama.cpp's converter writes it), and
// transformers' _compute_yarn_parameters puts exactly that on cos and sin in
// place of the derived magnitude, for every family here (DeepSeek's score
// correction is separate and stays). llama.cpp's runtime reads the key for
// grok alone and derives the magnitude for every other architecture: RULE 7m,
// chosen -- the model's reference, which parts from llama.cpp only on a file
// that states a factor the formula would not give.
func ropeScalingOf(f *meta.File, c *jlm.Config) error {
	kind := ""
	if kv, ok := f.Key("rope.scaling.type"); ok {
		s, ok := kv.String()
		if !ok {
			return fmt.Errorf("rope.scaling.type is not a string")
		}
		kind = s
	}
	factor := float32(f.FloatKey("rope.scaling.factor", 0))
	switch kind {
	case "", "none":
		return nil
	case "linear":
		if factor > 0 && factor != 1 {
			c.RopeLinear = factor
		}
		return nil
	case "yarn":
		if factor <= 1 {
			return nil
		}
		c.YarnFactor = factor
		c.YarnOrigCtx = uint32(f.UintKey("rope.scaling.original_context_length", uint64(c.NCtx)))
		c.YarnBetaFast = float32(f.FloatKey("rope.scaling.yarn_beta_fast", 32))
		c.YarnBetaSlow = float32(f.FloatKey("rope.scaling.yarn_beta_slow", 1))
		explicit, haveExplicit := 0.0, false
		if kv, ok := f.Key("rope.scaling.yarn_attn_factor"); ok {
			v, ok := kv.Float()
			if !ok || !(v > 0) || math.IsInf(v, 0) {
				return fmt.Errorf("rope.scaling.yarn_attn_factor is not a positive number")
			}
			explicit, haveExplicit = v, true
		}
		// The stored value is the coefficient of ln(factor) itself: llama.cpp
		// writes 0.1*mscale_all_dim here and divides by 0.1 at load, so it
		// replaces the 0.1 rather than scaling it.
		logMul := 0.1
		if c.Arch == jlm.ArchMistral3 {
			if haveExplicit {
				c.AttnFactor *= float32(explicit)
				return nil
			}
			mistral3Yarn(f, c, float64(factor))
			return nil
		}
		if kv, ok := f.Key("rope.scaling.yarn_log_multiplier"); ok {
			if v, ok := kv.Float(); ok {
				logMul = v
				c.YarnLogMul = float32(v)
			}
		}
		// The correction reaches the softmax as well as the rotary: the score is
		// scaled by its square (DeepSeek's softmax_scale, llama.cpp's kq_scale),
		// which the graph applies from c.YarnLogMul.
		//
		// DeepSeek does not scale cos and sin at all: its reference divides
		// yarn_get_mscale(factor, mscale) by yarn_get_mscale(factor,
		// mscale_all_dim), and every released DeepSeek has the two equal, so the
		// rotary is untouched and only the score takes the square. Folding it
		// here made DeepSeek-V2-Lite generate fluent nonsense.
		switch {
		case haveExplicit:
			c.AttnFactor *= float32(explicit)
		case c.Arch == jlm.ArchDeepseek4:
			// DeepSeek V4 scales neither: transformers forces the compressed
			// rotary's attention_factor to 1 and scores at head_dim^-1/2, and
			// llama.cpp's builder passes an attn_factor that cancels ggml's
			// own magnitude and a kq_scale of 1/sqrt(head_dim).
		case c.Arch != jlm.ArchDeepseek2 && c.Arch != jlm.ArchDeepseek32:
			c.AttnFactor *= float32(logMul*math.Log(float64(factor)) + 1)
		}
		return nil
	}
	return fmt.Errorf("rope.scaling.type %q is not implemented: %w", kind, ErrNotImplemented)
}

// allLocal is the SWAPeriod that makes every layer a sliding-window one.
// Config.SWA is il%period < period-1, so NLayer+1 puts every layer in the
// local run; giving period 1 a new meaning would have made older readers see
// no local layers in a newer container.
func allLocal(nLayer uint32) uint32 { return nLayer + 1 }

// mropeSectionsOf reads an M-RoPE model's split of a head's rotary pairs
// between the (t, h, w) coordinates of a position. The sum is checked because a
// short list leaves a tail of each head unrotated, which is fluent and which no
// self-comparing tier can see.
//
// An interleaved split (jlm.Arch.InterleavedMRope) is read as transformers
// reads it: pair i turns by h when i%3 == 1 and i < 3*sections[1], by w when
// i%3 == 2 and i < 3*sections[2], and by t otherwise. ggml's IMROPE sends a
// pair with i%3 == 0 at or past 3*sections[0] to a fourth axis instead; no
// published split reaches one, and one that would is refused rather than
// resolved, since the two references would then rotate it differently.
func mropeSectionsOf(f *meta.File, name string, c *jlm.Config) error {
	kv, ok := f.Key("rope.dimension_sections")
	if !ok {
		return fmt.Errorf("convert: %s: no rope.dimension_sections, which "+
			"an M-RoPE model must carry: %w", name, ErrNotImplemented)
	}
	secs, err := kv.Int32s()
	if err != nil {
		return fmt.Errorf("convert: %s: rope.dimension_sections: %w", name, err)
	}
	if len(secs) > len(c.RopeSections) {
		return fmt.Errorf("convert: %s: %d rope sections, this format defines %d",
			name, len(secs), len(c.RopeSections))
	}
	var sum uint32
	for i, v := range secs {
		if v < 0 {
			return fmt.Errorf("convert: %s: rope section %d is %d", name, i, v)
		}
		c.RopeSections[i] = uint32(v)
		sum += uint32(v)
	}
	if sum != c.NRot/2 {
		return fmt.Errorf("convert: %s: rope sections %v sum to %d, and a head "+
			"has %d rotary pairs -- the remainder would be left unrotated",
			name, secs, sum, c.NRot/2)
	}
	if c.Arch.InterleavedMRope() {
		if c.RopeSections[3] != 0 {
			return fmt.Errorf("convert: %s: an interleaved M-RoPE with a fourth section %d: %w",
				name, c.RopeSections[3], ErrNotImplemented)
		}
		for i := uint32(0); i < sum; i += 3 {
			if i >= 3*c.RopeSections[0] {
				return fmt.Errorf("convert: %s: interleaved rope sections %v leave pair %d to "+
					"a fourth axis in ggml and to t in transformers: %w", name, secs, i, ErrNotImplemented)
			}
		}
	}
	return nil
}

// projectorOf is the list of vision projectors (clip.projector_type) the
// converter implements: an unknown one is refused, as an unknown
// tokenizer.ggml.pre is. visionOf has a case for each.
var projectorOf = map[string]jlm.Projector{
	"idefics3":         jlm.ProjIdefics3,
	"mlp":              jlm.ProjMLP,
	"qwen2vl_merger":   jlm.ProjQwen2VL,
	"gemma3":           jlm.ProjGemma3,
	"internvl":         jlm.ProjInternVL,
	"resampler":        jlm.ProjResampler,
	"janus_pro":        jlm.ProjJanus,
	"qwen2.5vl_merger": jlm.ProjQwen25VL,
	"pixtral":          jlm.ProjPixtral,
	"phi4":             jlm.ProjPhi4,
	"gemma4v":          jlm.ProjGemma4V,
	"gemma3nv":         jlm.ProjGemma3nV,
	"llama4":           jlm.ProjLlama4,
	"glm4v":            jlm.ProjGLM4V,
	"hunyuanvl":        jlm.ProjHunyuanVL,
	"kimivl":           jlm.ProjKimiVL,
	"qwen3vl_merger":   jlm.ProjQwen3VL,
	"lfm2":             jlm.ProjLFM2,
}

// Projectors is every name in that list, sorted.
func Projectors() []string {
	out := make([]string, 0, len(projectorOf))
	for name := range projectorOf {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// ProjectorOf is the projector a clip.projector_type name converts to, and
// whether the name is in the list.
func ProjectorOf(name string) (jlm.Projector, bool) {
	p, ok := projectorOf[name]
	return p, ok
}

// visionOf reads a vision tower's description. A tower is its own container.
func visionOf(f *meta.File) (*jlm.Vision, error) {
	if f.Arch() != "clip" {
		return nil, nil
	}
	v := &jlm.Vision{
		NLayer:    uint32(f.UintKey("vision.block_count", 0)),
		NEmbd:     uint32(f.UintKey("vision.embedding_length", 0)),
		NHead:     uint32(f.UintKey("vision.attention.head_count", 0)),
		NFFN:      uint32(f.UintKey("vision.feed_forward_length", 0)),
		ImageSize: uint32(f.UintKey("vision.image_size", 0)),
		PatchSize: uint32(f.UintKey("vision.patch_size", 0)),
		Scale:     uint32(f.UintKey("vision.projector.scale_factor", 1)),
		Eps:       float32(f.FloatKey("vision.attention.layer_norm_epsilon", 1e-6)),
	}
	// The projector is a list and an unknown one is refused, as an unknown
	// tokenizer.ggml.pre is.
	name, _ := f.KV["clip.projector_type"].String()
	if name == "" {
		// A vision-and-audio mmproj (Gemma 4's) names each side's projector.
		name, _ = f.KV["clip.vision.projector_type"].String()
	}
	last := "mm.model.fc.weight"
	p, ok := projectorOf[name]
	if !ok {
		return nil, fmt.Errorf("convert: projector_type %q unsupported (only %s): %w",
			name, strings.Join(Projectors(), ", "), ErrNotImplemented)
	}
	v.Projector = p
	switch v.Projector {
	case jlm.ProjIdefics3:
	case jlm.ProjMLP:
		// llava's: linear, GELU, linear, and no pixel shuffle at all.
		last = "mm.2.weight"
	case jlm.ProjQwen2VL:
		// Qwen2-VL: the shuffle and the two-matrix MLP. The file does not state
		// the merge factor, so it is derived below from mm.0's k rather than
		// defaulted: a wrong one reshapes the sequence without faulting. The tower
		// has no v.position_embd: position is 2-D rotary over the patch grid.
		last = "mm.2.weight"
		v.Flags |= jlm.FlagVisionRope
	case jlm.ProjGemma3:
		// SigLIP, then average-pool Scale x Scale, RMSNorm and one matrix.
		// The pool is 4 when the file does not say (llama.cpp's default; only
		// a test model writes another). The matrix is stored [text, tower],
		// x @ W in the reference, and transposeGemma3Projection lays it out k
		// first; its row count before that is therefore the TEXT width.
		last = "mm.input_projection.weight"
		v.Scale = uint32(f.UintKey("vision.projector.scale_factor", 4))
	case jlm.ProjInternVL:
		// InternViT, the shuffle, then LayerNorm, linear, GELU, linear. The
		// image is cut into up to 12 tiles plus a thumbnail when the file does
		// not say (llama.cpp's defaults for an older conversion).
		last = "mm.model.mlp.3.weight"
		v.MinTiles = uint32(f.UintKey("vision.preproc_min_tiles", 1))
		v.MaxTiles = uint32(f.UintKey("vision.preproc_max_tiles", 12))
		if v.MinTiles == 0 || v.MinTiles > v.MaxTiles {
			return nil, fmt.Errorf("convert: clip: internvl tiles %d..%d", v.MinTiles, v.MaxTiles)
		}
		// InternViT-6B (InternVL 26B and up) is RMSNorm with a q/k norm, a
		// different block; refused by name rather than run as the 300M one.
		if _, ok := f.Get("v.blk.0.attn_q_norm.weight"); ok {
			return nil, fmt.Errorf("convert: clip: this InternViT has a q/k norm (the 6B tower); "+
				"only InternViT-300M is implemented: %w", ErrNotImplemented)
		}
	case jlm.ProjResampler:
		// MiniCPM-V's perceiver. Versions 3 to 6 (2.6, o-2.6, 4.0, 4.5) share
		// one graph and one token layout; 2 (Llama3-V 2.5) wraps its slices
		// differently and asks for 96 queries, so it is refused by name rather
		// than run with the wrong markers.
		ver := f.UintKey("minicpmv_version", 2)
		if ver < 3 || ver > 6 {
			return nil, fmt.Errorf("convert: resampler minicpmv_version %d is not implemented "+
				"(3 to 6 are: MiniCPM-V 2.6, o-2.6, 4.0, 4.5): %w", ver, ErrNotImplemented)
		}
		last = "resampler.proj.weight"
	case jlm.ProjJanus:
		// Janus-Pro's aligner is llava's two matrices with a GELU between, and
		// the file names the second one mm.1 (llama.cpp's clip.cpp reads
		// TN_LLAVA_PROJ 0 and 1); sourceOf renames it.
		last = "mm.1.weight"
	case jlm.ProjQwen25VL:
		// Qwen2.5-VL: Qwen2-VL's merger and 2-D rotary behind an RMSNorm,
		// gated-SiLU tower with window attention. The window is in pixels and
		// the file states only the pattern; llama.cpp's clip loader defaults the
		// size to 112 (transformers' window_size) when the key is absent, as
		// every published mmproj leaves it.
		last = "mm.2.weight"
		v.Flags |= jlm.FlagVisionRope
		v.WinPattern = uint32(f.UintKey("vision.n_wa_pattern", 0))
		v.WinSize = uint32(f.UintKey("vision.window_size", 112))
		if v.WinPattern == 0 {
			return nil, fmt.Errorf("convert: clip: qwen2.5vl_merger with no n_wa_pattern, so which " +
				"blocks attend inside windows is unknown")
		}
	case jlm.ProjPixtral:
		// Mistral's ViT and llava's MLP, with Mistral 3's patch merger when
		// the file states a merge size (Pixtral-12B states none: 1). The
		// picture is read at its own size, the longest side at most
		// image_size; the rotary's base is 10000, which llama.cpp's loader
		// sets for this projector and every published config states.
		last = "mm.2.weight"
		v.Flags |= jlm.FlagVisionRope
		v.Scale = uint32(f.UintKey("vision.spatial_merge_size", 1))
		if v.Scale == 0 {
			return nil, fmt.Errorf("convert: clip: pixtral states a spatial merge of 0")
		}
	case jlm.ProjPhi4:
		// Phi-4-reasoning-vision: the picture is read at its own size between
		// two pixel counts the file states; without them a picture has no
		// size. The converter already dropped the last block and the
		// post-LayerNorm (the reference reads hidden_states[-2]).
		last = "mm.2.weight"
		v.MinPixels = uint32(f.UintKey("vision.image_min_pixels", 0))
		v.MaxPixels = uint32(f.UintKey("vision.image_max_pixels", 0))
		if v.MinPixels == 0 || v.MaxPixels < v.MinPixels {
			return nil, fmt.Errorf("convert: clip: phi4 states image_min_pixels %d and "+
				"image_max_pixels %d; a picture read at its own size needs both", v.MinPixels, v.MaxPixels)
		}
	case jlm.ProjLFM2:
		// LFM2-VL's (and d1's): SigLIP2's NaFlex tower with its
		// post-LayerNorm, the pixel unshuffle the file states, an optional
		// LayerNorm and mm.1, GELU, mm.2. The tiling is Lfm2VlImageProcessor's,
		// which the file does not state and llama.cpp hardcodes: the class's
		// defaults, 64 to 256 tokens a piece, 2 to 10 tiles of 512 pixels
		// past twice the token budget. The resampling filter is the file's
		// image_resize_algo, bilinear (the class default) when absent.
		last = "mm.2.weight"
		v.Scale = uint32(f.UintKey("vision.projector.scale_factor", 2))
		if v.Scale == 0 || v.PatchSize == 0 || v.ImageSize%v.PatchSize != 0 {
			return nil, fmt.Errorf("convert: clip: lfm2 with a %d merge over %d-pixel patches of a %d table",
				v.Scale, v.PatchSize, v.ImageSize)
		}
		unit := v.PatchSize * v.Scale * v.PatchSize * v.Scale
		v.MinTiles, v.MaxTiles = 2, 10
		v.MinPixels, v.MaxPixels = 64*unit, 256*unit
		v.TileSize, v.Tolerance = 512, 2
		switch algo, _ := f.KV["clip.vision.image_resize_algo"].String(); algo {
		case "", "bilinear":
		case "bicubic":
			v.Flags |= jlm.FlagVisionBicubic
		default:
			return nil, fmt.Errorf("convert: clip: lfm2 resamples %q, which is not implemented: %w", algo, ErrNotImplemented)
		}
	case jlm.ProjGemma4V:
		// Gemma 4's tower. llama.cpp's graph scales the [0, 1] pixels to
		// 2x - 1 in place of a normalisation, which is mean 0.5 and std 0.5;
		// the MLP is gelu_pytorch_tanh, which the file does not state; the
		// pool is 3x3 unless the file says. The rotary's base (100) is the
		// loader's constant, as llama.cpp's is.
		last = "mm.input_projection.weight"
		v.Flags |= jlm.FlagVisionRope | jlm.FlagGELU
		v.Scale = uint32(f.UintKey("vision.projector.scale_factor", 3))
	case jlm.ProjGemma3nV:
		// Gemma 3n's MobileNet-V5: llama.cpp's converter writes a block count
		// of 128 and a patch of image_size/256 for its own loader, neither
		// the tower's. The blocks are the file's (stage, place) pairs, one
		// container block each, and a "patch" is the fusion adapter's output
		// cell, 16 to a side, so the tower's token count is its 256 soft
		// tokens. GELU is timm's tanh form; the pixels are [0, 1].
		last = "mm.input_projection.weight"
		v.Flags |= jlm.FlagGELU
		v.NLayer = uint32(len(mobilenetPlaces(f)))
		if v.NLayer == 0 || v.ImageSize%16 != 0 {
			return nil, fmt.Errorf("convert: clip: gemma3nv with %d blocks and a %d-pixel picture", v.NLayer, v.ImageSize)
		}
		v.PatchSize, v.Scale = v.ImageSize/16, 1
	case jlm.ProjLlama4:
		// Llama 4's ViT and its adapter: the pixel shuffle the file states,
		// two matrices each followed by a GELU, then the projector's matrix.
		// The file states neither tile bound; the processor's are one tile
		// and max_patches 16.
		last = "mm.model.fc.weight"
		v.Flags |= jlm.FlagVisionRope
		v.MinTiles, v.MaxTiles = 1, 16
	case jlm.ProjGLM4V:
		// GLM-4.1V / 4.5V / 4.6V: an RMSNorm, gated-SiLU tower with the 2-D
		// rotary and a learned table resampled BICUBIC, an RMSNorm after the
		// patch embedding, a 2x2 merging convolution, then linear, LayerNorm,
		// GELU and a gated MLP. See jlm.ProjGLM4V.
		last = "mm.down.weight"
		v.Flags |= jlm.FlagVisionRope
		mc, ok := f.Get("mm.patch_merger.weight")
		if !ok || len(mc.Dims) != 4 || mc.Dims[0] != mc.Dims[1] {
			return nil, fmt.Errorf("convert: clip: glm4v with no square mm.patch_merger.weight kernel")
		}
		v.Scale = uint32(mc.Dims[0])
		if _, ok := f.Get("v.position_embd.weight"); !ok {
			return nil, fmt.Errorf("convert: clip: glm4v with no v.position_embd")
		}
	case jlm.ProjHunyuanVL:
		// HunyuanVL's tower and merger. See jlm.ProjHunyuanVL. The picture is
		// read at its own size between the two pixel counts the file states,
		// to whole merge units; the table's class row is already gone
		// (llama.cpp's converter drops it).
		last = "mm.model.fc.weight"
		v.Scale = uint32(f.UintKey("vision.spatial_merge_size", 2))
		v.MinPixels = uint32(f.UintKey("vision.image_min_pixels", 0))
		v.MaxPixels = uint32(f.UintKey("vision.image_max_pixels", 0))
		if v.MinPixels == 0 || v.MaxPixels < v.MinPixels {
			return nil, fmt.Errorf("convert: clip: hunyuanvl states image_min_pixels %d and "+
				"image_max_pixels %d; a picture read at its own size needs both", v.MinPixels, v.MaxPixels)
		}
		if _, ok := f.Get("v.position_embd.weight"); !ok {
			return nil, fmt.Errorf("convert: clip: hunyuanvl with no v.position_embd")
		}
	case jlm.ProjKimiVL:
		// Kimi-VL's MoonViT and its projector. See jlm.ProjKimiVL. The file
		// states the 2x2 merge (projector.scale_factor) and names the
		// projector's matrices mm.1 and mm.2 (towerName).
		last = "mm.2.weight"
		v.Flags |= jlm.FlagVisionRope
		v.Scale = uint32(f.UintKey("vision.projector.scale_factor", 2))
		if _, ok := f.Get("v.position_embd.weight"); !ok {
			return nil, fmt.Errorf("convert: clip: kimivl with no v.position_embd")
		}
	case jlm.ProjQwen3VL:
		// Qwen3-VL: Qwen2-VL's merger and 2-D rotary behind a LayerNorm,
		// GELU-tanh tower that also adds a learned position table, resampled
		// to each picture's grid; and deepstack mergers on some blocks
		// (v.deepstack.N.*, which deepstackOf reads). See jlm.ProjQwen3VL.
		last = "mm.2.weight"
		v.Flags |= jlm.FlagVisionRope
		if _, ok := f.Get("v.position_embd.weight"); !ok {
			return nil, fmt.Errorf("convert: clip: qwen3vl_merger with no v.position_embd")
		}
	default:
		// A projector in projectorOf with no case here would convert with
		// another projector's tensor names; refuse it instead.
		return nil, fmt.Errorf("convert: projector_type %q has no conversion case", name)
	}
	// projection_dim is not the projector's output width: llava-phi-3 carries
	// CLIP's contrastive projection dim (768) there while mm.2 emits 3072. The
	// last projector matrix's row count is the fact.
	lt, ok := f.Get(last)
	if !ok || len(lt.Dims) < 2 {
		return nil, fmt.Errorf("convert: clip: %s is missing or not a matrix, so the "+
			"projector's output width cannot be derived", last)
	}
	v.ProjDim = uint32(lt.Dims[1])
	if v.Projector == jlm.ProjGemma3 {
		v.ProjDim = uint32(lt.Dims[0]) // stored transposed; see the case above
	}
	if v.Projector == jlm.ProjQwen2VL || v.Projector == jlm.ProjQwen25VL || v.Projector == jlm.ProjQwen3VL {
		// The merger consumes Scale*Scale patches at once, so mm.0's k is
		// NEmbd*Scale^2 (Qwen2-VL-2B: 5120 over 1280, a 2x2 merge).
		mm0, ok := f.Get("mm.0.weight")
		if !ok || len(mm0.Dims) < 2 {
			return nil, fmt.Errorf("convert: clip: qwen2vl_merger has no mm.0.weight matrix")
		}
		k := uint32(mm0.Dims[0])
		if k == 0 || v.NEmbd == 0 || k%v.NEmbd != 0 {
			return nil, fmt.Errorf("convert: clip: mm.0.weight k=%d is not a multiple of NEmbd=%d", k, v.NEmbd)
		}
		sq := k / v.NEmbd
		v.Scale = uint32(math.Sqrt(float64(sq)))
		if v.Scale*v.Scale != sq {
			return nil, fmt.Errorf("convert: clip: the merger takes %d patches at once, "+
				"which is not a square grouping", sq)
		}
	}
	// Three activations, matching llama.cpp's clip loader: GELU for use_gelu,
	// SiLU for use_silu, and quick-GELU when neither is set (openai CLIP,
	// llava-phi-3's mmproj). Defaulting to SiLU there would be fluent and wrong.
	gelu, _ := f.KV["clip.use_gelu"].Uint()
	silu, _ := f.KV["clip.use_silu"].Uint()
	switch {
	case gelu != 0 && silu != 0:
		return nil, fmt.Errorf("convert: clip: use_gelu and use_silu are both set")
	case gelu != 0:
		v.Flags |= jlm.FlagGELU
	case silu != 0:
		// SiLU is the zero value; naming the case keeps the table complete.
	default:
		v.Flags |= jlm.FlagQuickGELU
	}
	for i, key := range []string{"clip.vision.image_mean", "clip.vision.image_std"} {
		fs, err := f.KV[key].Float32s()
		if err != nil || len(fs) < 3 {
			return nil, fmt.Errorf("convert: %s must have at least 3 elements", key)
		}
		if i == 0 {
			copy(v.Mean[:], fs)
		} else {
			copy(v.Std[:], fs)
		}
	}
	if v.NLayer == 0 || v.NEmbd == 0 || v.NHead == 0 {
		return nil, fmt.Errorf("convert: clip: missing block_count/embedding_length/head_count")
	}
	// The FFN width comes from the matrix, not the key: mmproj-Qwen2-VL-2B
	// carries feed_forward_length 1536 (the text model's width) while its blocks
	// are 1280x5120. The width is whichever dimension is not NEmbd, since
	// Qwen2-VL's matrices are spelled the other way round.
	if ft, ok := f.Get("v.blk.0.ffn_up.weight"); ok && len(ft.Dims) >= 2 {
		a, b := uint32(ft.Dims[0]), uint32(ft.Dims[1])
		w := a
		if a == v.NEmbd {
			w = b
		}
		if w != 0 && w != v.NEmbd && w != v.NFFN {
			v.NFFN = w
		}
	}
	if v.NFFN == 0 {
		return nil, fmt.Errorf("convert: clip: no feed-forward width")
	}
	if v.Projector == jlm.ProjGemma4V {
		// The file's mean 0 and std 1 stand beside a graph that scales to
		// 2x - 1; the container states the normalisation that is, and the
		// activation the file leaves out (gelu_pytorch_tanh).
		v.Mean, v.Std = [3]float32{0.5, 0.5, 0.5}, [3]float32{0.5, 0.5, 0.5}
		v.Flags = v.Flags&^jlm.FlagQuickGELU | jlm.FlagGELU
	}
	if v.Projector == jlm.ProjGemma3nV {
		// timm's GELU is the tanh form, which the file does not state.
		v.Flags = v.Flags&^jlm.FlagQuickGELU | jlm.FlagGELU
	}
	return v, nil
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

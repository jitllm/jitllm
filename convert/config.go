package convert

import (
	"fmt"
	"maps"
	"math"
	"slices"
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

func archOf() map[string]Arch {
	return map[string]Arch{
		"llama":     ArchLlama, // llama2 and llama3 are this arch too
		"qwen2":     ArchQwen2,
		"qwen3":     ArchQwen3,
		"qwen3moe":  ArchQwen3MoE,
		"olmoe":     ArchOLMoE,
		"gemma":     ArchGemma,
		"phi3":      ArchPhi3, // phi4 uses this arch string too
		"clip":      ArchCLIP,
		"qwen3next": ArchQwen3Next,
		// Qwen3.5/3.6 are qwen3next's block re-cut, and every cut is in the file:
		// beta and alpha arrive as two tensors (fused back by fuseBetaAlpha), the
		// value heads are tiled (jlm.FlagDeltaKeyTiled), qwen35's FFN is dense, and
		// the interleaved M-RoPE is exactly NEOX for a text prompt. See qwen35Of.
		"qwen35":    ArchQwen3Next,
		"qwen35moe": ArchQwen3Next,
		// Clef-Flash: Qwen3.5-9B's text model under another name, with a joint
		// schema head after it that the decision readout reads (decisionOf,
		// jlm.DecisionClef).
		"clef":    ArchQwen3Next,
		"qwen2vl": ArchQwen2VL,
		"gemma2":  ArchGemma2,
		"gemma3":  ArchGemma3,
		"gpt-oss": ArchGPTOSS,
		// DeepSeek-V2/V3/R1, Kimi-K2 and Moonlight-16B-A3B. Not GLM: see
		// ArchDeepseek2 for which GLM is this graph and which is not.
		"deepseek2": ArchDeepseek2,
		// The BERT family of embedding models: all-minilm, mxbai-embed-large,
		// bge-large and the bert-sized snowflake-arctic-embed tags declare "bert";
		// nomic-embed-text declares "nomic-bert". See ArchBERT.
		"bert":       ArchBERT,
		"nomic-bert": ArchNomicBERT,
		// ModernBERT, and the decision encoders built on it (Laya). See
		// ArchModernBERT.
		"modern-bert": ArchModernBERT,
		// Llama 4 Scout (16 experts) and Maverick (128), text only. See
		// ArchLlama4 for the five things that make it not llama.
		"llama4": ArchLlama4,
		// IBM Granite 3.x/4.0, dense and mixture: llama's block with four scaling
		// constants. See ArchGranite. (granitehybrid, the Mamba-2 hybrid, is a
		// different graph and is not here.)
		"granite":    ArchGranite,
		"granitemoe": ArchGranite,
		// The classic-transformer blocks (C6): LayerNorm with biases, ungated
		// FFNs, parallel residuals. See ArchPhi2.
		"phi2":      ArchPhi2,
		"starcoder": ArchStarcoder,
		// Cohere: Command-R, Aya and Aya Expanse declare "command-r"; Command-R7B
		// and Command-A declare "cohere2". See ArchCommandR.
		"command-r":  ArchCommandR,
		"cohere2":    ArchCommandR,
		"stablelm":   ArchStableLM,
		"starcoder2": ArchStarcoder2,
		"falcon":     ArchFalcon,
		"nemotron":   ArchNemotron,
		// Databricks' DBRX: the C6 kit's bias-free LayerNorm on a mixture, with
		// q, k and v clamped. See ArchDBRX.
		"dbrx": ArchDBRX,
		// GLM-4.5, GLM-4.6 and GLM-4.5-Air: DeepSeek-V3's router on GQA attention.
		// See ArchGLM4MoE.
		"glm4moe": ArchGLM4MoE,
		// Qwen1.5-MoE and Qwen2-57B-A14B. See ArchQwen2MoE.
		"qwen2moe": ArchQwen2MoE,
		// ERNIE 4.5: the dense models are llama's graph exactly (interleaved
		// rotary, tied head, biases only as tensors), as llama3 is; the mixtures
		// are their own. See ArchErnie45MoE.
		"ernie4_5":     ArchLlama,
		"ernie4_5-moe": ArchErnie45MoE,
		// The dense llama-family group: each is llama's block with switches the
		// graph already runs. See convert/dense.go and ArchSmolLM3.
		"smollm3":  ArchSmolLM3,
		"arcee":    ArchArcee,
		"seed_oss": ArchSeedOSS,
		"olmo2":    ArchOLMo2, // OLMo 3 declares olmo2 too
		"exaone4":  ArchExaone4,
		"mistral3": ArchMistral3,
		// Tencent's Hunyuan, mixture and dense: one graph, q/k norm after the
		// rotary. See ArchHunyuan.
		"hunyuan-moe":   ArchHunyuan,
		"hunyuan-dense": ArchHunyuan,
		// MiniMax-M2/M2.1/M2.5: OLMoE's whole-projection q/k norm on GQA with
		// partial NEOX rotary, and DeepSeek-V3's sigmoid router without a shared
		// expert. See ArchMiniMaxM2.
		"minimax-m2": ArchMiniMaxM2,
		// Gemma 4 (text): gemma3's block with its global layers at another
		// attention geometry. See ArchGemma4.
		"gemma4": ArchGemma4,
		// DeepSeek V3.2: deepseek2's graph with DeepSeek Sparse Attention's
		// lightning indexer. See ArchDeepseek32.
		"deepseek32": ArchDeepseek32,
		// Gemma 3n (text): gemma3's attention with Gemma 4's E-model pieces,
		// inside AltUp's parallel residual streams. See ArchGemma3n.
		"gemma3n": ArchGemma3n,
		// The state-space hybrids: Mamba-2 blocks (jlm.LayerSSD), alone or beside
		// attention. See convert/ssm.go and ArchMamba2.
		"mamba2":        ArchMamba2,
		"granitehybrid": ArchGraniteHybrid,
		// Nemotron-H and Nemotron Nano 2: one mixer a source layer, merged into
		// blocks at conversion (convert/nemotronh.go).
		"nemotron_h": ArchNemotronH,
		// Nemotron 3 (Nano): the same blocks with the MLP a mixture of ungated
		// experts. See nemotronConfig.
		"nemotron_h_moe": ArchNemotronH,
		// Mamba-1: plain Mamba and FalconMamba (one arch, the latter's dt/B/C
		// norms written as ones), and Jamba beside attention. See convert/mamba1.go.
		"mamba": ArchMamba,
		"jamba": ArchJamba,
		// Falcon-H1: attention and Mamba-2 side by side in every block.
		"falcon-h1": ArchFalconH1,
		// LFM2: gated short convolutions beside attention (convert/lfm2.go).
		"lfm2": ArchLFM2,
		// LFM2's mixture (LFM2-8B-A1B, LFM2-24B-A2B): the same blocks with a
		// sigmoid top-k after a dense lead, read off the keys as for any mixture.
		"lfm2moe": ArchLFM2,
		// MiniMax-M3 (text): MiniMax-M2's attention with DeepSeek-V3's mixture
		// and MiniMax Sparse Attention. See ArchMiniMaxM3.
		"minimax-m3": ArchMiniMaxM3,
		// The modern text group: Ling 2.0, dots.llm1, Phi-3.5-MoE and Apertus.
		// See convert/moefamily2.go, convert/apertus.go and ArchBailingMoE2.
		"bailingmoe2": ArchBailingMoE2,
		"dots1":       ArchDots1,
		"phimoe":      ArchPhiMoE,
		"apertus":     ArchApertus,
		// BitNet b1.58 2B-4T as bitnet.cpp's converter names it (convert/bitnet.go,
		// ArchBitNet). llama.cpp's own "bitnet" is 1bitLLM's reproduction, a
		// SiLU gate, and its converter cannot write transformers' BitNetForCausalLM
		// at all (it maps none of its inner-norm names); it is not this graph.
		"bitnet-b1.58": ArchBitNet,
		// DeepSeek V4 (text): hyper-connected residual streams, single-key MQA
		// over a sliding window and a compressed history. See ArchDeepseek4.
		"deepseek4": ArchDeepseek4,

		// Qwen3-VL's text models: qwen3 and qwen3moe under an interleaved M-RoPE.
		// See ArchQwen3VL.
		"qwen3vl":    ArchQwen3VL,
		"qwen3vlmoe": ArchQwen3VLMoE,
		// GLM-4-0414 and the text model of GLM-4.1V / 4.6V-Flash. See ArchGLM4.
		// GLM-4.5V's and 4.6V's is glm4moe with M-RoPE sections, which configOf
		// carries as ArchGLM4VMoE.
		"glm4": ArchGLM4,
		// HunyuanVL's text model (HunyuanOCR): hunyuan-dense's block under
		// XD-RoPE. See ArchHunyuanVL.
		"hunyuan_vl": ArchHunyuanVL,
		// Kimi-K3 (text): Kimi-Linear's KDA + MLA hybrid with residual attention,
		// a latent mixture and the situ activation. See ArchKimiK3.
		"kimi-k3": ArchKimiK3,
		// Qwen3.8-Flash-Next (text): qwen3next's hybrid inside low-rank gated
		// hyper-connections, with QSA block selection and an n-gram embedding.
		// See ArchQwen4Exp.
		"qwen4exp": ArchQwen4Exp,
	}
}

// Supported reports whether the converter implements a GGUF architecture
// name -- the list above, which is the list (RULE 7).
func Supported(arch string) bool {
	_, ok := archOf()[arch]
	return ok
}

// Architectures is every name in that list, sorted.
//
// It exists so the list can be checked against, not just consulted:
// library.TestEveryArchitectureIsObtainable reads it to confirm everything the
// converter implements is reachable from the download catalogue.
func Architectures() []string {
	out := make([]string, 0, len(archOf()))
	for name := range archOf() {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// ArchitectureGraph is the graph a GGUF architecture name converts to, and
// whether the name is in the list. Several names share one graph. The file can
// refine it further (olmo2 with a sliding window is ArchOLMo3, glm4moe
// with M-RoPE sections ArchGLM4VMoE); this is the name's own entry.
func ArchitectureGraph(name string) (Arch, bool) {
	a, ok := archOf()[name]
	return a, ok
}

func configOf(f *meta.File) (*Config, error) {
	var c *Config
	var err error
	if head, ok := d1OmniHead(f); ok {
		c, err = d1OmniConfig(f, head)
	} else {
		c, err = configOfFile(f)
	}
	if err != nil {
		return nil, err
	}
	return c, localAxes(c)
}

// localAxes refuses rotary sections under a second rotary, the sliding
// layers' own base or a block's own width: a table over several position
// axes on the local layers, which no architecture in archOf or hfArchTable
// states.
func localAxes(c *Config) error {
	second := c.RopeBaseSWA != 0
	for b := range c.Geoms {
		second = second || c.Geoms[b].NRot != c.NRot
	}
	if c.RopeSections == [4]uint32{} || !second {
		return nil
	}
	return fmt.Errorf("convert: %s: rotary sections %v on the sliding layers' own rotary: %w",
		c.Arch, c.RopeSections, ErrNotImplemented)
}

// configOfFile is configOf for every file but the ones read through a view
// (d1OmniConfig).
func configOfFile(f *meta.File) (*Config, error) {
	name := f.Arch()
	arch, ok := archOf()[name]
	if !ok {
		return nil, fmt.Errorf("convert: architecture %q is not implemented (%s): %w",
			name, strings.Join(Architectures(), ", "), ErrNotImplemented)
	}
	// A state-space hybrid writes its head counts per layer with zeros on the
	// recurrent layers, and a model with no attention at all writes zero:
	// layerCount reads the attention layers' count and lets the all-recurrent
	// one through (ssdConfig fills a placeholder geometry).
	count := perLayerCount
	if ssdArch(arch) || arch == ArchLFM2 || mamba1Arch(arch) || arch == ArchKimiK3 {
		count = attnLayerCount
	}
	nhead, err := count(f, "attention.head_count", 0)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	c := &Config{Arch: arch, Config: &jlm.Config{
		NLayer:     uint32(f.UintKey("block_count", 0)),
		NEmbd:      uint32(f.UintKey("embedding_length", 0)),
		NHead:      nhead,
		NFFN:       uint32(f.UintKey("feed_forward_length", 0)),
		NCtx:       uint32(f.UintKey("context_length", 2048)),
		RMSEps:     float32(f.FloatKey("attention.layer_norm_rms_epsilon", 1e-5)),
		RopeBase:   float32(f.FloatKey("rope.freq_base", 10000)),
		EmbdScale:  1,
		AttnFactor: float32(f.FloatKey("rope.scaling.attn_factor", 1)),
	}}
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
	if arch != ArchGemma4 {
		if c.NKVHead, err = count(f, "attention.head_count_kv", c.NHead); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if c.NKVHead == 0 && arch != ArchGemma4 {
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
	// attention.scale is the score multiplier in place of 1/sqrt(head), for
	// every graph whose file states one; a graph that fixes its own sets it
	// below.
	if err := attnScaleKey(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}

	// What the library's file states for the architecture's container, over
	// the keys (mjStates).
	if err := stateContainer(f, c); err != nil {
		return nil, err
	}
	switch arch {
	case ArchDeepseek2, ArchDeepseek32:
		// The two latent ranks. The explicit key is the authority; value_length is
		// the fallback because on this arch it is kv_lora_rank (the absorbed row,
		// see above). The safetensors path reads these from config.json directly, so
		// only a real GGUF exercises this.
		c.KVLoraRank = uint32(f.UintKey("attention.kv_lora_rank",
			f.UintKey("attention.value_length", 0)))
		// Absent on V2-Lite, whose q_lora_rank is null: it projects the query
		// straight to NHead*HeadDim, and model.mlaProject asks the config.
		c.QLoraRank = uint32(f.UintKey("attention.q_lora_rank", 0))
		if arch == ArchDeepseek32 {
			if err := deepseek32Config(f, c); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
		}
	case ArchPhi3:
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
	case ArchGemma2, ArchGemma3:
		gemmaAttnScale(c)
	case ArchQwen3Next, ArchQwen4Exp:
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
		// The sections select nothing for a text prompt and are stored for
		// image rows, read as every M-RoPE's are; a text-only file states none.
		if _, ok := f.Key("rope.dimension_sections"); ok {
			if err := mropeSectionsOf(f, name, c); err != nil {
				return nil, err
			}
		}
		if arch == ArchQwen4Exp {
			if err := qwen4expConfig(f, c); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
		}
	case ArchQwen2VL, ArchQwen3VL, ArchQwen3VLMoE:
		// qwen2's, qwen3's and qwen3moe's blocks under a multimodal rope: the
		// sections split each head's rotary pairs between the (t, h, w)
		// components of a position (Qwen2-VL-2B is [16, 24, 24, 0], summing to
		// NRot/2 = 64; Qwen3-VL-2B's INTERLEAVED [24, 20, 20, 0] of 64 pairs).
		// See mropeSectionsOf.
		if err := mropeSectionsOf(f, name, c); err != nil {
			return nil, err
		}
	case ArchLlama4:
		if err := llama4Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchGranite:
		if err := graniteConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchStarcoder:
		if err := starcoderConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchStableLM:
		if err := stableLMConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchDBRX:
		if err := dbrxConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchGLM4MoE:
		if err := glm4moeConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		// GLM-4.5V / 4.6V: the same block under Qwen2-VL's M-RoPE, a code of
		// its own (ArchGLM4VMoE).
		if _, ok := f.Key("rope.dimension_sections"); ok {
			c.Arch = ArchGLM4VMoE
			if err := mropeSectionsOf(f, name, c); err != nil {
				return nil, err
			}
		}
	case ArchGLM4:
		// llama's block with biased q/k/v (tensors), a norm after the
		// attention and after the MLP (RolePostAttnNorm, RolePostFFNNorm), a
		// fused gate|up (unfuse), and a rotary on rope.dimension_count of each
		// head: NORM pairs, or with sections the NEOX M-RoPE llama.cpp's
		// converter permutes q and k into.
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
	case ArchQwen2MoE:
		if err := qwen2moeConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchErnie45MoE:
		if err := ernie45moeConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchSmolLM3, ArchArcee, ArchSeedOSS, ArchOLMo2, ArchExaone4, ArchMistral3, ArchOLMo3:
		if err := denseConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchHunyuan:
		if err := hunyuanConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchHunyuanVL:
		if err := hunyuanVLConfig(f, name, c); err != nil {
			return nil, err
		}
	case ArchMiniMaxM2:
		if err := minimaxM2Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchGemma4:
		if err := gemma4Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchGemma3n:
		if err := gemma3nConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchMamba2, ArchGraniteHybrid, ArchNemotronH, ArchFalconH1:
		if err := ssdConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchMamba, ArchJamba:
		if err := mamba1Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		if c.LayerKinds, err = mamba1Kinds(f, c.NLayer); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchLFM2:
		if err := lfm2Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchMiniMaxM3:
		if err := minimaxM3Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchBailingMoE2:
		if err := bailingmoe2Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchDots1:
		if err := dots1Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchPhiMoE:
		if err := phimoeConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchApertus:
		if err := apertusConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchDeepseek4:
		if err := deepseek4Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchKimiK3:
		if err := kimiK3Config(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	case ArchCLIP:
		return nil, fmt.Errorf("convert: %q is a vision tower, not a language model", name)
	case ArchBERT, ArchNomicBERT:
		// The encoder's norms are LayerNorms under attention.layer_norm_epsilon
		// (all-minilm's is 1e-12); layer_norm_rms_epsilon read above is absent.
		c.RMSEps = float32(f.FloatKey("attention.layer_norm_epsilon", 1e-12))
		// An encoder has no causal mask: every token attends to every token.
		c.Flags |= jlm.FlagNonCausal | jlm.FlagLayerNorm
		if arch == ArchNomicBERT {
			// Rotary on q and k in place of the position table, NEOX-paired
			// (llama.cpp's rope-type table groups nomic-bert with qwen2). The
			// base comes off rope.freq_base -- 1000 on nomic-embed-text-v1.5.
			c.Flags |= jlm.FlagRopeNeox
			if n := f.UintKey("moe_every_n_layers", 0); n > 0 {
				return nil, fmt.Errorf("convert: %s: a mixture every %d layers "+
					"(nomic-bert-moe) is not implemented: %w", name, n, ErrNotImplemented)
			}
		} else {
			// A learned position table in the input rows and no rotary.
			c.Flags |= jlm.FlagGELU | jlm.FlagNoPosEnc
		}
	case ArchModernBERT:
		if err := modernBERTConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}

	// A per-head q/k norm under LayerNorm blocks (command-r-plus, StableLM 2
	// 12B: a weight per head) is refused by name: running without it would be
	// fluent and wrong.
	if c.Flags.Has(jlm.FlagLayerNorm) {
		if _, ok := f.Get("blk.0.attn_q_norm.weight"); ok {
			return nil, fmt.Errorf("convert: %s: a per-head q/k LayerNorm: %w", name, ErrNotImplemented)
		}
	}
	// A rotary narrower than the head turns its leading pairs and passes the
	// rest through, whatever the architecture.
	if c.NRot != c.HeadDim {
		if err := partialRotary(c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}

	// A clamp on any other architecture is refused, not dropped: this converter
	// writes it only for ArchDBRX, whose code an older reader refuses.
	if arch != ArchDBRX {
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
	if (c.Arch == ArchQwen3Next || c.Arch == ArchQwen4Exp || ssdArch(c.Arch) || c.Arch == ArchLFM2) && c.LayerKinds == nil {
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
		// A stated expert_weights_norm decides over the graph's default, for
		// every graph: olmoe's case leaves the weights as they are, and a file
		// that says to renormalise (OLMoE's norm_topk_prob) is obeyed.
		// meta.Value.Int decodes Bool, so Uint is the accessor for a GGUF bool
		// and 0 is false.
		if kv, ok := f.Key("expert_weights_norm"); ok {
			switch v, ok := kv.Uint(); {
			case ok && v == 0:
				c.Flags |= jlm.FlagNoExpertNorm
			case ok:
				c.Flags &^= jlm.FlagNoExpertNorm
			}
		}
		// An absent expert_weights_norm is not one answer for every arch: llama.cpp's
		// qwen3moe builder passes true as a literal, its deepseek2 builder reads the
		// key with a default of false, and DeepSeek-V2-Lite's config.json says
		// norm_topk_prob: false. Renormalising it generates fluent nonsense from the
		// first token. See docs/engineering-history/model-correctness.md.
		if arch == ArchDeepseek2 || arch == ArchDeepseek32 || arch == ArchQwen2MoE {
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
				if arch != ArchDeepseek4 {
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
		if arch == ArchDeepseek32 {
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
		if arch == ArchKimiK3 && c.NExpertGroup == 0 && c.NExpertGroupUsed == 1 {
			c.NExpertGroupUsed = 0
		}
		if (c.NExpertGroup == 0) != (c.NExpertGroupUsed == 0) {
			return nil, fmt.Errorf("convert: %s: expert_group_count=%d with expert_group_used_count=%d",
				name, c.NExpertGroup, c.NExpertGroupUsed)
		}
		// A grouped selection picks whole groups of an even split.
		if c.NExpertGroup > 1 && (c.NExpert%c.NExpertGroup != 0 || c.NExpertGroupUsed > c.NExpertGroup) {
			return nil, fmt.Errorf("convert: %s: %d experts in %d groups, %d used", name, c.NExpert,
				c.NExpertGroup, c.NExpertGroupUsed)
		}
	}

	// The leading dense run is read for every architecture, as a property of the
	// file. Llama 4's shared expert is as wide as a routed one and the file does
	// not say so (llama.cpp sets n_ff_shexp = n_ff_exp), which needs NFFNExp.
	if arch == ArchLlama4 && c.NFFNShExp == 0 {
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
	if arch == ArchNemotronH {
		if err := nemotronMixture(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if err := sharedExpertWidth(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if arch == ArchGemma4 {
		if err := gemma4MoEConfig(f, c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if arch == ArchMiniMaxM3 {
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
func llama4Config(f *meta.File, c *Config) error {
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
	c.Flags |= jlm.FlagExpertSigmoid | jlm.FlagExpertWeightIn
	c.MoEStep = uint32(f.UintKey("interleave_moe_layer_step", 1))
	return neverRenormalises(f, c, "Llama 4's router")
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
func ropeScalingOf(f *meta.File, c *Config) error {
	// The context the rotary was trained to: YaRN scales from it, and
	// LongRoPE turns its long factors past it (model.RopeFor).
	if v := f.UintKey("rope.scaling.original_context_length", 0); v > 0 {
		c.YarnOrigCtx = uint32(v)
	}
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
	case "longrope", "su":
		// The factors are tensors: the file's own, or the publisher's
		// config.json's where the file dropped them (applyModelConfig).
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
		if c.Arch == ArchMistral3 {
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
		case c.Arch == ArchDeepseek4:
			// DeepSeek V4 scales neither: transformers forces the compressed
			// rotary's attention_factor to 1 and scores at head_dim^-1/2, and
			// llama.cpp's builder passes an attn_factor that cancels ggml's
			// own magnitude and a kq_scale of 1/sqrt(head_dim).
		case c.Arch != ArchDeepseek2 && c.Arch != ArchDeepseek32:
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
// An interleaved split (interleavedMRope) is read as transformers
// reads it: pair i turns by h when i%3 == 1 and i < 3*sections[1], by w when
// i%3 == 2 and i < 3*sections[2], and by t otherwise. ggml's IMROPE sends a
// pair with i%3 == 0 at or past 3*sections[0] to a fourth axis instead; no
// published split reaches one, and one that would is refused rather than
// resolved, since the two references would then rotate it differently.
func mropeSectionsOf(f *meta.File, name string, c *Config) error {
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
	if interleavedMRope(c.Arch) {
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
	lib, err := projectorLibrary()
	if err != nil {
		return nil, err
	}
	p, ok := lib[name]
	if !ok {
		return nil, fmt.Errorf("convert: projector_type %q unsupported (only %s): %w",
			name, strings.Join(slices.Sorted(maps.Keys(lib)), ", "), ErrNotImplemented)
	}
	if err := p.state(name, f, v); err != nil {
		return nil, err
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
	p.restate(v)
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

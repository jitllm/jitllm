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
		// the interleaved M-RoPE is exactly NEOX for a text prompt (convert/models/qwen3_5.json).
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
		// experts. See mergeMixers.
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

// Architectures is every GGUF architecture name the library's files name
// (convert/models), and the vision tower's (towerArch), sorted
// (TestArchitecturesAreTheLibrarys).
//
// It exists so the list can be checked against, not just consulted:
// library.TestEveryArchitectureIsObtainable reads it to confirm everything the
// converter implements is reachable from the download catalogue.
func Architectures() ([]string, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	out := []string{towerArch}
	for _, mj := range files {
		for _, n := range mj.GGUF {
			if !slices.Contains(out, n) {
				out = append(out, n)
			}
		}
	}
	sort.Strings(out)
	return out, nil
}

// ArchitectureGraph is the graph a GGUF architecture name converts to, and
// whether the name is in the list. Several names share one graph. The file can
// refine it further (glm4moe with M-RoPE sections is ArchGLM4VMoE); this is
// the name's own entry.
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
	if name == towerArch {
		return nil, fmt.Errorf("convert: %q is a vision tower, not a language model", name)
	}
	arch, ok := archOf()[name]
	if !ok {
		names, err := Architectures()
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("convert: architecture %q is not implemented (%s): %w",
			name, strings.Join(names, ", "), ErrNotImplemented)
	}
	nhead, err := uniformCount(f, "attention.head_count", 0)
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
	// A file stating no heads on any layer attends nowhere (plain Mamba): one
	// placeholder head as wide as the embedding keeps every attention-shaped
	// derivation finite, and no layer reads it (countsHeld refuses one that
	// does).
	if _, stated := f.Key("attention.head_count"); stated && c.NHead == 0 {
		c.NHead = 1
	}
	if c.NLayer == 0 || c.NEmbd == 0 || c.NHead == 0 {
		return nil, fmt.Errorf("convert: %s: missing block_count/embedding_length/head_count", name)
	}
	// Absent head_count_kv means MHA, not zero heads. One that differs by
	// layer is left for the blocks' own geometries to hold (countsHeld).
	nkv, perKV, err := layerCount(f, "attention.head_count_kv", c.NHead)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if c.NKVHead = nkv; perKV == nil && c.NKVHead == 0 {
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

	// A head narrower than embedding/heads is stated only by key_length; a
	// file that states none where its q projection says otherwise is refused
	// (llama.cpp refuses to load it too), and its safetensors input, which
	// reads head_dim, converts the model. A gated q projection carries the
	// gate's rows beside the heads'.
	if _, ok := f.Key("attention.key_length"); !ok && !c.Flags.Has(jlm.FlagAttnOutGate) {
		if q, ok := f.Layer(0, "attn_q.weight"); ok && len(q.Dims) == 2 &&
			q.Dims[1] != uint64(c.NHead)*uint64(c.HeadDim) {
			return nil, fmt.Errorf("convert: %s: blk.0.attn_q.weight has %d rows for %d heads of %d: the file "+
				"states no attention.key_length for its head width; convert the model from its safetensors: %w",
				name, q.Dims[1], c.NHead, c.HeadDim, ErrBadInput)
		}
	}
	// A mixture on every n'th block, the rest dense, is no graph this
	// converter states.
	if n := f.UintKey("moe_every_n_layers", 0); n > 0 {
		return nil, fmt.Errorf("convert: %s: a mixture every %d layers: %w", name, n, ErrNotImplemented)
	}
	// A learned position table is added to the input rows: as wide as they
	// are, and bounding the context, since a position past its rows has no
	// embedding.
	if t, ok := f.Get("position_embd.weight"); ok {
		if len(t.Dims) < 2 || uint32(t.Dims[0]) != c.NEmbd {
			return nil, fmt.Errorf("convert: %s: position_embd is %v, the model %d wide", name, t.Dims, c.NEmbd)
		}
		c.NCtx = min(c.NCtx, uint32(t.Dims[1]))
	}

	// A per-head q/k norm under LayerNorm blocks (command-r-plus, StableLM 2
	// 12B: a weight per head) is refused by name: running without it would be
	// fluent and wrong.
	if c.Flags.Has(jlm.FlagLayerNorm) {
		if _, ok := f.Get("blk.0.attn_q_norm.weight"); ok {
			return nil, fmt.Errorf("convert: %s: a per-head q/k LayerNorm: %w", name, ErrNotImplemented)
		}
	}
	if err := statedFeaturesFit(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := stateKindGeoms(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := sparsityOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	// A rotary narrower than the head turns its leading pairs and passes the
	// rest through, whatever the architecture.
	if c.NRot != c.HeadDim {
		if err := partialRotary(c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}

	// A clamp on q, k and v is read for every architecture whose file states
	// one; zero is none, and a bound that is not positive and finite clamps
	// nothing sensibly.
	if kv, ok := f.Key("attention.clamp_kqv"); ok {
		v, _ := kv.Float()
		if !(v >= 0) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("convert: %s: attention.clamp_kqv %g", name, v)
		}
		c.ClampKQV = float32(v)
	}
	// A per-head q/k norm normalises both or neither.
	_, q := f.Get("blk.0.attn_q_norm.weight")
	if _, k := f.Get("blk.0.attn_k_norm.weight"); q != k {
		return nil, fmt.Errorf("convert: %s: blk.0 carries a q norm %v and a k norm %v", name, q, k)
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

	// The per-layer kinds are read off the tensors each layer carries, as the
	// library states them, not off an "every fourth layer" rule.
	if err := stateLayers(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := selectiveNormsFit(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := countsHeld(c, f); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
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
		// norm_topk_prob: false, so latent.json states no_expert_norm where the
		// key is absent. Renormalising it generates fluent nonsense from the first
		// token. See docs/engineering-history/model-correctness.md.
		// 1 is softmax, 2 is sigmoid; anything else is refused rather than
		// approximated, because a wrong gate is a fluent model with another mixture.
		if kv, ok := f.Key("expert_gating_func"); ok {
			switch v, _ := kv.Uint(); v {
			case 0, 1:
			case 2:
				c.Flags |= jlm.FlagExpertSigmoid
			case 4:
				// sqrt(softplus), where the graph states that router (DeepSeek V4's).
				if !c.Flags2.Has(jlm.Flag2ExpertSqrtSoftplus) {
					return nil, fmt.Errorf("convert: %s: expert_gating_func 4 (sqrt-softplus) where the "+
						"graph states no sqrt-softplus router: %w", name, ErrNotImplemented)
				}
			default:
				return nil, fmt.Errorf("convert: %s: expert_gating_func %d is not implemented: %w",
					name, v, ErrNotImplemented)
			}
		}
		// Grouped selection. Both keys or neither; one alone cannot be acted on.
		c.NExpertGroup = uint32(f.UintKey("expert_group_count", 0))
		c.NExpertGroupUsed = uint32(f.UintKey("expert_group_used_count", 0))
		// One group, all of it used, is the ungrouped selection: llama.cpp's
		// kimi-k3 converter writes topk_group (1) as the used count and no group
		// count beside it.
		if c.NExpertGroup == 0 && c.NExpertGroupUsed == 1 {
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
	// file.
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
	if err := statedMixtureFits(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := sharedExpertWidth(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := denseMoEFits(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if kv, ok := f.Key("rope.scaling.finetuned"); ok && !c.Flags.Has(jlm.FlagNoPosEnc) {
		if v, ok := kv.Uint(); ok && v == 0 {
			return nil, fmt.Errorf("convert: %s: rope.scaling.finetuned false turns the rotary off, which the "+
				"architecture's graph does not state: %w", name, ErrNotImplemented)
		}
	}
	if err := decisionOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	return c, nil
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
		if c.Stated.yarnMagnitude == "mscale_ratio" {
			if haveExplicit {
				c.AttnFactor *= float32(explicit)
				return nil
			}
			yarnMscaleRatio(f, c, float64(factor))
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
		//
		// DeepSeek V4 scales neither: transformers forces the compressed
		// rotary's attention_factor to 1 and scores at head_dim^-1/2, and
		// llama.cpp's builder passes an attn_factor that cancels ggml's own
		// magnitude and a kq_scale of 1/sqrt(head_dim). Each states
		// yarn_magnitude softmax.
		switch {
		case haveExplicit:
			c.AttnFactor *= float32(explicit)
		case c.Stated.yarnMagnitude != "softmax":
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

// ropeSectionsFit sets an M-RoPE model's split secs of a head's rotary pairs
// between the (t, h, w) coordinates of a position in c (the rope_sections
// statement). The sum is checked because a short list leaves a tail of each
// head unrotated, which is fluent and which no self-comparing tier can see.
//
// An interleaved split (rope_layout "interleaved") is read as transformers
// reads it: pair i turns by h when i%3 == 1 and i < 3*sections[1], by w when
// i%3 == 2 and i < 3*sections[2], and by t otherwise. ggml's IMROPE sends a
// pair with i%3 == 0 at or past 3*sections[0] to a fourth axis instead; no
// published split reaches one, and one that would is refused rather than
// resolved, since the two references would then rotate it differently.
func ropeSectionsFit(name string, secs []int32, c *Config) error {
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
	if c.Stated.ropeLayout == "interleaved" {
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
func visionOf(f *meta.File) (*jlm.Vision, *projStates, error) {
	if f.Arch() != towerArch {
		return nil, nil, nil
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
		return nil, nil, err
	}
	p, ok := lib[name]
	if !ok {
		return nil, nil, fmt.Errorf("convert: projector_type %q unsupported (only %s): %w",
			name, strings.Join(slices.Sorted(maps.Keys(lib)), ", "), ErrNotImplemented)
	}
	if err := p.state(name, f, v); err != nil {
		return nil, nil, err
	}
	// Three activations, matching llama.cpp's clip loader: GELU for use_gelu,
	// SiLU for use_silu, and quick-GELU when neither is set (openai CLIP,
	// llava-phi-3's mmproj). Defaulting to SiLU there would be fluent and wrong.
	gelu, _ := f.KV["clip.use_gelu"].Uint()
	silu, _ := f.KV["clip.use_silu"].Uint()
	switch {
	case gelu != 0 && silu != 0:
		return nil, nil, fmt.Errorf("convert: clip: use_gelu and use_silu are both set")
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
			return nil, nil, fmt.Errorf("convert: %s must have at least 3 elements", key)
		}
		if i == 0 {
			copy(v.Mean[:], fs)
		} else {
			copy(v.Std[:], fs)
		}
	}
	if v.NLayer == 0 || v.NEmbd == 0 || v.NHead == 0 {
		return nil, nil, fmt.Errorf("convert: clip: missing block_count/embedding_length/head_count")
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
		return nil, nil, fmt.Errorf("convert: clip: no feed-forward width")
	}
	p.restate(v)
	return v, p, nil
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

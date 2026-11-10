package convert

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// kimiK3HF is Kimi-K3 (KimiK3ForConditionalGeneration) from safetensors, to
// the container its GGUF converts to (convert/kimik3.go is that half). The
// checkpoint is Moonshot's: the text model is KimiLinearForCausalLM under
// language_model., its hyperparameters under text_config, and the released
// weights carry the routed experts as compressed-tensors MXFP4
// (convert/hfmxfp4.go), which map onto the container's MXFP4 byte for byte.
//
// The vision tower (vision_tower.*, a MoonViT under mm_projector.'s
// patchmergerv2) is not converted: its tensors are named and skipped, as
// llama.cpp's converter skips them, and the container is the text model.
//
// The text model is Kimi-Linear's names with K3's additions: the residual
// attention's norm and projection per sublayer (gathered into their product,
// kimiK3Gather), the latent mixture's routed_expert_* projections and norm,
// g_proj (the MLA output gate in a full block, the full-rank KDA gate in a
// linear one), and the situ activation, a config key.
func kimiK3HF() *hfArch {
	return &hfArch{
		arch:        ArchKimiK3,
		blockPrefix: k3BlockPrefix,
		model: map[string]jlm.Role{
			"language_model.model.embed_tokens.weight": jlm.RoleTokenEmbd,
			"language_model.model.norm.weight":         jlm.RoleOutputNorm,
			"language_model.lm_head.weight":            jlm.RoleOutput,
		},
		ignorePrefix: []string{"vision_tower.", "mm_projector."},
		block: map[string]jlm.Role{
			"input_layernorm.weight":          jlm.RoleAttnNorm,
			"post_attention_layernorm.weight": jlm.RoleFFNNorm,

			// MLA with no positional encoding; q_proj is the one-step query a
			// null q_lora_rank states.
			"self_attn.q_proj.weight":             jlm.RoleAttnQ,
			"self_attn.q_a_proj.weight":           jlm.RoleAttnQA,
			"self_attn.q_a_layernorm.weight":      jlm.RoleAttnQANorm,
			"self_attn.q_b_proj.weight":           jlm.RoleAttnQB,
			"self_attn.kv_a_proj_with_mqa.weight": jlm.RoleAttnKVA,
			"self_attn.kv_a_layernorm.weight":     jlm.RoleAttnKVANorm,
			"self_attn.kv_b_proj.weight":          jlm.RoleAttnKB,
			"self_attn.o_proj.weight":             jlm.RoleAttnOut,
			"self_attn.g_proj.weight":             jlm.RoleAttnGate,

			// The dense lead, and the latent mixture.
			"mlp.gate_proj.weight":                             jlm.RoleFFNGate,
			"mlp.up_proj.weight":                               jlm.RoleFFNUp,
			"mlp.down_proj.weight":                             jlm.RoleFFNDown,
			"block_sparse_moe.gate.weight":                     jlm.RoleRouter,
			"block_sparse_moe.gate.e_score_correction_bias":    jlm.RoleExpProbsB,
			"block_sparse_moe.routed_expert_down_proj.weight":  jlm.RoleFFNRoutedDown,
			"block_sparse_moe.routed_expert_norm.weight":       jlm.RoleFFNRoutedNorm,
			"block_sparse_moe.routed_expert_up_proj.weight":    jlm.RoleFFNRoutedUp,
			"block_sparse_moe.shared_experts.gate_proj.weight": jlm.RoleShExpGate,
			"block_sparse_moe.shared_experts.up_proj.weight":   jlm.RoleShExpUp,
			"block_sparse_moe.shared_experts.down_proj.weight": jlm.RoleShExpDown,
		},
		blockLinear: map[string]jlm.Role{
			"input_layernorm.weight":          jlm.RoleAttnNorm,
			"post_attention_layernorm.weight": jlm.RoleFFNNorm,

			// q, k and v joined into one mixed projection and their convolutions
			// into one filter bank, in the order fuseOrder states: the layout
			// kimiK3Tensors builds from the GGUF.
			"self_attn.q_proj.weight":   jlm.RoleAttnQKV,
			"self_attn.k_proj.weight":   jlm.RoleAttnQKV,
			"self_attn.v_proj.weight":   jlm.RoleAttnQKV,
			"self_attn.q_conv1d.weight": jlm.RoleSSMConv1d,
			"self_attn.k_conv1d.weight": jlm.RoleSSMConv1d,
			"self_attn.v_conv1d.weight": jlm.RoleSSMConv1d,

			// beta alone, the forget gate's two halves, and the full-rank output
			// gate (K3's g_proj; Kimi-Linear's low-rank g_a/g_b are refused by
			// kimiK3HFConfig before any tensor is read).
			"self_attn.b_proj.weight":   jlm.RoleSSMBA,
			"self_attn.f_a_proj.weight": jlm.RoleSSMFA,
			"self_attn.f_b_proj.weight": jlm.RoleSSMFB,
			"self_attn.g_proj.weight":   jlm.RoleAttnGate,

			"self_attn.A_log":         jlm.RoleSSMA,
			"self_attn.dt_bias":       jlm.RoleSSMDtBias,
			"self_attn.o_norm.weight": jlm.RoleSSMNorm,
			"self_attn.o_proj.weight": jlm.RoleSSMOut,
		},
		fuseOrder:    kimiLinearHF().fuseOrder,
		expertPrefix: "block_sparse_moe.experts.",
		expert:       kimiLinearHF().expert,
		split: map[string]hfSplit{
			"self_attn.kv_b_proj.weight": deepseekSplitKVB,
			"self_attn.q_conv1d.weight":  kimiSqueezeConv1d,
			"self_attn.k_conv1d.weight":  kimiSqueezeConv1d,
			"self_attn.v_conv1d.weight":  kimiSqueezeConv1d,
			"self_attn.A_log":            kimiExpandALog,
		},
		config: kimiK3HFConfig,
		gather: kimiK3Gather,
	}
}

// k3BlockPrefix is where the text model's blocks live in the checkpoint.
const k3BlockPrefix = "language_model.model.layers."

// k3TextConfig is text_config: KimiLinearConfig's keys this converter reads
// or refuses on. A pointer is a key that may be absent, which then takes the
// CLASS's default (configuration_kimi_k3.py), never another class's.
type k3TextConfig struct {
	VocabSize              *uint32         `json:"vocab_size"`
	HiddenSize             *uint32         `json:"hidden_size"`
	IntermediateSize       *uint32         `json:"intermediate_size"`
	NumHiddenLayers        *uint32         `json:"num_hidden_layers"`
	NumAttentionHeads      *uint32         `json:"num_attention_heads"`
	HiddenAct              *string         `json:"hidden_act"`
	RMSNormEps             *float64        `json:"rms_norm_eps"`
	RopeTheta              *float64        `json:"rope_theta"`
	RopeScaling            json.RawMessage `json:"rope_scaling"`
	RopeParameters         json.RawMessage `json:"rope_parameters"`
	MaxPositionEmbeddings  *uint32         `json:"max_position_embeddings"`
	MoEIntermediateSize    *uint32         `json:"moe_intermediate_size"`
	MoERenormalize         *bool           `json:"moe_renormalize"`
	MoERouterActFunc       *string         `json:"moe_router_activation_func"`
	NumExperts             *uint32         `json:"num_experts"`
	NumExpertsPerToken     *uint32         `json:"num_experts_per_token"`
	NumSharedExperts       *uint32         `json:"num_shared_experts"`
	RoutedScalingFactor    *float64        `json:"routed_scaling_factor"`
	FirstKDenseReplace     *uint32         `json:"first_k_dense_replace"`
	MoELayerFreq           *uint32         `json:"moe_layer_freq"`
	UseGroupedTopk         *bool           `json:"use_grouped_topk"`
	NumExpertGroup         *uint32         `json:"num_expert_group"`
	TopkGroup              *uint32         `json:"topk_group"`
	TopkMethod             *string         `json:"topk_method"`
	QLoraRank              *uint32         `json:"q_lora_rank"`
	KVLoraRank             *uint32         `json:"kv_lora_rank"`
	QKNopeHeadDim          *uint32         `json:"qk_nope_head_dim"`
	QKRopeHeadDim          *uint32         `json:"qk_rope_head_dim"`
	VHeadDim               *uint32         `json:"v_head_dim"`
	MLAUseNope             *bool           `json:"mla_use_nope"`
	NumNextnPredictLayers  *uint32         `json:"num_nextn_predict_layers"`
	AttnResBlockSize       *uint32         `json:"attn_res_block_size"`
	ActivationSituBeta     *float64        `json:"activation_situ_beta"`
	ActivationSituLinBeta  *float64        `json:"activation_situ_linear_beta"`
	RoutedExpertHiddenSize *uint32         `json:"routed_expert_hidden_size"`
	LinearAttnConfig       *k3LinearAttn   `json:"linear_attn_config"`
}

// k3LinearAttn is linear_attn_config. KimiDeltaAttention reads
// use_full_rank_gate and gate_lower_bound with .get(), defaulting to false
// and None.
type k3LinearAttn struct {
	KDALayers       []uint32 `json:"kda_layers"`
	FullAttnLayers  []uint32 `json:"full_attn_layers"`
	NumHeads        uint32   `json:"num_heads"`
	HeadDim         uint32   `json:"head_dim"`
	ShortConv       uint32   `json:"short_conv_kernel_size"`
	GateLowerBound  *float64 `json:"gate_lower_bound"`
	UseFullRankGate *bool    `json:"use_full_rank_gate"`
}

// k3TextConfigOf reads text_config out of config.json. A Kimi-K3 checkpoint
// has no flat form: KimiK3Config builds its KimiLinearConfig from this dict.
func k3TextConfigOf(c *hfConfig) (*k3TextConfig, error) {
	raw, ok := c.keys["text_config"]
	if !ok || string(raw) == "null" {
		return nil, fmt.Errorf("convert: %s: KimiK3ForConditionalGeneration keeps the text model's "+
			"hyperparameters under text_config, and this one has none", c.path)
	}
	tc := &k3TextConfig{}
	if err := json.Unmarshal(raw, tc); err != nil {
		return nil, fmt.Errorf("convert: %s: text_config: %w", c.path, err)
	}
	return tc, nil
}

// kimiK3HFConfig reads text_config into the GGUF keys llama.cpp's
// converter (conversion/kimi_k3.py) writes from the same dict, and hands them
// to configOf: one reader, kimiK3Config's, decides what each key means, so
// the two inputs cannot drift apart. Every absent key takes
// KimiLinearConfig's default; every graph the engine does not run is refused
// by its key.
func kimiK3HFConfig(c *hfConfig, names []string) (*Config, error) {
	tc, err := k3TextConfigOf(c)
	if err != nil {
		return nil, err
	}
	bad := func(format string, a ...any) error {
		return fmt.Errorf("convert: %s: text_config: "+format, append([]any{c.path}, a...)...)
	}
	u := func(p *uint32, def uint32) uint32 {
		if p == nil {
			return def
		}
		return *p
	}
	f := func(p *float64, def float64) float64 {
		if p == nil {
			return def
		}
		return *p
	}
	str := func(p *string, def string) string {
		if p == nil {
			return def
		}
		return *p
	}
	b := func(p *bool, def bool) bool {
		if p == nil {
			return def
		}
		return *p
	}

	// The graph's switches. Each one the reference reads with a default the
	// engine does not run is refused rather than read as K3's.
	if act := str(tc.HiddenAct, "silu"); act != "situ" {
		return nil, bad("hidden_act %q; Kimi-K3's FFNs are situ: %w", act, ErrNotImplemented)
	}
	if !b(tc.MLAUseNope, false) {
		return nil, bad("mla_use_nope is false or absent (the class's default), and Kimi-K3's "+
			"latent attention has no positional encoding: %w", ErrNotImplemented)
	}
	if fn := str(tc.MoERouterActFunc, "sigmoid"); fn != "sigmoid" {
		return nil, bad("moe_router_activation_func %q; Kimi-K3 routes with a sigmoid: %w", fn, ErrNotImplemented)
	}
	if m := str(tc.TopkMethod, "noaux_tc"); m != "noaux_tc" {
		return nil, bad("topk_method %q; Kimi-K3 selects with the correction bias (noaux_tc): %w",
			m, ErrNotImplemented)
	}
	if n := u(tc.MoELayerFreq, 1); n != 1 {
		return nil, bad("moe_layer_freq %d: this container states the dense run as a length, and "+
			"a mixture every %d blocks is not one: %w", n, n, ErrNotImplemented)
	}
	if n := u(tc.NumNextnPredictLayers, 0); n != 0 {
		return nil, bad("num_nextn_predict_layers %d: Kimi-K3's prediction blocks are not "+
			"implemented: %w", n, ErrNotImplemented)
	}
	theta, err := k3RopeTheta(tc)
	if err != nil {
		return nil, bad("%v", err)
	}

	nl := u(tc.NumHiddenLayers, 32)
	la := tc.LinearAttnConfig
	if la == nil {
		return nil, bad("no linear_attn_config: Kimi-K3 is a hybrid and the layer kinds are read there")
	}
	if !b(la.UseFullRankGate, false) {
		return nil, bad("linear_attn_config.use_full_rank_gate is false or absent (the class's "+
			"default): Kimi-Linear's low-rank KDA output gate under Kimi-K3 is not implemented: %w",
			ErrNotImplemented)
	}
	heads := u(tc.NumAttentionHeads, 32)
	if la.NumHeads != heads {
		return nil, bad("linear_attn_config.num_heads %d beside num_attention_heads %d: the "+
			"container's KDA heads are the attention's (llama.cpp's n_head_kda): %w",
			la.NumHeads, heads, ErrNotImplemented)
	}
	// is_kda_layer reads kda_layers (one-indexed); llama.cpp's converter
	// reads full_attn_layers. They must partition the blocks, or the two
	// readers build different models from one file.
	full := make([]bool, nl)
	seen := make([]bool, nl)
	for _, set := range []struct {
		list   []uint32
		isFull bool
	}{{la.KDALayers, false}, {la.FullAttnLayers, true}} {
		for _, l := range set.list {
			if l < 1 || l > nl || seen[l-1] {
				return nil, bad("linear_attn_config names layer %d (one-indexed) twice or outside a "+
					"%d-layer model", l, nl)
			}
			seen[l-1], full[l-1] = true, set.isFull
		}
	}
	if i := slices.Index(seen, false); i >= 0 {
		return nil, bad("linear_attn_config names layer %d (one-indexed) in neither kda_layers nor "+
			"full_attn_layers", i+1)
	}
	kv := make([]int32, nl)
	for i, isFull := range full {
		if isFull {
			kv[i] = 1 // MLA served as MQA, as llama.cpp's converter writes it
		}
	}

	need := func(p *uint32, key string) (uint32, error) {
		if p == nil || *p == 0 {
			return 0, bad("no %s: Kimi-K3's full blocks are MLA", key)
		}
		return *p, nil
	}
	lora, err := need(tc.KVLoraRank, "kv_lora_rank")
	if err != nil {
		return nil, err
	}
	nope, err := need(tc.QKNopeHeadDim, "qk_nope_head_dim")
	if err != nil {
		return nil, err
	}
	rope, err := need(tc.QKRopeHeadDim, "qk_rope_head_dim")
	if err != nil {
		return nil, err
	}
	vdim, err := need(tc.VHeadDim, "v_head_dim")
	if err != nil {
		return nil, err
	}

	ns := "kimi-k3."
	kvs := map[string]meta.Value{
		"general.architecture":                  meta.MakeString("kimi-k3"),
		ns + "block_count":                      meta.MakeUint(uint64(nl)),
		ns + "context_length":                   meta.MakeUint(uint64(u(tc.MaxPositionEmbeddings, 4096))),
		ns + "embedding_length":                 meta.MakeUint(uint64(u(tc.HiddenSize, 4096))),
		ns + "feed_forward_length":              meta.MakeUint(uint64(u(tc.IntermediateSize, 11008))),
		ns + "attention.head_count":             meta.MakeUint(uint64(heads)),
		ns + "attention.head_count_kv":          meta.MakeInt32s(kv),
		ns + "rope.freq_base":                   meta.MakeFloat(float32(theta)),
		ns + "attention.layer_norm_rms_epsilon": meta.MakeFloat(float32(f(tc.RMSNormEps, 1e-6))),
		ns + "ssm.conv_kernel":                  meta.MakeUint(uint64(la.ShortConv)),
		ns + "kda.head_dim":                     meta.MakeUint(uint64(la.HeadDim)),
		ns + "attention.kv_lora_rank":           meta.MakeUint(uint64(lora)),
		ns + "rope.dimension_count":             meta.MakeUint(uint64(rope)),
		ns + "attention.key_length":             meta.MakeUint(uint64(lora + rope)),
		ns + "attention.value_length":           meta.MakeUint(uint64(lora)),
		ns + "attention.key_length_mla":         meta.MakeUint(uint64(nope + rope)),
		ns + "attention.value_length_mla":       meta.MakeUint(uint64(vdim)),
		ns + "leading_dense_block_count":        meta.MakeUint(uint64(u(tc.FirstKDenseReplace, 0))),
	}
	if q := u(tc.QLoraRank, 0); q != 0 {
		kvs[ns+"attention.q_lora_rank"] = meta.MakeUint(uint64(q))
	}
	if la.GateLowerBound != nil {
		kvs[ns+"kda.gate_lower_bound"] = meta.MakeFloat(float32(*la.GateLowerBound))
	}
	if tc.AttnResBlockSize != nil {
		kvs[ns+"attn_res.block_size"] = meta.MakeUint(uint64(*tc.AttnResBlockSize))
	}
	if tc.RoutedExpertHiddenSize != nil {
		kvs[ns+"expert_latent_length"] = meta.MakeUint(uint64(*tc.RoutedExpertHiddenSize))
	}
	// Absent situ bounds are None in the class; kimiK3Config refuses a pair
	// that is not the one the kernels bake, so a zero here is that refusal.
	kvs[ns+"activation.situ_beta"] = meta.MakeFloat(float32(f(tc.ActivationSituBeta, 0)))
	kvs[ns+"activation.situ_linear_beta"] = meta.MakeFloat(float32(f(tc.ActivationSituLinBeta, 0)))
	// The mixture. num_experts None is a dense model, which Kimi-K3 is not.
	if tc.NumExperts == nil || *tc.NumExperts == 0 {
		return nil, bad("no num_experts: Kimi-K3's blocks after the dense lead are mixtures")
	}
	kvs[ns+"expert_count"] = meta.MakeUint(uint64(*tc.NumExperts))
	kvs[ns+"expert_used_count"] = meta.MakeUint(uint64(u(tc.NumExpertsPerToken, 0)))
	kvs[ns+"expert_feed_forward_length"] = meta.MakeUint(uint64(u(tc.MoEIntermediateSize, 0)))
	kvs[ns+"expert_shared_count"] = meta.MakeUint(uint64(u(tc.NumSharedExperts, 0)))
	kvs[ns+"expert_weights_scale"] = meta.MakeFloat(float32(f(tc.RoutedScalingFactor, 1)))
	kvs[ns+"expert_weights_norm"] = meta.MakeBool(b(tc.MoERenormalize, true))
	kvs[ns+"expert_gating_func"] = meta.MakeUint(2)
	// Grouped selection: KimiMoEGate groups only when num_expert_group > 1
	// and exceeds topk_group. llama.cpp writes topk_group alone, which
	// configOf reads as the ungrouped selection when it is 1.
	groups, used := u(tc.NumExpertGroup, 1), u(tc.TopkGroup, 1)
	switch {
	case !b(tc.UseGroupedTopk, true) || groups <= 1 || groups <= used:
		kvs[ns+"expert_group_used_count"] = meta.MakeUint(1)
	default:
		kvs[ns+"expert_group_count"] = meta.MakeUint(uint64(groups))
		kvs[ns+"expert_group_used_count"] = meta.MakeUint(uint64(used))
	}

	// configOf reads the vocabulary's size and whether the head is tied off
	// the tensor table; the two it asks about stand in for the checkpoint's.
	embd := uint64(u(tc.HiddenSize, 4096))
	ts := []meta.Tensor{{Name: "token_embd.weight", Dims: []uint64{embd, uint64(u(tc.VocabSize, 163840))}}}
	if slices.Contains(names, "language_model.lm_head.weight") {
		ts = append(ts, meta.Tensor{Name: "output.weight", Dims: ts[0].Dims})
	}
	return configOf(meta.New(c.path, kvs, ts))
}

// k3RopeTheta is the rotary base, stated as rope_theta and, by transformers
// 5, inside rope_parameters; two values that disagree are refused. Kimi-K3's
// attention has no rotary at all (mla_use_nope), so a scaling type is a
// statement about nothing the graph runs, and anything but the default is
// refused rather than read.
func k3RopeTheta(tc *k3TextConfig) (float64, error) {
	theta := 10000.0
	if tc.RopeTheta != nil {
		theta = *tc.RopeTheta
	}
	for _, raw := range []json.RawMessage{tc.RopeParameters, tc.RopeScaling} {
		if len(raw) == 0 || string(raw) == "null" {
			continue
		}
		var rp struct {
			Theta    *float64 `json:"rope_theta"`
			Type     string   `json:"rope_type"`
			OldStyle string   `json:"type"`
		}
		if err := json.Unmarshal(raw, &rp); err != nil {
			return 0, fmt.Errorf("rope_parameters: %w", err)
		}
		if t := rp.Type + rp.OldStyle; t != "" && t != "default" {
			return 0, fmt.Errorf("rotary scaling %q on a model with no rotary: %w", t, ErrNotImplemented)
		}
		if rp.Theta != nil {
			if tc.RopeTheta != nil && *rp.Theta != *tc.RopeTheta {
				return 0, fmt.Errorf("rope_theta is %g and rope_parameters says %g", *tc.RopeTheta, *rp.Theta)
			}
			theta = *rp.Theta
		}
	}
	return theta, nil
}

// k3ResPairs are the residual attention's norm and projection, which the
// reference only ever multiplies: each pair becomes one F32 vector, their
// product, as llama.cpp's converter writes it (norm.float() * proj.float()).
func k3ResPairs() []struct {
	stem string
	role jlm.Role
} {
	return []struct {
		stem string
		role jlm.Role
	}{
		{"self_attention_res", jlm.RoleAttnResScore},
		{"mlp_res", jlm.RoleFFNResScore},
	}
}

// k3OutputRes is the head's pair, outside the blocks.
const k3OutputRes = "language_model.model.output_attn_res"

// kimiK3Gather folds each residual-attention pair into its score vector. A
// model with residual attention (attn_res_block_size) carries every pair;
// one without carries none, and a stray pair is left unconsumed for identify
// to refuse.
func kimiK3Gather(c *Config, names []string, read func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
	if c.AttnResBlock == 0 {
		return nil, nil, nil
	}
	used := map[string]bool{}
	var out []jlm.Tensor
	fold := func(stem string, role jlm.Role, block int32) error {
		norm, proj := stem+"_norm.weight", stem+"_proj.weight"
		if !slices.Contains(names, norm) || !slices.Contains(names, proj) {
			return fmt.Errorf("convert: residual attention every %d blocks, and %s or %s is missing",
				c.AttnResBlock, norm, proj)
		}
		n, err := read(norm)
		if err != nil {
			return err
		}
		p, err := read(proj)
		if err != nil {
			return err
		}
		if len(n) != int(c.NEmbd) || len(p) != int(c.NEmbd) {
			return fmt.Errorf("convert: %s and %s hold %d and %d values, want %d each",
				norm, proj, len(n), len(p), c.NEmbd)
		}
		v := make([]float32, len(n))
		for i := range v {
			v[i] = n[i] * p[i]
		}
		out = append(out, jlm.Tensor{Role: role, Block: block, Index: -1, Type: jlm.TypeF32, NDim: 1,
			Dims: [4]uint64{uint64(len(v))}, Data: f32AsBytes(v), Name: stem + "_norm*proj"})
		used[norm], used[proj] = true, true
		return nil
	}
	for b := range int(c.NLayer) {
		for _, r := range k3ResPairs() {
			if err := fold(k3BlockPrefix+itoa(b)+"."+r.stem, r.role, int32(b)); err != nil {
				return nil, nil, err
			}
		}
	}
	if err := fold(k3OutputRes, jlm.RoleOutputResScore, jlm.DenseBlock); err != nil {
		return nil, nil, err
	}
	for _, n := range names {
		if strings.HasSuffix(n, "_res_norm.weight") || strings.HasSuffix(n, "_res_proj.weight") {
			if !used[n] {
				return nil, nil, fmt.Errorf("convert: %s is a residual-attention tensor outside the "+
					"model's %d blocks", n, c.NLayer)
			}
		}
	}
	return out, used, nil
}

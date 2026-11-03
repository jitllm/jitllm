package convert

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// The dense llama family from HuggingFace safetensors: the same containers
// convert/dense.go writes from a GGUF, read off transformers' own classes
// (models/<arch>/modeling_*.py and configuration_*.py).
//
// The trap these share is the class default. A key config.json does not carry
// takes the default of ITS class, and the classes disagree: num_key_value_heads
// is 4 for SmolLM3, 8 for SeedOss and Ministral3, 32 for Exaone4 and the
// attention head count for the rest; head_dim is 128 for SeedOss and Ministral3
// and hidden/heads elsewhere; rope_theta is 2e6 for SmolLM3 and 5e5 for OLMo 3.
// llamaHFConfig's defaults are llama's, so each entry states its class's first.

// hfClassDefaults is what a class's config dataclass declares for the keys
// llamaHFConfig reads. A zero kvHeads or headDim means the generic derivation
// (MHA, hidden/heads), which is what those classes' __post_init__ does too.
type hfClassDefaults struct {
	kvHeads, headDim uint32
	ropeTheta        float64
	rmsEps           float64
	tied             bool
	nCtx             uint32
}

// apply writes d into the keys config.json does not carry. A key that is
// present, null included, is the file's answer.
func (d hfClassDefaults) apply(c *hfConfig) {
	if c.absent("num_key_value_heads") && d.kvHeads != 0 {
		v := d.kvHeads
		c.NumKeyValueHeads = &v
	}
	if c.absent("head_dim") && d.headDim != 0 {
		v := d.headDim
		c.HeadDim = &v
	}
	// RopeTheta is zero only when neither rope_theta nor rope_parameters
	// states a base (readHFConfig reads both).
	if c.RopeTheta == 0 {
		c.RopeTheta = d.ropeTheta
	}
	if c.absent("rms_norm_eps") {
		c.RMSNormEps = d.rmsEps
	}
	if c.absent("tie_word_embeddings") {
		t := d.tied
		c.TieWordEmbeddings = &t
	}
	if c.absent("max_position_embeddings") {
		c.MaxPositionEmbeddings = d.nCtx
	}
}

// hfLlamaBlockNoBias is llama's block for a class whose attention has no
// bias parameter at all (bias=False in the class): a bias in the file would be
// a tensor the reference never loads.
func hfLlamaBlockNoBias() map[string]jlm.Role {
	return map[string]jlm.Role{
		"input_layernorm.weight":          jlm.RoleAttnNorm,
		"post_attention_layernorm.weight": jlm.RoleFFNNorm,
		"self_attn.q_proj.weight":         jlm.RoleAttnQ,
		"self_attn.k_proj.weight":         jlm.RoleAttnK,
		"self_attn.v_proj.weight":         jlm.RoleAttnV,
		"self_attn.o_proj.weight":         jlm.RoleAttnOut,
		"mlp.gate_proj.weight":            jlm.RoleFFNGate,
		"mlp.up_proj.weight":              jlm.RoleFFNUp,
		"mlp.down_proj.weight":            jlm.RoleFFNDown,
	}
}

// hfPostNormBlock is OLMo 2's and EXAONE 4's block: no input_layernorm, and
// post_attention_layernorm is a real post-norm here (the attention OUTPUT is
// normalised before its residual add), not llama's FFN pre-norm.
func hfPostNormBlock() map[string]jlm.Role {
	return map[string]jlm.Role{
		"post_attention_layernorm.weight":   jlm.RolePostAttnNorm,
		"post_feedforward_layernorm.weight": jlm.RolePostFFNNorm,
		"self_attn.q_proj.weight":           jlm.RoleAttnQ,
		"self_attn.k_proj.weight":           jlm.RoleAttnK,
		"self_attn.v_proj.weight":           jlm.RoleAttnV,
		"self_attn.o_proj.weight":           jlm.RoleAttnOut,
		"self_attn.q_norm.weight":           jlm.RoleAttnQNorm,
		"self_attn.k_norm.weight":           jlm.RoleAttnKNorm,
		"mlp.gate_proj.weight":              jlm.RoleFFNGate,
		"mlp.up_proj.weight":                jlm.RoleFFNUp,
		"mlp.down_proj.weight":              jlm.RoleFFNDown,
	}
}

// smollm3: llama's block with no rotary on every no_rope_layer_interval-th
// layer, rotate_half rotary (permuted, as convert_hf_to_gguf.py does, so the
// container is ArchSmolLM3's adjacent-pair layout).
func smollm3HF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchSmolLM3,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block:       llamaHF().block,
		ignore:      llamaHF().ignore,
		config: func(c *hfConfig, names []string) (*jlm.Config, error) {
			hfClassDefaults{kvHeads: 4, ropeTheta: 2e6, rmsEps: 1e-6, tied: true, nCtx: 32768}.apply(c)
			// The window is SmolLM3Attention's only when use_sliding_window, which
			// the class defaults to false; llamaHFConfig refuses one in use.
			if c.UseSlidingWindow == nil {
				f := false
				c.UseSlidingWindow = &f
			}
			cfg, err := hfConfigWith(jlm.ArchSmolLM3, 0, "silu")(c, names)
			if err != nil {
				return nil, err
			}
			period, err := smolLM3NoPEPeriod(c, cfg.NLayer)
			if err != nil {
				return nil, err
			}
			// The period with no window names the NoPE layers and slides none
			// (convert/dense.go). llama.cpp hardcodes 4; config.json is read here.
			if period > 0 {
				cfg.SWAWindow, cfg.SWAPeriod = 0, period
				cfg.Flags |= jlm.FlagNoPEGlobal
			}
			return cfg, nil
		},
		fixup: llamaHFFixup,
	}
}

// smolLM3NoPEPeriod reads which layers skip the rotary: no_rope_layers (1
// rotates, 0 does not) or, absent, SmolLM3Config's derivation from
// no_rope_layer_interval, (i+1) % interval != 0. The container can say only a
// period, so any other list is refused. Zero means every layer rotates.
func smolLM3NoPEPeriod(c *hfConfig, n uint32) (uint32, error) {
	rope := c.NoRopeLayers
	if rope == nil {
		every := uint32(4)
		if c.NoRopeLayerInterval != nil {
			every = *c.NoRopeLayerInterval
		}
		if every == 0 {
			return 0, fmt.Errorf("convert: %s: no_rope_layer_interval 0", c.path)
		}
		for i := uint32(0); i < n; i++ {
			rope = append(rope, boolInt((i+1)%every != 0))
		}
	}
	if uint32(len(rope)) < n {
		return 0, fmt.Errorf("convert: %s: no_rope_layers has %d entries for %d layers",
			c.path, len(rope), n)
	}
	period := uint32(0)
	for i := uint32(0); i < n; i++ {
		if rope[i] != 0 && rope[i] != 1 {
			return 0, fmt.Errorf("convert: %s: no_rope_layers[%d] is %d", c.path, i, rope[i])
		}
		if rope[i] == 0 && period == 0 {
			period = i + 1
		}
	}
	if period == 0 {
		return 0, nil
	}
	if period == 1 {
		return 0, fmt.Errorf("convert: %s: no layer rotates (no_rope_layers %v): %w",
			c.path, rope[:n], ErrNotImplemented)
	}
	for i := uint32(0); i < n; i++ {
		if want := boolInt((i+1)%period != 0); rope[i] != want {
			return 0, fmt.Errorf("convert: %s: no_rope_layers %v is not every %d-th layer "+
				"(layer %d): %w", c.path, rope[:n], period, i, ErrNotImplemented)
		}
	}
	return period, nil
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// arcee: llama's block with an ungated up -> relu^2 -> down FFN (ArceeMLP has
// no gate_proj, so a gate in the file is refused as an unknown name).
func arceeHF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchArcee,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block: map[string]jlm.Role{
			"input_layernorm.weight":          jlm.RoleAttnNorm,
			"post_attention_layernorm.weight": jlm.RoleFFNNorm,
			"self_attn.q_proj.weight":         jlm.RoleAttnQ,
			"self_attn.k_proj.weight":         jlm.RoleAttnK,
			"self_attn.v_proj.weight":         jlm.RoleAttnV,
			"self_attn.o_proj.weight":         jlm.RoleAttnOut,
			"self_attn.q_proj.bias":           jlm.RoleAttnQBias,
			"self_attn.k_proj.bias":           jlm.RoleAttnKBias,
			"self_attn.v_proj.bias":           jlm.RoleAttnVBias,
			"self_attn.o_proj.bias":           jlm.RoleAttnOutBias,
			"mlp.up_proj.weight":              jlm.RoleFFNUp,
			"mlp.down_proj.weight":            jlm.RoleFFNDown,
		},
		ignore: llamaHF().ignore,
		config: func(c *hfConfig, names []string) (*jlm.Config, error) {
			hfClassDefaults{ropeTheta: 1e4, rmsEps: 1e-5, nCtx: 4096}.apply(c)
			// ArceeConfig's hidden_act defaults to relu2, and ArceeMLP applies
			// whatever it names ungated; relu2 is the one this graph computes.
			if c.absent("hidden_act") {
				c.HiddenAct = "relu2"
			}
			// The class never reads sliding_window.
			c.SlidingWindow = nil
			return hfConfigWith(jlm.ArchArcee, jlm.FlagReLU2, "relu2")(c, names)
		},
		fixup: llamaHFFixup,
	}
}

// seed_oss: llama's names (biased q/k/v, and o when attention_out_bias), NEOX
// rotary unpermuted, and a head wider than hidden/heads by class default.
func seedOssHF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchSeedOSS,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block:       llamaHF().block,
		ignore:      llamaHF().ignore,
		config: func(c *hfConfig, names []string) (*jlm.Config, error) {
			hfClassDefaults{kvHeads: 8, headDim: 128, ropeTheta: 1e4, rmsEps: 1e-6, nCtx: 524288}.apply(c)
			c.SlidingWindow = nil // the class never reads it
			return hfConfigWith(jlm.ArchSeedOSS, jlm.FlagRopeNeox, "silu")(c, names)
		},
	}
}

// olmo2: the post-norm block, q and k RMSNormed over the whole projection,
// NEOX rotary unpermuted. olmo3 is the same block with a window.
func olmo2Block() map[string]jlm.Role {
	return blockPlus(hfPostNormBlock(), map[string]jlm.Role{
		"self_attn.q_proj.bias": jlm.RoleAttnQBias,
		"self_attn.k_proj.bias": jlm.RoleAttnKBias,
		"self_attn.v_proj.bias": jlm.RoleAttnVBias,
		"self_attn.o_proj.bias": jlm.RoleAttnOutBias,
	})
}

func olmo2HF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchOLMo2,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block:       olmo2Block(),
		ignore:      llamaHF().ignore,
		config: func(c *hfConfig, names []string) (*jlm.Config, error) {
			hfClassDefaults{ropeTheta: 1e4, rmsEps: 1e-5, nCtx: 2048}.apply(c)
			c.SlidingWindow = nil // Olmo2Attention has no window
			return olmo2HFConfig(c, names)
		},
	}
}

// olmo2HFConfig is OLMo 2's config. The k norm spans NKVHead*HeadDim, which a
// GQA OLMo 2 (the 32B) makes narrower than q's; hfCheckShape holds each norm to
// its own projection's width.
func olmo2HFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
	return hfConfigWith(jlm.ArchOLMo2, jlm.FlagRopeNeox|jlm.FlagQKNorm|jlm.FlagQKNormWide, "silu")(c, names)
}

// olmo3: OLMo 2's block, a window on the layers layer_types calls
// sliding_attention (three in four by class default), and a rotary per layer
// type: the full layers' may be YaRN, the sliding layers' is a default rotary
// at their own base -- jlm.ArchOLMo3, as the GGUF path converts it.
func olmo3HF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchOLMo2,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block:       olmo2Block(),
		ignore:      llamaHF().ignore,
		config: func(c *hfConfig, names []string) (*jlm.Config, error) {
			hfClassDefaults{rmsEps: 1e-5, nCtx: 2048}.apply(c)
			theta, local, full, err := olmo3Rope(c)
			if err != nil {
				return nil, err
			}
			c.RopeTheta, c.RopeScaling = theta, full
			yarn, err := hfYarnConsume(c)
			if err != nil {
				return nil, err
			}
			// Olmo3Config defaults the window to 4096 and the kinds to three
			// sliding layers in four; an explicit null window slides nothing.
			win := uint32(0)
			switch {
			case c.absent("sliding_window"):
				win = 4096
			case c.SlidingWindow != nil:
				win = *c.SlidingWindow
			}
			c.SlidingWindow = nil
			cfg, err := olmo2HFConfig(c, names)
			if err != nil {
				return nil, err
			}
			hfYarnApply(yarn, cfg)
			if win == 0 {
				return cfg, nil
			}
			kinds := c.LayerTypes
			if kinds == nil {
				kinds = periodicKinds(cfg.NLayer, 4)
			}
			period, err := hfWindowPeriod(c, kinds, cfg.NLayer)
			if err != nil {
				return nil, err
			}
			if period > 0 {
				cfg.SWAWindow, cfg.SWAPeriod = win, period
				cfg.Arch, cfg.RopeBaseSWA = jlm.ArchOLMo3, float32(local)
			}
			return cfg, nil
		},
	}
}

// olmo3Rope is the rotary Olmo3Config.convert_rope_params_to_dict arrives at
// for each layer type: the full layers' base and scaling (full, a YaRN dict or
// nil), and the sliding layers' base, which carry no scaling.
//
// rope_parameters is a dict per layer type. Absent, both types are default
// rotary at rope_theta or 5e5 -- except that the class pops rope_theta for the
// full layers and so gives the sliding ones 5e5 whatever the file says, and a
// rope_scaling merges into the full layers only.
func olmo3Rope(c *hfConfig) (global, local float64, full json.RawMessage, err error) {
	type params struct {
		Type    string   `json:"rope_type"`
		Legacy  string   `json:"type"`
		Theta   *float64 `json:"rope_theta"`
		Partial *float64 `json:"partial_rotary_factor"`
	}
	var fullP, sliding params
	if raw, ok := c.keys["rope_parameters"]; ok && string(raw) != "null" {
		var per map[string]json.RawMessage
		if err := json.Unmarshal(raw, &per); err != nil {
			return 0, 0, nil, fmt.Errorf("convert: %s: rope_parameters: %w", c.path, err)
		}
		for k := range per {
			if k != "full_attention" && k != "sliding_attention" {
				// A flat dict: Olmo3Config files it beside two default
				// per-type dicts and never reads it.
				return 0, 0, nil, fmt.Errorf("convert: %s: rope_parameters key %q; OLMo 3's is a dict "+
					"per layer type: %w", c.path, k, ErrNotImplemented)
			}
		}
		for k, p := range map[string]*params{"full_attention": &fullP, "sliding_attention": &sliding} {
			if v, ok := per[k]; ok && string(v) != "null" {
				if err := json.Unmarshal(v, p); err != nil {
					return 0, 0, nil, fmt.Errorf("convert: %s: rope_parameters.%s: %w", c.path, k, err)
				}
				if k == "full_attention" {
					full = v
				}
			}
		}
	}
	if raw, ok := c.keys["rope_scaling"]; ok && string(raw) != "null" {
		if err := json.Unmarshal(raw, &fullP); err != nil {
			return 0, 0, nil, fmt.Errorf("convert: %s: rope_scaling: %w", c.path, err)
		}
		full = raw
	}
	top := 5e5
	if raw, ok := c.keys["rope_theta"]; ok {
		if err := json.Unmarshal(raw, &top); err != nil {
			return 0, 0, nil, fmt.Errorf("convert: %s: rope_theta: %w", c.path, err)
		}
	}
	tf, ts := top, 5e5
	if fullP.Theta != nil {
		tf = *fullP.Theta
	}
	if sliding.Theta != nil {
		ts = *sliding.Theta
	}
	for name, p := range map[string]params{"full_attention": fullP, "sliding_attention": sliding} {
		kind := p.Type
		if kind == "" {
			kind = p.Legacy
		}
		// YaRN on the full layers is the generic path's (hfYarnConsume); the
		// sliding layers' rotary is a default one in every OLMo 3.
		if kind != "" && kind != "default" && !(kind == "yarn" && name == "full_attention") {
			return 0, 0, nil, fmt.Errorf("convert: %s: OLMo 3 with %q rotary scaling on its %s layers: %w",
				c.path, kind, name, ErrNotImplemented)
		}
		if p.Partial != nil && *p.Partial != 1 {
			return 0, 0, nil, fmt.Errorf("convert: %s: partial_rotary_factor %v is not implemented",
				c.path, *p.Partial)
		}
	}
	if k := fullP.Type + fullP.Legacy; k == "" || k == "default" {
		full = nil // nothing for the generic path to read
	}
	return tf, ts, full, nil
}

// exaone4: the post-norm block with a per-head q/k norm, NEOX rotary
// unpermuted, and with a window: three sliding layers in four by class default,
// the global ones rotating not at all (Exaone4Attention rotates when
// sliding_window is None or the layer slides).
func exaone4HF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchExaone4,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block:       hfPostNormBlock(),
		ignore:      llamaHF().ignore,
		config: func(c *hfConfig, names []string) (*jlm.Config, error) {
			hfClassDefaults{kvHeads: 32, ropeTheta: 1e4, rmsEps: 1e-5, nCtx: 2048}.apply(c)
			win := uint32(0)
			switch {
			case c.absent("sliding_window"):
				win = 4096
			case c.SlidingWindow != nil:
				win = *c.SlidingWindow
			}
			c.SlidingWindow = nil
			cfg, err := hfConfigWith(jlm.ArchExaone4, jlm.FlagRopeNeox|jlm.FlagQKNorm, "silu")(c, names)
			if err != nil {
				return nil, err
			}
			if win == 0 {
				return cfg, nil
			}
			kinds := c.LayerTypes
			if kinds == nil {
				// Exaone4Config derives the kinds from an integer pattern; a string
				// one ("LLLG") it cannot read without layer_types.
				every := uint32(4)
				if len(c.SlidingWindowPattern) > 0 && string(c.SlidingWindowPattern) != "null" {
					if err := json.Unmarshal(c.SlidingWindowPattern, &every); err != nil || every == 0 {
						return nil, fmt.Errorf("convert: %s: sliding_window_pattern %s with no layer_types: %w",
							c.path, c.SlidingWindowPattern, ErrNotImplemented)
					}
				}
				kinds = periodicKinds(cfg.NLayer, every)
			}
			period, err := hfWindowPeriod(c, kinds, cfg.NLayer)
			if err != nil {
				return nil, err
			}
			if period == 0 {
				// A window and no sliding layer: every layer is global, and so
				// every layer skips the rotary.
				return nil, fmt.Errorf("convert: %s: sliding_window %d with no sliding layer leaves "+
					"no layer rotating: %w", c.path, win, ErrNotImplemented)
			}
			cfg.SWAWindow, cfg.SWAPeriod = win, period
			cfg.Flags |= jlm.FlagNoPEGlobal
			return cfg, nil
		},
	}
}

// periodicKinds is the layer_types a config class derives from a pattern:
// sliding except every every-th layer.
func periodicKinds(n, every uint32) []string {
	kinds := make([]string, n)
	for i := range kinds {
		kinds[i] = "sliding_attention"
		if (uint32(i)+1)%every == 0 {
			kinds[i] = "full_attention"
		}
	}
	return kinds
}

// hfWindowPeriod is swaPatternOf for layer_types: the SWAPeriod that makes
// Config.SWA true on exactly the sliding layers, allLocal when every layer
// slides, and 0 when none does. A pattern no period describes is refused.
func hfWindowPeriod(c *hfConfig, kinds []string, n uint32) (uint32, error) {
	if uint32(len(kinds)) < n {
		return 0, fmt.Errorf("convert: %s: layer_types has %d entries for %d layers", c.path, len(kinds), n)
	}
	kinds = kinds[:n]
	period := uint32(0)
	for i, k := range kinds {
		switch k {
		case "sliding_attention", "full_attention":
		default:
			return 0, fmt.Errorf("convert: %s: layer_types[%d] is %q: %w", c.path, i, k, ErrNotImplemented)
		}
		if k == "full_attention" && period == 0 {
			period = uint32(i) + 1
		}
	}
	switch period {
	case 0:
		return allLocal(n), nil
	case 1:
		for _, k := range kinds {
			if k != "full_attention" {
				return 0, fmt.Errorf("convert: %s: layer_types %v is not periodic: %w",
					c.path, kinds, ErrNotImplemented)
			}
		}
		return 0, nil
	}
	for i, k := range kinds {
		if want := uint32(i)%period < period-1; (k == "sliding_attention") != want {
			return 0, fmt.Errorf("convert: %s: layer_types %v is not periodic (layer %d): %w",
				c.path, kinds, i, ErrNotImplemented)
		}
	}
	return period, nil
}

// ministral3: llama's block with no biases (the class has none), rotate_half
// rotary permuted as convert_hf_to_gguf.py permutes it, YaRN, and
// llama_4_scaling_beta as the attention temperature on every layer.
func ministral3HF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchMistral3,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block:       hfLlamaBlockNoBias(),
		ignore:      llamaHF().ignore,
		config: func(c *hfConfig, names []string) (*jlm.Config, error) {
			hfClassDefaults{kvHeads: 8, headDim: 128, rmsEps: 1e-5, nCtx: 262144}.apply(c)
			r, err := ministral3Rope(c)
			if err != nil {
				return nil, err
			}
			c.RopeTheta, c.RopeScaling, c.PartialRotaryFactor = r.theta, nil, nil
			cfg, err := hfConfigWith(jlm.ArchMistral3, 0, "silu")(c, names)
			if err != nil {
				return nil, err
			}
			r.write(cfg)
			return cfg, nil
		},
		fixup: llamaHFFixup,
	}
}

// hfMinistral3Rope is Ministral3's rope_parameters as its class resolves them.
type hfMinistral3Rope struct {
	theta float64
	yarn  bool
	// factor, origCtx and the betas are YaRN's; attn is its magnitude
	// correction on cos and sin.
	factor, betaFast, betaSlow, attn float64
	origCtx                          uint32
	exact                            bool
	// temp and tempFloor are llama_4_scaling_beta and the position it floors by.
	temp      float64
	tempFloor uint32
}

// ministral3Rope reads the rotary the way Ministral3Config and
// _compute_yarn_parameters do.
//
// With neither rope_parameters nor rope_scaling the class supplies its own
// dict -- YaRN factor 16 over 16384 positions, beta 0.1, base 1e6 -- and a
// top-level rope_theta does not reach it. YaRN's magnitude is
// attention_factor where stated, else mscale(f, mscale)/mscale(f,
// mscale_all_dim) when both are set and non-zero, else mscale(f, 1); it scales
// cos and sin and nothing reaches the softmax. llama.cpp's mistral3 takes
// mscale as 1 (mistral3Yarn), which agrees wherever mscale is 1, as it is in
// every published Ministral 3.
func ministral3Rope(c *hfConfig) (*hfMinistral3Rope, error) {
	var p struct {
		Type         string   `json:"rope_type"`
		Legacy       string   `json:"type"`
		Theta        *float64 `json:"rope_theta"`
		Factor       *float64 `json:"factor"`
		OrigCtx      uint32   `json:"original_max_position_embeddings"`
		BetaFast     float64  `json:"beta_fast"`
		BetaSlow     float64  `json:"beta_slow"`
		MScale       float64  `json:"mscale"`
		MScaleAllDim float64  `json:"mscale_all_dim"`
		AttnFactor   *float64 `json:"attention_factor"`
		Beta         *float64 `json:"llama_4_scaling_beta"`
		Truncate     *bool    `json:"truncate"`
		Partial      *float64 `json:"partial_rotary_factor"`
	}
	raw := c.keys["rope_scaling"]
	if len(raw) == 0 || string(raw) == "null" {
		raw = c.keys["rope_parameters"]
	}
	if len(raw) == 0 || string(raw) == "null" {
		raw = json.RawMessage(`{"type": "yarn", "rope_theta": 1000000.0, "factor": 16.0,
			"original_max_position_embeddings": 16384, "beta_fast": 32.0, "beta_slow": 1.0,
			"mscale_all_dim": 1.0, "mscale": 1.0, "llama_4_scaling_beta": 0.1}`)
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, fmt.Errorf("convert: %s: rope_parameters: %w", c.path, err)
	}
	r := &hfMinistral3Rope{theta: 1e4}
	if raw, ok := c.keys["rope_theta"]; ok {
		if err := json.Unmarshal(raw, &r.theta); err != nil {
			return nil, fmt.Errorf("convert: %s: rope_theta: %w", c.path, err)
		}
	}
	if p.Theta != nil {
		r.theta = *p.Theta
	}
	if p.Partial != nil && *p.Partial != 1 {
		return nil, fmt.Errorf("convert: %s: partial_rotary_factor %v is not implemented", c.path, *p.Partial)
	}
	// Ministral3Attention multiplies q by 1 + beta*ln(1 + floor(pos/orig))
	// with both read unconditionally; the class has no form without them.
	if p.Beta == nil {
		return nil, fmt.Errorf("convert: %s: rope_parameters has no llama_4_scaling_beta, which "+
			"Ministral3Attention reads on every layer", c.path)
	}
	if r.temp = *p.Beta; r.temp != 0 {
		if p.OrigCtx == 0 {
			return nil, fmt.Errorf("convert: %s: llama_4_scaling_beta %g with no "+
				"original_max_position_embeddings to floor positions by", c.path, r.temp)
		}
		r.tempFloor = p.OrigCtx
	}
	kind := p.Type
	if kind == "" {
		kind = p.Legacy
	}
	switch kind {
	case "", "default":
		return r, nil
	case "yarn":
	default:
		return nil, fmt.Errorf("convert: %s: rope_scaling %q is not implemented for Ministral3", c.path, kind)
	}
	if p.Factor == nil || p.OrigCtx == 0 {
		return nil, fmt.Errorf("convert: %s: a YaRN rope_parameters needs factor and "+
			"original_max_position_embeddings", c.path)
	}
	if *p.Factor < 1 {
		return nil, fmt.Errorf("convert: %s: YaRN factor %v", c.path, *p.Factor)
	}
	if *p.Factor == 1 {
		// The identity: interpolation and extrapolation are the same table.
		return r, nil
	}
	f := *p.Factor
	ms := func(k float64) float64 { return 0.1*k*math.Log(f) + 1 }
	r.yarn, r.factor, r.origCtx = true, f, p.OrigCtx
	// `get("beta_fast") or 32`: zero is the default too.
	r.betaFast, r.betaSlow = 32, 1
	if p.BetaFast != 0 {
		r.betaFast = p.BetaFast
	}
	if p.BetaSlow != 0 {
		r.betaSlow = p.BetaSlow
	}
	switch {
	case p.AttnFactor != nil:
		r.attn = *p.AttnFactor
	case p.MScale != 0 && p.MScaleAllDim != 0:
		r.attn = ms(p.MScale) / ms(p.MScaleAllDim)
	default:
		r.attn = ms(1)
	}
	r.exact = p.Truncate != nil && !*p.Truncate
	return r, nil
}

// write sets what the container carries of r, as denseConfig and
// ropeScalingOf set it from a GGUF.
func (r *hfMinistral3Rope) write(cfg *jlm.Config) {
	if r.temp != 0 {
		cfg.AttnTempScale, cfg.AttnTempFloor, cfg.AttnTempOffset = float32(r.temp), r.tempFloor, 0
	}
	if !r.yarn {
		return
	}
	cfg.YarnFactor, cfg.YarnOrigCtx = float32(r.factor), r.origCtx
	cfg.YarnBetaFast, cfg.YarnBetaSlow = float32(r.betaFast), float32(r.betaSlow)
	cfg.AttnFactor *= float32(r.attn)
	if r.exact {
		cfg.Flags |= jlm.FlagYarnExact
	}
}

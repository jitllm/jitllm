package convert

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// c6HFDir is where scripts/c6gold.py keeps the HuggingFace directories it
// writes its GGUFs from (its CV_HF, by default $JITLLM_MODELS/hf), so one set of weights reaches the
// container through both converters.
func c6HFDir() string {
	if d := os.Getenv("JITLLM_C6_HF"); d != "" {
		return d
	}
	return testmodels.Path("hf")
}

// TestDenseSafetensorsMatchesItsGGUF converts each dense llama-family fixture
// twice -- the HuggingFace directory directly, and the GGUF llama.cpp's own
// convert_hf_to_gguf.py wrote from the same directory -- and demands the same
// container: every Config field, and every tensor byte for byte (both are F32,
// so a permutation, a role or a dropped tensor is an inequality, not a
// tolerance). The GGUF path is gated against transformers on the host and every
// device; this is what makes the safetensors path the same answer rather than a
// second reading of the same config.json.
func TestDenseSafetensorsMatchesItsGGUF(t *testing.T) {
	compared := 0
	for _, name := range []string{"synth-smollm3", "synth-arcee", "synth-seedoss", "synth-olmo2",
		"synth-exaone4", "synth-mistral3", "synth-olmo2-gqa", "synth-olmo3", "synth-apertus"} {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(c6HFDir(), name)
			gp := testmodels.Path(name + ".gguf")
			for _, p := range []string{filepath.Join(dir, "config.json"), gp} {
				if _, err := os.Stat(p); err != nil {
					testmodels.Missing(t, "MODEL MISSING: %v (scripts/c6gold.py %s writes both; JITLLM_C6_HF names "+
						"its HuggingFace directory) -- this gate proved nothing", err, name)
				}
			}
			a, b := hfSource(t, dir), ggufSource(t, gp)
			ca, cb := reflect.ValueOf(*a.Config), reflect.ValueOf(*b.Config)
			for i := 0; i < ca.NumField(); i++ {
				x, y := ca.Field(i).Interface(), cb.Field(i).Interface()
				if !reflect.DeepEqual(x, y) {
					t.Errorf("Config.%s: safetensors %v, gguf %v", ca.Type().Field(i).Name, x, y)
				}
			}
			type key struct {
				role         jlm.Role
				block, index int32
			}
			ta := map[key]*jlm.Tensor{}
			for i := range a.Tensors {
				ta[key{a.Tensors[i].Role, a.Tensors[i].Block, a.Tensors[i].Index}] = &a.Tensors[i]
			}
			same := 0
			for i := range b.Tensors {
				e := &b.Tensors[i]
				k := key{e.Role, e.Block, e.Index}
				x, ok := ta[k]
				if !ok {
					t.Errorf("the gguf has %v block %d and safetensors does not", e.Role, e.Block)
					continue
				}
				delete(ta, k)
				switch {
				case x.Type != e.Type || x.NDim != e.NDim || x.Dims != e.Dims:
					t.Errorf("%v block %d: safetensors %v %v, gguf %v %v",
						e.Role, e.Block, x.Type, x.Dims[:x.NDim], e.Type, e.Dims[:e.NDim])
				case !bytes.Equal(x.Data, e.Data):
					t.Errorf("%v block %d: the bytes differ", e.Role, e.Block)
				default:
					same++
				}
			}
			for k := range ta {
				t.Errorf("safetensors has %v block %d and the gguf does not", k.role, k.block)
			}
			if same == 0 {
				t.Fatal("no tensor compared -- this gate proved nothing")
			}
			compared++
			if !t.Failed() {
				t.Logf("%v: every Config field and %d tensors identical", a.Config.Arch, same)
			}
		})
	}
	if compared == 0 {
		t.Skip("no pair on this box -- this gate proved nothing")
	}
}

// hfConfigOf is readHFConfig over config.json text.
func hfConfigOf(t *testing.T, js string) *hfConfig {
	t.Helper()
	c, err := readHFConfig(hfDir{fstest.MapFS{"config.json": {Data: []byte(js)}}, "test"})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// TestDenseHFClassDefaults gates the class defaults, which no fixture can:
// save_pretrained writes every key, so a fixture never has one absent. Each
// case is a config.json missing the keys whose default differs from llama's,
// against what transformers' own config class resolves them to (checked with
// <Class>Config.from_dict in the reference venv). A converter that took
// llama's default reads a different number in every row.
func TestDenseHFClassDefaults(t *testing.T) {
	const base = `"hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
		"num_attention_heads": 4, "vocab_size": 100`
	head := []string{"lm_head.weight"}
	for _, tc := range []struct {
		class string
		extra string
		names []string
		check func(c *jlm.Config) string
	}{
		{"SmolLM3ForCausalLM", ``, nil, func(c *jlm.Config) string {
			// kv 4, base 2e6, eps 1e-6, tied, NoPE every 4th layer.
			return denseOK(c.NKVHead == 4 && c.RopeBase == 2e6 && c.RMSEps == 1e-6 &&
				c.Flags.Has(jlm.FlagTiedEmbd) && c.NCtx == 32768 &&
				c.Flags.Has(jlm.FlagNoPEGlobal) && c.SWAPeriod == 4 && c.SWAWindow == 0)
		}},
		{"SmolLM3ForCausalLM", `, "no_rope_layer_interval": 2, "sliding_window": 4096`, nil, func(c *jlm.Config) string {
			// use_sliding_window defaults to false, so the window is unread.
			return denseOK(c.SWAPeriod == 2 && c.SWAWindow == 0)
		}},
		{"ArceeForCausalLM", ``, head, func(c *jlm.Config) string {
			return denseOK(c.NKVHead == 4 && c.HeadDim == 16 && c.Flags.Has(jlm.FlagReLU2) && c.NCtx == 4096 &&
				c.RMSEps == 1e-5 && !c.Flags.Has(jlm.FlagTiedEmbd))
		}},
		{"SeedOssForCausalLM", ``, head, func(c *jlm.Config) string {
			return denseOK(c.NKVHead == 8 && c.HeadDim == 128 && c.NRot == 128 && c.RMSEps == 1e-6 &&
				c.NCtx == 524288 && c.Flags.Has(jlm.FlagRopeNeox))
		}},
		{"Olmo2ForCausalLM", ``, head, func(c *jlm.Config) string {
			return denseOK(c.NKVHead == 4 && c.HeadDim == 16 && c.RopeBase == 1e4 && c.SWAWindow == 0 &&
				c.Flags.Has(jlm.FlagQKNormWide))
		}},
		{"Olmo3ForCausalLM", ``, head, func(c *jlm.Config) string {
			// Window 4096 on three layers in four, base 5e5 for both kinds.
			return denseOK(c.Arch == jlm.ArchOLMo3 && c.SWAWindow == 4096 && c.SWAPeriod == 4 &&
				c.RopeBase == 5e5 && c.RopeBaseSWA == 5e5 && !c.Flags.Has(jlm.FlagNoPEGlobal))
		}},
		{"Olmo3ForCausalLM", `, "rope_theta": 10000.0`, head, func(c *jlm.Config) string {
			// The class pops rope_theta for the full layers alone: the
			// sliding ones keep 5e5.
			return denseOK(c.Arch == jlm.ArchOLMo3 && c.RopeBase == 1e4 && c.RopeBaseSWA == 5e5)
		}},
		{"Olmo3ForCausalLM", `, "num_key_value_heads": 2, "rope_scaling": {"rope_type": "yarn", "factor": 8.0,
			"original_max_position_embeddings": 8192}`, head, func(c *jlm.Config) string {
			// YaRN on the full layers, with its magnitude; the sliding
			// layers' plain rotary is the arch's (SWARopePlain).
			return denseOK(c.Arch == jlm.ArchOLMo3 && c.NKVHead == 2 && c.YarnFactor == 8 &&
				c.YarnOrigCtx == 8192 && c.AttnFactor > 1.2 && c.AttnFactor < 1.21 && c.RopeBaseSWA == 5e5)
		}},
		{"Olmo2ForCausalLM", `, "num_key_value_heads": 2`, head, func(c *jlm.Config) string {
			return denseOK(c.Arch == jlm.ArchOLMo2 && c.NKVHead == 2 && c.Flags.Has(jlm.FlagQKNormWide))
		}},
		{"Olmo3ForCausalLM", `, "sliding_window": null`, head, func(c *jlm.Config) string {
			return denseOK(c.SWAWindow == 0 && c.SWAPeriod == 0)
		}},
		{"Exaone4ForCausalLM", `, "num_key_value_heads": 2`, head, func(c *jlm.Config) string {
			// Window 4096 on three layers in four, NoPE on the fourth.
			return denseOK(c.SWAWindow == 4096 && c.SWAPeriod == 4 && c.Flags.Has(jlm.FlagNoPEGlobal) &&
				!c.Flags.Has(jlm.FlagQKNormWide) && c.NCtx == 2048)
		}},
		{"Exaone4ForCausalLM", `, "num_key_value_heads": 2, "sliding_window": null,
			"layer_types": ["full_attention", "full_attention", "full_attention", "full_attention"]`, head,
			func(c *jlm.Config) string {
				return denseOK(c.SWAWindow == 0 && !c.Flags.Has(jlm.FlagNoPEGlobal))
			}},
		{"Ministral3ForCausalLM", `, "num_key_value_heads": 2, "rope_theta": 5.0`, head, func(c *jlm.Config) string {
			// No rope_parameters: the class's own YaRN dict, whose base is
			// 1e6 whatever rope_theta says; head_dim 128.
			return denseOK(c.HeadDim == 128 && c.RopeBase == 1e6 && c.YarnFactor == 16 &&
				c.YarnOrigCtx == 16384 && c.AttnTempScale == float32(0.1) && c.AttnTempFloor == 16384 &&
				c.AttnFactor == 1 && c.YarnLogMul == 0 && c.NCtx == 262144)
		}},
	} {
		t.Run(tc.class, func(t *testing.T) {
			c := hfConfigOf(t, fmt.Sprintf(`{"architectures": [%q], %s%s}`, tc.class, base, tc.extra))
			ha, err := hfArchOf(c)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := ha.config(c, tc.names)
			if err != nil {
				t.Fatal(err)
			}
			if why := tc.check(cfg); why != "" {
				t.Errorf("%s%s: %+v", tc.class, tc.extra, *cfg)
			}
		})
	}
}

func denseOK(ok bool) string {
	if ok {
		return ""
	}
	return "wrong"
}

// TestDenseHFRefusals: what the GGUF path refuses by name, the safetensors
// path refuses by name too.
func TestDenseHFRefusals(t *testing.T) {
	const base = `"hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
		"num_attention_heads": 4, "vocab_size": 100`
	head := []string{"lm_head.weight"}
	for _, tc := range []struct{ class, extra, want string }{
		{"Olmo3ForCausalLM", `, "num_key_value_heads": 4, "rope_parameters": {"full_attention":
			{"rope_type": "default", "rope_theta": 5e5}, "sliding_attention": {"rope_type": "yarn", "factor": 8.0,
			"original_max_position_embeddings": 8192, "rope_theta": 5e5}}`, "rotary scaling on its sliding_attention"},
		{"Olmo3ForCausalLM", `, "num_key_value_heads": 4, "rope_scaling": {"rope_type": "linear", "factor": 8.0}`,
			"rotary scaling on its full_attention"},
		{"SmolLM3ForCausalLM", `, "no_rope_layers": [1, 0, 1, 1], "tie_word_embeddings": false`, "not every"},
		{"Exaone4ForCausalLM", `, "num_key_value_heads": 2, "layer_types": ["sliding_attention",
			"full_attention", "full_attention", "sliding_attention"]`, "not periodic"},
		{"Ministral3ForCausalLM", `, "num_key_value_heads": 2, "rope_parameters": {"rope_type": "default"}`,
			"llama_4_scaling_beta"},
		{"ArceeForCausalLM", `, "hidden_act": "silu"`, "hidden_act"},
		{"SeedOssForCausalLM", `, "mlp_bias": true`, "mlp_bias"},
	} {
		c := hfConfigOf(t, fmt.Sprintf(`{"architectures": [%q], %s%s}`, tc.class, base, tc.extra))
		ha, err := hfArchOf(c)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ha.config(c, head); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s%s: %v, want a refusal naming %q", tc.class, tc.extra, err, tc.want)
		}
	}
}

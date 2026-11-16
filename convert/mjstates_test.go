package convert

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestTheLibraryStatesTheContainer holds each GGUF architecture whose
// container the library states to what configOfFile set for it by name
// before the files did: the flags, the activation, the norm epsilon read
// off its key with its default, the embedding scale off the width, the
// window and its pattern, the windowed layers' rotary base and the caps.
func TestTheLibraryStatesTheContainer(t *testing.T) {
	neox, qk := jlm.FlagRopeNeox, jlm.FlagRopeNeox|jlm.FlagQKNorm
	ln := jlm.FlagLayerNorm | jlm.FlagRopeNeox | jlm.FlagGELU
	gemma := map[string]meta.Value{"embedding_length": meta.MakeUint(2048)}
	with := func(m map[string]meta.Value, k string, v meta.Value) map[string]meta.Value {
		out := map[string]meta.Value{k: v}
		for k, v := range m {
			out[k] = v
		}
		return out
	}
	for _, tc := range []struct {
		arch        string
		keys        map[string]meta.Value
		flags       jlm.Flags
		eps, scale  float32
		win, period uint32
		ropeSWA     float32
		acap, fcap  float32
		logit       float32
	}{
		{arch: "qwen2", flags: neox},
		{arch: "qwen2vl", flags: neox},
		{arch: "qwen3", flags: qk},
		{arch: "qwen3vl", flags: qk | jlm.FlagRopeInterleaved},
		{arch: "qwen3moe", flags: qk},
		{arch: "qwen3vlmoe", flags: qk | jlm.FlagRopeInterleaved},
		{arch: "qwen35", flags: jlm.FlagRopeInterleaved},
		{arch: "qwen35moe", flags: jlm.FlagRopeInterleaved},
		{arch: "qwen3next", flags: jlm.FlagRopeInterleaved},
		{arch: "qwen4exp", flags: jlm.FlagRopeInterleaved},
		{arch: "clef", flags: jlm.FlagRopeInterleaved},
		{arch: "hunyuan_vl", flags: jlm.FlagRopeXD},
		{arch: "bert", flags: jlm.FlagNonCausal},
		{arch: "nomic-bert", flags: jlm.FlagNonCausal},
		{arch: "modern-bert", flags: jlm.FlagNonCausal},
		{arch: "olmoe", flags: qk | jlm.FlagQKNormWide | jlm.FlagNoExpertNorm},
		{arch: "bitnet-b1.58", flags: neox | jlm.FlagReLU2},
		{arch: "gemma", keys: map[string]meta.Value{"embedding_length": meta.MakeUint(2048)},
			flags: neox | jlm.FlagGELU, scale: float32(math.Sqrt(2048))},
		{arch: "starcoder2", flags: ln, eps: 1e-5},
		{arch: "starcoder2", keys: map[string]meta.Value{"attention.layer_norm_epsilon": meta.MakeFloat(1e-6)},
			flags: ln, eps: 1e-6},
		{arch: "falcon", keys: map[string]meta.Value{"attention.layer_norm_epsilon": meta.MakeFloat(3e-7)},
			flags: ln | jlm.FlagParallel, eps: 3e-7},
		{arch: "nemotron", flags: jlm.FlagLayerNorm | neox | jlm.FlagReLU2, eps: 1e-5},
		{arch: "phi2", flags: ln | jlm.FlagParallel, eps: 1e-5},
		{arch: "command-r", keys: map[string]meta.Value{"logit_scale": meta.MakeFloat(0.0625)},
			flags: jlm.FlagLayerNorm | jlm.FlagParallel, eps: 1e-5, logit: 16},
		{arch: "command-r", keys: map[string]meta.Value{"logit_scale": meta.MakeFloat(1)},
			flags: jlm.FlagLayerNorm | jlm.FlagParallel, eps: 1e-5},
		{arch: "cohere2", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(4096), "logit_scale": meta.MakeFloat(0.25)},
			flags: jlm.FlagLayerNorm | jlm.FlagParallel | jlm.FlagNoPEGlobal, eps: 1e-5, logit: 4, win: 4096, period: 4},
		{arch: "cohere2", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(4096), "rope.freq_base_swa": meta.MakeFloat(1e4),
			"rope.freq_base": meta.MakeFloat(5e4)},
			flags: jlm.FlagLayerNorm | jlm.FlagParallel | jlm.FlagNoPEGlobal, eps: 1e-5, win: 4096, period: 4, ropeSWA: 1e4},
		{arch: "gemma2", keys: gemma, flags: neox | jlm.FlagGELU, scale: float32(math.Sqrt(2048)),
			win: 4096, period: 2, ropeSWA: 10000},
		{arch: "gemma2", keys: with(with(with(gemma, "rope.freq_base", meta.MakeFloat(5e5)), "attn_logit_softcapping",
			meta.MakeFloat(50)), "final_logit_softcapping", meta.MakeFloat(30)),
			flags: neox | jlm.FlagGELU, scale: float32(math.Sqrt(2048)), win: 4096, period: 2, ropeSWA: 5e5, acap: 50, fcap: 30},
		{arch: "gemma3", keys: gemma, flags: qk | jlm.FlagGELU, scale: float32(math.Sqrt(2048))},
		{arch: "gemma3", keys: with(with(gemma, "attention.sliding_window", meta.MakeUint(512)),
			"attention.sliding_window_pattern", meta.MakeUint(4)),
			flags: qk | jlm.FlagGELU, scale: float32(math.Sqrt(2048)), win: 512, period: 4, ropeSWA: 10000},
		{arch: "gpt-oss", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(128)},
			flags: neox | jlm.FlagSwiGLUOAI | jlm.FlagYarnExact, win: 128, period: 2},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString(tc.arch)}
		for k, v := range tc.keys {
			kv[tc.arch+"."+k] = v
		}
		c := &Config{Config: &jlm.Config{}}
		if err := stateContainer(meta.New(tc.arch+".gguf", kv, nil), c); err != nil {
			t.Fatalf("%s: %v", tc.arch, err)
		}
		if c.Flags != tc.flags || c.RMSEps != tc.eps || c.EmbdScale != tc.scale {
			t.Errorf("%s: flags %v, eps %g, scale %g; want %v, %g, %g", tc.arch, c.Flags, c.RMSEps, c.EmbdScale,
				tc.flags, tc.eps, tc.scale)
		}
		if c.LogitScale != tc.logit {
			t.Errorf("%s: logits divided by %g, want %g", tc.arch, c.LogitScale, tc.logit)
		}
		if c.SWAWindow != tc.win || c.SWAPeriod != tc.period || c.RopeBaseSWA != tc.ropeSWA ||
			c.AttnSoftcap != tc.acap || c.FinalSoftcap != tc.fcap {
			t.Errorf("%s: window %d every %d at base %g, caps %g %g; want %d every %d at %g, %g %g", tc.arch,
				c.SWAWindow, c.SWAPeriod, c.RopeBaseSWA, c.AttnSoftcap, c.FinalSoftcap,
				tc.win, tc.period, tc.ropeSWA, tc.acap, tc.fcap)
		}
	}
	kv := map[string]meta.Value{"general.architecture": meta.MakeString("command-r"), "command-r.logit_scale": meta.MakeFloat(-2)}
	if err := stateContainer(meta.New("command-r.gguf", kv, nil), &Config{Config: &jlm.Config{}}); err == nil ||
		!strings.Contains(err.Error(), "logit_div is -0.5") {
		t.Errorf("a negative logit scale: %v, want it refused", err)
	}
	for want, keys := range map[string]map[string]meta.Value{
		"is undefined":                {},
		"no attention.sliding_window": {"attention.sliding_window": meta.MakeUint(0)},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString("gpt-oss")}
		for k, v := range keys {
			kv["gpt-oss."+k] = v
		}
		err := stateContainer(meta.New("gpt-oss.gguf", kv, nil), &Config{Config: &jlm.Config{}})
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("gpt-oss with %v: %v, want an error naming %q", keys, err, want)
		}
	}
}

// TestContainerStatementsAreChecked refuses a statement for an architecture
// the file does not name, a flag no reader states, an activation that is
// not the container's, and an expression reading anything but the keys; a
// statement of the wrong type fails where it is read.
func TestContainerStatementsAreChecked(t *testing.T) {
	file := func(states string) string {
		return `{"gguf": ["x"], "container": {` + states + `}}`
	}
	for _, tc := range []struct{ states, want string }{
		{`"y": {}`, "names no GGUF architecture"},
		{`"x": {"flags": {"rope_sideways": "true"}}`, "is no graph feature"},
		{`"x": {"activation": "gelu_fast"}`, "is not one the container states"},
		{`"x": {"router": {"score": "softmax"}}`, "router {Score:softmax Weight:} is not one"},
		{`"x": {"router": {"weight": "softmax"}}`, "is not one the container states"},
		{`"x": {"activation": "xielu"}`, "carries its parameters"},
		{`"x": {"norm_eps": "eps | default(1e-5)"}`, "is not the keys"},
		{`"x": {"norm_eps": "src[["}`, "jinja expression"},
		{`"x": {"window": {"period": "2"}}`, "states its size"},
		{`"x": {"window": {"size": "8", "period": "2", "pattern": "2"}}`, "not both"},
		{`"x": {"revision": {"act_quant": "int4"}}`, "activation quantization \"int4\" is not one"},
		{`"x": {"revision": {"rope_scale_swa": "scale"}}`, "is not the keys"},
	} {
		_, err := parseModelJSON([]byte(file(tc.states)), "test")
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error naming %q", tc.states, err, tc.want)
		}
	}
	f := meta.New("x.gguf", map[string]meta.Value{"general.architecture": meta.MakeString("x"),
		"x.heads": meta.MakeUint(8)}, nil)
	for _, tc := range []struct {
		st   mjStates
		want string
	}{
		{mjStates{Flags: map[string]string{"qk_norm": "src['heads']"}}, "not true or false"},
		{mjStates{NormEps: "'small'"}, "not a number"},
		{mjStates{EmbdScale: "src['width']"}, "is undefined"},
		{mjStates{Flags: map[string]string{"post_norm": "true"}}, "not implemented"},
		{mjStates{Revision: &mjStatesRevision{SWAGlobalFirst: "src['heads']"}}, "swa_global_first is 8, not true or false"},
		{mjStates{Revision: &mjStatesRevision{RopeScaleSWA: "'one'"}}, "rope_scale_swa is"},
		{mjStates{Revision: &mjStatesRevision{PictureBidirSliding: "src['heads'] + 1"}}, "picture_bidir_sliding is 9"},
	} {
		srcs := []string{tc.st.Flags["qk_norm"], tc.st.Flags["post_norm"], tc.st.NormEps, tc.st.EmbdScale}
		if r := tc.st.Revision; r != nil {
			srcs = append(srcs, r.SWAGlobalFirst, r.RopeScaleSWA, r.PictureBidirSliding)
		}
		if err := tc.st.compile(srcs); err != nil {
			t.Fatal(err)
		}
		err := tc.st.apply("test", f, &Config{Config: &jlm.Config{}})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%+v: %v, want an error naming %q", tc.st, err, tc.want)
		}
	}
}

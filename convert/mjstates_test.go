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
// off its key with its default, and the embedding scale off the width.
func TestTheLibraryStatesTheContainer(t *testing.T) {
	neox, qk := jlm.FlagRopeNeox, jlm.FlagRopeNeox|jlm.FlagQKNorm
	ln := jlm.FlagLayerNorm | jlm.FlagRopeNeox | jlm.FlagGELU
	for _, tc := range []struct {
		arch  string
		keys  map[string]meta.Value
		flags jlm.Flags
		eps   float32
		scale float32
	}{
		{arch: "qwen2", flags: neox},
		{arch: "qwen2vl", flags: neox},
		{arch: "qwen3", flags: qk},
		{arch: "qwen3vl", flags: qk},
		{arch: "qwen3moe", flags: qk},
		{arch: "qwen3vlmoe", flags: qk},
		{arch: "olmoe", flags: qk | jlm.FlagQKNormWide | jlm.FlagNoExpertNorm},
		{arch: "bitnet-b1.58", flags: neox | jlm.FlagReLU2},
		{arch: "gemma", keys: map[string]meta.Value{"embedding_length": meta.MakeUint(2048)},
			flags: neox | jlm.FlagGELU, scale: float32(math.Sqrt(2048))},
		{arch: "starcoder2", flags: ln, eps: 1e-5},
		{arch: "starcoder2", keys: map[string]meta.Value{"attention.layer_norm_epsilon": meta.MakeFloat(1e-6)},
			flags: ln, eps: 1e-6},
		{arch: "falcon", keys: map[string]meta.Value{"attention.layer_norm_epsilon": meta.MakeFloat(3e-7)},
			flags: ln | jlm.FlagParallel, eps: 3e-7},
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
		{`"x": {"activation": "gelu"}`, "is not one the container states"},
		{`"x": {"activation": "xielu"}`, "carries its parameters"},
		{`"x": {"norm_eps": "eps | default(1e-5)"}`, "is not the keys"},
		{`"x": {"norm_eps": "src[["}`, "jinja expression"},
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
	} {
		if err := tc.st.compile([]string{tc.st.Flags["qk_norm"], tc.st.Flags["post_norm"], tc.st.NormEps, tc.st.EmbdScale}); err != nil {
			t.Fatal(err)
		}
		err := tc.st.apply("test", f, &Config{Config: &jlm.Config{}})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%+v: %v, want an error naming %q", tc.st, err, tc.want)
		}
	}
}

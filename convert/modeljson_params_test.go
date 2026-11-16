package convert

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestModelJSONSoftplus: a map's softplus is torch's, log(1 + e^x) and x
// itself past 20 (800 would overflow the logarithm's form), and a function
// no map calls is refused.
func TestModelJSONSoftplus(t *testing.T) {
	env := mjEnv(map[string]any{})
	v, err := mjRun(t, env, `[{"map": "softplus(x)"}]`, mjF32([]uint64{4}, -30, 0.5, 20, 800))
	if err != nil {
		t.Fatal(err)
	}
	sp := func(x float64) float32 { return float32(math.Log1p(math.Exp(x))) }
	want := []float32{sp(-30), sp(0.5), sp(20), 800}
	if got := f32s(v.data); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("softplus: %v, want %v", got, want)
	}
	if _, err := mjRun(t, env, `[{"map": "softplu(x)"}]`, mjF32([]uint64{1}, 1)); err == nil {
		t.Error("a map called softplu")
	}
}

// xieluGather is a gather of the four xIELU numbers a block carries, folded
// as XIELUActivation computes them, written with operands of their own.
const xieluGather = `{"role": "ffn_xielu", "from": [
	{"from": ["a"], "do": [{"map": "softplus(x)"}]},
	{"from": ["b", "c"], "do": [{"map": "y + softplus(x)"}]},
	"c", "d"], "do": [{"concat": 0}]}`

// TestModelJSONGatherOperands: a gather's operand may be a tensor composed of
// sources by primitives of its own, each rounded once; the result is named
// as the container names the role, and here is the GGUF path's own fold of
// the same numbers, byte for byte. A composed operand without its
// primitives is refused, and so is a block missing a source.
func TestModelJSONGatherOperands(t *testing.T) {
	var g mjGather
	if err := json.Unmarshal([]byte(xieluGather), &g); err != nil {
		t.Fatal(err)
	}
	step := hfGatherStep{role: jlm.RoleXIELU, from: g.From, do: g.Do}
	vals := map[string]float32{
		"m.0.a": 0.5, "m.0.b": 0.25, "m.0.c": 0.5, "m.0.d": -1e-6,
		"m.1.a": -1, "m.1.b": 30, "m.1.c": 0.4, "m.1.d": -0.1,
	}
	read := func(n string) ([]float32, error) {
		v, ok := vals[n]
		if !ok {
			return nil, fmt.Errorf("no %s", n)
		}
		return []float32{v}, nil
	}
	names := func() []string {
		var ns []string
		for n := range vals {
			ns = append(ns, n)
		}
		sort.Strings(ns)
		return ns
	}
	out, used, err := gatherBlocks(mjEnv(map[string]any{}), "m.", []hfGatherStep{step}, names(), read, nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || len(used) != len(vals) {
		t.Fatalf("%d tensors and %d sources used, want 2 and %d", len(out), len(used), len(vals))
	}
	for _, e := range out {
		b := fmt.Sprint(e.Block)
		f := func(k string) float64 { return float64(vals["m."+b+"."+k]) }
		want := xieluFold(int(e.Block), f("a"), f("b"), f("c"), f("d"))
		if e.Name != want.Name || e.Role != want.Role || e.NDim != 1 || e.Dims != want.Dims || e.Type != want.Type ||
			string(e.Data) != string(want.Data) {
			t.Errorf("block %s: %s %v %v %v, want the GGUF fold's %s %v %v %v",
				b, e.Name, e.Dims[:e.NDim], e.Type, f32s(e.Data), want.Name, want.Dims[:1], want.Type, f32s(want.Data))
		}
	}
	var bad mjGather
	if err := json.Unmarshal([]byte(strings.Replace(xieluGather, `, "do": [{"map": "y + softplus(x)"}]`, ``, 1)), &bad); err == nil {
		t.Error("a composed operand without its primitives parsed")
	}
	var lone mjGather
	if err := json.Unmarshal([]byte(strings.Replace(xieluGather, `, "do": [{"map": "softplus(x)"}]`, `, "keep": true`, 1)), &lone); err != nil {
		t.Errorf("one source read and kept, with no primitives, did not parse: %v", err)
	}
	delete(vals, "m.1.c")
	if _, _, err := gatherBlocks(mjEnv(map[string]any{}), "m.", []hfGatherStep{step}, names(), read, nil, 2); err == nil {
		t.Error("a block without one of its sources gathered")
	}
}

// apertusLlama3 is llama 3's rope scaling as an Apertus config.json states it.
const apertusLlama3 = `{"rope_type": "llama3", "rope_theta": 10000, "factor": 8, "low_freq_factor": 1,
	"high_freq_factor": 4, "original_max_position_embeddings": 64}`

// apertusTestConfig is an ApertusForCausalLM config.json of two blocks with
// the rope parameters given; extra is added to it.
func apertusTestConfig(t *testing.T, rope, extra string) *hfConfig {
	return hfConfigOf(t, `{"architectures": ["ApertusForCausalLM"], "vocab_size": 100, "hidden_size": 64,
	  "intermediate_size": 96, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
	  "max_position_embeddings": 512, "rope_parameters": `+rope+extra+`}`)
}

// TestActivationParametersAndLlama3: through convert/models/apertus.json,
// the rope scaling statement writes llama 3's factors as the rope_freqs
// vector; qk_norm is a stated flag and post_norm one refused by name; an
// activation whose parameters a block carries finds them in every block, and
// a file stating another activation is refused when the blocks carry them.
func TestActivationParametersAndLlama3(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib["ApertusForCausalLM"]
	if mj == nil {
		t.Fatal("no library file converts ApertusForCausalLM")
	}
	run := func(rope, extra string, blocks int) (*Config, *hfConfig, error) {
		a, err := mj.hfArch()
		if err != nil {
			t.Fatal(err)
		}
		hc := apertusTestConfig(t, rope, extra)
		cfg, err := a.config(hc, []string{"lm_head.weight"})
		if err != nil {
			return nil, hc, err
		}
		vals := map[string]float32{}
		var names []string
		for b := range blocks {
			for i, k := range []string{"alpha_p", "alpha_n", "beta", "eps"} {
				n := "model.layers." + itoa(b) + ".mlp.act_fn." + k
				vals[n] = float32(i+1) / 8
				names = append(names, n)
			}
		}
		sort.Strings(names)
		read := func(n string) ([]float32, error) { return []float32{vals[n]}, nil }
		out, _, err := a.gather(cfg, names, read)
		if err == nil && len(out) != blocks {
			t.Fatalf("%d xIELU vectors for %d blocks", len(out), blocks)
		}
		return cfg, hc, err
	}
	cfg, hc, err := run(apertusLlama3, "", 2)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Flags.Has(jlm.FlagQKNorm) || !cfg.Flags.Has(jlm.FlagRopeNeox) || cfg.Flags&mjActivationFlags != 0 {
		t.Errorf("flags %v", cfg.Flags)
	}
	want := llama3Freqs(16, 10000, 8, 1, 4, 64)
	if len(hc.extra) != 1 || hc.extra[0].Role != jlm.RoleRopeFreqs || string(hc.extra[0].Data) != string(f32AsBytes(want)) {
		t.Errorf("rope_freqs: %d extra tensors, want one holding %v", len(hc.extra), want)
	}
	if cfg, _, err := run(apertusLlama3, `, "qk_norm": false`, 2); err != nil || cfg.Flags.Has(jlm.FlagQKNorm) {
		t.Errorf("qk_norm false: %v", err)
	}
	for _, r := range []struct{ name, rope, extra, err string }{
		{"post_norm", apertusLlama3, `, "post_norm": true`, "post_norm is true"},
		{"silu with xIELU numbers", apertusLlama3, `, "hidden_act": "silu"`, "the parameters of activation xielu"},
	} {
		if _, _, err := run(r.rope, r.extra, 2); err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		}
	}
	if _, _, err := run(apertusLlama3, "", 1); err == nil || !strings.Contains(err.Error(), "block 1 carries no ffn_xielu") {
		t.Errorf("a block without its xIELU numbers: %v", err)
	}
}

// xieluFold is block b's RoleXIELU vector as XIELUActivation computes it each
// call: alpha_p = softplus(a_p), alpha_n = beta + softplus(a_n), torch's
// softplus (x past 20 is itself), in binary64, each stored as binary32.
func xieluFold(b int, aP, aN, beta, eps float64) jlm.Tensor {
	sp := func(x float64) float64 {
		if x > 20 {
			return x
		}
		return math.Log1p(math.Exp(x))
	}
	p := [4]float32{float32(sp(aP)), float32(beta + sp(aN)), float32(beta), float32(eps)}
	t := jlm.Tensor{Role: jlm.RoleXIELU, Block: int32(b), Index: -1, Type: jlm.TypeF32, NDim: 1,
		Data: f32AsBytes(p[:]), Name: "blk." + itoa(b) + ".ffn_xielu"}
	t.Dims[0] = 4
	return t
}

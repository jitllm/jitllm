package convert

import (
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestDS4ClampIsTheFiles holds DeepSeek V4's SwiGLU clamp to the limits the
// file states, block by block: a file clamping its blocks at 7, 10 and 12
// states each block's activation at its own limit, one stating none
// transformers' 10; routed and shared limits that differ are refused.
func TestDS4ClampIsTheFiles(t *testing.T) {
	file := func(kv map[string]meta.Value) *meta.File {
		kv["general.architecture"] = meta.MakeString("deepseek4")
		return meta.New("ds4.gguf", kv, nil)
	}
	clamped := func(lim float32) jlm.Expr {
		var e expr
		e.Fn(jlm.FnMul, e.SiLU(e.Fn(jlm.FnMin, operand(0), num(lim))), e.clamp(operand(1), lim))
		return jlm.Expr{Terms: e.T}
	}
	for _, c := range []struct {
		name string
		kv   map[string]meta.Value
		want []float32
	}{
		{"per block", map[string]meta.Value{
			"deepseek4.swiglu_clamp_exp":   meta.MakeFloat32s([]float32{7, 10, 12}),
			"deepseek4.swiglu_clamp_shexp": meta.MakeFloat32s([]float32{7, 10, 12})}, []float32{7, 10, 12}},
		{"one value for every block", map[string]meta.Value{"deepseek4.swiglu_clamp_exp": meta.MakeFloat(8)},
			[]float32{8, 8, 8}},
		{"none stated", map[string]meta.Value{}, []float32{10, 10, 10}},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := Unstated(&jlm.Config{NLayer: 3, Flags2: jlm.Flag2SwiGLUClamp})
			if err := swigluClamps(file(c.kv), cfg); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(cfg.SwiGLULimits, c.want) {
				t.Fatalf("limits %v, want %v", cfg.SwiGLULimits, c.want)
			}
			for li, lim := range c.want {
				got, err := Activation(cfg, li, true, false)
				if err != nil {
					t.Fatal(err)
				}
				if want := clamped(lim); !reflect.DeepEqual(got.Terms, want.Terms) {
					t.Fatalf("block %d's activation is %+v, want the SwiGLU clamped at %g", li, got, lim)
				}
			}
		})
	}
	if reflect.DeepEqual(clamped(7).Terms, clamped(10).Terms) {
		t.Fatal("the clamped SwiGLU does not depend on its limit")
	}
	for _, c := range []struct {
		name, want string
		kv         map[string]meta.Value
	}{
		{"routed and shared differ", "routed", map[string]meta.Value{
			"deepseek4.swiglu_clamp_exp":   meta.MakeFloat32s([]float32{10, 10, 10}),
			"deepseek4.swiglu_clamp_shexp": meta.MakeFloat32s([]float32{10, 9, 10})}},
		{"a list that is not a block each", "limits for", map[string]meta.Value{
			"deepseek4.swiglu_clamp_exp": meta.MakeFloat32s([]float32{10, 10})}},
		{"a zero clamp", "positive", map[string]meta.Value{
			"deepseek4.swiglu_clamp_exp": meta.MakeFloat32s([]float32{10, 0, 10})}},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := Unstated(&jlm.Config{NLayer: 3})
			if err := swigluClamps(file(c.kv), cfg); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("got %v, want a refusal naming %q", err, c.want)
			}
		})
	}
}

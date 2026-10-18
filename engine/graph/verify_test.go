package graph

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// llama and qwen3 as their containers state them, the inventory written out
// role by role rather than derived from Dense, so the verifier is held to
// something Build did not produce.
func stated(qk bool) Facts {
	d := Dense{Block: 3, NEmbd: 64, NHead: 4, NKVHead: 2, HeadDim: 16, NFFN: 96, Eps: 1e-6, NRot: 16,
		Base: 1e6, Layout: RopeNorm, Scale: 0.25, Act: ActSiLU}
	inv := map[jlm.Role]Shape{
		jlm.RoleAttnNorm: {64, 0, 0}, jlm.RoleAttnQ: {64, 64, 0}, jlm.RoleAttnK: {64, 32, 0}, jlm.RoleAttnV: {64, 32, 0},
		jlm.RoleAttnOut: {64, 64, 0}, jlm.RoleFFNNorm: {64, 0, 0}, jlm.RoleFFNGate: {64, 96, 0}, jlm.RoleFFNUp: {64, 96, 0},
		jlm.RoleFFNDown: {96, 64, 0},
	}
	if qk {
		d.QKNorm, d.Layout = true, RopeNeox
		inv[jlm.RoleAttnQNorm], inv[jlm.RoleAttnKNorm] = Shape{16, 0, 0}, Shape{16, 0, 0}
	}
	return Facts{Config: d, Inventory: inv}
}

func TestDenseGraphsVerify(t *testing.T) {
	for _, qk := range []bool{false, true} {
		f := stated(qk)
		g := f.Config.Build()
		if err := Verify(g, f); err != nil {
			t.Fatalf("qk=%v: %v\n%s", qk, err, g)
		}
		heads := 0
		for _, o := range g.Ops {
			if o.Kind == OpHeadNorm {
				heads++
			}
		}
		if want := map[bool]int{false: 0, true: 2}[qk]; heads != want {
			t.Fatalf("qk=%v: %d head norms, want %d", qk, heads, want)
		}
	}
}

// Each violation the design names must fail, with its reason (RULE 10).
func TestVerifierRefusesWrongGraphs(t *testing.T) {
	cases := []struct {
		name string
		qk   bool
		mut  func(g *Graph, f *Facts)
		want string
	}{
		{"missing q/k norm", true, func(g *Graph, f *Facts) {
			d := f.Config
			d.QKNorm = false
			*g = *d.Build()
		}, "attn_q_norm, which no op binds"},
		{"rotary layout changed", true, func(g *Graph, f *Facts) {
			for i := range g.Ops {
				if g.Ops[i].Kind == OpRope {
					g.Ops[i].Attr.Layout = RopeNorm
				}
			}
		}, "the configuration states neox"},
		{"no kv write", false, func(g *Graph, f *Facts) {
			ops := g.Ops[:0]
			for _, o := range g.Ops {
				if o.Kind != OpKVAppend {
					ops = append(ops, o)
				}
			}
			g.Ops = ops
		}, "reads kv@1, but kv@0"},
		{"attention reads the history before the write", false, func(g *Graph, f *Facts) {
			for i := range g.Ops {
				if g.Ops[i].Kind == OpAttend {
					g.Ops[i].Reads[0].Version = 0
				}
			}
		}, "never in the history"},
		{"gate bound to up's tensor", false, func(g *Graph, f *Facts) {
			for i := range g.Ops {
				if len(g.Ops[i].Weight) > 0 && g.Ops[i].Weight[0].Role == jlm.RoleFFNGate {
					g.Ops[i].Weight[0].Role = jlm.RoleFFNUp
				}
			}
		}, "ffn_gate, which no op binds"},
		{"binding of the wrong shape", false, func(g *Graph, f *Facts) {
			f.Inventory[jlm.RoleAttnK] = Shape{64, 64, 0}
		}, "stored 64x64"},
		{"epsilon not the stated one", false, func(g *Graph, f *Facts) {
			g.Ops[0].Attr.Eps = 1e-5
		}, "epsilon"},
		{"unknown version", false, func(g *Graph, f *Facts) { g.Version = 2 }, "v2"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := stated(tc.qk)
			g := f.Config.Build()
			if err := Verify(g, f); err != nil {
				t.Fatalf("unmutated: %v", err)
			}
			tc.mut(g, &f)
			err := Verify(g, f)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

// classic is a parallel C6 block with a biased LayerNorm, biased projections,
// an ungated GELU FFN, a clamp, sinks and a window, stated role by role.
func classic() Facts {
	d := Dense{Block: 1, NEmbd: 64, NHead: 4, NKVHead: 2, HeadDim: 16, NFFN: 96, Eps: 1e-5, NRot: 8,
		Base: 1e4, Layout: RopeNeox, Scale: 0.25, Act: ActGELUTanh, LayerNorm: true, NormBias: true,
		Parallel: true, SharedNorm: true, OutBias: true, UpBias: true, DownBias: true, Ungated: true,
		Clamp: 8, Sinks: true, Window: 32, RopeFreqs: true, Mscale: 1.25, TempScale: 0.1, TempFloor: 8192}
	inv := map[jlm.Role]Shape{
		jlm.RoleAttnNorm: {64, 0, 0}, jlm.RoleAttnNormBias: {64, 0, 0}, jlm.RoleAttnQ: {64, 64, 0},
		jlm.RoleAttnK: {64, 32, 0}, jlm.RoleAttnV: {64, 32, 0}, jlm.RoleAttnOut: {64, 64, 0},
		jlm.RoleAttnOutBias: {64, 0, 0}, jlm.RoleFFNUp: {64, 96, 0}, jlm.RoleFFNUpBias: {96, 0, 0},
		jlm.RoleFFNDown: {96, 64, 0}, jlm.RoleFFNDownBias: {64, 0, 0}, jlm.RoleAttnSinks: {4, 0, 0},
	}
	return Facts{Config: d, Inventory: inv}
}

// The classic block's ops verify, and each of its facts misstated fails.
func TestClassicBlockVerifies(t *testing.T) {
	f := classic()
	if err := Verify(f.Config.Build(), f); err != nil {
		t.Fatalf("%v\n%s", err, f.Config.Build())
	}
	for _, c := range []struct {
		name string
		mut  func(d *Dense)
		want string
	}{
		{"no norm bias", func(d *Dense) { d.NormBias = false }, "attn_norm.bias, which no op binds"},
		{"no sinks", func(d *Dense) { d.Sinks = false }, "attn_sinks, which no op binds"},
		{"gated", func(d *Dense) { d.Ungated = false }, "ffn_gate"},
		{"no out bias", func(d *Dense) { d.OutBias = false }, "attn_out.bias, which no op binds"},
	} {
		d := f.Config
		c.mut(&d)
		err := Verify(d.Build(), f)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: %v, want %q", c.name, err, c.want)
		}
	}
	// The graph Build wrote, held to facts that state another bound, window
	// or scale.
	for _, c := range []struct {
		name string
		mut  func(d *Dense)
		want string
	}{
		{"clamp", func(d *Dense) { d.Clamp = 4 }, "bound 8"},
		{"window", func(d *Dense) { d.Window = 0 }, "window 32"},
		{"temperature", func(d *Dense) { d.TempScale = 0 }, "window 32"},
		{"mscale", func(d *Dense) { d.Mscale = 0 }, "pairs over"},
	} {
		g := f.Config.Build()
		ff := f
		c.mut(&ff.Config)
		err := Verify(g, ff)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: %v, want %q", c.name, err, c.want)
		}
	}
}

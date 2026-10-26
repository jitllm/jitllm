package jit_test

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// llamaBlock is a Llama-3.2-1B-shaped block: the attention half (a norm, q,
// k and v, the rotary on q and k, the history appended and attended, the
// output and the residual) and the MLP.
func llamaBlock() *graph.Graph { return blockOf(2048, 32, 8, 64, 8192) }

// blockOf is llamaBlock at a width of embd, heads query heads and kv key
// heads of dim, and an MLP of ffn.
func blockOf(embd, heads, kv, dim, ffn int) *graph.Graph {
	b := graph.New()
	b.G.Block = 0
	x := b.In("x", embd)
	n := b.Norm(x, jlm.RoleAttnNorm, 1e-5)
	q := b.MatVec(n, jlm.RoleAttnQ, heads*dim)
	k := b.MatVec(n, jlm.RoleAttnK, kv*dim)
	v := b.MatVec(n, jlm.RoleAttnV, kv*dim)
	rope := graph.Attrs{HeadDim: dim, NRot: dim, Layout: graph.RopeNeox}
	qr := b.Emit(graph.OpRope, []graph.ValueID{q}, []graph.Result{{Name: "qr", Width: heads * dim}}, nil, rope, nil, nil)[0]
	kr := b.Emit(graph.OpRope, []graph.ValueID{k}, []graph.Result{{Name: "kr", Width: kv * dim}}, nil, rope, nil, nil)[0]
	h := b.KVAppend(kr, v, 0)
	a := b.Attend(qr, h, graph.Attrs{NHead: heads, NKVHead: kv, HeadDim: dim, Scale: 0.125})
	x2 := b.Add(x, b.MatVec(a, jlm.RoleAttnOut, embd))
	m := b.Norm(x2, jlm.RoleFFNNorm, 1e-5)
	gate := b.MatVec(m, jlm.RoleFFNGate, ffn)
	up := b.MatVec(m, jlm.RoleFFNUp, ffn)
	b.Out(b.Add(x2, b.MatVec(siluMul(b, gate, up, ffn), jlm.RoleFFNDown, embd)))
	return b.G
}

// TestCapacityFormsHoldTheCallsRows forms a Llama-shaped block at a
// capacity of rows on a device of CUDA's statement and one of Vulkan's,
// with int8 and with float activations, at 64 and at 128 rows: every kernel
// whose launch grows with the capacity -- its rows' work -- is held to the
// call's rows (Code.Live), its launch at a call of one row no wider than at
// the capacity's and a call of the whole capacity launching the capacity's;
// every kernel writing the history is held, so a call's stale rows store
// nothing. A form of one row is decode's and holds none. A form whose
// held kernels lose their hold (the pass off) fails.
func TestCapacityFormsHoldTheCallsRows(t *testing.T) {
	cuda := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, Regs: 32}
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp} {
		formats[r] = jlm.TypeQ4S
	}
	formats[jlm.RoleFFNDown] = jlm.TypeQ6S
	formats[jlm.RoleAttnNorm], formats[jlm.RoleFFNNorm] = jlm.TypeF32, jlm.TypeF32
	form := func(reg *jit.Registry, d jit.Device, R int) []jit.Code {
		t.Helper()
		_, codes, err := reg.Form(jit.Request{Graph: llamaBlock(), Formats: formats,
			Step: jit.Step{Rows: R, Page: 64, Keys: 1024, ActWindow: 32, Seqs: 1}, Device: d,
			History: jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: true}}})
		if err != nil {
			t.Fatalf("at %d rows: %v", R, err)
		}
		return codes
	}
	history := func(c jit.Code) bool {
		for _, b := range c.Binds {
			if b.Writes && b.Operand.Kind == jit.OperandTable && (b.Operand.Name == "kv.k" || b.Operand.Name == "kv.v") {
				return true
			}
		}
		return false
	}
	check := func(reg *jit.Registry, d jit.Device) error {
		for _, c := range form(reg, d, 1) {
			if c.Live != 0 {
				return fmt.Errorf("decode's %s holds %d rows", c.Name, c.Live)
			}
		}
		a, b := form(reg, d, 64), form(reg, d, 128)
		if len(a) != len(b) {
			return fmt.Errorf("%d kernels at 64 rows, %d at 128", len(a), len(b))
		}
		held := 0
		for i := range a {
			grows := b[i].Groups > a[i].Groups
			for _, c := range []jit.Code{a[i], b[i]} {
				switch {
				case (grows || history(c)) && c.Live == 0:
					return fmt.Errorf("%s %v at %d groups is not held to the call's rows", c.Name, c.Passes, c.Groups)
				case c.Live > 0 && (c.LiveGroups(c.Live) != c.Groups || c.LiveGroups(1) > c.Groups || c.LiveGroups(1) < 1):
					return fmt.Errorf("%s launches %d groups at one row, %d at its %d, of %d", c.Name, c.LiveGroups(1),
						c.LiveGroups(c.Live), c.Live, c.Groups)
				}
			}
			// A held axis that grows is the rows': it doubles with the capacity.
			if grows && a[i].Live > 0 && (b[i].LiveUnits != 2*a[i].LiveUnits || b[i].LiveInner != a[i].LiveInner) {
				return fmt.Errorf("%s holds an axis of %d units of %d at 64 rows, %d of %d at 128: not the rows'", a[i].Name,
					a[i].LiveUnits, a[i].LiveInner, b[i].LiveUnits, b[i].LiveInner)
			}
			if grows {
				held++
			}
		}
		if held == 0 {
			return fmt.Errorf("no kernel grows with the rows: the gate selected nothing")
		}
		t.Logf("units %d: %d kernels, %d held to the call's rows", d.Units, len(b), held)
		return nil
	}
	for _, d := range []jit.Device{cuda, {Kind: jit.DeviceGPU, Lanes: 32}} {
		for _, float := range []bool{false, true} {
			reg := jit.Default()
			if float {
				reg = reg.WithoutInt8(jit.DeviceGPU)
			}
			if err := check(reg, d); err != nil {
				t.Fatalf("units %d, float %v: %v", d.Units, float, err)
			}
		}
	}
	reg := jit.Default()
	p := reg.Profiles[jit.DeviceGPU]
	p.Off = map[string]bool{"callrows": true}
	reg.Profiles[jit.DeviceGPU] = p
	if err := check(reg, cuda); err == nil {
		t.Fatal("with the call's rows unheld the gate passes: it cannot see it")
	} else {
		t.Logf("with the call's rows unheld: %v", err)
	}
}

// TestCPURowsFormsHoldTheCallsRows forms a stories15M-shaped block on the CPU
// at 2 and at 64 rows, with Q8 and with Q4 weights, its kernels fused and
// with no kernel joined to another (as a part the registers refused is
// formed): every kernel of a form of several rows runs the units of its
// calls' leading axis a call's rows reach (Code.Live), as the CPU's call
// site asks of every kernel of a rows form. A form whose kernels lose their
// hold (rowcalls off) fails.
func TestCPURowsFormsHoldTheCallsRows(t *testing.T) {
	d := jit.Device{Kind: jit.DeviceCPU, Slots: 6, Lanes: 8, ExactDiv: true}
	check := func(reg *jit.Registry, typ jlm.Type, R int) (int, error) {
		formats := map[jlm.Role]jlm.Type{jlm.RoleAttnNorm: jlm.TypeF32, jlm.RoleFFNNorm: jlm.TypeF32}
		for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp, jlm.RoleFFNDown} {
			formats[r] = typ
		}
		_, codes, err := reg.Form(jit.Request{Graph: blockOf(288, 6, 6, 48, 768), Formats: formats,
			Step: jit.Step{Rows: R, Page: 64, Keys: 256, ActWindow: 32, Seqs: 1}, Device: d})
		if err != nil {
			return 0, err
		}
		for _, c := range codes {
			if c.Live == 0 || len(c.Calls) == 0 || c.Calls[0] != c.LiveUnits {
				return 0, fmt.Errorf("%s %v: its calls %v hold no rows of a call (live %d of %d units)", c.Name, c.Passes, c.Calls, c.Live, c.LiveUnits)
			}
		}
		return len(codes), nil
	}
	ran := 0
	for _, fused := range []bool{true, false} {
		reg := jit.Default()
		if !fused {
			p := reg.Profiles[jit.DeviceCPU]
			p.Off = map[string]bool{"fuse": true}
			reg.Profiles[jit.DeviceCPU] = p
		}
		for _, typ := range []jlm.Type{jlm.TypeQ8, jlm.TypeQ4} {
			for _, R := range []int{2, 64} {
				n, err := check(reg, typ, R)
				if err != nil {
					t.Fatalf("fused %v, type %d, %d rows: %v", fused, typ, R, err)
				}
				t.Logf("fused %v, type %d, %d rows: %d kernels, each held to the call's rows", fused, typ, R, n)
				ran++
			}
		}
	}
	reg := jit.Default()
	p := reg.Profiles[jit.DeviceCPU]
	p.Off = map[string]bool{"rowcalls": true}
	reg.Profiles[jit.DeviceCPU] = p
	if _, err := check(reg, jlm.TypeQ8, 64); err == nil {
		t.Fatal("with the call's rows unheld the gate passes: it cannot see it")
	} else {
		t.Logf("with the call's rows unheld: %v", err)
	}
	if ran != 8 {
		t.Fatalf("%d of 8 forms held", ran)
	}
}

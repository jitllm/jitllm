package composer

import (
	"runtime"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestTailedKernelsKeepTheFormThatSpillsLess forms the CPU's expert matvec
// at a mixture's gate and down shapes both ways -- its sheet base taken past
// the loops' tails (Kernel.Tailed), and with the tails counting -- and holds
// Form to the one with fewer stack operands. The gate's spills less with the
// base taken on every host; on x86 the down's spills more, so the choice goes
// each way there (on arm64 its tails agree and there is none to make). A
// Form keeping the other is the violation.
func TestTailedKernelsKeepTheFormThatSpillsLess(t *testing.T) {
	dev := cpuDevice(hostArm())
	dev.Slots = 12
	r := Default()
	if r.Profiles[DeviceCPU].Int8 == nil {
		r = withCPUInt8()
	}
	prof := r.Profiles[DeviceCPU].on(dev)
	tookBase, keptTails := 0, 0
	for _, sh := range []struct {
		name    string
		role    jlm.Role
		typ     quant.Type
		K, rows int
	}{
		{"gate q4_k", jlm.RoleExpGateBank, quant.Q4_K, 2048, 1024},
		{"down q6_k", jlm.RoleExpDownBank, quant.Q6_K, 1024, 2048},
		{"down q4_k", jlm.RoleExpDownBank, quant.Q4_K, 1024, 2048},
	} {
		const E, P = 64, 8
		jt, _ := jlm.TypeOf(sh.typ)
		b := graph.New()
		x := b.In("x", P*sh.K)
		sel := b.In("sel", P)
		b.G.Values[sel].Elem = graph.ElemU32
		y := b.Emit(graph.OpExpertMatVec, []graph.ValueID{x, sel}, []graph.Result{{Name: "y", Width: P * sh.rows}},
			[]graph.Binding{{Role: sh.role, K: sh.K, Rows: sh.rows, Sheets: E}}, graph.Attrs{}, nil, nil)
		b.Out(y[0])
		req := Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{sh.role: jt},
			Step: Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: dev}
		ks, err := r.formWith(req, prof, nil)
		if err != nil {
			t.Fatal(err)
		}
		tailed := map[KernelAt]bool{}
		for _, k := range ks {
			if k.Tailed {
				tailed[k.at()] = true
			}
		}
		past, err := r.Lower(ks, dev)
		if err != nil {
			t.Fatal(err)
		}
		counting := past
		if len(tailed) > 0 {
			ks, err = r.formWith(req, prof, tailed)
			if err != nil {
				t.Fatal(err)
			}
			if counting, err = r.Lower(ks, dev); err != nil {
				t.Fatal(err)
			}
		}
		_, codes, err := r.Form(req)
		if err != nil {
			t.Fatal(err)
		}
		got := 0
		for _, c := range codes {
			got += c.Regs.Stack
		}
		sp, sc := stackRefs(past), stackRefs(counting)
		t.Logf("%s: %d stack operands with the base taken past the tails, %d with the tails counting; Form keeps %d", sh.name, sp, sc, got)
		if got != min(sp, sc) {
			t.Errorf("%s: Form keeps a form of %d stack operands, the two spill %d and %d", sh.name, got, sp, sc)
		}
		switch {
		case sp < sc:
			tookBase++
		case sc < sp:
			keptTails++
		}
	}
	if tookBase == 0 || runtime.GOARCH == "amd64" && keptTails == 0 {
		t.Fatalf("%d shapes took the base past the tails and %d kept the tails counting: the choice went no way it must", tookBase, keptTails)
	}
}

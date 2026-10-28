package jit

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/x86"
)

// rowJamRegistry is the default registry with the CPU's RowJam at n rows a
// call and the passes named in off turned off.
func rowJamRegistry(n int, off ...string) *Registry {
	reg := speedRegistry(-1, n)
	p := reg.Profiles[DeviceCPU]
	p.Off = map[string]bool{}
	for _, o := range off {
		p.Off[o] = true
	}
	reg.Profiles[DeviceCPU] = p
	return reg
}

// rowsForm is the jammed kernel a prompt's matvec of typ forms at rows rows
// a step, and the weight words a launch of it loads.
func rowsForm(t *testing.T, reg *Registry, typ quant.Type, isa uint64, rows int) (*Kernel, int) {
	t.Helper()
	jt, _ := jlm.TypeOf(typ)
	req := Request{
		Graph:   graph.MatVec(jlm.RoleOutput, 64, 2048),
		Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
		Step:    Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32},
		Device:  Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: isa, ExactDiv: true},
	}
	ks, err := reg.formKernels(req)
	if err != nil {
		t.Fatalf("%v at %d rows: %v", typ, rows, err)
	}
	n, err := WeightLoads(reg, req)
	if err != nil {
		t.Fatal(err)
	}
	var rk *Kernel
	for _, k := range ks {
		if k.Jam > 1 || rk == nil && slices.Contains(k.Passes, "workers") {
			rk = k
		}
	}
	return rk, n
}

// TestCPURowJamReadsTheWeightOnce holds a prompt's matvec on the CPU to
// reading each weight word once for the rows a call runs: its jammed form,
// two rows a call, loads half the weight words the same step loads with
// RowJam off, on every packed type the prompts run (the weight's words and
// the scales its sub-blocks decode alike, shared by both rows and ordered
// so the registers hold them), with and without VNNI. A form the registers
// refuse at eight rows a call halves its rows and keeps Jam's lanes. With
// Jam off, the halving check must part.
func TestCPURowJamReadsTheWeightOnce(t *testing.T) {
	host := uint64(cpu.CPU().ISA())
	isas := map[string]uint64{"host": host}
	if vnni := uint64(x86.ISAVNNI); host&vnni != 0 {
		isas["without vnni"] = host &^ vnni
	}
	checked := 0
	for _, typ := range []quant.Type{quant.Q4_K, quant.Q5_K, quant.Q6_K, quant.Q8_0, quant.Q3_K, quant.Q4_0, quant.Q5_0, quant.Q5_1, quant.TQ2_0, quant.TQ2_F32, quant.MXFP4} {
		for name, isa := range isas {
			name := fmt.Sprintf("%v/%s", typ, name)
			k, jam := rowsForm(t, rowJamRegistry(2), typ, isa, 8)
			if k == nil || k.Jam != 2 {
				t.Errorf("%s: the prompt's matvec does not run two rows a call", name)
				continue
			}
			_, off := rowsForm(t, rowJamRegistry(2, "rowjam"), typ, isa, 8)
			if jam == 0 || 2*jam > off {
				t.Errorf("%s: two rows a call load %d weight words a launch, against %d a row a call", name, jam, off)
				continue
			}
			t.Logf("%s: %d weight words a launch two rows a call, %d a row a call", name, jam, off)
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no jammed form checked: the gate proved nothing")
	}
	halves := func(reg *Registry) error {
		k, _ := rowsForm(t, reg, quant.Q8_0, host, 8)
		switch {
		case k == nil || k.Unjam == 0:
			return fmt.Errorf("eight rows a call were not refused and halved")
		case k.Jam != 8>>k.Unjam:
			return fmt.Errorf("halved %d times, the form runs %d rows a call", k.Unjam, k.Jam)
		case !k.wide():
			return fmt.Errorf("halved to %d rows a call, it lost Jam's lanes", k.Jam)
		}
		t.Logf("eight rows a call refused %d times: %d rows a call, Jam's lanes kept", k.Unjam, k.Jam)
		return nil
	}
	if err := halves(rowJamRegistry(8)); err != nil {
		t.Error(err)
	}
	if err := halves(rowJamRegistry(8, "jam")); err == nil {
		t.Error("with Jam off the halving check passed: it cannot see a form that lost its lanes")
	}
}

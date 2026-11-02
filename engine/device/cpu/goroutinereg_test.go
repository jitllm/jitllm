//go:build amd64

package cpu

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestGeneratedKernelsLeaveTheGoroutineAlone disassembles every amd64 kernel
// this package emits and fails on a write to R14, RBP or RSP. Go's amd64
// register ABI pins R14 to the current goroutine, and a corrupted g fails far
// from its cause (the quantizer once wrote R14 with every numeric gate green).
// ssegate_test.go applies the same check to SSE kernels.
func TestGeneratedKernelsLeaveTheGoroutineAlone(t *testing.T) {
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Skip("objdump not installed -- this gate is a disassembly check")
	}
	dir := t.TempDir()
	check := func(name string, code []byte) {
		t.Helper()
		if len(code) == 0 {
			return
		}
		for _, in := range disasmWide(t, dir, name, code) {
			fields := strings.Fields(in.text)
			if len(fields) == 0 {
				continue
			}
			mn, ops := fields[0], strings.Join(fields[1:], " ")
			switch mn {
			case "push", "pop", "enter", "leave", "pushf", "popf", "pushfq", "popfq":
				t.Errorf("%s: %q moves RSP; the trampoline requires RSP identical "+
					"on every path through a kernel", name, in.text)
				continue
			}
			if !writesFirstOperand(mn) {
				continue
			}
			dst := strings.SplitN(ops, ",", 2)[0]
			if strings.Contains(dst, "[") { // a memory destination writes no register
				continue
			}
			switch strings.TrimSpace(dst) {
			case "r14", "r14d", "r14w", "r14b":
				t.Errorf("%s: %q writes R14, which is Go's goroutine pointer -- "+
					"trampoline_amd64.s offers it on no terms, and a corrupted g "+
					"does not fail near its cause", name, in.text)
			case "rbp", "ebp", "bp", "bpl":
				t.Errorf("%s: %q writes RBP, which callKernel does not save", name, in.text)
			case "rsp", "esp", "sp", "spl":
				t.Errorf("%s: %q writes RSP", name, in.text)
			}
		}
	}

	// The sweep is the kernels a token runs, with the quantizer's half-sum
	// forms named explicitly (emitted only when half is baked true).
	for _, half := range []bool{false, true} {
		check("quantact", EmitQuantAct(half))
		check("quantact_narrow", EmitQuantActNarrow(half))
		for _, tok := range []int{1, 4} {
			check("packact", EmitPackAct(tok, half))
		}
	}
	check("argmax", EmitArgmax())
	em := EmittersFor(TierAVX2)
	for _, q := range quant.PackedTypes {
		if b, err := em.PackedFused(q); err == nil {
			check("packed_fused_"+q.String(), b)
		}
		if b, err := em.PackedMatVec(q, PackedRows); err == nil {
			check("packed_matvec_"+q.String(), b)
		}
	}
	if b, err := em.Softmax(); err == nil {
		check("softmax", b)
	}
	if b, err := em.RMSNorm(2048); err == nil {
		check("rmsnorm", b)
	}
	// The rotary table, at a whole-vector width and a ragged one: the ragged
	// arm emits the body a second time with the cursors moved back, so it is
	// its own code and not a suffix of the other's.
	for _, np := range []int{64, 26} {
		if b, err := em.RopeTable(np); err == nil {
			check("rope_table", b)
		}
	}
	// The sampler's three kernels, on both x86 tiers: the widest general
	// register users in this package.
	for _, tier := range []Tier{TierAVX2, TierSSE} {
		e, suffix := EmittersFor(tier), "_"+tier.String()
		for _, first := range []bool{true, false} {
			for _, idsMem := range []bool{false, true} {
				if b, err := e.SampleSegMax(first, idsMem); err == nil {
					check("sample_segmax"+suffix, b)
				}
			}
		}
		if b, err := e.SampleDraw(); err == nil {
			check("sample_draw"+suffix, b)
		}
		if b, err := e.SamplePenalty(); err == nil {
			check("sample_penalty"+suffix, b)
		}
	}
}

// TestGoroutineRegGateCatchesItsViolation runs that gate against the bug it was
// written for: a kernel that loads a pointer into R14, which is exactly what
// the quantizer did.
func TestGoroutineRegGateCatchesItsViolation(t *testing.T) {
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Skip("objdump not installed")
	}
	var a Buf
	a.MOVLoad(R14, At(RDI, 104)) // the line that was in packact.go
	a.RET()
	found := false
	for _, in := range disasmWide(t, t.TempDir(), "violation", a.Bytes()) {
		fields := strings.Fields(in.text)
		if len(fields) == 0 || !writesFirstOperand(fields[0]) {
			continue
		}
		if strings.HasPrefix(strings.Join(fields[1:], " "), "r14") {
			found = true
		}
	}
	if !found {
		t.Fatal("the disassembly check did not see a write to r14, so the gate above " +
			"cannot catch the bug it was written for")
	}
}

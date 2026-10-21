package cpu

import "testing"

func regsOf(t *testing.T, name string) KernelRegs {
	t.Helper()
	for _, k := range KernelRegisters() {
		if k.Emitter == name {
			return k
		}
	}
	t.Fatalf("no ledger entry for %s in %+v", name, KernelRegisters())
	return KernelRegs{}
}

func emitSpillsOnce() []byte {
	var a Buf
	a.VADDPS(Y14, Y2, Y3)
	a.MOVStore(At(RSP, -8), RAX)
	a.RET()
	return a.Bytes()
}

func emitNoSpill() []byte {
	var a Buf
	a.VADDPS(Y4, Y2, Y3)
	a.RET()
	return a.Bytes()
}

func emitA64Spills() []byte {
	var a A64
	a.STRq(V3, SP, 16)
	a.LDRq(V3, SP, 16)
	a.LDRq(V4, X1, 0) // off a pointer: not the stack
	a.RET()
	return a.Bytes()
}

// TestRegisterLedgerCounts holds the host register report to what a kernel
// emitted, both ways: a stack store is counted and its absence is not, and the
// highest vector register named is the one asked for.
func TestRegisterLedgerCounts(t *testing.T) {
	emitSpillsOnce()
	emitNoSpill()
	emitA64Spills()
	if k := regsOf(t, "cpu.emitSpillsOnce"); k.StackRefs != 1 || k.VecUsed != 15 || k.ISA != "avx2" {
		t.Errorf("emitSpillsOnce: %+v, want 1 stack ref and 15 of 16 vector registers on avx2", k)
	}
	if k := regsOf(t, "cpu.emitNoSpill"); k.StackRefs != 0 || k.VecUsed != 5 {
		t.Errorf("emitNoSpill: %+v, want no stack ref and 5 vector registers", k)
	}
	if k := regsOf(t, "cpu.emitA64Spills"); k.StackRefs != 2 || k.ISA != "neon" {
		t.Errorf("emitA64Spills: %+v, want 2 stack refs on neon", k)
	}
}

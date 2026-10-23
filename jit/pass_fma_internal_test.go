package jit

import (
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
)

// TestFMAFusesOnlyFloatSums holds the fma pass to fusing a float sum of
// products and leaving an integer one's multiply and add: OpFma is a float
// operation, and an integer dot fused into it lowered to an fma.rn.f32 over
// integer registers, which ptxas refused for every K-quant model.
func TestFMAFusesOnlyFloatSums(t *testing.T) {
	sum := func(ty ir.Type) *Kernel {
		k := &Kernel{Name: "sum", Space: []int{1}, Loops: []Loop{{Trip: 4, Parent: -1}}}
		x := k.add(Node{Op: OpConst, Type: ty, Imm: 3})
		y := k.add(Node{Op: OpConst, Type: ty, Imm: 5})
		zero := k.add(Node{Op: OpConst, Type: ty})
		fold := k.add(Node{Op: OpFold, Type: ty, B: zero, At: 0})
		prod := k.add(Node{Op: OpMul, Type: ty, A: x, B: y})
		k.Nodes[fold].A = k.add(Node{Op: OpAdd, Type: ty, A: fold, B: prod})
		at := k.add(Node{Op: OpCoord, Type: ir.U32})
		k.Bufs = append(k.Bufs, Buf{Kind: BufOut, Elem: ty, Operand: Operand{Kind: OperandScratch, Name: "out"}, Shape: []int{1}})
		k.Stores = append(k.Stores, Store{Buf: 0, Val: fold, At: -1, Ax: []int{at}})
		return k
	}
	fused := func(ks []*Kernel) bool {
		for _, k := range ks {
			for _, nd := range k.Nodes {
				if nd.Op == OpFma {
					return true
				}
			}
		}
		return false
	}
	for _, c := range []struct {
		ty   ir.Type
		want bool
	}{{ir.F32, true}, {ir.I32, false}, {ir.U32, false}} {
		ks, err := FMA{}.Apply([]*Kernel{sum(c.ty)}, Device{Kind: DeviceGPU})
		if err != nil {
			t.Fatal(err)
		}
		if got := fused(ks); got != c.want {
			t.Errorf("a %v sum of products fused: %v, want %v", c.ty, got, c.want)
		}
	}
}

// TestFMAFoldsAnAddIntoTheByteDot holds the fma pass to running acc +
// dot4(a, b, 0) as dot4(a, b, acc), and to leaving a dot that does not
// start from zero, or that something else reads, as it is.
func TestFMAFoldsAnAddIntoTheByteDot(t *testing.T) {
	chain := func(start int64, shared bool) *Kernel {
		k := &Kernel{Name: "dots", Space: []int{1}}
		a := k.add(Node{Op: OpConst, Type: ir.U32, Imm: 0x01020304})
		b := k.add(Node{Op: OpConst, Type: ir.U32, Imm: 0x05060708})
		acc := k.add(Node{Op: OpConst, Type: ir.I32, Imm: 9})
		from := k.add(Node{Op: OpConst, Type: ir.I32, Imm: start})
		dot := k.add(Node{Op: OpDot4, Type: ir.I32, A: a, B: b, C: from})
		sum := k.add(Node{Op: OpAdd, Type: ir.I32, A: acc, B: dot})
		at := k.add(Node{Op: OpCoord, Type: ir.U32})
		k.Bufs = append(k.Bufs, Buf{Kind: BufOut, Elem: ir.I32, Operand: Operand{Kind: OperandScratch, Name: "out"}, Shape: []int{2}})
		k.Stores = append(k.Stores, Store{Buf: 0, Val: sum, At: -1, Ax: []int{at}})
		if shared {
			k.Stores = append(k.Stores, Store{Buf: 0, Val: dot, At: -1, Ax: []int{at}})
		}
		return k
	}
	folded := func(k *Kernel) bool {
		ks, err := FMA{}.Apply([]*Kernel{k}, Device{Kind: DeviceGPU})
		if err != nil {
			t.Fatal(err)
		}
		for _, nd := range ks[0].Nodes {
			if nd.Op == OpAdd {
				return false
			}
		}
		return true
	}
	if !folded(chain(0, false)) {
		t.Error("acc + dot4(a, b, 0) kept its add")
	}
	if folded(chain(7, false)) {
		t.Error("acc + dot4(a, b, 7) lost its add")
	}
	if folded(chain(0, true)) {
		t.Error("a dot something else reads lost its add")
	}
}

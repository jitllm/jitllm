package composer

import (
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestSharedBytesCountsWhatTheKernelDeclares holds the workgroup memory a
// kernel is bounded by to the arrays its lowering declares: a plane no
// access names any more (Stage moves the warps' sums' accesses to its own)
// is declared nowhere and holds nothing (gemma4-wide's staged block on an
// Apple GPU, refused for 34816 bytes it never held).
func TestSharedBytesCountsWhatTheKernelDeclares(t *testing.T) {
	k := &Kernel{Bufs: []Buf{
		{Kind: BufIn},
		{Kind: BufShared, Elem: ir.F32, Size: 256},
		{Kind: BufShared, Elem: ir.F32, Size: 4096, Plane: true},
	}}
	ix := k.u32(0)
	v := k.add(Node{Op: OpLoad, Type: ir.F32, Buf: 1, A: ix})
	k.Stores = []Store{{Buf: 1, Ix: ix, Val: v, At: -1}}
	if got := sharedBytes(k); got != 4*256 {
		t.Errorf("a kernel declaring %d bytes of workgroup memory is bounded at %d (%s)", 4*256, got, sharedParts(k))
	}
}

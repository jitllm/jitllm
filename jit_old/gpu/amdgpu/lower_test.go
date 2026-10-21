package amdgpu

import (
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

func mustTarget(t *testing.T, arch string) Target {
	t.Helper()
	tg, err := TargetFor(arch)
	if err != nil {
		t.Fatal(err)
	}
	return tg
}

func lowerOK(t *testing.T, k *ir.Kernel, arch string) string {
	t.Helper()
	s, err := Lower(k, mustTarget(t, arch))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// TestTargetsByGeneration: wave64 on gfx9, wave32 from gfx10, and the dot
// instruction each generation has. An unknown generation is refused rather
// than given a guessed wave size.
func TestTargetsByGeneration(t *testing.T) {
	for _, c := range []struct {
		arch string
		wave int
		dot  Dot4Kind
	}{
		{"gfx90a:sramecc+:xnack-", 64, Dot4Signed},
		{"gfx942", 64, Dot4Signed},
		{"gfx906", 64, Dot4Bytes},
		{"gfx1030", 32, Dot4Signed},
		{"gfx1010", 32, Dot4Bytes},
		{"gfx1100", 32, Dot4Mixed},
		{"gfx1151", 32, Dot4Mixed},
		{"gfx1201", 32, Dot4Mixed},
	} {
		tg := mustTarget(t, c.arch)
		if tg.Wave != c.wave || tg.Dot4 != c.dot {
			t.Errorf("%s: wave %d dot4 %s, want %d %s", c.arch, tg.Wave, tg.Dot4, c.wave, c.dot)
		}
	}
	for _, bad := range []string{"", "sm_86", "gfx", "gfx803"} {
		if _, err := TargetFor(bad); err == nil {
			t.Errorf("%q was accepted as a target", bad)
		}
	}
}

// TestShiftsKeepPTXsAnswerPastThirtyOne: LLVM's shl and lshr are poison at a
// shift of 32 or more, PTX's are 0. The lowering selects 0, so a kernel does
// not mean two things on two back ends.
func TestShiftsKeepPTXsAnswerPastThirtyOne(t *testing.T) {
	b := ir.New("shifts", [3]int{64, 1, 1})
	in := b.Param("in", ir.U32)
	out := b.Param("out", ir.U32)
	tid := b.TID()
	x := b.Load(ir.U32, in, tid, 0)
	b.Store(out, tid, b.Shr(ir.U32, b.Shl(ir.U32, x, tid), tid), 0)
	s := lowerOK(t, b.Done(), "gfx1100")
	if strings.Count(s, "icmp ult i32") < 2 || strings.Count(s, ", i32 0\n") < 2 {
		t.Errorf("a shift is not guarded against 32 and past:\n%s", s)
	}
}

// TestConvertSaturates: OpCvtI32 is PTX's saturating cvt.rzi, never a bare
// fptosi (poison out of range).
func TestConvertSaturates(t *testing.T) {
	b := ir.New("cvt", [3]int{64, 1, 1})
	in := b.Param("in", ir.F32)
	out := b.Param("out", ir.I32)
	tid := b.TID()
	b.Store(out, tid, b.CvtI32(b.Load(ir.F32, in, tid, 0)), 0)
	s := lowerOK(t, b.Done(), "gfx942")
	if !strings.Contains(s, "llvm.fptosi.sat.i32.f32") || strings.Contains(s, " fptosi ") {
		t.Errorf("OpCvtI32 is not the saturating conversion:\n%s", s)
	}
}

// TestDot4PerTarget: the dot instruction is the one the target has.
func TestDot4PerTarget(t *testing.T) {
	b := ir.New("dot", [3]int{64, 1, 1})
	in := b.Param("in", ir.U32)
	out := b.Param("out", ir.I32)
	tid := b.TID()
	x := b.Load(ir.U32, in, tid, 0)
	b.Store(out, tid, b.Dot4(x, x, b.Const(ir.I32, 7)), 0)
	k := b.Done()
	for arch, want := range map[string]string{
		"gfx90a": "llvm.amdgcn.sdot4", "gfx1100": "llvm.amdgcn.sudot4(i1 true", "gfx906": "sext <4 x i8>",
	} {
		if s := lowerOK(t, k, arch); !strings.Contains(s, want) {
			t.Errorf("%s: dot4 is not %s:\n%s", arch, want, s)
		}
	}
}

// TestBarrierFences: s_barrier alone does not order LDS accesses under the
// AMDGPU memory model; HIP's __syncthreads brackets it with workgroup fences.
func TestBarrierFences(t *testing.T) {
	b := ir.New("bar", [3]int{64, 1, 1})
	in := b.Param("in", ir.F32)
	out := b.Param("out", ir.F32)
	sh := b.Shared("t", ir.F32, 64)
	tid := b.TID()
	b.Store(sh, tid, b.Load(ir.F32, in, tid, 0), 0)
	b.Barrier()
	b.Store(out, tid, b.Load(ir.F32, sh, b.Xor(ir.U32, tid, b.Const(ir.U32, 1)), 0), 0)
	s := lowerOK(t, b.Done(), "gfx1201")
	want := "fence syncscope(\"workgroup\") release\n  call void @llvm.amdgcn.s.barrier()\n  fence syncscope(\"workgroup\") acquire"
	if !strings.Contains(s, want) {
		t.Errorf("the barrier is not fenced:\n%s", s)
	}
	if !strings.Contains(s, "internal addrspace(3) global [64 x float]") {
		t.Errorf("the shared array is not in LDS:\n%s", s)
	}
}

// TestLoopTestsBeforeTheBody: a runtime count of zero runs the body zero
// times, and a phi read after the loop is its final value -- the header's,
// which is what a while loop gives.
func TestLoopTestsBeforeTheBody(t *testing.T) {
	b := ir.New("loop", [3]int{64, 1, 1})
	in := b.Param("in", ir.U32)
	out := b.Param("out", ir.F32)
	tid := b.TID()
	n := b.Load(ir.U32, in, tid, 0)
	zero := b.ConstF32(0)
	b.LoopN(n)
	acc := b.Phi(ir.F32, zero)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.ConstF32(1)))
	b.EndLoop()
	b.Store(out, tid, acc, 0)
	k := b.Done()
	s := lowerOK(t, k, "gfx1100")
	for _, want := range []string{
		"phi i32 [ 0, %entry ]", "icmp ult i32 %c", "label %L", "store float %v" + strconv.Itoa(int(acc)),
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q:\n%s", want, s)
		}
	}
	if strings.Contains(s, "@@") {
		t.Errorf("a back edge was left unpatched:\n%s", s)
	}
}

// TestShuffleStaysInItsHalf: on a 64-wide wave the lane comes from both mbcnt
// halves, and on a 32-wide one from the low half alone; the source lane is
// lane^mask either way.
func TestShuffleStaysInItsHalf(t *testing.T) {
	b := ir.New("shfl", [3]int{64, 1, 1})
	in := b.Param("in", ir.F32)
	out := b.Param("out", ir.F32)
	tid := b.TID()
	x := b.Load(ir.F32, in, tid, 0)
	b.Store(out, tid, b.ShuffleXor(ir.F32, x, 16), 0)
	k := b.Done()
	s64 := lowerOK(t, k, "gfx942")
	s32 := lowerOK(t, k, "gfx1100")
	if !strings.Contains(s64, "mbcnt.hi") || strings.Contains(s32, "mbcnt.hi") {
		t.Error("the lane index does not follow the wave size")
	}
	for _, s := range []string{s64, s32} {
		if !strings.Contains(s, "xor i32 %lane, 16") || !strings.Contains(s, "ds.bpermute") {
			t.Errorf("the shuffle is not a bpermute of lane^16:\n%s", s)
		}
	}
	if !strings.Contains(s64, "+wavefrontsize64") || !strings.Contains(s32, "+wavefrontsize32") {
		t.Error("the wave size is not stated in the target features")
	}
}

// TestRefusesTheFragmentMMA: OpMMA is NVIDIA's lane layout.
func TestRefusesTheFragmentMMA(t *testing.T) {
	b := ir.New("mma", [3]int{32, 1, 1})
	p := b.Param("p", ir.I32)
	tid := b.TID()
	sh := ir.MMAShape{M: 16, N: 8, K: 16}
	na, nb, nc := sh.Frags()
	frag := func(n int, t ir.Type) []ir.Value {
		out := make([]ir.Value, n)
		for i := range out {
			out[i] = b.Const(t, 0)
		}
		return out
	}
	d := b.MMA(sh, frag(na, ir.U32), frag(nb, ir.U32), frag(nc, ir.I32))
	b.Store(p, tid, d[0], 0)
	if _, err := Lower(b.Done(), mustTarget(t, "gfx942")); err == nil {
		t.Error("an MMA kernel was lowered")
	}
}

package backend_test

import (
	"encoding/binary"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/ptx"
)

// TestSharedPastTheStaticLimit runs, on every CUDA device, a workgroup whose
// two shared arrays are past the 48 KiB a kernel declares statically: the
// launch asks for them as dynamic shared memory (ptx.DynamicShared,
// cuda.Module.SetDynamicShared). Each thread fills its share of both arrays
// keyed on the workgroup, and after a barrier reads another thread's words
// of each: arrays that overlapped, or a launch given less than they need,
// read the wrong words or fail.
func TestSharedPastTheStaticLimit(t *testing.T) {
	gpuLock(t)
	const width, groups, big, small = 256, 4, 16384, 512
	b := ir.New("dynshared", [3]int{width, 1, 1})
	pOut := b.Param("pOut", ir.U32)
	shA := b.Shared("a", ir.U32, big)
	shB := b.Shared("b", ir.U32, small)
	tid, cta := b.TID(), b.CTAID()
	u := func(v int) ir.Value { return b.Const(ir.U32, int64(v)) }
	key := b.Mul(ir.U32, cta, u(1000003))
	for j := 0; j < big/width; j++ {
		i := b.Add(ir.U32, tid, u(j*width))
		b.Store(shA, i, b.Add(ir.U32, b.Mul(ir.U32, i, u(3)), key), 0)
	}
	for j := 0; j < small/width; j++ {
		i := b.Add(ir.U32, tid, u(j*width))
		b.Store(shB, i, b.Add(ir.U32, b.Mul(ir.U32, i, u(7)), key), 0)
	}
	b.Barrier()
	ra := b.Rem(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, tid, u(61)), u(3)), u(big))
	rb := b.Sub(ir.U32, u(small-1), tid)
	at := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, cta, u(width)), tid), u(2))
	b.Store(pOut, at, b.Load(ir.U32, shA, ra, 0), 0)
	b.Store(pOut, at, b.Load(ir.U32, shB, rb, 0), 1)
	k := mustBuild(t, b)
	if n := ptx.DynamicShared(k); n < 48<<10 {
		t.Fatalf("the kernel's %d bytes of shared memory are not past the static limit: the gate would not ask for dynamic shared memory", n)
	}
	ran := 0
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		kk, err := d.Compile(k)
		if err != nil {
			t.Fatalf("%s: compile: %v", d.Name(), err)
		}
		defer kk.Close()
		n := 2 * groups * width
		buf := make([]byte, 4*n)
		out, err := d.Alloc(len(buf))
		if err != nil {
			t.Fatal(err)
		}
		defer out.Free()
		if err := out.Write(buf); err != nil {
			t.Fatal(err)
		}
		if err := kk.Launch(groups, width, out); err != nil {
			t.Fatalf("%s: launch: %v", d.Name(), err)
		}
		if err := out.Read(buf); err != nil {
			t.Fatal(err)
		}
		bad := 0
		for i := 0; i < n; i++ {
			f := i / 2
			cta, tid := f/width, f%width
			want := uint32((tid*61+3)%big*3 + cta*1000003)
			if i%2 == 1 {
				want = uint32((small-1-tid)*7 + cta*1000003)
			}
			if v := binary.LittleEndian.Uint32(buf[4*i:]); v != want {
				if bad++; bad <= 3 {
					t.Errorf("%s: word %d = %d, want %d", d.Name(), i, v, want)
				}
			}
		}
		ran++
	}
	if ran == 0 {
		t.Fatal("no CUDA device ran the kernel")
	}
}

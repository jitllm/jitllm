package backend_test

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestEveryBackendRefusesALaunchWiderThanItsKernel holds every backend to one
// launch contract: the width a kernel is launched at is the workgroup it
// declares. kernels.HCMix declared 64 threads and the tier launched it 128
// wide; CUDA ran the launch's 128 and was right, Vulkan ran the declared 64
// and left half of every group's rows stale with no error. The refusal is the
// only safety property, so a kernel declared 64 wide is launched at 64 (and
// must write every row, or the gate proves nothing), then at 128 and at 32,
// through Kernel.Launch and through a Session: each mismatch is an error that
// names the kernel, and the buffer keeps its poison.
func TestEveryBackendRefusesALaunchWiderThanItsKernel(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const declared, rows = 64, 256
	b := ir.New("widthprobe", [3]int{declared, 1, 1})
	out := b.Param("pOut", ir.F32)
	i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	b.Store(out, i, b.CvtF32(i), 0)
	k := built(t, b)
	if err := k.Validate(); err != nil {
		t.Fatal(err)
	}
	poison := make([]byte, 4*rows)
	for j := 0; j < rows; j++ {
		binary.LittleEndian.PutUint32(poison[4*j:], math.Float32bits(float32(math.NaN())))
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			kern, err := d.Compile(k)
			if err != nil {
				t.Fatal(err)
			}
			defer kern.Close()
			buf, err := d.Alloc(4 * rows)
			if err != nil {
				t.Fatal(err)
			}
			defer buf.Free()
			read := func() []byte {
				got := make([]byte, 4*rows)
				if err := buf.Read(got); err != nil {
					t.Fatal(err)
				}
				return got
			}
			// The declared width runs, and runs every row.
			if err := buf.Write(poison); err != nil {
				t.Fatal(err)
			}
			if err := kern.Launch(rows/declared, declared, buf); err != nil {
				t.Fatalf("launched at its declared %d: %v", declared, err)
			}
			got := read()
			for j := 0; j < rows; j++ {
				if v := math.Float32frombits(binary.LittleEndian.Uint32(got[4*j:])); v != float32(j) {
					t.Fatalf("row %d reads %v at the declared width, want %d", j, v, j)
				}
			}
			for _, w := range []int{2 * declared, declared / 2} {
				if err := buf.Write(poison); err != nil {
					t.Fatal(err)
				}
				refused := func(how string, err error) {
					if err == nil || !strings.Contains(err.Error(), "widthprobe") {
						t.Errorf("%s at %d wide, declared %d: %v; want a refusal naming the kernel", how, w,
							declared, err)
					}
				}
				refused("Kernel.Launch", kern.Launch((rows+w-1)/w, w, buf))
				var serr error
				d.Session(func(s backend.Session) {
					serr = s.Launch(kern, (rows+w-1)/w, w, buf)
					if err := s.Sync(); err != nil && serr == nil {
						serr = err
					}
				})
				refused("Session.Launch", serr)
				got := read()
				for j := 0; j < rows; j++ {
					if v := math.Float32frombits(binary.LittleEndian.Uint32(got[4*j:])); !math.IsNaN(float64(v)) {
						t.Fatalf("%d wide: row %d was written (%v) by a launch that was refused", w, j, v)
					}
				}
			}
		})
	}
}

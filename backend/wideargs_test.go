package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/ir"
)

// wideArgs is out[i] = the sum over n inputs of in_j[i]: a kernel binding n+1
// buffers, as a persistent kernel composed from a block's launches binds every
// buffer they do (ir.Persist).
func wideArgs(t testing.TB, n, width int) *ir.Kernel {
	b := ir.New(fmt.Sprintf("wide%d", n), [3]int{width, 1, 1})
	ins := make([]ir.Value, n)
	for j := range ins {
		ins[j] = b.Param(fmt.Sprintf("in%d", j), ir.F32)
	}
	out := b.Param("out", ir.F32)
	i := b.TID()
	acc := b.ConstF32(0)
	for _, in := range ins {
		acc = b.Fma(b.Load(ir.F32, in, i, 0), b.ConstF32(1), acc)
	}
	b.Store(out, i, acc, 0)
	return mustBuild(t, b)
}

// TestAKernelBindsAsManyBuffersAsTheDeviceTakes launches a kernel of 201
// buffers on every device, on the legacy stream, in a session and on a queue:
// a kernel's parameter list is bound by the device's toolchain (CUDA's 4 KB of
// parameters, 512 pointers), not by an array of this code's. Every input j is
// j+1 everywhere, so every output word is 200*201/2 exactly, and the output
// starts poisoned. On a tree whose argument arrays held 64 the launch was
// refused ("kernel arguments is too many").
func TestAKernelBindsAsManyBuffersAsTheDeviceTakes(t *testing.T) {
	const n, width = 200, 32
	gpuLock(t)
	devs := backend.OpenForTest()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			k, err := d.Compile(wideArgs(t, n, width))
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer k.Close()
			bufs := make([]backend.Buf, n+1)
			raw := make([]byte, 4*width)
			for j := range bufs {
				b, err := d.Alloc(len(raw))
				if err != nil {
					t.Fatal(err)
				}
				defer b.Free()
				v := float32(j + 1)
				if j == n {
					v = float32(math.NaN())
				}
				for i := 0; i < width; i++ {
					binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
				}
				if err := b.Write(raw); err != nil {
					t.Fatal(err)
				}
				bufs[j] = b
			}
			check := func(how string) {
				t.Helper()
				if err := bufs[n].Read(raw); err != nil {
					t.Fatal(err)
				}
				for i := 0; i < width; i++ {
					if got := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])); got != n*(n+1)/2 {
						t.Fatalf("%s: word %d is %v, want %d", how, i, got, n*(n+1)/2)
					}
				}
				poison := make([]byte, len(raw))
				for i := 0; i < width; i++ {
					binary.LittleEndian.PutUint32(poison[4*i:], math.Float32bits(float32(math.NaN())))
				}
				if err := bufs[n].Write(poison); err != nil {
					t.Fatal(err)
				}
			}
			if err := k.Launch(1, width, bufs...); err != nil {
				t.Fatalf("launch: %v", err)
			}
			check("launch")
			var serr error
			d.Session(func(s backend.Session) {
				if serr = s.Launch(k, 1, width, bufs...); serr == nil {
					serr = s.Sync()
				}
			})
			if serr != nil {
				t.Fatalf("session: %v", serr)
			}
			check("session")
			if qd, ok := d.(backend.Queued); ok {
				q, err := qd.NewQueue()
				if err != nil {
					t.Fatalf("NewQueue: %v", err)
				}
				defer q.Close()
				qd.SessionOn(q, func(s backend.Session) {
					if serr = s.Launch(k, 1, width, bufs...); serr == nil {
						serr = s.Sync()
					}
				})
				if serr != nil {
					t.Fatalf("queue: %v", serr)
				}
				check("queue")
			}
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no device ran the kernel")
	}
}

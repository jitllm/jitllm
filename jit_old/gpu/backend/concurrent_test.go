package backend_test

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/metal"
)

// TestConcurrentEncoderKeepsTheChain holds Metal's concurrent encoder
// (metal.Opts.Concurrent) to the dependencies it must keep: a chain of
// dispatches in one session, each reading what the one before it wrote and
// half of them overwriting a buffer the one before read, ends with the value
// a serial run gives. A large grid makes a missing barrier show: the next
// dispatch starts while the last groups of the one before still run.
//
// JITLLM_VIOLATE_CONCURRENT=1 tells the encoder every kernel writes nothing,
// so no barrier goes in, which must fail.
func TestConcurrentEncoderKeepsTheChain(t *testing.T) {
	testdevices.Require(t, "metal")
	gpuLock(t)
	d, err := backend.OpenMetalWith(backend.Opts{Metal: metal.Opts{Concurrent: true}})
	if err != nil {
		t.Skipf("no Metal: %v", err)
	}
	defer d.Close()
	const n, steps = 1 << 22, 40
	// step: dst[i] = src[i]*3 + 1, in u32 arithmetic.
	b := ir.New("chainstep", [3]int{256, 1, 1})
	src := b.Param("src", ir.U32)
	dst := b.Param("dst", ir.U32)
	i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.Const(ir.U32, 256)), b.TID())
	v := b.Load(ir.U32, src, i, 0)
	b.Store(dst, i, b.Add(ir.U32, b.Mul(ir.U32, v, b.Const(ir.U32, 3)), b.Const(ir.U32, 1)), 0)
	k := b.Done()
	kern, err := d.Compile(k)
	if err != nil {
		t.Fatal(err)
	}
	defer kern.Close()
	if os.Getenv("JITLLM_VIOLATE_CONCURRENT") != "" {
		backend.MetalKernelWrites(kern, 0)
	}
	g := newGPU(t, d)
	defer g.free()
	a := g.up(make([]byte, 4*n))
	c := g.up(make([]byte, 4*n))
	var serr error
	d.Session(func(s backend.Session) {
		for st := 0; st < steps && serr == nil; st++ {
			from, to := a, c
			if st%2 == 1 {
				from, to = c, a
			}
			serr = s.Launch(kern, n/256, 256, from, to)
		}
	})
	if serr != nil {
		t.Fatal(serr)
	}
	out := make([]byte, 4*n)
	if err := a.Read(out); err != nil {
		t.Fatal(err)
	}
	var want uint32
	for st := 0; st < steps; st++ {
		want = want*3 + 1
	}
	bad := 0
	for j := 0; j < n; j++ {
		if binary.LittleEndian.Uint32(out[4*j:]) != want {
			bad++
		}
	}
	t.Logf("%d of %d words differ from the serial chain's %d after %d steps (%d barriers so far)",
		bad, n, want, steps, metal.Barriers())
	if bad != 0 {
		t.Fatalf("%d words broke the chain", bad)
	}
}

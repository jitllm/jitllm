package backend

import (
	"math"
	"strings"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/backend/hip"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/ir"
)

// hipOrSkip opens HIP device 0, or skips by name: without an AMD card the
// device half of the AMD backend did not run, and the line says so.
func hipOrSkip(t *testing.T) Device {
	t.Helper()
	if HIPCount(openDefaults.HIP) == 0 {
		t.Skipf("NO AMD DEVICE: %s -- the HIP device gate did not run", HIPMissing(openDefaults.HIP))
	}
	d, err := OpenHIPWith(testdevices.Ordinal(t, "hip", 0), openDefaults)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	return d
}

func bytesOf[T any](s []T) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(s[0])))
}

// TestHIPDeviceRunsTheIR runs one kernel with every surface the lowering has --
// shared memory and a barrier, the 32-lane shuffle, a runtime loop, f16 pack
// and unpack, a saturating convert and a dot4 -- against the same arithmetic
// in Go, on a real AMD device.
func TestHIPDeviceRunsTheIR(t *testing.T) {
	d := hipOrSkip(t)
	const n = 256
	b := ir.New("hip_surface", [3]int{n, 1, 1})
	pIn := b.Param("in", ir.F32)
	pQ := b.Param("q", ir.U32)
	pOut := b.Param("out", ir.F32)
	pI := b.Param("outi", ir.I32)
	sh := b.Shared("t", ir.F32, n)
	tid := b.TID()
	x := b.Load(ir.F32, pIn, tid, 0)
	b.Store(sh, tid, x, 0)
	b.Barrier()
	nb := b.Load(ir.F32, sh, b.Rem(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, 1)), b.Const(ir.U32, n)), 0)
	s := x
	for m := int64(16); m >= 1; m /= 2 {
		s = b.Add(ir.F32, s, b.ShuffleXor(ir.F32, s, m))
	}
	cnt := b.And(ir.U32, tid, b.Const(ir.U32, 7))
	zero := b.ConstF32(0)
	one := b.ConstF32(1)
	b.LoopN(cnt)
	acc := b.Phi(ir.F32, zero)
	b.SetPhi(acc, b.Add(ir.F32, acc, one))
	b.EndLoop()
	h := b.CvtF16H(b.PackF16(nb, nb))
	b.Store(pOut, tid, b.Add(ir.F32, b.Add(ir.F32, s, acc), h), 0)
	q := b.Load(ir.U32, pQ, tid, 0)
	big := b.CvtI32(b.Mul(ir.F32, x, b.ConstF32(1e10)))
	b.Store(pI, tid, b.Add(ir.I32, b.Dot4(q, q, b.Const(ir.I32, 3)), big), 0)
	built, err := b.Done()
	if err != nil {
		t.Fatal(err)
	}
	k, err := d.Compile(built)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()

	in := make([]float32, n)
	qs := make([]uint32, n)
	for i := range in {
		in[i] = float32(i%37)*0.25 - 4
		qs[i] = uint32(i)*2654435761 ^ 0x80808080
	}
	bufs := make([]Buf, 4)
	for i := range bufs {
		if bufs[i], err = d.Alloc(4 * n); err != nil {
			t.Fatal(err)
		}
		defer bufs[i].Free()
	}
	if err := bufs[0].Write(bytesOf(in)); err != nil {
		t.Fatal(err)
	}
	if err := bufs[1].Write(bytesOf(qs)); err != nil {
		t.Fatal(err)
	}
	if err := k.Launch(1, n, bufs...); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, n)
	outi := make([]int32, n)
	if err := bufs[2].Read(bytesOf(out)); err != nil {
		t.Fatal(err)
	}
	if err := bufs[3].Read(bytesOf(outi)); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for i := 0; i < n; i++ {
		var s float32
		for j := i &^ 31; j < i&^31+32; j++ {
			s += in[j]
		}
		// The neighbour is a multiple of a quarter, exact through f16.
		want := s + float32(i&7) + in[(i+1)%n]
		if math.Abs(float64(out[i]-want)) > 1e-3 {
			bad++
			if bad < 8 {
				t.Errorf("out[%d] = %v, want %v", i, out[i], want)
			}
		}
		dot := int32(3)
		for k := 0; k < 4; k++ {
			v := int32(int8(qs[i] >> (8 * k)))
			dot += v * v
		}
		sat := int32(0)
		switch {
		case in[i] > 0:
			sat = math.MaxInt32
		case in[i] < 0:
			sat = math.MinInt32
		}
		if got := outi[i]; got != dot+sat {
			bad++
			if bad < 8 {
				t.Errorf("outi[%d] = %d, want %d", i, got, dot+sat)
			}
		}
	}
	t.Logf("%s: %d of %d lanes wrong", d.Name(), bad, 2*n)
}

// TestExplicitHIPWithoutROCmIsAnError: hip:N named with no ROCm behind the
// path is an error naming the cause, and the default set holds no HIP
// device; neither falls back to another backend for the request.
func TestExplicitHIPWithoutROCmIsAnError(t *testing.T) {
	testdevices.SkipSelected(t, "it names hip:0 on a box that has none")
	o := Opts{HIP: hip.Config{Path: t.TempDir()}}
	_, err := OpenHIPWith(0, o)
	if err == nil || !strings.Contains(err.Error(), "hip:0 requested but ROCm was not found (searched "+o.HIP.Path) {
		t.Errorf("err = %v", err)
	}
	if n := HIPCount(o.HIP); n != 0 {
		t.Errorf("%d HIP devices from an empty directory", n)
	}
	if m := HIPMissing(o.HIP); !strings.HasPrefix(m, "ROCm not found (searched "+o.HIP.Path) {
		t.Errorf("HIPMissing = %q", m)
	}
	for _, d := range OpenWith(o) {
		if d.API() == "amdgcn" {
			t.Error("Open returned a HIP device with no ROCm")
		}
		d.Close()
	}
}

// TestHIPOrdinalPastTheCountIsAnError: an ordinal ROCm does not have names the
// count. It needs a ROCm that loads; on one with no card the count is 0.
func TestHIPOrdinalPastTheCountIsAnError(t *testing.T) {
	testdevices.SkipSelected(t, "it opens an ordinal past the box's last card, which no selection holds")
	if !hip.Loaded(openDefaults.HIP) {
		t.Skipf("NO ROCM: %s -- the ordinal refusal did not run", HIPMissing(openDefaults.HIP))
	}
	n := HIPCount(openDefaults.HIP)
	_, err := OpenHIPWith(n+1, openDefaults)
	if want := "HIP devices"; err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("err = %v", err)
	}
}

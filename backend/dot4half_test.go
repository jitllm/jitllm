package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestDot4AndHalfAgreeEverywhere runs the integer dot of four signed bytes
// and the low binary16 half of a word as a float on every device a backend
// opens, on words held to arithmetic done by hand: negative bytes, and halves
// whose word's upper bits are another number.
func TestDot4AndHalfAgreeEverywhere(t *testing.T) {
	a := []uint32{0x01020304, 0xff80017f, 0x80808080, 0x7f7f7f7f, 0x00ff00ff}
	x := []uint32{0x04030201, 0x01017f80, 0x7f7f7f7f, 0x81818181, 0xff01ff01}
	acc := []int32{0, 5, -7, 100, 0}
	sb := func(w uint32, i int) int32 { return int32(int8(w >> (8 * i))) }
	h := []uint32{0x3c00, 0xabcd3c00, 0xc000, 0x12347bff, 0x0001bc00}
	want := func(i int) (int32, float32) {
		d := acc[i]
		for j := range 4 {
			d += sb(a[i], j) * sb(x[i], j)
		}
		return d, halfToFloat(uint16(h[i]))
	}
	n := len(a)
	b := ir.New("dot4half", [3]int{32, 1, 1})
	pa, px, pc, ph := b.Param("a", ir.U32), b.Param("x", ir.U32), b.Param("c", ir.I32), b.Param("h", ir.U32)
	pd, pf := b.Param("d", ir.I32), b.Param("f", ir.F32)
	i := b.Min(ir.U32, b.TID(), b.Const(ir.U32, int64(n-1)))
	b.Store(pd, i, b.Dot4(b.Load(ir.U32, pa, i, 0), b.Load(ir.U32, px, i, 0), b.Load(ir.I32, pc, i, 0)), 0)
	b.Store(pf, i, b.CvtF16H(b.Load(ir.U32, ph, i, 0)), 0)
	k := built(t, b)
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	words := func(w []uint32) []byte {
		out := make([]byte, 4*len(w))
		for i, v := range w {
			binary.LittleEndian.PutUint32(out[4*i:], v)
		}
		return out
	}
	accw := make([]uint32, n)
	for i, v := range acc {
		accw[i] = uint32(v)
	}
	for _, dev := range devs {
		kern, err := dev.Compile(k)
		if err != nil {
			t.Fatalf("%s: %v", dev.API(), err)
		}
		var bufs []backend.Buf
		for _, data := range [][]byte{words(a), words(x), words(accw), words(h), make([]byte, 4*n), make([]byte, 4*n)} {
			buf, err := dev.Alloc(len(data))
			if err != nil {
				t.Fatal(err)
			}
			if err := buf.Write(data); err != nil {
				t.Fatal(err)
			}
			bufs = append(bufs, buf)
		}
		if err := kern.Launch(1, 32, bufs...); err != nil {
			t.Fatalf("%s: %v", dev.API(), err)
		}
		d, f := make([]byte, 4*n), make([]byte, 4*n)
		bufs[4].Read(d)
		bufs[5].Read(f)
		for i := range n {
			wd, wf := want(i)
			gd, gf := int32(binary.LittleEndian.Uint32(d[4*i:])), math.Float32frombits(binary.LittleEndian.Uint32(f[4*i:]))
			if gd != wd {
				t.Errorf("%s %s: dot4 of %#08x and %#08x plus %d is %d, want %d", dev.API(), dev.Name(), a[i], x[i], acc[i], gd, wd)
			}
			if gf != wf {
				t.Errorf("%s %s: the low half of %#08x is %v, want %v", dev.API(), dev.Name(), h[i], gf, wf)
			}
		}
		for _, buf := range bufs {
			buf.Free()
		}
		kern.Close()
	}
}

func halfToFloat(h uint16) float32 {
	sign, exp, frac := uint32(h>>15), uint32(h>>10)&31, uint32(h)&1023
	switch {
	case exp == 0:
		v := float32(frac) / 1024 / 16384
		if sign != 0 {
			v = -v
		}
		return v
	case exp == 31:
		return math.Float32frombits(sign<<31 | 0xff<<23 | frac<<13)
	}
	return math.Float32frombits(sign<<31 | (exp+112)<<23 | frac<<13)
}

// TestSignedConversionAgreesEverywhere converts words bitcast to i32 to
// floats on every device a backend opens: the IR's conversion is of a signed
// integer, so a word with its top bit set is negative. Metal once converted
// it unsigned, and a Q8_0 embedding's negative bytes became 4.29e9 times
// their scale.
func TestSignedConversionAgreesEverywhere(t *testing.T) {
	w := []uint32{0xffffff80, 0xffffffff, 0x7f, 0x80000000, 0x7fffffff, 0}
	n := len(w)
	b := ir.New("cvtsigned", [3]int{32, 1, 1})
	pw, pf := b.Param("w", ir.U32), b.Param("f", ir.F32)
	i := b.Min(ir.U32, b.TID(), b.Const(ir.U32, int64(n-1)))
	b.Store(pf, i, b.CvtF32(b.Bitcast(ir.I32, b.Load(ir.U32, pw, i, 0))), 0)
	k := built(t, b)
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	in := make([]byte, 4*n)
	for j, v := range w {
		binary.LittleEndian.PutUint32(in[4*j:], v)
	}
	for _, dev := range devs {
		kern, err := dev.Compile(k)
		if err != nil {
			t.Fatalf("%s: %v", dev.API(), err)
		}
		bw, err := dev.Alloc(len(in))
		if err != nil {
			t.Fatal(err)
		}
		bf, err := dev.Alloc(len(in))
		if err != nil {
			t.Fatal(err)
		}
		if err := bw.Write(in); err != nil {
			t.Fatal(err)
		}
		if err := kern.Launch(1, 32, bw, bf); err != nil {
			t.Fatalf("%s: %v", dev.API(), err)
		}
		out := make([]byte, len(in))
		bf.Read(out)
		for j, v := range w {
			if got, want := math.Float32frombits(binary.LittleEndian.Uint32(out[4*j:])), float32(int32(v)); got != want {
				t.Errorf("%s %s: %#08x as i32 converts to %v, want %v", dev.API(), dev.Name(), v, got, want)
			}
		}
		bw.Free()
		bf.Free()
		kern.Close()
	}
}

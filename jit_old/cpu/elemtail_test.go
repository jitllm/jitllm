//go:build amd64 || arm64

package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// TestElementwiseEveryWidth runs every runtime-length elementwise kernel at
// widths 1..5*ElemLanes and asserts two things per width:
//
//   - element i is bit-identical to element i of a whole-vector run. The ops
//     are lane-independent, so the tail -- one element in lane 0 through the
//     same body -- has no excuse to round differently.
//   - nothing past the last element is written: a NaN sentinel with a payload
//     follows the slice and must come back with the same bits.
func TestElementwiseEveryWidth(t *testing.T) {
	consts := actConsts()
	alpha := float32(-0.625)
	capC := [2]float32{2.0 / 3, 3} // a cap of 3: the inputs reach 11, so it bites
	type kern struct {
		name string
		code []byte
		src  bool // reads AScale as a second vector
		scr  unsafe.Pointer
	}
	var ks []kern
	host := hostTable()
	ks = append(ks, kern{"axpy", hostBytes(t)(host.Axpy()), true, unsafe.Pointer(&alpha)})
	ks = append(ks, kern{"scale", hostBytes(t)(host.Scale()), false, unsafe.Pointer(&alpha)})
	// softcap reads {2/c, c} through W, set in call below.
	ks = append(ks, kern{"softcap", hostBytes(t)(host.Softcap()), false, unsafe.Pointer(&consts[0])})
	clampLH := [2]float32{-2.5, 4} // the inputs reach +-11, so both bounds bite
	ks = append(ks, kern{"clamp", hostBytes(t)(host.Clamp()), false, unsafe.Pointer(&clampLH[0])})
	ks = append(ks, kern{"sigmoidmul", hostBytes(t)(host.SigmoidMul()), true, unsafe.Pointer(&consts[0])})
	for _, k := range Gated {
		ks = append(ks, kern{"actmul/" + k.String(), hostBytes(t)(host.ActMul(k)), true, unsafe.Pointer(&consts[0])})
	}
	for _, k := range Ungated {
		ks = append(ks, kern{"act/" + k.String(), hostBytes(t)(host.Act(k)), false, unsafe.Pointer(&consts[0])})
	}
	const guard = 2 * ElemLanes
	sentinel := math.Float32frombits(0x7fc0beef)
	max := 5 * ElemLanes
	input := func(n int) (dst, src []float32) {
		dst = make([]float32, n+guard)
		src = make([]float32, n+guard)
		for i := 0; i < n; i++ {
			dst[i] = float32(math.Sin(float64(i)*0.37) * float64(1+i%11))
			src[i] = float32(math.Cos(float64(i)*0.23) * float64(1+i%9))
		}
		for i := n; i < n+guard; i++ {
			dst[i], src[i] = sentinel, sentinel
		}
		return
	}
	for _, k := range ks {
		c := onHost(t)(k.code, nil)
		call := func(dst, src []float32, n int) {
			args := Args{
				Out:  &dst[0],
				Scr:  (*byte)(k.scr),
				K:    int64(n / ElemLanes),
				Rows: int64(n % ElemLanes),
			}
			if k.src {
				args.AScale = &src[0]
			}
			if k.name == "softcap" {
				args.W = (*byte)(unsafe.Pointer(&capC[0]))
			}
			c.Call(&args)
		}
		full, fsrc := input(max)
		call(full, fsrc, max)
		for n := 1; n <= max; n++ {
			dst, src := input(n)
			call(dst, src, n)
			for i := 0; i < n; i++ {
				if math.Float32bits(dst[i]) != math.Float32bits(full[i]) {
					t.Fatalf("%s width %d: element %d is %v, the whole-vector run gives %v", k.name, n, i, dst[i], full[i])
				}
			}
			for i := n; i < n+guard; i++ {
				if math.Float32bits(dst[i]) != 0x7fc0beef {
					t.Fatalf("%s width %d: wrote element %d past the end (%v)", k.name, n, i, dst[i])
				}
			}
		}
		c.Close()
	}
	t.Logf("%d kernels x widths 1..%d: every element matches the whole-vector run, nothing written past the end", len(ks), max)
}

// TestEmitSoftcapMatchesReference gates the cap against float64 tanh, over a
// range that saturates both ways.
func TestEmitSoftcapMatchesReference(t *testing.T) {
	c := onHost(t)(hostTable().Softcap())
	defer c.Close()
	consts := actConsts()
	for _, cap := range []float32{30, 50} {
		kc := [2]float32{2 / cap, cap}
		n := 203
		x := make([]float32, n)
		want := make([]float64, n)
		sat := 0
		for i := range x {
			v := float32(math.Sin(float64(i)*0.37) * float64(i) * 20) // |x| to ~4000
			x[i] = v
			want[i] = float64(cap) * math.Tanh(float64(v)/float64(cap))
			if math.Abs(float64(v)) > 20*float64(cap) {
				sat++
			}
		}
		if sat == 0 {
			t.Fatalf("cap %v: no input saturates, so the clamps are untested", cap)
		}
		c.Call(&Args{Out: &x[0], W: (*byte)(unsafe.Pointer(&kc[0])),
			Scr: (*byte)(unsafe.Pointer(&consts[0])), K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
		for i := range x {
			if d := math.Abs(float64(x[i]) - want[i]); d > 2e-5*float64(cap) {
				t.Fatalf("cap %v: x[%d] = %v, want %v", cap, i, x[i], want[i])
			}
		}
	}
}

// TestEmitClampMatchesReference gates DBRX's clip_qkv kernel against
// min(max(x, lo), hi) at every ragged width, with inputs past both bounds and
// inside them, so a kernel that applies one bound only, or swaps them, or
// clamps to [-hi, hi] when lo != -hi, fails.
func TestEmitClampMatchesReference(t *testing.T) {
	c := onHost(t)(hostTable().Clamp())
	defer c.Close()
	for _, lh := range [][2]float32{{-1.5, 1.5}, {-0.25, 3}} {
		inside, below, above := 0, 0, 0
		for n := 1; n <= 5*ElemLanes+3; n++ {
			x := make([]float32, n)
			want := make([]float32, n)
			for i := range x {
				v := float32(math.Sin(float64(i)*0.71+float64(n)) * 4)
				x[i] = v
				w := v
				switch {
				case v < lh[0]:
					w, below = lh[0], below+1
				case v > lh[1]:
					w, above = lh[1], above+1
				default:
					inside++
				}
				want[i] = w
			}
			c.Call(&Args{Out: &x[0], Scr: (*byte)(unsafe.Pointer(&lh[0])),
				K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
			for i := range x {
				if math.Float32bits(x[i]) != math.Float32bits(want[i]) {
					t.Fatalf("[%v,%v] width %d: element %d is %v, want %v", lh[0], lh[1], n, i, x[i], want[i])
				}
			}
		}
		if inside == 0 || below == 0 || above == 0 {
			t.Fatalf("[%v,%v]: %d inside, %d below, %d above -- a bound went untested", lh[0], lh[1], inside, below, above)
		}
	}
}

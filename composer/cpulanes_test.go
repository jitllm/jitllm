package composer

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestComposedCPULanesAsDataEveryCopy executes on the CPU the radix sort's
// count of each element's digit before it in its block: j = the block's
// start + u reads word j as an address and is compared with i as data,
// and the two digits compared are the words' low four bits. The induction
// pass rewrites each unrolled copy's compare onto the lanes' iteration; a
// copy whose rewrite was lost compared every lane at the trip's first u,
// and the sort scattered most elements onto a few places. Held to the
// definition exactly, blocks of 16 and of 32, the
// digits of the sort's first, second and last passes.
func TestComposedCPULanesAsDataEveryCopy(t *testing.T) {
	U, F := ir.U32, ir.F32
	const R = 1
	ran := 0
	for _, c := range []struct{ S, s int }{{2000, 0}, {2000, 4}, {2048, 0}, {4096, 28}} {
		S, ds := c.S, uint32(c.s)
		B := selectBlock(S)
		sh := uint32(bits.TrailingZeros(uint(B)))
		in := Operand{Kind: OperandValue, Value: 0, Elem: U}
		out := Operand{Kind: OperandScratch, Name: "lanes.out", Elem: F}
		var x exprs
		r, i, u := x.coord(0), x.coord(1), x.coord(2)
		j := x.op(OpAdd, U, x.op(OpShl, U, x.op(OpShr, U, i, x.u(sh)), x.u(sh)), u)
		digit := func(k int) int { return x.op(OpSub, U, x.u(15), x.op(OpAnd, U, x.op(OpShr, U, k, x.u(ds)), x.u(15))) }
		dj := digit(x.load(0, x.op(OpAdd, U, x.op(OpMul, U, r, x.u(uint32(S))), j)))
		di := digit(x.at(0, []int{R, S}, r, i))
		hit := x.sel(F, x.op(OpLess, ir.Pred, dj, di), x.f(0), x.sel(F, x.op(OpLess, ir.Pred, di, dj), x.f(0), x.f(1)))
		x.sel(F, x.op(OpLess, ir.Pred, j, i), hit, x.f(0))
		p := Prim{Kind: PrimReduce, Shape: []int{R, S, B}, In: []Operand{in}, Out: out, Combine: OpAdd, Type: F,
			Expr: x.e, Facts: Facts{Extents: map[int]int{0: R * S}}}
		dev := cpuDevice(hostArm())
		reg := Default()
		kn, err := generate(p)
		if err != nil {
			t.Fatal(err)
		}
		ks, err := reg.Profiles[DeviceCPU].on(dev).run([]*Kernel{kn}, dev)
		if err != nil {
			t.Fatal(err)
		}
		vec := false
		for _, k := range ks {
			for _, l := range k.Loops {
				vec = vec || l.Lanes > 0 && l.Copies > 1
			}
		}
		pcs, err := reg.Lower(ks, dev)
		if err != nil {
			t.Fatal(err)
		}
		var codes []Code
		for _, c := range pcs {
			codes = append(codes, c.Code)
		}
		rng := rand.New(rand.NewSource(int64(S)))
		words := make([]uint32, R*S)
		xs := make([]float32, R*S)
		for i := range words {
			words[i] = rng.Uint32()
			xs[i] = math.Float32frombits(words[i])
		}
		poison := math.Float32bits(float32(math.NaN()))
		got := make([]uint32, R*S+16)
		for i := range got {
			got[i] = poison
		}
		if _, err := runCPUBound(hostArm(), codes, xs, 1, [3][]uint32{}, nil, func(o Operand) []uint32 {
			switch {
			case o.Name == out.Name:
				return got
			case o.Kind == OperandValue && o.Value == 0:
				return words
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		name := fmt.Sprintf("a row of %d, blocks of %d, digits from bit %d", S, B, ds)
		bad := 0
		for i := range S {
			s := float32(0)
			for j := i &^ (B - 1); j < i; j++ {
				if words[j]>>ds&15 == words[i]>>ds&15 {
					s++
				}
			}
			if g := math.Float32frombits(got[i]); g != s {
				if bad == 0 {
					t.Errorf("%s: element %d counts %v, want %v", name, i, g, s)
				}
				bad++
			}
		}
		for _, g := range got[R*S:] {
			if g != poison {
				t.Fatalf("%s: written past the row", name)
			}
		}
		if bad > 0 {
			t.Errorf("%s: %d of %d elements part", name, bad, S)
		}
		if !vec {
			t.Errorf("%s: no vector loop of several vectors a trip: the form under test was not selected", name)
		}
		ran++
	}
	if ran != 4 {
		t.Fatalf("%d of 4 shapes ran", ran)
	}
}

// TestComposedCPULanesAsData executes on the CPU a reduction whose vector
// loop's iteration is read both as an address and as data: each element's
// count of the words before it in its block of B above a threshold (the
// top-k selection's prefix counts), j = the block's start + u reading word
// j and compared with i. The vector loop over u, unrolled to several
// vectors a trip, must give each lane its own u in the compare -- once it
// gave every lane the trip's first, and stepped a trip by one vector where
// it covers several. Held to the definition exactly over a row of blocks
// of 16, 32 and 64, the vocabularies' widths among them.
func TestComposedCPULanesAsData(t *testing.T) {
	U, F := ir.U32, ir.F32
	ran := 0
	for _, R := range []int{1} {
		for _, S := range []int{2048, 4096, 51200, 65536, 151936} {
			B := selectBlock(S)
			sh := bits.TrailingZeros(uint(B))
			in := Operand{Kind: OperandValue, Value: 0, Elem: F}
			th := Operand{Kind: OperandScratch, Name: "lanes.t", Elem: U}
			out := Operand{Kind: OperandScratch, Name: "lanes.out", Elem: F}
			var x exprs
			r, i, u := x.coord(0), x.coord(1), x.coord(2)
			j := x.op(OpAdd, U, x.op(OpShl, U, x.op(OpShr, U, i, x.u(uint32(sh))), x.u(uint32(sh))), u)
			k := x.orderKey(x.load(0, x.op(OpAdd, U, x.op(OpMul, U, r, x.u(uint32(S))), j)))
			tt := x.at(1, []int{R}, r)
			w := x.sel(F, x.op(OpLess, ir.Pred, tt, k), x.f(1), x.sel(F, x.op(OpLess, ir.Pred, k, tt), x.f(0), x.f(1<<16)))
			x.sel(F, x.op(OpLess, ir.Pred, j, i), w, x.f(0))
			p := Prim{Kind: PrimReduce, Shape: []int{R, S, B}, In: []Operand{in, th}, Out: out, Combine: OpAdd, Type: F,
				Expr: x.e, Facts: Facts{Extents: map[int]int{0: R * S}}}
			dev := cpuDevice(hostArm())
			reg := Default()
			kn, err := generate(p)
			if err != nil {
				t.Fatal(err)
			}
			ks, err := reg.Profiles[DeviceCPU].on(dev).run([]*Kernel{kn}, dev)
			if err != nil {
				t.Fatal(err)
			}
			vec := false
			for _, k := range ks {
				for _, l := range k.Loops {
					vec = vec || l.Lanes > 0 && l.Copies > 1
				}
			}
			pcs, err := reg.Lower(ks, dev)
			if err != nil {
				t.Fatal(err)
			}
			var codes []Code
			for _, c := range pcs {
				codes = append(codes, c.Code)
			}
			rng := rand.New(rand.NewSource(int64(S)))
			xs := make([]float32, R*S)
			for i := range xs {
				xs[i] = float32(math.Round(rng.NormFloat64()*8) / 8)
			}
			key := func(v float32) uint32 {
				b := math.Float32bits(v)
				if b^0x80000000 < 1 {
					b = 0
				}
				if b > 0x7fffffff {
					return b ^ 0xffffffff
				}
				return b | 0x80000000
			}
			thr := make([]uint32, R+16)
			for r := range R {
				thr[r] = key(xs[r*S+rng.Intn(S)])
			}
			poison := math.Float32bits(float32(math.NaN()))
			got := make([]uint32, R*S+16)
			for i := range got {
				got[i] = poison
			}
			if _, err := runCPUBound(hostArm(), codes, xs, 1, [3][]uint32{}, nil, func(o Operand) []uint32 {
				switch {
				case o.Name == th.Name:
					return thr
				case o.Name == out.Name:
					return got
				case o.Kind == OperandValue && o.Value == 0:
					return unsafe.Slice((*uint32)(unsafe.Pointer(&xs[0])), len(xs))
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			name := fmt.Sprintf("%d rows of %d, blocks of %d", R, S, B)
			bad := 0
			for r := range R {
				for i := range S {
					s := float32(0)
					for j := i &^ (B - 1); j < i; j++ {
						switch kk := key(xs[r*S+j]); {
						case thr[r] < kk:
							s++
						case kk == thr[r]:
							s += 1 << 16
						}
					}
					if g := math.Float32frombits(got[r*S+i]); g != s {
						if bad == 0 {
							t.Errorf("%s: element %d of row %d counts %v, want %v", name, i, r, g, s)
						}
						bad++
					}
				}
			}
			for _, g := range got[R*S:] {
				if g != poison {
					t.Fatalf("%s: written past the rows", name)
				}
			}
			if bad > 0 {
				t.Errorf("%s: %d of %d elements part", name, bad, R*S)
			}
			if !vec {
				t.Errorf("%s: no vector loop of several vectors a trip: the form under test was not selected", name)
			}
			ran++
		}
	}
	if ran != 5 {
		t.Fatalf("%d of 5 shapes ran", ran)
	}
}

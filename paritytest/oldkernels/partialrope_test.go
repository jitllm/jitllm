package oldkernels_test

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestPartialRotaryMatchesTheHost runs nRot < headDim, which
// TestAttentionAgainstRef (nRot = hd) does not. RoPERows writes only the
// dimensions it rotates, so the tier must copy the tail first; the host rotates
// in place and passes the tail through, which is what this compares against.
// Both pairings are covered, because NEOX (i, i+nRot/2) and normal (2i, 2i+1)
// disagree about which elements rotate.
func TestPartialRotaryMatchesTheHost(t *testing.T) {
	d := dev(t)
	const (
		nh, hd = 4, 8
		base   = 10000.0
		pos    = 3
	)
	for _, neox := range []bool{false, true} {
		for _, nRot := range []int{hd, hd / 2, hd / 4} {
			name := "norm"
			if neox {
				name = "neox"
			}
			t.Run(name+"/nRot"+itoa(nRot), func(t *testing.T) {
				src := ramp(nh*hd, func(i int) float32 {
					return float32(math.Sin(float64(i)*0.7)) + 1
				})
				// The host: rotate the first nRot dims of each head in place and
				// leave the rest exactly as they were.
				want := append([]float32(nil), src...)
				for h := 0; h < nh; h++ {
					ropeRef(nn.Rope{NRot: nRot, Base: base, Neox: neox}, want[h*hd:(h+1)*hd], pos)
				}
				cs := make([]float32, nRot)
				theta := float64(pos)
				rs := math.Pow(base, -2/float64(nRot))
				for p := 0; p < nRot/2; p++ {
					cs[2*p], cs[2*p+1] = float32(math.Cos(theta)), float32(math.Sin(theta))
					theta *= rs
				}

				// The device: the tier's sequence -- copy the whole head, then
				// rotate the prefix over the top of it.
				dst, _ := d.Alloc(nh * hd * 4)
				bsrc, _ := d.Alloc(nh * hd * 4)
				bcs, _ := d.Alloc(nRot * 4)
				bz, _ := d.Alloc(4)
				defer func() { dst.Free(); bsrc.Free(); bcs.Free(); bz.Free() }()
				// Poison the destination (RULE 13): a fresh buffer is usually zero,
				// which would let a missing tail copy pass as a small number.
				poison := make([]float32, nh*hd)
				for i := range poison {
					poison[i] = float32(math.NaN())
				}
				dst.Write(f32b(poison))
				bsrc.Write(f32b(src))
				bcs.Write(f32b(cs))
				bz.Write(u32b1(0))

				if nRot < hd {
					cp, err := oldkernels.CopyAtRows(nh*hd, 1)
					if err != nil {
						t.Fatal(err)
					}
					kc, err := d.Compile(cp)
					if err != nil {
						t.Fatalf("compile CopyAtRows: %v", err)
					}
					if err := kc.Launch((nh*hd+127)/128, 128, bsrc, dst, bz); err != nil {
						t.Fatal(err)
					}
					kc.Close()
				}
				rope, err := oldkernels.RoPERows(nh, hd, nRot, neox, 1)
				if err != nil {
					t.Fatal(err)
				}
				kr, err := d.Compile(rope)
				if err != nil {
					t.Fatalf("compile RoPE: %v", err)
				}
				defer kr.Close()
				if err := kr.Launch(1, 128, bsrc, bcs, bz, dst); err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, nh*hd*4)
				dst.Read(raw)
				got := make([]float32, nh*hd)
				b2f(raw, got)

				for i := range got {
					// Non-finite first: `NaN > tol` is false, so a tolerance alone
					// accepts an unwritten element.
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
						t.Fatalf("%s nRot=%d: element %d (head %d, dim %d) is %g -- "+
							"never written; the tail copy is missing",
							name, nRot, i, i/hd, i%hd, got[i])
					}
					if math.Abs(float64(got[i])-float64(want[i])) > 1e-5 {
						t.Fatalf("%s nRot=%d: element %d (head %d, dim %d) = %g, host says %g",
							name, nRot, i, i/hd, i%hd, got[i], want[i])
					}
				}
			})
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestPartialRotaryKCacheMatchesTheHost is the same question for the K cache,
// which the q test cannot reach. The cache is transposed: CopyAtRowsT and
// RoPERowsT write element i of a row at i*stride + pos, so a single position
// can look right with a wrong stride. This writes two positions and checks
// both.
func TestPartialRotaryKCacheMatchesTheHost(t *testing.T) {
	d := dev(t)
	const (
		nkv, hd = 2, 8
		kvDim   = nkv * hd
		maxSeq  = 16
		stride  = maxSeq + 1 // the tier's kStride: one slot past MaxSeq
		base    = 10000.0
		nRot    = hd / 2
	)
	for _, neox := range []bool{false, true} {
		name := "norm"
		if neox {
			name = "neox"
		}
		t.Run(name, func(t *testing.T) {
			cache, _ := d.Alloc(kvDim * stride * 4)
			bsrc, _ := d.Alloc(kvDim * 4)
			bcs, _ := d.Alloc(nRot * 4)
			boff, _ := d.Alloc(4)
			defer func() { cache.Free(); bsrc.Free(); bcs.Free(); boff.Free() }()

			poison := make([]float32, kvDim*stride)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			cache.Write(f32b(poison))

			cp, err := oldkernels.CopyAtRowsT(kvDim, 1, stride)
			if err != nil {
				t.Fatal(err)
			}
			kc, err := d.Compile(cp)
			if err != nil {
				t.Fatalf("compile CopyAtRowsT: %v", err)
			}
			defer kc.Close()
			rope, err := oldkernels.RoPERowsT(nkv, hd, nRot, neox, 1, stride)
			if err != nil {
				t.Skipf("no transposed rope for this shape: %v", err)
			}
			kr, err := d.Compile(rope)
			if err != nil {
				t.Fatalf("compile RoPERowsT: %v", err)
			}
			defer kr.Close()

			want := map[[2]int][]float32{}
			for _, pos := range []int{0, 1} {
				src := ramp(kvDim, func(i int) float32 {
					return float32(math.Cos(float64(i+pos*13)*0.31)) + 1
				})
				w := append([]float32(nil), src...)
				for h := 0; h < nkv; h++ {
					ropeRef(nn.Rope{NRot: nRot, Base: base, Neox: neox}, w[h*hd:(h+1)*hd], pos)
				}
				want[[2]int{pos, 0}] = w

				cs := make([]float32, nRot)
				theta := float64(pos)
				rs := math.Pow(base, -2/float64(nRot))
				for p := 0; p < nRot/2; p++ {
					cs[2*p], cs[2*p+1] = float32(math.Cos(theta)), float32(math.Sin(theta))
					theta *= rs
				}
				bsrc.Write(f32b(src))
				bcs.Write(f32b(cs))
				boff.Write(u32b1(uint32(pos)))
				if err := kc.Launch((kvDim+127)/128, 128, bsrc, cache, boff); err != nil {
					t.Fatal(err)
				}
				if err := kr.Launch(1, 128, bsrc, bcs, boff, cache); err != nil {
					t.Fatal(err)
				}
			}

			raw := make([]byte, kvDim*stride*4)
			cache.Read(raw)
			got := make([]float32, kvDim*stride)
			b2f(raw, got)
			for _, pos := range []int{0, 1} {
				w := want[[2]int{pos, 0}]
				for i := 0; i < kvDim; i++ {
					g := got[i*stride+pos]
					if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
						t.Fatalf("%s pos=%d: kv element %d (head %d, dim %d) is %g -- "+
							"never written", name, pos, i, i/hd, i%hd, g)
					}
					if math.Abs(float64(g)-float64(w[i])) > 1e-5 {
						t.Fatalf("%s pos=%d: kv element %d (head %d, dim %d) = %g, host says %g",
							name, pos, i, i/hd, i%hd, g, w[i])
					}
				}
			}
		})
	}
}

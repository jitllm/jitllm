package oldkernels_test

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

func u32b1(v uint32) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
}

// TestAttentionAgainstRef runs RoPE, the score loop, softmax and the weighted
// sum on the device and compares against nn's reference, at a GQA shape (4 query
// heads over 2 kv heads) because that is where the head-to-kv-head division can
// be backwards and still produce numbers.
func TestAttentionAgainstRef(t *testing.T) {
	d := dev(t)
	const (
		nh, hd, nkv, maxSeq = 4, 8, 2, 16
		gqa                 = nh / nkv
		kvDim               = nkv * hd
		n                   = 5 // positions in the cache
		nRot                = hd
		base                = 10000.0
		pos                 = 3
	)
	scale := float32(1 / math.Sqrt(hd))

	q := ramp(nh*hd, func(i int) float32 { return float32(math.Sin(float64(i) * 0.7)) })
	kc := ramp(maxSeq*kvDim, func(i int) float32 { return float32(math.Cos(float64(i) * 0.31)) })
	vc := ramp(maxSeq*kvDim, func(i int) float32 { return float32(math.Sin(float64(i) * 0.19)) })

	// --- RoPE, against nn.RoPE on the same input.
	wantQ := make([]float32, nh*hd)
	for i, v := range q {
		wantQ[i] = v
	}
	for h := 0; h < nh; h++ {
		ropeRef(nn.Rope{NRot: nRot, Base: base}, wantQ[h*hd:(h+1)*hd], pos)
	}
	cs := make([]float32, nRot) // cos,sin interleaved, nRot/2 pairs
	theta := float64(pos)
	rs := math.Pow(base, -2/float64(nRot))
	for p := 0; p < nRot/2; p++ {
		cs[2*p], cs[2*p+1] = float32(math.Cos(theta)), float32(math.Sin(theta))
		theta *= rs
	}
	rope, err := oldkernels.RoPERows(nh, hd, nRot, false, 1)
	if err != nil {
		t.Fatal(err)
	}
	kr, err := d.Compile(rope)
	if err != nil {
		t.Fatalf("compile RoPE: %v", err)
	}
	defer kr.Close()
	bsrc, _ := d.Alloc(nh * hd * 4)
	bcs, _ := d.Alloc(nRot * 4)
	bzero, _ := d.Alloc(4)
	bq, _ := d.Alloc(nh * hd * 4)
	defer func() { bsrc.Free(); bcs.Free(); bzero.Free(); bq.Free() }()
	bsrc.Write(f32b(q))
	bcs.Write(f32b(cs))
	bzero.Write(u32b1(0))
	if err := kr.Launch(1, 128, bsrc, bcs, bzero, bq); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, nh*hd*4)
	bq.Read(raw)
	gotQ := make([]float32, nh*hd)
	b2f(raw, gotQ)
	// Non-finite is checked first, because `NaN > tol` is false and a
	// tolerance comparison would accept a NaN. The rope's destination is the
	// buffer a partial rotary leaves partly unwritten.
	for i := range gotQ {
		if math.IsNaN(float64(gotQ[i])) || math.IsInf(float64(gotQ[i]), 0) {
			t.Fatalf("rope[%d] = %g, which is not finite -- a tolerance test "+
				"would have accepted it", i, gotQ[i])
		}
		if math.Abs(float64(gotQ[i])-float64(wantQ[i])) > 1e-5 {
			t.Fatalf("rope[%d] = %g, nn.RoPE says %g", i, gotQ[i], wantQ[i])
		}
	}

	// --- scores, softmax, accumulate.
	sc, err := oldkernels.AttnScoresTiled(nh, hd, kvDim, gqa, maxSeq, scale, 1, 1, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	ac, err := oldkernels.AttnAcc(nh, hd, kvDim, gqa, maxSeq)
	if err != nil {
		t.Fatal(err)
	}
	ksc, err := d.Compile(sc)
	if err != nil {
		t.Fatalf("compile AttnScores: %v", err)
	}
	defer ksc.Close()
	kac, err := d.Compile(ac)
	if err != nil {
		t.Fatalf("compile AttnAcc: %v", err)
	}
	defer kac.Close()

	bk, _ := d.Alloc(maxSeq * kvDim * 4)
	bv, _ := d.Alloc(maxSeq * kvDim * 4)
	bn, _ := d.Alloc(4)
	batt, _ := d.Alloc(nh * maxSeq * 4)
	bo, _ := d.Alloc(nh * hd * 4)
	defer func() { bk.Free(); bv.Free(); bn.Free(); batt.Free(); bo.Free() }()
	bk.Write(f32b(kc))
	bv.Write(f32b(vc))
	bn.Write(u32b1(n))
	if err := ksc.Launch((nh*n+127)/128, 128, bq, bk, bn, batt); err != nil {
		t.Fatal(err)
	}
	bprob, _ := d.Alloc(nh * maxSeq * 4)
	defer bprob.Free()

	// Both softmax shapes, through the same attention: the warp reduction that
	// ships where a device carries a subgroup shuffle, and the thread-per-head
	// fallback. They must reach the same attention output, not merely each
	// produce a plausible one.
	for _, lanes := range []int{1, 32} {
		sm, err := oldkernels.SoftmaxRows(nh, maxSeq, lanes, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		ksm, err := d.Compile(sm)
		if err != nil {
			t.Fatalf("compile Softmax(lanes=%d): %v", lanes, err)
		}
		groups, width := smLaunch(nh, lanes)
		if err := ksm.Launch(groups, width, batt, bn, bprob); err != nil {
			ksm.Close()
			t.Fatal(err)
		}
		ksm.Close()
		if err := kac.Launch((nh*hd+127)/128, 128, bprob, bv, bn, bo); err != nil {
			t.Fatal(err)
		}
		raw = make([]byte, nh*hd*4)
		bo.Read(raw)
		got := make([]float32, nh*hd)
		b2f(raw, got)

		// The reference, exactly as engine/model/forward.go does it.
		for h := 0; h < nh; h++ {
			kvh := h / gqa
			a := make([]float64, n)
			for tt := 0; tt < n; tt++ {
				var s float64
				for i := 0; i < hd; i++ {
					s += float64(gotQ[h*hd+i]) * float64(kc[tt*kvDim+kvh*hd+i])
				}
				a[tt] = s * float64(scale)
			}
			oracle.Softmax(a)
			for i := 0; i < hd; i++ {
				var acc float64
				for tt := 0; tt < n; tt++ {
					acc += a[tt] * float64(vc[tt*kvDim+kvh*hd+i])
				}
				if g := float64(got[h*hd+i]); math.Abs(g-acc) > 1e-5 {
					t.Fatalf("lanes=%d head %d elem %d: got %g, ref %g", lanes, h, i, g, acc)
				}
			}
		}
	}
	t.Logf("%s: RoPE, scores, softmax and accumulate agree with nn at gqa=%d, n=%d", d.API(), gqa, n)
}

// smLaunch is the geometry tier.Layers uses: one workgroup per head for the
// warp kernel, one 64-wide group for the scalar one.
func smLaunch(nHeads, lanes int) (groups, width int) {
	if lanes == 32 {
		return nHeads, 32
	}
	return (nHeads + 63) / 64, 64
}

// TestSoftmaxWarpMatchesScalar walks the row lengths where a strided warp
// reduction can be wrong and a single-thread pass cannot.
//
// The interesting lengths are not the round ones: n=31 leaves one lane idle,
// n=32 gives every lane one position, n=33 gives lane 0 two, and n=137
// spreads the maximum across lanes so a reduction that only sees its own
// stride picks the wrong one.
func TestSoftmaxWarpMatchesScalar(t *testing.T) {
	d := dev(t)
	const nh, maxSeq = 3, 256
	a := ramp(nh*maxSeq, func(i int) float32 {
		return float32(4 * math.Sin(0.37*float64(i)))
	})
	ba, _ := d.Alloc(len(a) * 4)
	bn, _ := d.Alloc(4)
	bo, _ := d.Alloc(len(a) * 4)
	defer func() { ba.Free(); bn.Free(); bo.Free() }()
	ba.Write(f32b(a))

	sm, err := oldkernels.SoftmaxRows(nh, maxSeq, 32, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	k, err := d.Compile(sm)
	if err != nil {
		t.Fatalf("compile Softmax(lanes=32): %v", err)
	}
	defer k.Close()

	for _, n := range []int{1, 2, 31, 32, 33, 64, 100, 137, 256} {
		bn.Write(u32b1(uint32(n)))
		groups, width := smLaunch(nh, 32)
		if err := k.Launch(groups, width, ba, bn, bo); err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		raw := make([]byte, len(a)*4)
		bo.Read(raw)
		got := make([]float32, len(a))
		b2f(raw, got)
		for h := 0; h < nh; h++ {
			want := make([]float64, n)
			for i := range want {
				want[i] = float64(a[h*maxSeq+i])
			}
			oracle.Softmax(want)
			for i := range want {
				if g := float64(got[h*maxSeq+i]); math.Abs(g-want[i]) > 1e-5 {
					t.Fatalf("n=%d head %d pos %d: got %g, nn.Softmax says %g",
						n, h, i, g, want[i])
				}
			}
		}
	}
	t.Logf("%s: the 32-lane softmax agrees with nn.Softmax at every row length", d.API())
}

// TestLoopNRunsZeroTimes pins the contract LoopN advertises. PTX's loop is a
// countdown do-while, so a zero count would wrap to 2^32 iterations without the
// entry guard -- and nothing in attention ever passes zero, which is exactly why
// it would not have been noticed.
func TestLoopNRunsZeroTimes(t *testing.T) {
	d := dev(t)
	const nh, maxSeq = 2, 8
	// Both arms: the warp kernel derives a per-lane trip count from n, so zero
	// has to survive that arithmetic as well as the loop.
	for _, lanes := range []int{1, 32} {
		sm, err := oldkernels.SoftmaxRows(nh, maxSeq, lanes, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		k, err := d.Compile(sm)
		if err != nil {
			t.Fatalf("compile lanes=%d: %v", lanes, err)
		}
		a := ramp(nh*maxSeq, func(i int) float32 { return float32(i) })
		ba, _ := d.Alloc(len(a) * 4)
		bn, _ := d.Alloc(4)
		bo, _ := d.Alloc(len(a) * 4)
		ba.Write(f32b(a))
		bo.Write(f32b(a))
		bn.Write(u32b1(0))
		groups, width := smLaunch(nh, lanes)
		if err := k.Launch(groups, width, ba, bn, bo); err != nil {
			t.Fatalf("launch lanes=%d: %v", lanes, err)
		}
		raw := make([]byte, len(a)*4)
		bo.Read(raw)
		got := make([]float32, len(a))
		b2f(raw, got)
		for i := range got {
			if got[i] != a[i] {
				t.Fatalf("lanes=%d: n=0 modified a[%d]: %g -> %g", lanes, i, a[i], got[i])
			}
		}
		k.Close()
		ba.Free()
		bn.Free()
		bo.Free()
	}
}

// ropeRef rotates head by r at pos the way the host does: the table from
// nn.Rope, the rotation from the oracle.
func ropeRef(r nn.Rope, head []float32, pos int) {
	cs := make([]float32, r.NRot)
	r.Table(cs, pos)
	oracle.RopeApply(head, cs, r.Neox)
}

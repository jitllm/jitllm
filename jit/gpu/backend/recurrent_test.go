package backend_test

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestRecurrentKernelsOnEveryDevice runs the hybrid's recurrent kernels on real
// hardware, against the float64 oracle, on every backend this host has (not
// just the "gpu" default, which resolves to CUDA). A whole-model comparison
// cannot localise a bug to one kernel; this can.
func TestRecurrentKernelsOnEveryDevice(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			t.Logf("device: %s", d.Name())
			t.Run("conv1d", func(t *testing.T) { convCase(t, d) })
			t.Run("delta", func(t *testing.T) { deltaCase(t, d, false) })
			t.Run("delta-chan", func(t *testing.T) { deltaCase(t, d, true) })
			t.Run("splitheadgate", func(t *testing.T) { splitGateCase(t, d) })
			// A norm whose partial covers one element: partsFor picks the
			// largest power of two up to 64 that divides NEmbd, so NEmbd 32
			// or 64 gives a trip count of 1, reached only by tiny fixtures.
			for _, c := range []struct{ k, parts int }{{32, 32}, {64, 64}, {2048, 64}, {96, 32}} {
				t.Run("normpart-"+itoa32(c.k)+"x"+itoa32(c.parts),
					func(t *testing.T) { normPartCase(t, d, c.k, c.parts) })
			}
			// Partial rotary (qwen3next: 64 of 256): nRot < headDim, so the
			// tail of every head is left unrotated and must have been copied
			// there first.
			for _, neox := range []bool{false, true} {
				n := "norm"
				if neox {
					n = "neox"
				}
				t.Run("rope-partial-"+n, func(t *testing.T) { ropeCase(t, d, 256, 64, neox) })
				t.Run("rope-full-"+n, func(t *testing.T) { ropeCase(t, d, 64, 64, neox) })
			}
			// XD-RoPE (HunyuanVL): a pair's halves by tables of their own, over
			// two rows, a partial tail and a whole head.
			t.Run("rope-split-partial", func(t *testing.T) { ropeSplitCase(t, d, 256, 64) })
			t.Run("rope-split-full", func(t *testing.T) { ropeSplitCase(t, d, 32, 32) })
		})
	}
}

// convCase runs Conv1dRows and Conv1dShift and checks both outputs. A kernel
// that convolves correctly but does not shift the state is right for the first
// token and drifts from the second.
//
// The window sits at slot 2 of a three-slot pool, the others noise in and
// poison out: a kernel that reads or writes slot 0 regardless of the
// descriptor is right for a session that happens to own it and wrong for every
// other.
func convCase(t *testing.T, d backend.Device) {
	const taps, chans, rows = 4, 64, 1
	rng := rand.New(rand.NewSource(11))
	st := make([]float32, (taps-1)*chans)
	x := make([]float32, chans)
	w := make([]float32, taps*chans)
	for i := range st {
		st[i] = float32(rng.NormFloat64())
	}
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64() * 0.3)
	}

	const slots, slot = 3, 2
	per := len(st)
	g := newGPU(t, d)
	defer g.free()
	bSt := g.up(f32bytes(inPool(rng, slots, slot, st)))
	bX, bW := g.up(f32bytes(x)), g.up(f32bytes(w))
	bOut := g.up(make([]byte, chans*4))
	bStOut := g.up(f32bytes(poisonPool(slots * per)))
	bRows := g.up(recDesc(rows, slot, slot))

	kr, err := mainkernels.Conv1dRows(taps, chans, rows)
	if err != nil {
		t.Fatal(err)
	}
	ks, err := mainkernels.Conv1dShift(taps, chans, rows)
	if err != nil {
		t.Fatal(err)
	}
	cr, err := d.Compile(kr)
	if err != nil {
		t.Fatal(err)
	}
	defer cr.Close()
	cs, err := d.Compile(ks)
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()
	// Each at the workgroup it was lowered for: Metal's NTID is the declared
	// width, so a narrower launch leaves part of the state unwritten.
	wr, ws := kr.Group[0], ks.Group[0]
	if err := cr.Launch((rows*chans+wr-1)/wr, wr, bSt, bX, bW, bOut, bRows); err != nil {
		t.Fatal(err)
	}
	if err := cs.Launch((((taps-1)*chans)+ws-1)/ws, ws, bSt, bX, bStOut, bRows); err != nil {
		t.Fatal(err)
	}

	wantOut := make([]float64, chans)
	wantSt := f64of(st)
	oracle.Conv1d(wantOut, wantSt, f64of(w), f64of(x), taps, chans)
	cmpF32(t, d, "conv out", readF32(t, bOut, chans), wantOut, 1e-4)
	pool := readF32(t, bStOut, slots*per)
	cmpF32(t, d, "conv state", pool[slot*per:(slot+1)*per], wantSt, 1e-4)
	poisonKept(t, "conv state", pool, per, slot)
}

// poison is what an output pool holds where no slot was written.
const poison = float32(-7.25)

// inPool is a pool of slots states of len(st) floats with st at slot and noise
// elsewhere.
func inPool(rng *rand.Rand, slots, slot int, st []float32) []float32 {
	p := make([]float32, slots*len(st))
	for i := range p {
		p[i] = float32(rng.NormFloat64())
	}
	copy(p[slot*len(st):], st)
	return p
}

func poisonPool(n int) []float32 {
	p := make([]float32, n)
	for i := range p {
		p[i] = poison
	}
	return p
}

// poisonKept fails when a slot other than the written one moved.
func poisonKept(t *testing.T, what string, pool []float64, per int, written ...int) {
	t.Helper()
	for i, v := range pool {
		if slices.Contains(written, i/per) {
			continue
		}
		if float32(v) != poison {
			t.Fatalf("%s: slot %d element %d is %g -- a slot the descriptor did not name was written",
				what, i/per, i%per, v)
		}
	}
}

// recDesc is a recurrent descriptor: m real rows, then each sequence's in and
// out slot.
func recDesc(m uint32, slots ...uint32) []byte {
	b := u32le(m)
	for _, s := range slots {
		b = append(b, u32le(s)...)
	}
	return b
}

// deltaCase runs one token of the delta rule and checks the output and the
// state, per-head or per-channel decay. The state is read at slot 2 of a pool
// and written at slot 0 of another: the shared form's in and out slots differ.
func deltaCase(t *testing.T, d backend.Device, chanDecay bool) {
	const vHeads, vDim, kDim, rep = 2, 32, 32, 1
	rng := rand.New(rand.NewSource(23))
	st := make([]float32, vHeads*vDim*kDim)
	k := make([]float32, vHeads*kDim)
	q := make([]float32, vHeads*kDim)
	v := make([]float32, vHeads*vDim)
	beta := make([]float32, vHeads)
	nDec := vHeads
	if chanDecay {
		nDec = vHeads * kDim
	}
	decay := make([]float32, nDec)
	for i := range st {
		st[i] = float32(rng.NormFloat64())
	}
	for _, s := range [][]float32{k, q, v} {
		for i := range s {
			s[i] = float32(rng.NormFloat64() * 0.2)
		}
	}
	for i := range beta {
		beta[i] = float32(rng.Float64())
	}
	for i := range decay {
		decay[i] = float32(0.5 + 0.4*rng.Float64())
	}

	const slots, in, out = 3, 2, 0
	per := len(st)
	g := newGPU(t, d)
	defer g.free()
	bS := g.up(f32bytes(inPool(rng, slots, in, st)))
	bK, bQ, bV := g.up(f32bytes(k)), g.up(f32bytes(q)), g.up(f32bytes(v))
	bD, bB := g.up(f32bytes(decay)), g.up(f32bytes(beta))
	bOut := g.up(make([]byte, vHeads*vDim*4))
	bSOut := g.up(f32bytes(poisonPool(slots * per)))
	bN := g.up(recDesc(1, in, out))

	mk := mainkernels.GatedDeltaStep
	if chanDecay {
		mk = mainkernels.GatedDeltaStepChan
	}
	kk, err := mk(vHeads, vDim, kDim, rep)
	if err != nil {
		t.Fatal(err)
	}
	c, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	n := vHeads * vDim
	if err := c.Launch((n+127)/128, 128, bS, bK, bQ, bV, bD, bB, bOut, bSOut, bN); err != nil {
		t.Fatal(err)
	}

	// The oracle, per value head, on the transposed state this tree keeps.
	wantOut := make([]float64, vHeads*vDim)
	wantSt := f64of(st)
	for vh := 0; vh < vHeads; vh++ {
		kh := vh / rep
		o := wantOut[vh*vDim : (vh+1)*vDim]
		mat := wantSt[vh*vDim*kDim:][:vDim*kDim]
		kv := f64of(k[kh*kDim : (kh+1)*kDim])
		qv := f64of(q[kh*kDim : (kh+1)*kDim])
		vv := f64of(v[vh*vDim : (vh+1)*vDim])
		if chanDecay {
			oracle.GatedDeltaChan(o, mat, kv, qv, vv,
				f64of(decay[vh*kDim:(vh+1)*kDim]), float64(beta[vh]))
			continue
		}
		oracle.GatedDelta(o, mat, kv, qv, vv, float64(decay[vh]), float64(beta[vh]))
	}
	cmpF32(t, d, "delta out", readF32(t, bOut, vHeads*vDim), wantOut, 1e-4)
	pool := readF32(t, bSOut, slots*per)
	cmpF32(t, d, "delta state", pool[out*per:(out+1)*per], wantSt, 1e-4)
	poisonKept(t, "delta state", pool, per, out)
}

// splitGateCase runs the attention out-gate's deinterleave.
//
// It is the only kernel in this family that loads at a non-zero constant
// offset (b.Load(pSrc, src, headDim)); a backend that drops or mis-scales it
// reads the query where the gate should be.
func splitGateCase(t *testing.T, d backend.Device) {
	const nHead, headDim, rows = 4, 32, 1
	qdim := nHead * headDim
	rng := rand.New(rand.NewSource(37))
	src := make([]float32, rows*2*qdim)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	g := newGPU(t, d)
	defer g.free()
	bSrc := g.up(f32bytes(src))
	bQ, bG := g.up(make([]byte, rows*qdim*4)), g.up(make([]byte, rows*qdim*4))
	kk, err := mainkernels.SplitHeadGate(nHead, headDim, rows)
	if err != nil {
		t.Fatal(err)
	}
	c, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if err := c.Launch((rows*qdim+127)/128, 128, bSrc, bQ, bG); err != nil {
		t.Fatal(err)
	}
	wantQ := make([]float64, rows*qdim)
	wantG := make([]float64, rows*qdim)
	for r := 0; r < rows; r++ {
		for h := 0; h < nHead; h++ {
			for j := 0; j < headDim; j++ {
				o := r*qdim + h*headDim + j
				s := r*2*qdim + h*2*headDim + j
				wantQ[o] = float64(src[s])
				wantG[o] = float64(src[s+headDim])
			}
		}
	}
	cmpF32(t, d, "splitheadgate q", readF32(t, bQ, rows*qdim), wantQ, 0)
	cmpF32(t, d, "splitheadgate gate", readF32(t, bG, rows*qdim), wantG, 0)
}

// ropeCase runs RoPERows at a head width and a rotated width, and checks that
// the unrotated tail is left exactly as the destination already held it: the
// tier copies the tail first, and a kernel that wrote it would be wrong only on
// architectures that rotate part of a head.
func ropeCase(t *testing.T, d backend.Device, headDim, nRot int, neox bool) {
	const nHeads, rows = 2, 1
	n := nHeads * headDim
	rng := rand.New(rand.NewSource(int64(headDim*100 + nRot)))
	src := make([]float32, n)
	dst := make([]float32, n)
	cs := make([]float32, nRot)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
		dst[i] = float32(rng.NormFloat64()) // a SENTINEL: the tail must survive
	}
	for p := 0; p < nRot/2; p++ {
		a := 0.3 + 0.1*float64(p)
		cs[2*p], cs[2*p+1] = float32(math.Cos(a)), float32(math.Sin(a))
	}
	g := newGPU(t, d)
	defer g.free()
	bSrc, bCS := g.up(f32bytes(src)), g.up(f32bytes(cs))
	bOff := g.up(u32le(0))
	bDst := g.up(f32bytes(dst))
	kk, err := mainkernels.RoPERows(nHeads, headDim, nRot, neox, rows)
	if err != nil {
		t.Fatal(err)
	}
	c, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	threads := nHeads * (nRot / 2) * rows
	if err := c.Launch((threads+127)/128, 128, bSrc, bCS, bOff, bDst); err != nil {
		t.Fatal(err)
	}
	want := make([]float64, n)
	for i := range want {
		want[i] = float64(dst[i]) // untouched unless rotated
	}
	for h := 0; h < nHeads; h++ {
		for p := 0; p < nRot/2; p++ {
			var i0, i1 int
			if neox {
				i0, i1 = h*headDim+p, h*headDim+p+nRot/2
			} else {
				i0, i1 = h*headDim+2*p, h*headDim+2*p+1
			}
			co, si := float64(cs[2*p]), float64(cs[2*p+1])
			x0, x1 := float64(src[i0]), float64(src[i1])
			want[i0], want[i1] = x0*co-x1*si, x0*si+x1*co
		}
	}
	cmpF32(t, d, "rope", readF32(t, bDst, n), want, 1e-5)
}

// ropeSplitCase runs RoPERowsSplit over two rows, each with its own tables A
// and B, and checks each pair's halves took their own table's angle -- the
// first x0*cA - x1*sA, the second x0*sB + x1*cB -- and the tail survived. A
// kernel that read A for both halves fails the second half of every pair.
func ropeSplitCase(t *testing.T, d backend.Device, headDim, nRot int) {
	const nHeads, rows = 2, 2
	n := nHeads * headDim
	rng := rand.New(rand.NewSource(int64(headDim*7 + nRot)))
	src := make([]float32, rows*n)
	dst := make([]float32, rows*n)
	cs := make([]float32, rows*2*nRot)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
		dst[i] = float32(rng.NormFloat64())
	}
	for r := 0; r < rows; r++ {
		for p := 0; p < nRot/2; p++ {
			a, b := 0.3+0.1*float64(p)+float64(r), 1.7-0.05*float64(p)+0.3*float64(r)
			o := r * 2 * nRot
			cs[o+2*p], cs[o+2*p+1] = float32(math.Cos(a)), float32(math.Sin(a))
			cs[o+nRot+2*p], cs[o+nRot+2*p+1] = float32(math.Cos(b)), float32(math.Sin(b))
		}
	}
	offs := make([]byte, 0, 4*rows)
	for r := 0; r < rows; r++ {
		offs = append(offs, u32le(uint32(r*n))...)
	}
	g := newGPU(t, d)
	defer g.free()
	bSrc, bCS, bOff, bDst := g.up(f32bytes(src)), g.up(f32bytes(cs)), g.up(offs), g.up(f32bytes(dst))
	kk, err := mainkernels.RoPERowsSplit(nHeads, headDim, nRot, rows, 0)
	if err != nil {
		t.Fatal(err)
	}
	c, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	threads := nHeads * (nRot / 2) * rows
	if err := c.Launch((threads+127)/128, 128, bSrc, bCS, bOff, bDst); err != nil {
		t.Fatal(err)
	}
	want := make([]float64, rows*n)
	for i := range want {
		want[i] = float64(dst[i])
	}
	for r := 0; r < rows; r++ {
		o := r * 2 * nRot
		for h := 0; h < nHeads; h++ {
			for p := 0; p < nRot/2; p++ {
				i0, i1 := r*n+h*headDim+p, r*n+h*headDim+p+nRot/2
				ca, sa := float64(cs[o+2*p]), float64(cs[o+2*p+1])
				cb, sb := float64(cs[o+nRot+2*p]), float64(cs[o+nRot+2*p+1])
				x0, x1 := float64(src[i0]), float64(src[i1])
				want[i0], want[i1] = x0*ca-x1*sa, x0*sb+x1*cb
			}
		}
	}
	cmpF32(t, d, "rope split", readF32(t, bDst, rows*n), want, 1e-5)
}

// normPartCase runs the norm's partial pass and checks every partial.
//
// The trip count is the point: a backend that lowered a one-trip ir.Loop as
// zero-trip would leave every partial at zero. Only small fixtures reach it.
func normPartCase(t *testing.T, d backend.Device, k, parts int) {
	const rows = 1
	rng := rand.New(rand.NewSource(int64(k*31 + parts)))
	x := make([]float32, k*rows)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	g := newGPU(t, d)
	defer g.free()
	bX := g.up(f32bytes(x))
	bOut := g.up(make([]byte, parts*rows*4))
	kk, err := mainkernels.NormPartRows(k, parts, rows)
	if err != nil {
		t.Fatal(err)
	}
	c, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if err := c.Launch((parts*rows+127)/128, 128, bX, bOut); err != nil {
		t.Fatal(err)
	}
	per := k / parts
	want := make([]float64, parts*rows)
	for p := range want {
		var acc float64
		for i := 0; i < per; i++ {
			v := float64(x[p*per+i])
			acc += v * v
		}
		want[p] = acc
	}
	cmpF32(t, d, "normpart", readF32(t, bOut, parts*rows), want, 1e-4)
}

func f64of(x []float32) []float64 {
	o := make([]float64, len(x))
	for i, v := range x {
		o[i] = float64(v)
	}
	return o
}

func cmpF32(t *testing.T, d backend.Device, what string, got, want []float64, tol float64) {
	t.Helper()
	worst, at := 0.0, -1
	for i := range want {
		if e := math.Abs(got[i] - want[i]); e > worst {
			worst, at = e, i
		}
	}
	if worst > tol {
		t.Errorf("%s: %s worst |delta| %.3e at %d (got %v, want %v)",
			d.API(), what, worst, at, got[at], want[at])
		return
	}
	t.Logf("%s: %s worst |delta| %.3e", d.API(), what, worst)
}

func u32le(v uint32) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
}

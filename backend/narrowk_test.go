package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestFloatMatVecBelowItsSubBlock runs a float matvec whose k is below the
// format's activation sub-block against a float64 evaluation of the same values.
//
// For a quantized format the 32-element sub-block is load-bearing (scales,
// minima and activation sums are laid out against it); for a float weight it is
// only a loop stride. Multi-head latent attention reaches it: its absorbed banks
// have k equal to the latent rank or the nope half.
//
// The oracle is an independent float64 dot, not the kernel at another Sub,
// which would share a wrong-stride mistake.
func TestFloatMatVecBelowItsSubBlock(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, q := range []kernels.Quant{kernels.Float32, kernels.Float16, kernels.BFloat16} {
				for _, c := range []struct{ rows, k, sub int }{
					// MLA's own shapes on the fixture: NHead sheets of 16x16.
					{16, 16, 16}, {64, 16, 16},
					// A k that is neither the sub-block nor a multiple of it.
					{32, 20, 20}, {8, 4, 4},
					// And a k the format could walk whole, narrowed anyway: the
					// claim is that Sub changes the stride and never the answer,
					// so it has to be checked where both are legal.
					{64, 64, 16}, {64, 64, 4},
				} {
					name := q.String() + "/" + itoa(c.rows) + "x" + itoa(c.k) + "/sub" + itoa(c.sub)
					t.Run(name, func(t *testing.T) {
						narrowKCase(t, d, q, c.rows, c.k, c.sub, false)
					})
				}
			}
		})
	}
}

// TestFloatMatVecBelowItsSubBlockIsGated runs the same comparison against a
// violation and requires it to fail. The violation perturbs the last element of
// the last row, which only the final narrowed step reads.
func TestFloatMatVecBelowItsSubBlockIsGated(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			narrowKCase(t, d, kernels.Float32, 16, 16, 16, true)
			narrowKCase(t, d, kernels.Float32, 64, 64, 4, true)
		})
	}
}

// TestMatVecSubIsRefusedWhereItCannotHold runs the contract's own guards against
// violations, because a guard that has never been tripped is not a guard either.
//
// The quantized refusal matters most: a k-quant's scales, minima and sums are
// addressed per sub-block, so smaller steps would read another run's scale,
// finite and wrong.
func TestMatVecSubIsRefusedWhereItCannotHold(t *testing.T) {
	for _, c := range []struct {
		name string
		s    kernels.MatVecShape
	}{
		{"quantized", kernels.MatVecShape{T: kernels.Q4_K, K: 256, Rows: 64, Sub: 16}},
		{"not a multiple of four", kernels.MatVecShape{T: kernels.Float32, K: 18, Rows: 8, Sub: 18}},
		{"does not divide k", kernels.MatVecShape{T: kernels.Float32, K: 20, Rows: 8, Sub: 8}},
		{"wider than the format's own", kernels.MatVecShape{T: kernels.Float32, K: 64, Rows: 8, Sub: 64}},
	} {
		if _, err := kernels.MatVec(c.s); err == nil {
			t.Fatalf("%s: MatVec accepted Sub=%d for %v K=%d", c.name, c.s.Sub, c.s.T, c.s.K)
		} else {
			t.Logf("%s: refused by name -- %v", c.name, err)
		}
	}
	// And the shape the whole field exists for is accepted, or the four
	// refusals above would be a gate on a capability that does not exist.
	if _, err := kernels.MatVec(kernels.MatVecShape{T: kernels.Float32, K: 16, Rows: 16, Sub: 16}); err != nil {
		t.Fatalf("the narrowed float shape was refused: %v", err)
	}
}

// narrowKCase builds a float weight of `rows x k`, runs the matvec at the given
// sub-block and compares against a float64 dot of the same values. violate
// perturbs one weight the final narrowed step is the only reader of, and then
// requires the comparison to fail.
func narrowKCase(t *testing.T, d backend.Device, q kernels.Quant, rows, k, sub int, violate bool) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(rows*31337 + k*97 + sub + int(q))))
	raw := make([]byte, rows*k*4)
	exact := make([][]float64, rows)
	for r := range exact {
		exact[r] = make([]float64, k)
	}
	floatWeights(q, raw, exact, rows, k, rng)
	// floatWeights writes q's own width; trim to it so the upload is the
	// buffer the kernel reads.
	w := 4
	if q != kernels.Float32 {
		w = 2
	}
	raw = raw[:rows*k*w]

	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	// The perturbation is the last element of the last row, which is read by
	// exactly one thread on exactly its final sub-block step.
	if violate {
		o := (rows-1)*k + (k - 1)
		v := float32(exact[rows-1][k-1]) + 7
		switch q {
		case kernels.Float32:
			binary.LittleEndian.PutUint32(raw[4*o:], math.Float32bits(v))
		default:
			binary.LittleEndian.PutUint16(raw[2*o:], uint16(math.Float32bits(v)>>16))
		}
	}

	qs, dw, scw, err := kernels.PackWeights(q, raw, rows, k)
	if err != nil {
		t.Fatalf("PackWeights %v %dx%d: %v", q, rows, k, err)
	}
	_ = dw
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	// The int8 activation plane is still handed over, unread: a float weight's
	// scale-plane slot carries the float activation, and the launch keeps its
	// arity, as in the tier.
	av := make([]uint32, max(1, (k+3)/4))
	ax := make([]float32, max(1, 3*((k+31)/32)))

	kk, err := kernels.MatVec(kernels.MatVecShape{T: q, K: k, Rows: rows, Sub: sub})
	if err != nil {
		t.Fatalf("MatVec %v %dx%d sub=%d: %v", q, rows, k, sub, err)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer kern.Close()

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(p []byte) backend.Buf {
		b, err := d.Alloc(len(p))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Write(p); err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, b)
		return b
	}
	// Every buffer is allocated before the Session opens: device calls from
	// inside a Session are to be avoided (see backend.OwnerCalls).
	out := make([]float32, rows)
	bQS := up(u32bytes(qs))
	bX := up(f32bytes(x))
	bSC := up(u32bytes(scw))
	bA := up(u32bytes(av))
	bAX := up(f32bytes(ax))
	bOut := up(f32bytes(out))
	var rerr error
	d.Session(func(s backend.Session) {
		if e := s.Launch(kern, (rows+127)/128, 128, bQS, bX, bSC, bA, bAX, bOut); e != nil {
			rerr = e
			return
		}
		rerr = s.Read(bOut, f32bytes(out))
	})
	if rerr != nil {
		t.Fatal(rerr)
	}

	var num, den float64
	for r := 0; r < rows; r++ {
		var acc float64
		for i := 0; i < k; i++ {
			acc += exact[r][i] * float64(x[i])
		}
		dv := float64(out[r]) - acc
		num, den = num+dv*dv, den+acc*acc
	}
	// A degenerate reference is a failure, not a pass: NaN > bound is false
	// (RULE 10).
	nmse := num / den
	if den == 0 || math.IsNaN(nmse) {
		t.Fatalf("degenerate oracle: num %v den %v", num, den)
	}
	const bound = 1e-10
	if violate {
		if nmse < bound {
			t.Fatalf("the perturbed weight survived at NMSE %.3e: the last narrowed "+
				"sub-block is not being read", nmse)
		}
		t.Logf("violation caught at NMSE %.3e", nmse)
		return
	}
	if !(nmse < bound) {
		t.Fatalf("%v %dx%d sub=%d: NMSE %.3e against a float64 dot", q, rows, k, sub, nmse)
	}
}

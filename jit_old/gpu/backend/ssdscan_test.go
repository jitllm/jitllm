package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// ssdRef steps Mamba-2's selective update over rows of convolved (pre-bias,
// pre-SiLU) channels in float64, from st (one sequence's [VHeads][VDim][KDim]
// state), the way engine/model/delta.go's ssdAttn does: the bias and the
// SiLU, C | B | x sliced off the row, dt and the decay, and oracle.GatedSSD per
// head. It returns the outputs, row by row, and the final state.
func ssdRef(c mainkernels.DeltaScan, conv, dtRaw, cb, dtb, aa, dd, st []float32, rows int) ([]float64, []float64) {
	kHeads := c.VHeads / c.Rep
	qk, vw := kHeads*c.KDim, c.VHeads*c.VDim
	chans := 2*qk + vw
	s := f64of(st)
	out := make([]float64, rows*vw)
	for t := 0; t < rows; t++ {
		x := make([]float64, chans)
		for ch := range x {
			z := float64(conv[t*chans+ch]) + float64(cb[ch])
			x[ch] = z / (1 + math.Exp(-z))
		}
		for vh := 0; vh < c.VHeads; vh++ {
			kh := vh / c.Rep
			z := float64(dtRaw[t*c.VHeads+vh]) + float64(dtb[vh])
			dt := math.Max(z, 0) + math.Log1p(math.Exp(-math.Abs(z)))
			dec := math.Exp(float64(aa[vh]) * dt)
			sl := c.VDim * c.KDim
			oracle.GatedSSD(out[t*vw+vh*c.VDim:t*vw+(vh+1)*c.VDim], s[vh*sl:(vh+1)*sl],
				x[qk+kh*c.KDim:qk+(kh+1)*c.KDim], x[kh*c.KDim:(kh+1)*c.KDim],
				x[2*qk+vh*c.VDim:2*qk+(vh+1)*c.VDim], dec, dt, float64(dd[vh]))
		}
	}
	return out, s
}

func ssdNMSE(got []float32, want []float64) float64 {
	var se, sy float64
	for i := range want {
		d := float64(got[i]) - want[i]
		se, sy = se+d*d, sy+want[i]*want[i]
	}
	return se / math.Max(sy, 1e-30)
}

// TestGatedSSDMatchesTheOracle holds mainkernels.GatedDeltaFused's Mamba-2 form to
// a float64 transcription of ssdAttn over oracle.GatedSSD, on every backend: a
// chunk of consecutive tokens at its full length, a ragged one and decode, the
// lane-group and one-thread forms, one group and several (a value head reading
// key head vh/Rep), and state sizes under, at and over the subgroup. The state
// is checked as well as the output: a kernel that dropped the update is right
// on the first token only.
//
// Then the run form against each sequence alone, and three violations the
// oracle must see: no D skip, no convolution bias and dt left out of the
// update, each of which a kernel could plausibly omit and still produce
// finite, fluent numbers.
func TestGatedSSDMatchesTheOracle(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	ran := 0
	for _, c := range []mainkernels.DeltaScan{
		{VHeads: 8, VDim: 16, KDim: 16, Rep: 8, Rows: 5, Lanes: ir.SubgroupLanes},
		{VHeads: 8, VDim: 16, KDim: 64, Rep: 4, Rows: 5, Lanes: ir.SubgroupLanes},
		{VHeads: 4, VDim: 64, KDim: 128, Rep: 2, Rows: 3, Lanes: ir.SubgroupLanes},
		{VHeads: 8, VDim: 16, KDim: 16, Rep: 8, Rows: 5, Lanes: 1},
		{VHeads: 6, VDim: 9, KDim: 12, Rep: 3, Rows: 4, Lanes: 1},
		{VHeads: 4, VDim: 64, KDim: 128, Rep: 2, Rows: 3, Lanes: 1},
	} {
		for _, d := range devs {
			if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok && c.Lanes != 1 {
				continue
			}
			for _, m := range []int{c.Rows, c.Rows - 1, 1} {
				t.Run(fmt.Sprintf("%s/v%dx%dx%d-rep%d-lanes%d/m%d", d.API(), c.VHeads, c.VDim, c.KDim,
					c.Rep, c.Lanes, m), func(t *testing.T) {
					ssdCase(t, d, c, m)
				})
				ran++
			}
		}
	}
	if ran == 0 {
		t.Skip("no device")
	}
}

func ssdCase(t *testing.T, d backend.Device, c mainkernels.DeltaScan, m int) {
	kHeads := c.VHeads / c.Rep
	qk, vw := kHeads*c.KDim, c.VHeads*c.VDim
	chans := 2*qk + vw
	rows := c.Rows
	rng := rand.New(rand.NewSource(int64(c.VHeads*1000 + c.KDim*10 + m)))
	fill := func(n int, f func() float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(f())
		}
		return x
	}
	sLen := c.VHeads * c.VDim * c.KDim
	st := fill(sLen, func() float64 { return rng.NormFloat64() * 0.3 })
	conv := fill(rows*chans, rng.NormFloat64)
	dtRaw := fill(rows*c.VHeads, rng.NormFloat64)
	cb := fill(chans, func() float64 { return rng.NormFloat64() * 0.3 })
	dtb := fill(c.VHeads, func() float64 { return -2 + rng.NormFloat64() })
	aa := fill(c.VHeads, func() float64 { return -0.1 - 2*rng.Float64() })
	dd := fill(c.VHeads, rng.NormFloat64)

	g := newGPU(t, d)
	defer g.free()
	read := func(b backend.Buf, n int) []float32 {
		p := make([]byte, n*4)
		if err := b.Read(p); err != nil {
			t.Fatal(err)
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = math.Float32frombits(uint32(p[4*i]) | uint32(p[4*i+1])<<8 | uint32(p[4*i+2])<<16 | uint32(p[4*i+3])<<24)
		}
		return v
	}
	k, err := mainkernels.GatedDeltaFused(c, mainkernels.DeltaFuse{Chans: chans, SSD: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := k.Validate(); err != nil {
		t.Fatalf("%s does not validate: %v", k.Name, err)
	}
	ck, err := d.Compile(k)
	if err != nil {
		t.Fatal(err)
	}
	defer ck.Close()
	out, sOut := g.up(f32bytes(poisonPool(rows*vw))), g.up(f32bytes(poisonPool(sLen)))
	if err := ck.Launch((c.Threads()+127)/128, 128, g.up(f32bytes(st)), g.up(f32bytes(conv)),
		g.up(f32bytes(dtRaw)), g.up(f32bytes(dtb)), g.up(f32bytes(aa)), g.up(f32bytes(cb)),
		g.up(f32bytes(dd)), out, sOut, g.up(recDesc(uint32(m), 0, 0))); err != nil {
		t.Fatal(err)
	}
	gotO, gotS := read(out, m*vw), read(sOut, sLen)
	wantO, wantS := ssdRef(c, conv, dtRaw, cb, dtb, aa, dd, st, m)
	// The softplus and the exp are series and approximations on every
	// backend; 1e-9 is far under what any violation below moves.
	if e := ssdNMSE(gotO, wantO); math.IsNaN(e) || e > 1e-9 {
		t.Fatalf("%s: output NMSE %.3e against the oracle", d.API(), e)
	}
	if e := ssdNMSE(gotS, wantS); math.IsNaN(e) || e > 1e-9 {
		t.Fatalf("%s: STATE NMSE %.3e against the oracle", d.API(), e)
	}
	// The violations, as the oracle sees them: each must move the output far
	// past the bound the kernel met.
	for _, v := range []struct {
		name string
		ref  func() []float64
	}{
		{"no D skip", func() []float64 {
			o, _ := ssdRef(c, conv, dtRaw, cb, dtb, aa, make([]float32, c.VHeads), st, m)
			return o
		}},
		{"no convolution bias", func() []float64 {
			o, _ := ssdRef(c, conv, dtRaw, make([]float32, chans), dtb, aa, dd, st, m)
			return o
		}},
		{"A positive", func() []float64 {
			neg := make([]float32, len(aa))
			for i := range aa {
				neg[i] = -aa[i]
			}
			o, _ := ssdRef(c, conv, dtRaw, cb, dtb, neg, dd, st, m)
			return o
		}},
	} {
		if e := ssdNMSE(gotO, v.ref()); !(e > 1e-5) {
			t.Fatalf("%s: the kernel agrees with the violation at %.3e", v.name, e)
		}
	}
	if m == rows {
		ssdRunsCase(t, d, c, conv, dtRaw, cb, dtb, aa, dd, read)
	}
}

// ssdRunsCase is the run form: three sequences over the chunk's rows -- one
// decoding, two chunks -- each from its own state at its own slots, held to
// the oracle stepping each alone.
func ssdRunsCase(t *testing.T, d backend.Device, c mainkernels.DeltaScan, conv, dtRaw, cb, dtb, aa, dd []float32,
	read func(backend.Buf, int) []float32) {
	R := c.Rows
	var runs stepRuns
	switch R {
	case 5:
		runs = stepRuns{{3, 0}, {4}, {1, 2}}
	case 4:
		runs = stepRuns{{2, 0}, {3}, {1}}
	default:
		runs = stepRuns{{2, 0}, {1}}
	}
	S := len(runs)
	kHeads := c.VHeads / c.Rep
	qk, vw := kHeads*c.KDim, c.VHeads*c.VDim
	chans := 2*qk + vw
	sLen := c.VHeads * c.VDim * c.KDim
	rng := rand.New(rand.NewSource(int64(R*7 + c.KDim)))
	P := S + 3
	pool := make([]float32, P*sLen)
	for i := range pool {
		pool[i] = float32(rng.NormFloat64() * 0.3)
	}
	in, out := make([]int, S), make([]int, S)
	for j := range in {
		in[j], out[j] = j+1, P-1-j
	}
	g := newGPU(t, d)
	defer g.free()
	cr := c
	cr.Runs = true
	k, err := mainkernels.GatedDeltaFused(cr, mainkernels.DeltaFuse{Chans: chans, SSD: true})
	if err != nil {
		t.Fatal(err)
	}
	ck, err := d.Compile(k)
	if err != nil {
		t.Fatal(err)
	}
	defer ck.Close()
	o, sOut := g.up(f32bytes(poisonPool(R*vw))), g.up(f32bytes(poisonPool(P*sLen)))
	if err := ck.Launch((cr.Threads()+127)/128, 128, g.up(f32bytes(pool)), g.up(f32bytes(conv)),
		g.up(f32bytes(dtRaw)), g.up(f32bytes(dtb)), g.up(f32bytes(aa)), g.up(f32bytes(cb)),
		g.up(f32bytes(dd)), o, sOut, g.up(runs.desc(R, in, out))); err != nil {
		t.Fatal(err)
	}
	gotO, gotPool := read(o, R*vw), read(sOut, P*sLen)
	for j, rs := range runs {
		var rc, rd []float32
		for _, r := range rs {
			rc = append(rc, conv[r*chans:(r+1)*chans]...)
			rd = append(rd, dtRaw[r*c.VHeads:(r+1)*c.VHeads]...)
		}
		wo, ws := ssdRef(c, rc, rd, cb, dtb, aa, dd, pool[in[j]*sLen:(in[j]+1)*sLen], len(rs))
		for i, r := range rs {
			if e := ssdNMSE(gotO[r*vw:(r+1)*vw], wo[i*vw:(i+1)*vw]); math.IsNaN(e) || e > 1e-9 {
				t.Fatalf("%s runs: run %d row %d output NMSE %.3e against the sequence alone", d.API(), j, r, e)
			}
		}
		if e := ssdNMSE(gotPool[out[j]*sLen:(out[j]+1)*sLen], ws); math.IsNaN(e) || e > 1e-9 {
			t.Fatalf("%s runs: run %d state NMSE %.3e against the sequence alone", d.API(), j, e)
		}
	}
}

// TestGroupNormRowsNormsEachGroupAgainstItsSlice holds mainkernels.GroupNormRows
// to a float64 grouped RMSNorm on every backend, both lane forms, several rows:
// each group of width normalised on its own and scaled by its own slice of the
// weight. The violation is HeadNorm's shared weight (every group reading the
// first slice), which the reference must tell apart.
func TestGroupNormRowsNormsEachGroupAgainstItsSlice(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	const eps = 1e-5
	for _, d := range devs {
		for _, c := range []struct{ groups, width, rows, lanes int }{
			{1, 128, 1, 32}, {4, 64, 3, 32}, {2, 32, 2, 32}, {1, 128, 3, 1}, {3, 24, 2, 1}, {8, 7, 1, 1},
		} {
			if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok && c.lanes != 1 {
				continue
			}
			t.Run(fmt.Sprintf("%s/g%d-w%d-r%d-lanes%d", d.API(), c.groups, c.width, c.rows, c.lanes), func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(c.groups*100 + c.width)))
				n := c.groups * c.width
				x, w := make([]float32, c.rows*n), make([]float32, n)
				for i := range x {
					x[i] = float32(rng.NormFloat64() * 2)
				}
				for i := range w {
					w[i] = float32(1 + 0.5*rng.NormFloat64())
				}
				k, err := mainkernels.GroupNormRows(c.groups, c.width, eps, c.rows, c.lanes)
				if err != nil {
					t.Fatal(err)
				}
				if err := k.Validate(); err != nil {
					t.Fatal(err)
				}
				ck, err := d.Compile(k)
				if err != nil {
					t.Fatal(err)
				}
				defer ck.Close()
				g := newGPU(t, d)
				defer g.free()
				out := g.up(f32bytes(poisonPool(c.rows * n)))
				grid, wid := c.rows*c.groups, 32
				if c.lanes == 1 {
					grid, wid = (c.rows*c.groups+63)/64, 64
				}
				if err := ck.Launch(grid, wid, g.up(f32bytes(x)), g.up(f32bytes(w)), out); err != nil {
					t.Fatal(err)
				}
				p := make([]byte, c.rows*n*4)
				if err := out.Read(p); err != nil {
					t.Fatal(err)
				}
				got := make([]float32, c.rows*n)
				for i := range got {
					got[i] = math.Float32frombits(uint32(p[4*i]) | uint32(p[4*i+1])<<8 | uint32(p[4*i+2])<<16 | uint32(p[4*i+3])<<24)
				}
				ref := func(shared bool) []float64 {
					o := make([]float64, c.rows*n)
					for h := 0; h < c.rows*c.groups; h++ {
						var ss float64
						for i := 0; i < c.width; i++ {
							v := float64(x[h*c.width+i])
							ss += v * v
						}
						inv := 1 / math.Sqrt(ss/float64(c.width)+eps)
						wb := (h % c.groups) * c.width
						if shared {
							wb = 0
						}
						for i := 0; i < c.width; i++ {
							o[h*c.width+i] = float64(x[h*c.width+i]) * inv * float64(w[wb+i])
						}
					}
					return o
				}
				if e := ssdNMSE(got, ref(false)); math.IsNaN(e) || e > 1e-11 {
					t.Fatalf("NMSE %.3e against the grouped norm", e)
				}
				if c.groups > 1 {
					if e := ssdNMSE(got, ref(true)); !(e > 1e-4) {
						t.Fatalf("the kernel agrees with a shared weight at %.3e", e)
					}
				}
			})
		}
	}
}

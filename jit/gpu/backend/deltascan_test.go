package backend_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestGatedDeltaScanOnEveryDevice runs the chunked gated delta rule on every
// backend this host has and holds it to the float64 oracle applied one token at
// a time -- every token's output and the state the chunk leaves behind.
//
// The ragged chunk matters most: a scan that ran padded rows as tokens would
// decay the state once per zero row while every output up to m stayed right,
// so the state is compared too.
//
// The shapes are the ones the kernel branches on: a head narrower than a
// subgroup (kDim 8, eight lanes a row), rep > 1 with the grouped and the tiled
// key pairing (Qwen3.5 tiles, qwen3next groups -- the wrong one is fluent),
// KDA's per-channel decay, Qwen3-Next-80B's own 32x128x128 at rep 2, and the
// one-thread-per-row form a device without a guaranteed subgroup takes.
func TestGatedDeltaScanOnEveryDevice(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	cases := []oldkernels.DeltaScan{
		{VHeads: 4, VDim: 8, KDim: 8, Rep: 2, Rows: 6, Lanes: ir.SubgroupLanes},
		{VHeads: 4, VDim: 32, KDim: 32, Rep: 2, TiledK: 2, Rows: 6, Lanes: ir.SubgroupLanes},
		{VHeads: 2, VDim: 32, KDim: 32, Rep: 1, PerChan: true, Rows: 5, Lanes: ir.SubgroupLanes},
		{VHeads: 4, VDim: 128, KDim: 128, Rep: 2, PerChan: true, Rows: 3, Lanes: ir.SubgroupLanes},
		{VHeads: 32, VDim: 128, KDim: 128, Rep: 2, Rows: 4, Lanes: ir.SubgroupLanes},
		{VHeads: 4, VDim: 16, KDim: 16, Rep: 2, Rows: 5, Lanes: 1},
	}
	for _, c := range cases {
		for _, m := range []int{c.Rows, c.Rows - 1, 1} {
			name := fmt.Sprintf("v%dx%dx%d-rep%d-tiled%d-chan%v-lanes%d/m%d-of-%d",
				c.VHeads, c.VDim, c.KDim, c.Rep, c.TiledK, c.PerChan, c.Lanes, m, c.Rows)
			t.Run(name, func(t *testing.T) {
				for _, d := range devs {
					deltaScanCase(t, d, c, m)
				}
			})
		}
	}
}

func deltaScanCase(t *testing.T, d backend.Device, c oldkernels.DeltaScan, m int) {
	kHeads := c.VHeads / c.Rep
	if c.TiledK > 0 {
		kHeads = c.TiledK
	}
	qk, vw := kHeads*c.KDim, c.VHeads*c.VDim
	nDec := c.VHeads
	if c.PerChan {
		nDec = c.VHeads * c.KDim
	}
	rng := rand.New(rand.NewSource(int64(c.VHeads*1000 + c.KDim*10 + m)))
	fill := func(n int, sc float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * sc)
		}
		return x
	}
	st := fill(c.VHeads*c.VDim*c.KDim, 0.5)
	k := fill(c.Rows*qk, 0.2)
	q := fill(c.Rows*qk, 0.2)
	v := fill(c.Rows*vw, 0.5)
	beta := make([]float32, c.Rows*c.VHeads)
	decay := make([]float32, c.Rows*nDec)
	for i := range beta {
		beta[i] = float32(rng.Float64())
	}
	for i := range decay {
		decay[i] = float32(0.5 + 0.45*rng.Float64())
	}

	kk, err := oldkernels.GatedDeltaScan(c)
	if err != nil {
		t.Fatal(err)
	}
	if err := kk.Validate(); err != nil {
		t.Fatalf("does not validate: %v", err)
	}
	if c.Lanes == ir.SubgroupLanes {
		if ok, why := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok {
			t.Logf("%s: no guaranteed 32-lane subgroup (%s); skipping the warp form", d.API(), why)
			return
		}
	}
	// The sequence's state is slot 1 of a two-slot pool, read and written there.
	const slots, slot = 2, 1
	per := len(st)
	g := newGPU(t, d)
	defer g.free()
	bS := g.up(f32bytes(inPool(rng, slots, slot, st)))
	bK, bQ, bV := g.up(f32bytes(k)), g.up(f32bytes(q)), g.up(f32bytes(v))
	bD, bB := g.up(f32bytes(decay)), g.up(f32bytes(beta))
	bOut := g.up(make([]byte, c.Rows*vw*4))
	bSOut := g.up(f32bytes(poisonPool(slots * per)))
	bN := g.up(recDesc(uint32(m), slot, slot))
	ck, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer ck.Close()
	n := c.Threads()
	if err := ck.Launch((n+127)/128, 128, bS, bK, bQ, bV, bD, bB, bOut, bSOut, bN); err != nil {
		t.Fatal(err)
	}

	// The oracle, token by token, on the transposed state this tree keeps.
	wantSt := f64of(st)
	wantOut := make([]float64, m*vw)
	for tk := 0; tk < m; tk++ {
		for vh := 0; vh < c.VHeads; vh++ {
			kh := vh / c.Rep
			if c.TiledK > 0 {
				kh = vh % c.TiledK
			}
			o := wantOut[tk*vw+vh*c.VDim : tk*vw+(vh+1)*c.VDim]
			mat := wantSt[vh*c.VDim*c.KDim:][:c.VDim*c.KDim]
			kv := f64of(k[tk*qk+kh*c.KDim : tk*qk+(kh+1)*c.KDim])
			qv := f64of(q[tk*qk+kh*c.KDim : tk*qk+(kh+1)*c.KDim])
			vv := f64of(v[tk*vw+vh*c.VDim : tk*vw+(vh+1)*c.VDim])
			b := float64(beta[tk*c.VHeads+vh])
			if c.PerChan {
				oracle.GatedDeltaChan(o, mat, kv, qv, vv,
					f64of(decay[tk*nDec+vh*c.KDim:tk*nDec+(vh+1)*c.KDim]), b)
				continue
			}
			oracle.GatedDelta(o, mat, kv, qv, vv, float64(decay[tk*nDec+vh]), b)
		}
	}
	cmpF32(t, d, "scan out", readF32(t, bOut, m*vw), wantOut, 2e-4)
	pool := readF32(t, bSOut, slots*per)
	cmpF32(t, d, "scan state", pool[slot*per:(slot+1)*per], wantSt, 2e-4)
	poisonKept(t, "scan state", pool, per, slot)
}

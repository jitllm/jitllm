package backend_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGatedDeltaFusedMatchesTheUnfusedChain holds mainkernels.GatedDeltaFused to
// the eleven launches it replaces -- Act, three SliceRows, two HeadNormRows,
// three Scales, SplitDeltaGatesRows and DeltaGateRows (KDA: two DeltaGateRows)
// -- followed by GatedDeltaScan, run on the device in that order from the same
// inputs, on every backend.
//
// Bit for bit where the unfused norm is HeadNorm's warp form, which is every
// real hybrid (kDim a multiple of 32 on a 32-lane device): the fused kernel
// forms every value with the same operations in the same order, so the state
// and the outputs must be the same floats. Elsewhere -- kDim 8, or a device
// without the subgroup, where HeadNorm sums serially and the fused norm does
// not -- and on SPIR-V (see exact below) they are held to fusedBand.
//
// The cases carry the tiled key pairing, the beta/alpha interleave at rep 2,
// KDA's per-channel decay, Qwen3-Next-80B's own 32x128x128, a ragged chunk
// (m < rows) and decode (m = 1).
func TestGatedDeltaFusedMatchesTheUnfusedChain(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	cases := []mainkernels.DeltaScan{
		{VHeads: 4, VDim: 32, KDim: 32, Rep: 2, Rows: 5, Lanes: ir.SubgroupLanes},
		{VHeads: 4, VDim: 32, KDim: 32, Rep: 2, TiledK: 2, Rows: 5, Lanes: ir.SubgroupLanes},
		{VHeads: 2, VDim: 32, KDim: 32, Rep: 1, PerChan: true, Rows: 4, Lanes: ir.SubgroupLanes},
		{VHeads: 32, VDim: 128, KDim: 128, Rep: 2, Rows: 3, Lanes: ir.SubgroupLanes},
		{VHeads: 4, VDim: 8, KDim: 8, Rep: 2, Rows: 6, Lanes: ir.SubgroupLanes},
		{VHeads: 4, VDim: 16, KDim: 16, Rep: 2, Rows: 4, Lanes: 1},
	}
	for _, c := range cases {
		for _, m := range []int{c.Rows, c.Rows - 1, 1} {
			name := fmt.Sprintf("v%dx%dx%d-rep%d-tiled%d-chan%v-lanes%d/m%d-of-%d",
				c.VHeads, c.VDim, c.KDim, c.Rep, c.TiledK, c.PerChan, c.Lanes, m, c.Rows)
			t.Run(name, func(t *testing.T) {
				for _, d := range devs {
					deltaFusedCase(t, d, c, m)
				}
			})
		}
	}
}

func deltaFusedCase(t *testing.T, d backend.Device, c mainkernels.DeltaScan, m int) {
	if c.Lanes == ir.SubgroupLanes {
		if ok, why := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok {
			t.Logf("%s: no guaranteed 32-lane subgroup (%s)", d.API(), why)
			return
		}
	}
	kHeads := c.VHeads / c.Rep
	if c.TiledK > 0 {
		kHeads = c.TiledK
	}
	rows := c.Rows
	qk, vw := kHeads*c.KDim, c.VHeads*c.VDim
	chans := 2*qk + vw
	baRep := c.VHeads / kHeads
	nGate := c.VHeads
	if c.PerChan {
		nGate = c.VHeads * c.KDim
	}
	const rmsEps = 1e-6
	eps := float32(rmsEps / float64(c.KDim))
	sc := float32(1 / math.Sqrt(float64(c.KDim)))

	rng := rand.New(rand.NewSource(int64(c.VHeads*100 + c.KDim + m)))
	fill := func(n int, f func() float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(f())
		}
		return x
	}
	st := fill(c.VHeads*c.VDim*c.KDim, func() float64 { return rng.NormFloat64() * 0.3 })
	conv := fill(rows*chans, rng.NormFloat64)
	ba := fill(rows*2*c.VHeads, rng.NormFloat64)                        // qwen3next's interleaved beta|alpha
	alpha := fill(rows*nGate, rng.NormFloat64)                          // KDA's forget gate
	bRaw := fill(rows*c.VHeads, rng.NormFloat64)                        // KDA's beta logits
	dt := fill(nGate, rng.NormFloat64)                                  // the block's dt bias
	aa := fill(nGate, func() float64 { return -0.1 - 2*rng.Float64() }) // -exp(A_log)
	ones := fill(c.KDim, func() float64 { return 1 })

	g := newGPU(t, d)
	defer g.free()
	buf := func(n int) backend.Buf { return g.up(make([]byte, n*4)) }
	bS, bConv := g.up(f32bytes(st)), g.up(f32bytes(conv))
	bDt, bA, bOnes := g.up(f32bytes(dt)), g.up(f32bytes(aa)), g.up(f32bytes(ones))
	bN := g.up(recDesc(uint32(m), 0, 0))
	bScale := g.up(f32bytes([]float32{sc}))
	run := func(k *ir.Kernel, err error, grid, width int, args ...backend.Buf) {
		t.Helper()
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
		if err := ck.Launch(grid, width, args...); err != nil {
			t.Fatal(err)
		}
	}
	lin := func(k *ir.Kernel, err error, n int, args ...backend.Buf) {
		t.Helper()
		run(k, err, (n+127)/128, 128, args...)
	}

	// The unfused chain, as emitLinear issues it.
	mixed := buf(rows * chans)
	k, err := mainkernels.Act(rows*chans, kernels.ActSiLU)
	lin(k, err, rows*chans, bConv, mixed)
	bQ, bK, bV := buf(rows*qk), buf(rows*qk), buf(rows*vw)
	k, err = mainkernels.SliceRows(qk, chans, 0, rows)
	lin(k, err, rows*qk, mixed, bQ)
	k, err = mainkernels.SliceRows(qk, chans, qk, rows)
	lin(k, err, rows*qk, mixed, bK)
	k, err = mainkernels.SliceRows(vw, chans, 2*qk, rows)
	lin(k, err, rows*vw, mixed, bV)
	lanes := c.Lanes
	if lanes == ir.SubgroupLanes && c.KDim%ir.SubgroupLanes != 0 {
		lanes = 1 // the tier's headLanes: HeadNorm's warp form needs whole subgroups
	}
	bQn, bKn := buf(rows*qk), buf(rows*qk)
	k, err = mainkernels.HeadNormRows(kHeads, c.KDim, float64(eps), rows, lanes)
	if lanes == 1 {
		run(k, err, (rows*kHeads+63)/64, 64, bQ, bOnes, bQn)
		run(k, err, (rows*kHeads+63)/64, 64, bK, bOnes, bKn)
	} else {
		run(k, err, rows*kHeads, 32, bQ, bOnes, bQn)
		run(k, err, rows*kHeads, 32, bK, bOnes, bKn)
	}
	k, err = mainkernels.Scale(rows * qk)
	lin(k, err, rows*qk, bKn, bScale, bK)
	lin(k, err, rows*qk, bQn, bScale, bQ)
	lin(k, err, rows*qk, bQ, bScale, bQn)
	bDecay, bBeta, bWaste := buf(rows*nGate), buf(rows*c.VHeads), buf(rows*nGate)
	var bBA, bAlpha, bBRaw backend.Buf
	if c.PerChan {
		bAlpha, bBRaw = g.up(f32bytes(alpha)), g.up(f32bytes(bRaw))
		k, err = mainkernels.DeltaGateRows(nGate, rows)
		lin(k, err, rows*nGate, bAlpha, bDt, bA, bAlpha, bDecay, bWaste)
		k, err = mainkernels.DeltaGateRows(c.VHeads, rows)
		lin(k, err, rows*c.VHeads, bAlpha, bDt, bA, bBRaw, bWaste, bBeta)
	} else {
		bBA = g.up(f32bytes(ba))
		bBRaw, bAlpha = buf(rows*c.VHeads), buf(rows*c.VHeads)
		k, err = mainkernels.SplitDeltaGatesRows(c.VHeads/baRep, baRep, rows)
		lin(k, err, rows*c.VHeads, bBA, bBRaw, bAlpha)
		k, err = mainkernels.DeltaGateRows(c.VHeads, rows)
		lin(k, err, rows*c.VHeads, bAlpha, bDt, bA, bBRaw, bDecay, bBeta)
	}
	out0, s0 := buf(rows*vw), buf(len(st))
	k, err = mainkernels.GatedDeltaScan(c)
	lin(k, err, c.Threads(), bS, bK, bQn, bV, bDecay, bBeta, out0, s0, bN)

	// The fused kernel, from the same raw inputs.
	out1, s1 := buf(rows*vw), buf(len(st))
	k, err = mainkernels.GatedDeltaFused(c, mainkernels.DeltaFuse{Chans: chans, Eps: eps, Scale: sc, BARep: baRep})
	if c.PerChan {
		lin(k, err, c.Threads(), bS, bConv, bAlpha, bBRaw, bDt, bA, out1, s1, bN)
	} else {
		lin(k, err, c.Threads(), bS, bConv, bBA, bDt, bA, out1, s1, bN)
	}

	read := func(b backend.Buf, n int) []byte {
		p := make([]byte, n*4)
		if err := b.Read(p); err != nil {
			t.Fatal(err)
		}
		return p
	}
	// Exact on PTX only: a SPIR-V consumer may contract a multiply and add
	// into an fma per kernel, so two different kernels can round a product
	// differently there, while ptx's fma is explicit.
	exact := d.API() == "ptx" && c.Lanes == ir.SubgroupLanes && c.KDim%ir.SubgroupLanes == 0
	for _, pair := range []struct {
		what string
		a, b []byte
	}{{"out", read(out0, m*vw), read(out1, m*vw)}, {"state", read(s0, len(st)), read(s1, len(st))}} {
		if exact {
			if !bytes.Equal(pair.a, pair.b) {
				t.Fatalf("%s: fused %s is not the unfused chain's, bit for bit (worst %.3e)",
					d.API(), pair.what, worstRel(pair.a, pair.b))
			}
			continue
		}
		if w := worstRel(pair.a, pair.b); !(w <= fusedBand) {
			t.Fatalf("%s: fused %s differs from the unfused chain by %.3e", d.API(), pair.what, w)
		}
	}
}

// worstRel is the largest |a-b| / max(|a|, 1e-3) over two f32 buffers.
// fusedBand is how far the fused and unfused chains may part where they are not
// bit for bit. It is a band of driver rounding, not of the fusion: the PTX arm
// holds the same kernels to equality, which is what convicts a wrong operation.
// On SPIR-V each kernel's consumer may contract an fma and runs exp and
// inversesqrt at the precision Vulkan allows (several ulp, input-dependent), so
// the band is the drivers'. Over every case here a discrete card's Vulkan
// driver reads at worst 1.04e-05 and an integrated GPU's 5.02e-05 (the
// 128-wide head at two of three rows).
const fusedBand = 1e-4

func worstRel(a, b []byte) float64 {
	w := 0.0
	for i := 0; i+4 <= len(a); i += 4 {
		x := float64(math.Float32frombits(binary.LittleEndian.Uint32(a[i:])))
		y := float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i:])))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return math.Inf(1)
		}
		w = math.Max(w, math.Abs(x-y)/math.Max(math.Abs(x), 1e-3))
	}
	return w
}

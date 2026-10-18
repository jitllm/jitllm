package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// attn70Case is one attention geometry: heads, head width, kv heads, query
// rows, the chunk's widest count, and a sliding window (0 for none).
type attn70Case struct {
	nHeads, headDim, nKV, rows, cap, window int
	// rowMajor is a [position][kvDim] K (kStride 0): MLA's latent cache.
	rowMajor bool
	// kvRow overrides kvDim (nKV*headDim) with a cache row WIDER than the
	// heads it holds: MLA's 576-float row whose first 512 are the value.
	kvRow int
}

var attn70Cases = []attn70Case{
	{32, 128, 8, 512, 512, 0, false, 0},  // Llama-3.1-8B, one 512-token chunk
	{32, 128, 8, 128, 700, 0, false, 0},  // a later chunk, a cap that is no tile multiple
	{32, 64, 4, 128, 384, 0, false, 0},   // tinyllama
	{8, 256, 1, 32, 67, 0, false, 0},     // gemma: one kv head, wide, ragged
	{16, 128, 8, 64, 200, 100, false, 0}, // a sliding window
}

// attn70RowMajorCases are the scores over a row-major K: DeepSeek-V2's latent
// row (16 heads, one 576-wide row shared by every head) at a whole chunk and at
// a ragged one, and a grouped-query shape so a kv head's offset inside the row
// is exercised too.
var attn70RowMajorCases = []attn70Case{
	{16, 576, 1, 512, 512, 0, true, 0},
	{16, 576, 1, 64, 200, 0, true, 0},
	{32, 128, 8, 128, 700, 0, true, 0},
}

// attn70Counts is pN: element 0 the widest count, then each row's own.
func attn70Counts(rows, cap int) []uint32 {
	pn := make([]uint32, 1+rows)
	pn[0] = uint32(cap)
	for r := 0; r < rows; r++ {
		pn[1+r] = uint32(max(cap-rows+1+r, 1))
	}
	return pn
}

// TestAttnScoresMMA70MatchesFMA holds sm_70's m8n8k4 scores kernel to the FMA
// tile on the same inputs, as TestAttnScoresMMAMatchesFMA does for m16n8k16,
// with the same bar (binary16 operands, float32 sums) and the same mask check:
// every position past a row's count (or before its window) must be -inf.
func TestAttnScoresMMA70MatchesFMA(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	ran := 0
	for _, d := range mmaDevices(t, devs, ir.MMAVolta()) {
		for _, c := range append(append([]attn70Case{}, attn70Cases...), attn70RowMajorCases...) {
			t.Run(fmt.Sprintf("%s/h%d/d%d/kv%d/rows%d/cap%d/w%d/rm%v", d.API(), c.nHeads, c.headDim, c.nKV, c.rows,
				c.cap, c.window, c.rowMajor),
				func(t *testing.T) { scores70Case(t, d, c) })
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}

func scores70Case(t *testing.T, d backend.Device, c attn70Case) {
	const maxSeq, mt, nt = 1024, 2, 4
	kvDim, gqa := c.nKV*c.headDim, c.nHeads/c.nKV
	sstride, kStride := maxSeq+32, maxSeq+1
	if c.rowMajor {
		kStride = 0
	}
	scale := float32(1) / float32(math.Sqrt(float64(c.headDim)))
	mk, err := kernels.AttnScoresMMA70(c.nHeads, c.headDim, kvDim, gqa, sstride, scale, c.rows, kStride, mt, nt, c.window)
	if err != nil {
		t.Fatal(err)
	}
	mkern, err := d.Compile(mk)
	if err != nil {
		t.Fatal(err)
	}
	defer mkern.Close()
	fk, err := kernels.AttnScoresTiledW(c.nHeads, c.headDim, kvDim, gqa, sstride, scale, c.rows, 4, 2, kStride, c.window)
	if err != nil {
		t.Fatal(err)
	}
	fkern, err := d.Compile(fk)
	if err != nil {
		t.Fatal(err)
	}
	defer fkern.Close()

	rng := rand.New(rand.NewSource(int64(c.nHeads*17 + c.headDim + c.rows + c.cap)))
	q := make([]float32, c.rows*c.nHeads*c.headDim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	kc := make([]float32, kvDim*max(kStride, maxSeq))
	for i := range kc {
		kc[i] = float32(rng.NormFloat64())
		// A row-major cache past the chunk's width is poisoned: the kernel
		// clamps its key loads to the last counted row, and a load past it
		// would carry NaN into a score the mask does not replace.
		if c.rowMajor && i/kvDim >= c.cap {
			kc[i] = float32(math.NaN())
		}
	}
	pn := attn70Counts(c.rows, c.cap)
	g := newGPU(t, d)
	defer g.free()
	bQ, bK, bN := g.up(f32bytes(q)), g.up(f32bytes(kc)), g.up(u32bytes(pn))
	nOut := c.rows * c.nHeads * sstride
	run := func(kern backend.Kernel, threads int) []float32 {
		out := g.up(f32bytes(nanFill(nOut)))
		if err := kern.Launch((threads+127)/128, 128, bQ, bK, bN, out); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, nOut*4)
		if err := out.Read(raw); err != nil {
			t.Fatal(err)
		}
		v := make([]float32, nOut)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return v
	}
	gotM := run(mkern, kernels.AttnScoresMMA70Warps(c.nHeads, c.rows, c.cap, mt, nt)*32)
	gotF := run(fkern, (c.rows/4)*c.nHeads*((c.cap+1)/2))
	var sse, sy2 float64
	skipped := 0
	for r := 0; r < c.rows; r++ {
		// Only up to the group's widest count. The kernel skips every key tile
		// at or past the widest count of its 8*nt query rows -- nothing
		// downstream reads there (SoftmaxRows' qt and AttnAccMMA70 walk the
		// same group width) -- so a row's scores are defined up to that count
		// and poison beyond it is the contract, not a fault.
		g := r / (8 * nt)
		gmax := int(pn[1+min(8*nt*(g+1), c.rows)-1])
		skipped += c.cap - gmax
		for h := 0; h < c.nHeads; h++ {
			for p := 0; p < gmax; p++ {
				i := (r*c.nHeads+h)*sstride + p
				m, f := gotM[i], gotF[i]
				if math.IsInf(float64(f), -1) {
					if !math.IsInf(float64(m), -1) {
						t.Fatalf("row %d head %d pos %d: %v where the FMA kernel masks (-inf)", r, h, p, m)
					}
					continue
				}
				if math.IsNaN(float64(f)) || math.IsNaN(float64(m)) || math.IsInf(float64(m), 0) {
					t.Fatalf("row %d head %d pos %d: fma %v mma70 %v", r, h, p, f, m)
				}
				e := float64(m) - float64(f)
				sse += e * e
				sy2 += float64(f) * float64(f)
			}
		}
	}
	if sy2 == 0 {
		t.Fatal("the reference is all zero or masked; the oracle is degenerate")
	}
	nmse := sse / sy2
	t.Logf("NMSE %.3e (%d row-positions past their group's widest count left unscored)", nmse, skipped)
	if nmse > 1e-5 {
		t.Fatalf("NMSE %.3e against the FMA kernel", nmse)
	}
}

// attn70LatentAccCases are the accumulate over MLA's latent row: 16 heads
// sharing one 576-float row whose first 512 floats are the value, at a whole
// chunk and a ragged one.
var attn70LatentAccCases = []attn70Case{
	{nHeads: 16, headDim: 512, nKV: 1, rows: 512, cap: 512, kvRow: 576},
	{nHeads: 16, headDim: 512, nKV: 1, rows: 64, cap: 200, kvRow: 576},
}

// TestAttnAccMMA70MatchesFMA holds sm_70's m8n8k4 weighted sum of V to the FMA
// tile (AttnAccTiled at the same 32-row group) on the same probabilities, with
// every V slot past the chunk's widest count poisoned: the masked last step has
// to read nothing it should not, because 0 * NaN is NaN.
func TestAttnAccMMA70MatchesFMA(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	ran := 0
	for _, d := range mmaDevices(t, devs, ir.MMAVolta()) {
		for _, c := range append(append([]attn70Case{}, attn70Cases...), attn70LatentAccCases...) {
			if c.window > 0 {
				continue // the accumulate has no window; its zeros are the softmax's
			}
			if c.kvRow > 0 {
				// A latent row: acc70Case takes the tier's latent tile itself.
				t.Run(fmt.Sprintf("%s/h%d/d%d/kv%d/rows%d/cap%d/row%d", d.API(), c.nHeads, c.headDim, c.nKV, c.rows,
					c.cap, c.kvRow),
					func(t *testing.T) { acc70Case(t, d, c, 0) })
				ran++
				continue
			}
			// mt 1 is what the tier ships (volta70AccMT); the whole head is the
			// widest tile the kernel takes, kept so both ends stay gated.
			for _, mt := range []int{1, c.headDim / 32} {
				if mt == 1 && c.headDim == 32 {
					continue
				}
				t.Run(fmt.Sprintf("%s/h%d/d%d/kv%d/rows%d/cap%d/mt%d", d.API(), c.nHeads, c.headDim, c.nKV, c.rows, c.cap, mt),
					func(t *testing.T) { acc70Case(t, d, c, mt) })
				ran++
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}

func acc70Case(t *testing.T, d backend.Device, c attn70Case, mt int) {
	const maxSeq, nt = 1024, 4
	kvDim, gqa := c.nKV*c.headDim, c.nHeads/c.nKV
	if c.kvRow > 0 {
		// The tier's latent tile: four 32-dim m-tiles a warp (volta70LatMT).
		kvDim, mt = c.kvRow, min(c.headDim/32, 4)
	}
	sstride := maxSeq + 32
	mk, err := kernels.AttnAccMMA70(c.nHeads, c.headDim, kvDim, gqa, sstride, c.rows, mt, nt)
	if err != nil {
		t.Fatal(err)
	}
	mkern, err := d.Compile(mk)
	if err != nil {
		t.Fatal(err)
	}
	defer mkern.Close()
	const qt = 8 * nt
	fk, err := kernels.AttnAccTiled(c.nHeads, c.headDim, kvDim, gqa, sstride, c.rows, qt)
	if err != nil {
		t.Fatal(err)
	}
	fkern, err := d.Compile(fk)
	if err != nil {
		t.Fatal(err)
	}
	defer fkern.Close()

	rng := rand.New(rand.NewSource(int64(c.nHeads*31 + c.headDim + c.rows + c.cap)))
	pn := attn70Counts(c.rows, c.cap)
	// Probabilities as a softmax leaves them: positive below the row's own
	// count, zero from there to its group's widest, NaN beyond (never read).
	prob := nanFill(c.rows * c.nHeads * sstride)
	for r := 0; r < c.rows; r++ {
		widest := int(pn[1+(r/qt)*qt+qt-1])
		for h := 0; h < c.nHeads; h++ {
			for p := 0; p < widest; p++ {
				v := float32(0)
				if p < int(pn[1+r]) {
					v = float32(rng.Float64())
				}
				prob[(r*c.nHeads+h)*sstride+p] = v
			}
		}
	}
	vc := nanFill(maxSeq * kvDim)
	for i := 0; i < c.cap*kvDim; i++ {
		vc[i] = float32(rng.NormFloat64())
	}
	g := newGPU(t, d)
	defer g.free()
	bP, bV, bN := g.up(f32bytes(prob)), g.up(f32bytes(vc)), g.up(u32bytes(pn))
	nOut := c.rows * c.nHeads * c.headDim
	run := func(kern backend.Kernel, threads int) []float32 {
		out := g.up(f32bytes(nanFill(nOut)))
		if err := kern.Launch((threads+127)/128, 128, bP, bV, bN, out); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, nOut*4)
		if err := out.Read(raw); err != nil {
			t.Fatal(err)
		}
		v := make([]float32, nOut)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return v
	}
	gotM := run(mkern, kernels.AttnAccMMA70Warps(c.nHeads, c.headDim, c.rows, mt, nt)*32)
	gotF := run(fkern, c.rows*c.nHeads*c.headDim/qt)
	var sse, sy2 float64
	for i := range gotF {
		m, f := float64(gotM[i]), float64(gotF[i])
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("the FMA arm is not finite at %d", i)
		}
		if math.IsNaN(m) || math.IsInf(m, 0) {
			t.Fatalf("output %d (row %d) is %v", i, i/(c.nHeads*c.headDim), m)
		}
		sse += (m - f) * (m - f)
		sy2 += f * f
	}
	if sy2 == 0 {
		t.Fatal("the reference is all zero; the oracle is degenerate")
	}
	nmse := sse / sy2
	t.Logf("NMSE %.3e", nmse)
	if nmse > 1e-5 {
		t.Fatalf("NMSE %.3e against the FMA kernel", nmse)
	}
}

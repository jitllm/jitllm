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

// flash70Case is one geometry for FlashPrefill70.
type flash70Case struct {
	heads, dim, kv, rows, cap, window int
	softcap                           float32
}

var flash70Cases = []flash70Case{
	{16, 128, 8, 512, 512, 0, 0}, // Qwen3-1.7B, one 512-token chunk
	{32, 64, 8, 128, 700, 0, 0},  // Llama-3.2-1B, a later chunk, a cap no tile multiple
	{8, 128, 1, 64, 67, 0, 0},    // one kv head, ragged
	{16, 128, 8, 64, 200, 100, 0},
	{8, 128, 4, 128, 150, 0, 50}, // a softcap
}

// TestFlashPrefill70MatchesReference holds the one-kernel prompt attention to a
// float64 evaluation of softmax(scale*QK^T, causal, window, softcap) V on the
// same inputs. Every K and V slot past the chunk's widest count is poisoned:
// the kernel stages whole 32-key tiles, and a key it does not count must
// contribute nothing -- 0 * NaN is NaN, so a V row it failed to zero shows.
func TestFlashPrefill70MatchesReference(t *testing.T) {
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
		for _, c := range flash70Cases {
			t.Run(fmt.Sprintf("%s/h%d/d%d/kv%d/rows%d/cap%d/w%d/cap%g", d.API(), c.heads, c.dim, c.kv, c.rows, c.cap, c.window, c.softcap),
				func(t *testing.T) { flash70Run(t, d, c, false) })
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}

// TestFlashPrefill70IsGated requires the comparison to fail against a
// violation (RULE 10): the causal counts handed to the kernel are one short,
// so every row drops the key it should have seen last.
func TestFlashPrefill70IsGated(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	for _, d := range mmaDevices(t, devs, ir.MMAVolta()) {
		t.Run(d.API(), func(t *testing.T) { flash70Run(t, d, flash70Cases[1], true) })
	}
}

// TestFlashPrefillTileMatchesReference is the same reference and the same
// poisoned slots for kernels.FlashPrefillTile, on every device that lowers the
// ir tile ops at its 8x8 shape (Metal's simdgroup_matrix, PTX's mma.sync), at
// the blocking FlashTileFor picks per head width.
func TestFlashPrefillTileMatchesReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	ran := 0
	for _, d := range devs {
		if ok, why := backend.LowersTile(d, tile8f32); !ok {
			t.Logf("%s: no 8x8 float32 tile: %s", d.API(), why)
			continue
		}
		for _, c := range append(flash70Cases, flash70Case{8, 256, 1, 64, 90, 0, 0}) {
			t.Run(fmt.Sprintf("%s/h%d/d%d/kv%d/rows%d/cap%d/w%d/cap%g", d.API(), c.heads, c.dim, c.kv, c.rows, c.cap, c.window, c.softcap),
				func(t *testing.T) { flashRun(t, d, c, false, flashTileBuild) })
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no device here lowers an 8x8 collective tile")
	}
}

// TestFlashPrefillTileIsGated is TestFlashPrefill70IsGated for the tile kernel.
func TestFlashPrefillTileIsGated(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	for _, d := range devs {
		if ok, _ := backend.LowersTile(d, tile8f32); ok {
			t.Run(d.API(), func(t *testing.T) { flashRun(t, d, flash70Cases[1], true, flashTileBuild) })
		}
	}
}

// tile8 is the collective shape GemmTile and FlashPrefillTile multiply in.
var tile8 = backend.TileShape{M: 8, N: 8, K: 8, Elem: ir.TileF16, Acc: ir.TileF32}

// tile8f32 is FlashPrefillTile's P*V product: float32 operands.
var tile8f32 = backend.TileShape{M: 8, N: 8, K: 8, Elem: ir.TileF32, Acc: ir.TileF32}

// tile16x8 is TileGemm.Tall's shape: a 16-row A tile by an 8x8 B.
var tile16x8 = backend.TileShape{M: 16, N: 8, K: 8, Elem: ir.TileF16, Acc: ir.TileF32}

// flashBuild returns a flash kernel for a shape and its launch geometry.
type flashBuild func(sh kernels.FlashPrefill70Shape) (*ir.Kernel, int, int, error)

func flash70Build(sh kernels.FlashPrefill70Shape) (*ir.Kernel, int, int, error) {
	k, err := kernels.FlashPrefill70(sh)
	return k, kernels.FlashPrefill70Groups(sh), kernels.FlashPrefill70Threads, err
}

func flashTileBuild(sh kernels.FlashPrefill70Shape) (*ir.Kernel, int, int, error) {
	tl, ok := kernels.FlashTileFor(sh.Dim)
	if !ok {
		return nil, 0, 0, fmt.Errorf("no blocking fits head width %d", sh.Dim)
	}
	k, err := kernels.FlashPrefillTile(sh, tl)
	return k, kernels.FlashTileGroups(sh.Heads, sh.Rows, tl), tl.Threads(), err
}

func flash70Run(t *testing.T, d backend.Device, c flash70Case, violate bool) {
	flashRun(t, d, c, violate, flash70Build)
}

func flashRun(t *testing.T, d backend.Device, c flash70Case, violate bool, build flashBuild) {
	const maxSeq = 1024
	kStride := maxSeq + 1
	kvDim := c.kv * c.dim
	gqa := c.heads / c.kv
	scale := float32(1 / math.Sqrt(float64(c.dim)))
	sh := kernels.FlashPrefill70Shape{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, KStride: kStride,
		Scale: scale, Softcap: c.softcap, Window: c.window}
	kk, groups, threads, err := build(sh)
	if err != nil {
		t.Fatal(err)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kern.Close()

	rng := rand.New(rand.NewSource(int64(c.heads*13 + c.dim + c.rows + c.cap)))
	q := make([]float32, c.rows*c.heads*c.dim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	kc := nanFill(kvDim * kStride)
	vc := nanFill(maxSeq * kvDim)
	for e := 0; e < kvDim; e++ {
		for p := 0; p < c.cap; p++ {
			kc[e*kStride+p] = float32(rng.NormFloat64())
		}
	}
	for i := 0; i < c.cap*kvDim; i++ {
		vc[i] = float32(rng.NormFloat64())
	}
	pn := attn70Counts(c.rows, c.cap)
	send := append([]uint32(nil), pn...)
	if violate {
		for r := 1; r < len(send); r++ {
			send[r]--
		}
	}
	g := newGPU(t, d)
	defer g.free()
	nOut := c.rows * c.heads * c.dim
	out := g.up(f32bytes(nanFill(nOut)))
	if err := kern.Launch(groups, threads,
		g.up(f32bytes(q)), g.up(f32bytes(kc)), g.up(f32bytes(vc)), g.up(u32bytes(send)), out); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, nOut*4)
	if err := out.Read(raw); err != nil {
		t.Fatal(err)
	}
	var sse, sy2 float64
	sc := make([]float64, c.cap)
	for r := 0; r < c.rows; r++ {
		cnt := int(pn[1+r])
		lo := 0
		if c.window > 0 && cnt > c.window {
			lo = cnt - c.window
		}
		for h := 0; h < c.heads; h++ {
			kvh := (h / gqa) * c.dim
			mx := math.Inf(-1)
			for p := lo; p < cnt; p++ {
				var dot float64
				for e := 0; e < c.dim; e++ {
					dot += float64(q[(r*c.heads+h)*c.dim+e]) * float64(kc[(kvh+e)*kStride+p])
				}
				x := dot * float64(scale)
				if c.softcap > 0 {
					x = float64(c.softcap) * math.Tanh(x/float64(c.softcap))
				}
				sc[p] = x
				mx = math.Max(mx, x)
			}
			var sum float64
			for p := lo; p < cnt; p++ {
				sc[p] = math.Exp(sc[p] - mx)
				sum += sc[p]
			}
			for e := 0; e < c.dim; e++ {
				var acc float64
				for p := lo; p < cnt; p++ {
					acc += sc[p] * float64(vc[p*kvDim+kvh+e])
				}
				want := acc / sum
				i := (r*c.heads+h)*c.dim + e
				got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
				if math.IsNaN(got) || math.IsInf(got, 0) {
					t.Fatalf("row %d head %d dim %d: %v", r, h, e, got)
				}
				sse += (got - want) * (got - want)
				sy2 += want * want
			}
		}
	}
	if sy2 == 0 {
		t.Fatal("the reference is all zero; the oracle is degenerate")
	}
	nmse := sse / sy2
	if violate {
		if nmse < 1e-5 {
			t.Fatalf("the violation (every count one short) read NMSE %.3e: the gate cannot see a dropped key", nmse)
		}
		t.Logf("violation NMSE %.3e (fails the bound, as it must)", nmse)
		return
	}
	t.Logf("NMSE %.3e", nmse)
	// binary16 operands (Q, K, V and P), float32 sums: the bound the kernels
	// this replaces are held to against their own FMA references.
	if nmse > 1e-5 {
		t.Fatalf("NMSE %.3e against the float64 reference", nmse)
	}
}

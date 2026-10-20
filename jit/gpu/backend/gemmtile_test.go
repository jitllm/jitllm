package backend_test

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// gemmTileCases are the shapes TestGemmTileMatchesTheReference sweeps: every
// staging nesting (rows a multiple of the threads, threads a multiple of the
// rows), one to four sub-blocks a trip, tiles that are tall, wide and square,
// and a bias on each family.
var gemmTileCases = []struct {
	rows, k, ntok int
	tl            oldkernels.TileGemm
	bias          bool
}{
	{64, 512, 32, oldkernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: 2}, false},
	{128, 512, 64, oldkernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: 2}, true},
	{128, 1024, 64, oldkernels.TileGemm{MT: 4, NT: 4, WM: 2, WN: 2, KB: 1}, false},
	{256, 768, 32, oldkernels.TileGemm{MT: 4, NT: 4, WM: 4, WN: 1, KB: 1}, true},
	{64, 1024, 128, oldkernels.TileGemm{MT: 2, NT: 4, WM: 2, WN: 2, KB: 2}, true},
}

// TestGemmTileMatchesTheReference holds the staged collective-tile GEMM to the
// float64 reference and to the dp4a kernel it replaces, on every format that
// has a binary16 dequant. A device that lowers an 8x8 tile from workgroup
// memory (backend.LowersTile: Metal's simdgroup_matrix, PTX's mma.sync) must
// build every case; anywhere else a refusal is the expected answer.
func TestGemmTileMatchesTheReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran, split, tall := 0, 0, 0
	for _, d := range devs {
		defer d.Close()
		// The 16-row tiles (TileGemm.Tall) where the device lowers them.
		talls := []bool{false}
		if ok, _ := backend.LowersTile(d, tile16x8); ok {
			talls = append(talls, true)
		}
		for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0, kernels.Q5_K,
			kernels.Q4_0, kernels.Q5_0, kernels.Q3_K, kernels.Q5_1} {
			for _, sh := range gemmTileCases {
				for _, isTall := range talls {
					sub, _, _, _ := kernels.Layout(q)
					tl := sh.tl
					tl.KB = max(tl.KB*32/sub, 1)
					tl.Tall = isTall
					s := oldkernels.MatVecShape{T: q, K: sh.k, Rows: sh.rows, NTok: sh.ntok, Bias: sh.bias}
					kk, err := oldkernels.GemmTile(s, tl)
					if err != nil {
						t.Fatalf("%v %+v: %v", q, sh, err)
					}
					kern, err := d.Compile(kk)
					if err != nil {
						if ok, _ := backend.LowersTile(d, tile8); ok {
							t.Fatalf("%s %v %+v: %v", d.API(), q, sh, err)
						}
						t.Logf("%s: %v", d.API(), err)
						break
					}
					ran++
					if strings.Contains(kk.Name, "_p") {
						split++
					}
					if isTall {
						tall++
					}
					mma70Case(t, d, kern, s, fmt.Sprintf("gemmtile %+v", tl),
						oldkernels.GemmTileGroups(s, tl), tl.Threads(), true)
					kern.Close()
				}
			}
		}
	}
	t.Logf("%d cases, %d of them on 16-row tiles", ran, tall)
	if ran == 0 {
		t.Skip("no backend lowers the collective tile GEMM here")
	}
	// The split staging (a sub-block across threads) must have been among
	// the kernels checked, or this gate says nothing about it.
	if split == 0 {
		t.Fatal("no case built the split staging")
	}
}

// TestGemmTileGroupedMatchesPerExpert is the grouped (mixture) GemmTile held
// to the dense GemmTile on each expert's own sheet bit for bit, and to the
// float64 reference, with a block past `used` required to stay poisoned --
// TestGemmVoltaGroupedMatchesPerExpert's contract, on Metal's tiles.
func TestGemmTileGroupedMatchesPerExpert(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.Q4_K, kernels.MXFP4, kernels.Q6_K, kernels.Q8_0, kernels.Q5_0,
			kernels.Q5_K, kernels.Q4_0, kernels.Q5_1} {
			sub, _, _, _ := kernels.Layout(q)
			kb := max(32/sub, 1)
			for _, c := range []struct {
				rows, k, experts int
				tl               oldkernels.TileGemm
				sel              []uint32
				used             int
				bias             bool
			}{
				{128, 512, 4, oldkernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: kb}, []uint32{2, 0, 0, 3, 1}, 5, false},
				{192, 256, 4, oldkernels.TileGemm{MT: 4, NT: 2, WM: 2, WN: 2, KB: kb}, []uint32{3, 3, 1, 0}, 3, true},
				{256, 512, 3, oldkernels.TileGemm{MT: 4, NT: 4, WM: 2, WN: 2, KB: kb}, []uint32{1, 2, 0}, 3, false},
				{96, 512, 4, oldkernels.TileGemm{MT: 2, NT: 2, WM: 2, WN: 2, KB: kb}, []uint32{0, 3, 3, 1}, 4, true},
			} {
				s := oldkernels.MatVecShape{T: q, K: c.k, Rows: c.rows, NTok: c.tl.Toks() * len(c.sel), Experts: c.experts,
					Bias: c.bias}
				kk, err := oldkernels.GemmTile(s, c.tl)
				if err != nil {
					t.Fatalf("%v %+v: %v", q, c.tl, err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					if ok, _ := backend.LowersTile(d, tile8); ok {
						t.Fatalf("%s %v %+v: %v", d.API(), q, c.tl, err)
					}
					t.Logf("%s: %v", d.API(), err)
					break
				}
				ran++
				groupedGemmCase(t, d, kern, s, tileGrouped(c.tl), c.sel, c.used)
				kern.Close()
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the collective tile GEMM here")
	}
}

// TestGemmTileSpeed times GemmTile's blockings, split and unsplit staging,
// against the dp4a batched twin (JITLLM_GEMMTILE_BENCH=1). A probe, not a
// gate. JITLLM_GEMMTILE_SHAPES names the shapes as fmt:rowsxk,... (default
// Llama-3.2-1B's Q4_K projections) and JITLLM_GEMMTILE_NTOK the tokens (512).
func TestGemmTileSpeed(t *testing.T) {
	if os.Getenv("JITLLM_GEMMTILE_BENCH") == "" {
		t.Skip("set JITLLM_GEMMTILE_BENCH=1")
	}
	type shape struct {
		q       kernels.Quant
		rows, k int
	}
	shapes := []shape{{kernels.Q4_K, 8192, 2048}, {kernels.Q4_K, 2048, 8192}, {kernels.Q4_K, 2048, 2048}, {kernels.Q4_K, 512, 2048}}
	if v := os.Getenv("JITLLM_GEMMTILE_SHAPES"); v != "" {
		shapes = nil
		for _, f := range strings.Split(v, ",") {
			var name string
			var sh shape
			if n, err := fmt.Sscanf(strings.Replace(f, ":", " ", 1), "%s %dx%d", &name, &sh.rows, &sh.k); n != 3 {
				t.Fatalf("shape %q: %v", f, err)
			}
			for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0, kernels.Q5_K,
				kernels.Q4_0, kernels.Q5_0, kernels.Q3_K, kernels.Q5_1} {
				if q.String() == name {
					sh.q = q
				}
			}
			if sh.q.String() != name {
				t.Fatalf("shape %q: no format %q", f, name)
			}
			shapes = append(shapes, sh)
		}
	}
	ntok := 512
	if v := os.Getenv("JITLLM_GEMMTILE_NTOK"); v != "" {
		fmt.Sscan(v, &ntok)
	}
	gpuLock(t)
	for _, d := range backend.Open() {
		defer d.Close()
		if ok, _ := backend.LowersTile(d, tile8); !ok {
			continue
		}
		for _, sh := range shapes {
			q, rows, k := sh.q, sh.rows, sh.k
			rng := rand.New(rand.NewSource(1))
			g := newGPU(t, d)
			qs, dw, scw, _ := kernels.PackWeights(q, rawWeights(q, rows, k, rng), rows, k)
			bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
			out := g.up(make([]byte, rows*ntok*4))
			bB := g.up(make([]byte, ntok*k*2))
			sync := func() { out.Read(make([]byte, 4)) }
			time1 := func(name string, f func() error) {
				if err := f(); err != nil {
					t.Logf("%s: %v", name, err)
					return
				}
				sync()
				const n = 20
				t0 := time.Now()
				for i := 0; i < n; i++ {
					f()
				}
				sync()
				el := time.Since(t0) / n
				t.Logf("%v %5dx%-5d t%d %-34s %8.1f us  %.2f TFLOP/s", q, rows, k, ntok, name, float64(el.Microseconds()),
					2*float64(rows)*float64(k)*float64(ntok)/el.Seconds()/1e12)
			}
			sub, _, _, _ := kernels.Layout(q)
			for _, tl := range []oldkernels.TileGemm{
				{MT: 4, NT: 2, WM: 2, WN: 2, KB: 1}, {MT: 4, NT: 2, WM: 2, WN: 2, KB: 2},
				{MT: 4, NT: 4, WM: 2, WN: 2, KB: 1}, {MT: 4, NT: 4, WM: 2, WN: 2, KB: 2},
				{MT: 2, NT: 4, WM: 2, WN: 2, KB: 1}, {MT: 4, NT: 4, WM: 4, WN: 1, KB: 1},
			} {
				tl.KB = max(tl.KB*32/sub, 1)
				for _, ns := range []bool{true, false} {
					tl.NoSplit = ns
					s := oldkernels.MatVecShape{T: q, K: k, Rows: rows, NTok: ntok}
					kk, err := oldkernels.GemmTile(s, tl)
					if err != nil {
						continue
					}
					if !ns && !strings.Contains(kk.Name, "_p") {
						continue // no split for this blocking: the same kernel
					}
					kern, err := d.Compile(kk)
					if err != nil {
						t.Fatalf("%+v: %v", tl, err)
					}
					time1(fmt.Sprintf("tile m%dn%d w%dx%d k%d split %v", tl.MT, tl.NT, tl.WM, tl.WN, tl.KB, !ns), func() error {
						return kern.Launch(oldkernels.GemmTileGroups(s, tl), tl.Threads(), bQS, bD, bSC, bB, out)
					})
					kern.Close()
				}
			}
			g.free()
		}
	}
}

package backend_test

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestGemmVoltaMatchesTheReference holds the shared-memory-staged sm_70 GEMM to
// the same float64 reference and dp4a bound as MatVecMMA70, over every tile
// family the tier picks from, both staging nestings (a workgroup's rows a
// multiple of its threads and the other way round), k staged one and two
// sub-blocks a trip, a k-split with a bias, and the three formats that shape
// the dequant differently (a min array, a secondary plane, a flat byte).
func TestGemmVoltaMatchesTheReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0, kernels.Q5_K, kernels.Q4_0, kernels.Q5_0, kernels.Q5_1} {
			for _, sh := range []struct {
				rows, k, ntok, split int
				tl                   kernels.VoltaTile
				bias                 bool
			}{
				{128, 512, 128, 1, kernels.VoltaTile{MT: 2, NT: 8, WM: 2, WN: 2, KB: 1}, false},
				{256, 512, 128, 2, kernels.VoltaTile{MT: 2, NT: 8, WM: 2, WN: 2, KB: 1}, true},
				{128, 1024, 64, 1, kernels.VoltaTile{MT: 4, NT: 4, WM: 1, WN: 2, KB: 2}, true},
				{64, 512, 64, 1, kernels.VoltaTile{MT: 1, NT: 2, WM: 2, WN: 2, KB: 2}, false},
				{32, 1024, 32, 2, kernels.VoltaTile{MT: 1, NT: 4, WM: 1, WN: 1, KB: 4}, true},
				{256, 768, 32, 3, kernels.VoltaTile{MT: 1, NT: 4, WM: 4, WN: 1, KB: 2}, false},
				// A 96-row block, whose 96 staging items do not nest in 128
				// threads (gpt-oss's 2880-row projection).
				{192, 512, 128, 2, kernels.VoltaTile{MT: 3, NT: 4, WM: 1, WN: 4, KB: 1}, true},
				// The 16-token block of a prompt at an odd number of 16s.
				{256, 512, 112, 1, kernels.VoltaTile{MT: 4, NT: 2, WM: 1, WN: 1, KB: 1}, true},
				{192, 1024, 48, 2, kernels.VoltaTile{MT: 3, NT: 2, WM: 1, WN: 1, KB: 1}, false},
			} {
				s := kernels.MatVecShape{T: q, K: sh.k, Rows: sh.rows, NTok: sh.ntok, MT: sh.tl.MT, NT: sh.tl.NT,
					Split: sh.split, Bias: sh.bias}
				kk, err := kernels.GemmVolta(s, sh.tl)
				if err != nil {
					t.Fatalf("%v %+v: %v", q, sh, err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					// Only PTX lowers the shape; anywhere else a refusal is the
					// expected answer. On PTX it is a broken kernel, not a skip.
					if d.API() == "ptx" {
						t.Fatalf("%s %+v: %v", d.API(), sh, err)
					}
					t.Logf("%s: %v", d.API(), err)
					break
				}
				ran++
				mma70Case(t, d, kern, s, fmt.Sprintf("gemmvolta %+v", sh.tl),
					kernels.GemmVoltaGroups(s, sh.tl), sh.tl.Threads(), true)
				kern.Close()
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}

// TestGemmF16MatchesTheReference holds GemmVolta's m16n8 forms (F16K 16 on
// sm_80 on, 8 on sm_75) to the reference MatVecMMA70 is held to: every
// staging nesting, a block of one trip and of three, a narrow tile of one
// m-tile and one n-tile, 96-row blocks, a k-split with and without a bias,
// and the formats whose dequant differs. A shape the device does not lower
// is refused on every k; on a card that lowers m16n8k16 a refusal of it is
// a failure.
func TestGemmF16MatchesTheReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, f16k := range []int{16, 8} {
			for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0, kernels.Q5_K, kernels.Q4_0, kernels.Q5_0, kernels.Q5_1} {
				sub, _, _, _ := kernels.Layout(q)
				kb := max(32/sub, 1)
				refused := false
				for _, sh := range []struct {
					rows, k, ntok, split int
					tl                   kernels.VoltaTile
					bias                 bool
				}{
					{128, 512, 64, 1, kernels.VoltaTile{MT: 4, NT: 4, WM: 2, WN: 2}, false},
					{256, 768, 128, 1, kernels.VoltaTile{MT: 4, NT: 8, WM: 2, WN: 2}, true},
					{32, 256, 32, 1, kernels.VoltaTile{MT: 1, NT: 2, WM: 2, WN: 2}, true},
					{192, 512, 32, 2, kernels.VoltaTile{MT: 3, NT: 2, WM: 2, WN: 2}, true},
					{256, 768, 32, 3, kernels.VoltaTile{MT: 1, NT: 4, WM: 4, WN: 1}, false},
					{64, 1024, 128, 1, kernels.VoltaTile{MT: 4, NT: 4, WM: 1, WN: 4}, false},
				} {
					sh.tl.KB, sh.tl.F16K = kb, f16k
					s := kernels.MatVecShape{T: q, K: sh.k, Rows: sh.rows, NTok: sh.ntok, Split: sh.split, Bias: sh.bias}
					kk, err := kernels.GemmVolta(s, sh.tl)
					if err != nil {
						// A staging nesting the block does not admit (a 96-row
						// block of two-sub-block trips) is the tier's next tile.
						if !strings.Contains(err.Error(), "do not tile") {
							t.Fatalf("%v %+v: %v", q, sh, err)
						}
						continue
					}
					kern, err := d.Compile(kk)
					if err != nil {
						t.Logf("%s %s f%d: %v", d.API(), d.Name(), f16k, err)
						refused = true
						break
					}
					ran++
					mma70Case(t, d, kern, s, fmt.Sprintf("gemm f%d %+v", f16k, sh.tl),
						kernels.GemmVoltaGroups(s, sh.tl), sh.tl.Threads(), true)
					kern.Close()
				}
				if refused && d.API() == "ptx" && ptxSM(d.Name()) >= mmaMinSM(ir.MMAShape{M: 16, N: 8, K: f16k, Kind: ir.MMAF16}) {
					t.Fatalf("%s lowers m16n8k%d and refused it", d.Name(), f16k)
				}
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers an m16n8 binary16 shape here")
	}
}

// TestGemmVoltaSpeed times GemmVolta against MatVecMMA70 on Llama-3.1-8B's
// shapes (JITLLM_MMA70_BENCH=1). A probe, not a gate.
func TestGemmVoltaSpeed(t *testing.T) {
	if os.Getenv("JITLLM_MMA70_BENCH") == "" {
		t.Skip("set JITLLM_MMA70_BENCH=1")
	}
	gpuLock(t)
	devs := backend.Open()
	for _, d := range devs {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		for _, sh := range [][3]int{{14336, 4096, 512}, {4096, 14336, 512}, {4096, 4096, 512}, {1024, 4096, 512}} {
			rows, k, ntok := sh[0], sh[1], sh[2]
			rng := rand.New(rand.NewSource(1))
			g := newGPU(t, d)
			qs, dw, scw, _ := kernels.PackWeights(kernels.Q4_K, rawWeights(kernels.Q4_K, rows, k, rng), rows, k)
			bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
			out := g.up(make([]byte, 8*rows*ntok*4))
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
				t.Logf("%5dx%-5d t%d %-34s %8.1f us  %.2f TMAC/s", rows, k, ntok, name, float64(el.Microseconds()),
					float64(rows)*float64(k)*float64(ntok)/el.Seconds()/1e12)
			}
			kk, _ := kernels.MatVecMMA70(kernels.MatVecShape{T: kernels.Q4_K, K: k, Rows: rows, NTok: ntok, MT: 2, NT: 4})
			if kern, err := d.Compile(kk); err == nil {
				warps := rows / 64 * ntok / 32
				time1("mma70 mt2 nt4", func() error { return kern.Launch((warps*32+127)/128, 128, bQS, bD, bSC, bB, out) })
			}
			tiles := []kernels.VoltaTile{
				{MT: 2, NT: 8, WM: 2, WN: 2, KB: 1}, {MT: 4, NT: 4, WM: 1, WN: 4, KB: 1},
				{MT: 4, NT: 4, WM: 2, WN: 4, KB: 1}, {MT: 4, NT: 2, WM: 1, WN: 8, KB: 1},
				{MT: 2, NT: 4, WM: 2, WN: 4, KB: 1}, {MT: 4, NT: 4, WM: 1, WN: 2, KB: 1},
				{MT: 2, NT: 4, WM: 1, WN: 4, KB: 1}, {MT: 4, NT: 2, WM: 1, WN: 4, KB: 1},
			}
			if v := os.Getenv("JITLLM_GEMM_TILE"); v != "" {
				var tl kernels.VoltaTile
				fmt.Sscanf(v, "%d,%d,%d,%d,%d", &tl.MT, &tl.NT, &tl.WM, &tl.WN, &tl.KB)
				tiles = []kernels.VoltaTile{tl}
			}
			for _, tl := range tiles {
				for _, split := range []int{1, 2, 4} {
					s := kernels.MatVecShape{T: kernels.Q4_K, K: k, Rows: rows, NTok: ntok, Split: split}
					kk, err := kernels.GemmVolta(s, tl)
					if err != nil {
						continue
					}
					kern, err := d.Compile(kk)
					if err != nil {
						t.Logf("%+v: %v", tl, err)
						continue
					}
					time1(fmt.Sprintf("gemm %dx%d w%dx%d k%d s%d", tl.MT, tl.NT, tl.WM, tl.WN, tl.KB, split), func() error {
						return kern.Launch(kernels.GemmVoltaGroups(s, tl), tl.Threads(), bQS, bD, bSC, bB, out)
					})
				}
			}
			g.free()
		}
	}
}

// TestGemmF16DeepStageMatchesTheReference holds GemmVolta's deep m16n8 stage
// (llama.cpp MMQ's shape: eight warps a workgroup, 64 to 256 elements of k a
// trip, one buffer or two) to MatVecMMA70's reference: both 8-warp grids,
// every trip depth the shared memory admits, a k-split with a bias, a k of
// one trip and of several, and every format whose dequant differs. A refusal
// on a card that lowers the instruction is a failure.
func TestGemmF16DeepStageMatchesTheReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		for _, f16k := range []int{16, 8} {
			if ptxSM(d.Name()) < mmaMinSM(ir.MMAShape{M: 16, N: 8, K: f16k, Kind: ir.MMAF16}) {
				continue
			}
			for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0, kernels.Q5_K, kernels.Q4_0, kernels.Q5_1} {
				sub, _, _, _ := kernels.Layout(q)
				for _, sh := range []struct {
					rows, k, ntok, split, depth int
					tl                          kernels.VoltaTile
					bias                        bool
				}{
					{128, 512, 64, 1, 64, kernels.VoltaTile{MT: 4, NT: 2, WM: 2, WN: 4}, false},
					{128, 512, 64, 1, 128, kernels.VoltaTile{MT: 4, NT: 2, WM: 2, WN: 4, Single: true}, true},
					{256, 768, 128, 1, 128, kernels.VoltaTile{MT: 2, NT: 4, WM: 4, WN: 2, Single: true}, false},
					{128, 1024, 64, 2, 128, kernels.VoltaTile{MT: 2, NT: 4, WM: 4, WN: 2, Single: true}, true},
					{128, 256, 32, 1, 128, kernels.VoltaTile{MT: 4, NT: 1, WM: 2, WN: 4, Single: true}, true},
					{128, 768, 64, 3, 64, kernels.VoltaTile{MT: 2, NT: 4, WM: 4, WN: 2, Single: true}, false},
				} {
					sh.tl.KB, sh.tl.F16K = sh.depth/sub, f16k
					s := kernels.MatVecShape{T: q, K: sh.k, Rows: sh.rows, NTok: sh.ntok, Split: sh.split, Bias: sh.bias}
					kk, err := kernels.GemmVolta(s, sh.tl)
					if err != nil {
						t.Fatalf("%v %+v: %v", q, sh, err)
					}
					kern, err := d.Compile(kk)
					if err != nil {
						t.Fatalf("%s %s f%d %+v: %v", d.API(), d.Name(), f16k, sh, err)
					}
					ran++
					mma70Case(t, d, kern, s, fmt.Sprintf("gemm f%d k%d %+v", f16k, sh.depth, sh.tl),
						kernels.GemmVoltaGroups(s, sh.tl), sh.tl.Threads(), true)
					kern.Close()
				}
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend lowers an m16n8 binary16 shape here")
	}
}

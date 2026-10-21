package backend_test

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGemmVoltaGroupedSpeed times the grouped tensor-core GEMM against the
// grouped dp4a matvec it replaces on the mixtures' real shapes, with a routing
// drawn at random over the bank (JITLLM_MMA70_BENCH=1). A probe, not a gate.
func TestGemmVoltaGroupedSpeed(t *testing.T) {
	if os.Getenv("JITLLM_MMA70_BENCH") == "" {
		t.Skip("set JITLLM_MMA70_BENCH=1")
	}
	gpuLock(t)
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		for _, sh := range []struct {
			name             string
			q                mainkernels.Quant
			rows, k, e, used int
			pairs            int
		}{
			{"gpt-oss gate/up/down", mainkernels.MXFP4, 2880, 2880, 32, 4, 2048},
			{"V2-Lite gate/up", mainkernels.Q4_K, 1408, 2048, 64, 6, 3072},
			{"V2-Lite down q8", mainkernels.Q8_0, 2048, 1408, 64, 6, 3072},
			{"Qwen3-30B gate/up", mainkernels.Q4_K, 768, 2048, 128, 8, 4096},
			{"Mixtral gate/up", mainkernels.Q4_K, 14336, 4096, 8, 2, 1024},
		} {
			rng := rand.New(rand.NewSource(1))
			g := newGPU(t, d)
			raw := randWeights(sh.q, sh.rows, sh.k, rng)
			q1, d1, s1, err := mainkernels.PackWeights(sh.q, raw, sh.rows, sh.k)
			if err != nil {
				t.Fatal(err)
			}
			rep := func(v []uint32) []uint32 {
				if len(v) == 0 {
					return []uint32{0}
				}
				out := make([]uint32, 0, len(v)*sh.e)
				for i := 0; i < sh.e; i++ {
					out = append(out, v...)
				}
				return out
			}
			bQS, bD, bSC := g.up(u32bytes(rep(q1))), g.up(u32bytes(rep(d1))), g.up(u32bytes(rep(s1)))
			// A random routing's per-expert counts, each run padded to the block.
			count := make([]int, sh.e)
			for i := 0; i < sh.pairs; i++ {
				count[rng.Intn(sh.e)]++
			}
			selFor := func(unit int) []uint32 {
				var sel []uint32
				for e, n := range count {
					for j := 0; j < (n+unit-1)/unit; j++ {
						sel = append(sel, uint32(e))
					}
				}
				return sel
			}
			maxCols := sh.pairs + sh.e*128
			out := g.up(make([]byte, maxCols*sh.rows*4))
			bB := g.up(make([]byte, maxCols*sh.k*2))
			bA := g.up(make([]byte, maxCols*sh.k))
			bAX := g.up(make([]byte, 3*maxCols*(sh.k/32)*4))
			sync := func() { out.Read(make([]byte, 4)) }
			time1 := func(name string, cols int, f func() error) {
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
				t.Logf("%-22s %5dx%-5d e%-3d %-30s %8.1f us  %.2f TMAC/s useful (%d cols)", sh.name, sh.rows, sh.k, sh.e,
					name, float64(el.Microseconds()), float64(sh.rows)*float64(sh.k)*float64(sh.pairs)/el.Seconds()/1e12, cols)
			}
			// The dp4a grouped matvec at its shipping Tok.
			{
				sel := selFor(4)
				ntok := len(sel) * 4
				kk, err := mainkernels.MatVec(mainkernels.MatVecShape{T: sh.q, K: sh.k, Rows: sh.rows, Experts: sh.e, NTok: ntok, Tok: 4})
				if err == nil {
					if kern, err := d.Compile(kk); err == nil {
						bSel := g.up(u32bytes(sel))
						time1("dp4a grouped tok4", ntok, func() error {
							return kern.Launch((sh.rows*len(sel)+127)/128, 128, bQS, bD, bSC, bA, bAX, out, bSel)
						})
						kern.Close()
					}
				}
			}
			sub, _, _, _ := mainkernels.Layout(sh.q)
			kb := max(32/sub, 1)
			for _, tl := range []mainkernels.VoltaTile{
				{MT: 4, NT: 1, WM: 1, WN: 4, KB: kb}, {MT: 2, NT: 1, WM: 1, WN: 4, KB: kb},
				{MT: 4, NT: 2, WM: 1, WN: 4, KB: kb}, {MT: 2, NT: 2, WM: 1, WN: 4, KB: kb},
				{MT: 2, NT: 1, WM: 2, WN: 2, KB: kb}, {MT: 4, NT: 1, WM: 2, WN: 2, KB: kb},
				{MT: 1, NT: 1, WM: 1, WN: 4, KB: kb}, {MT: 4, NT: 4, WM: 1, WN: 4, KB: kb},
				{MT: 3, NT: 1, WM: 1, WN: 4, KB: kb}, {MT: 3, NT: 2, WM: 1, WN: 4, KB: kb},
				{MT: 6, NT: 1, WM: 1, WN: 4, KB: kb}, {MT: 5, NT: 2, WM: 1, WN: 4, KB: kb},
			} {
				sel := selFor(tl.Toks())
				ntok := len(sel) * tl.Toks()
				s := mainkernels.MatVecShape{T: sh.q, K: sh.k, Rows: sh.rows, NTok: ntok, Experts: sh.e}
				kk, err := mainkernels.GemmVolta(s, tl)
				if err != nil {
					t.Logf("%+v: %v", tl, err)
					continue
				}
				kern, err := d.Compile(kk)
				if err != nil {
					t.Logf("%+v: %v", tl, err)
					continue
				}
				bSel := g.up(u32bytes(sel))
				time1(fmt.Sprintf("gemm %dx%d w%dx%d (%dx%d)", tl.MT, tl.NT, tl.WM, tl.WN, tl.Rows(), tl.Toks()), ntok, func() error {
					return kern.Launch(mainkernels.GemmVoltaGrouped(s, tl, len(sel)), tl.Threads(), bQS, bD, bSC, bB, out, bSel)
				})
				kern.Close()
			}
			g.free()
		}
	}
}

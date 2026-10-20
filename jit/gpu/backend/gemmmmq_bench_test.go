package backend_test

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGemmMMQSweep sets the staged int8 GEMM in llama.cpp MMQ's shape
// (GemmInt8 with deep trips, eight warps, one buffer; m8n8k16 below sm_80)
// against the shipping binary16 GEMM and its best deep tile on
// Llama-3.1-8B Q4_K_M's projections at 512 tokens, interleaved as
// TestGemmDeepSweep: the median per-round ratio to the shipping tile and
// its IQR, the first row the A/A control. It also times each arm family's
// operand preparation alone (ActF16T against Quantize over the same rows):
// in the engine both are fused into the norm or activation before the GEMM,
// so the difference between them, not their sum, is what an int8 arm adds.
// JITLLM_GEMMMMQ_BENCH=1; a probe, not a gate.
func TestGemmMMQSweep(t *testing.T) {
	if os.Getenv("JITLLM_GEMMMMQ_BENCH") == "" {
		t.Skip("set JITLLM_GEMMMMQ_BENCH=1")
	}
	gpuLock(t)
	rounds := 24
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		sm := ptxSM(d.Name())
		f16k := 16
		if sm < 80 {
			f16k = 8
		}
		type arm struct {
			name   string
			launch func() error
			kern   []backend.Kernel
			ratios []float64
			us     []float64
		}
		for _, sh := range []struct {
			what    string
			q       mainkernels.Quant
			rows, k int
		}{
			{"q/o", mainkernels.Q4_K, 4096, 4096},
			{"k", mainkernels.Q4_K, 1024, 4096},
			{"v", mainkernels.Q6_K, 1024, 4096},
			{"gate/up", mainkernels.Q4_K, 14336, 4096},
			{"down q4k", mainkernels.Q4_K, 4096, 14336},
			{"down q6k", mainkernels.Q6_K, 4096, 14336},
		} {
			const ntok = 512
			sub, _, _, _ := mainkernels.Layout(sh.q)
			rng := rand.New(rand.NewSource(1))
			g := newGPU(t, d)
			qs, dw, scw, _ := mainkernels.PackWeights(sh.q, rawWeights(sh.q, sh.rows, sh.k, rng), sh.rows, sh.k)
			if len(scw) == 0 {
				scw = []uint32{0}
			}
			x := make([]float32, ntok*sh.k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
			out := g.up(make([]byte, 4*sh.rows*ntok*4))
			bX := g.up(f32bytes(x))
			bB := g.up(make([]byte, ntok*sh.k*2))
			bA := g.up(make([]byte, ntok*sh.k))
			bAX := g.up(make([]byte, 4*(ntok*sh.k/32+ntok*sh.k/16)))
			sync := func() { out.Read(make([]byte, 4)) }
			var arms []*arm
			f16 := func(name string, tl mainkernels.VoltaTile, depth, split int) {
				tl.KB, tl.F16K = max(depth/sub, 1), f16k
				s := mainkernels.MatVecShape{T: sh.q, K: sh.k, Rows: sh.rows, NTok: ntok, Split: split}
				kk, err := mainkernels.GemmVolta(s, tl)
				if err != nil {
					return
				}
				kern, err := d.Compile(kk)
				if err != nil {
					t.Logf("%s: %v", name, err)
					return
				}
				r, _ := backend.KernelResourcesOf(kern)
				arms = append(arms, &arm{name: fmt.Sprintf("%-40s regs %3d spill %d res %d", name, r.Regs, r.SpillBytes, r.Resident),
					kern: []backend.Kernel{kern}, launch: func() error {
						return kern.Launch(mainkernels.GemmVoltaGroups(s, tl), tl.Threads(), bQS, bD, bSC, bB, out)
					}})
			}
			i8 := func(name string, tl mainkernels.Int8Tile, split int) {
				s := mainkernels.MatVecShape{T: sh.q, K: sh.k, Rows: sh.rows, NTok: ntok, Split: split}
				kk, err := mainkernels.GemmInt8(s, tl)
				if err != nil {
					return
				}
				kern, err := d.Compile(kk)
				if err != nil {
					t.Logf("%s: %v", name, err)
					return
				}
				r, _ := backend.KernelResourcesOf(kern)
				arms = append(arms, &arm{name: fmt.Sprintf("%-40s regs %3d spill %d res %d", name, r.Regs, r.SpillBytes, r.Resident),
					kern: []backend.Kernel{kern}, launch: func() error {
						return kern.Launch(mainkernels.GemmInt8Groups(s, tl), tl.Threads(), bQS, bD, bSC, bA, bAX, out)
					}})
			}
			ship := mainkernels.VoltaTile{MT: 4, NT: 4, WM: 2, WN: 2}
			deep := mainkernels.VoltaTile{MT: 4, NT: 4, WM: 2, WN: 4, Single: true}
			f16("f16 ship s1", ship, 32, 1)
			f16("f16 ship s1 (A/A)", ship, 32, 1)
			f16("f16 ship s2", ship, 32, 2)
			f16("f16 d64 w8 4x4 1b s1", deep, 64, 1)
			f16("f16 d64 w8 4x4 1b s2", deep, 64, 2)
			for _, m8 := range []bool{sm < 80, false} {
				if !m8 && sm < 80 {
					continue
				}
				for _, dd := range []int{1, 2, 4} {
					for _, single := range []bool{false, true} {
						for _, tl := range []mainkernels.Int8Tile{
							{MT: 4, NT: 4, WM: 2, WN: 2}, {MT: 2, NT: 2, WM: 2, WN: 2}, {MT: 2, NT: 4, WM: 2, WN: 2},
							{MT: 4, NT: 2, WM: 2, WN: 4}, {MT: 2, NT: 4, WM: 4, WN: 2}, {MT: 2, NT: 2, WM: 4, WN: 2},
						} {
							for _, ldm := range []bool{false, true} {
								if !ldm && os.Getenv("JITLLM_GEMMMMQ_LDM_ONLY") != "" {
									continue
								}
								tl.D, tl.Single, tl.M8, tl.LdM = dd, single, m8, ldm
								for _, split := range []int{1, 2} {
									i8(fmt.Sprintf("i8 m8=%v ldm=%v d%d w%d %dx%d w%dx%d 1b=%v s%d", m8, ldm, 32*dd, tl.WM*tl.WN,
										tl.MT, tl.NT, tl.WM, tl.WN, single, split), tl, split)
								}
							}
						}
					}
				}
			}
			// The operand preparations, alone.
			if kk, err := mainkernels.ActF16T(sh.q, ntok, sh.k); err == nil {
				if kern, err := d.Compile(kk); err == nil {
					n := ntok * sh.k / sub
					arms = append(arms, &arm{name: "prep ActF16T (binary16 operand)", kern: []backend.Kernel{kern},
						launch: func() error { return kern.Launch((n+127)/128, 128, bX, bB) }})
				}
			}
			if kk, err := mainkernels.Quantize(ntok*sh.k, 32); err == nil {
				if kern, err := d.Compile(kk); err == nil {
					n := mainkernels.QuantizeThreads(ntok * sh.k / 32)
					arms = append(arms, &arm{name: "prep Quantize (int8 operand)", kern: []backend.Kernel{kern},
						launch: func() error { return kern.Launch((n+127)/128, 128, bX, bA, bAX) }})
				}
			}
			for _, a := range arms {
				if err := a.launch(); err != nil {
					t.Fatalf("%s: %v", a.name, err)
				}
			}
			sync()
			const reps = 5
			for r := 0; r < rounds; r++ {
				var base float64
				for i, a := range arms {
					sync()
					t0 := time.Now()
					for j := 0; j < reps; j++ {
						a.launch()
					}
					sync()
					us := float64(time.Since(t0).Microseconds()) / reps
					if i == 0 {
						base = us
					}
					a.us = append(a.us, us)
					a.ratios = append(a.ratios, us/base)
				}
			}
			for _, a := range arms {
				med, iqr := medIQR(a.ratios)
				umed, _ := medIQR(a.us)
				t.Logf("MMQ %-9s %v %5dx%-5d t%d %-62s %8.1f us  ratio %.3f IQR/med %.3f", sh.what, sh.q, sh.rows, sh.k, ntok,
					a.name, umed, med, iqr/med)
				for _, k := range a.kern {
					k.Close()
				}
			}
			g.free()
		}
	}
}

package backend_test

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestGemmDeepSweep times GemmVolta's m16n8 tiles on Llama-3.1-8B
// Q4_K_M's projection shapes at a 512-token chunk, interleaved: every round
// launches each arm in turn, and each arm's figure is the median of its
// per-round ratio to the shipping tile (f16Tiles' first) with its IQR. The
// shipping tile runs twice, so the first row is the A/A self-control.
// JITLLM_GEMMDEEP_BENCH=1; a probe, not a gate. Registers, spill, shared
// bytes and resident workgroups are the driver's.
func TestGemmDeepSweep(t *testing.T) {
	if os.Getenv("JITLLM_GEMMDEEP_BENCH") == "" {
		t.Skip("set JITLLM_GEMMDEEP_BENCH=1")
	}
	gpuLock(t)
	rounds := 24
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		f16k := 16
		if ptxSM(d.Name()) < 80 {
			f16k = 8
		}
		type arm struct {
			name   string
			tl     kernels.VoltaTile
			depth  int
			split  int
			kern   backend.Kernel
			s      kernels.MatVecShape
			ratios []float64
			us     []float64
		}
		for _, sh := range []struct {
			what    string
			q       kernels.Quant
			rows, k int
		}{
			{"q/o", kernels.Q4_K, 4096, 4096},
			{"k", kernels.Q4_K, 1024, 4096},
			{"v", kernels.Q6_K, 1024, 4096},
			{"gate/up", kernels.Q4_K, 14336, 4096},
			{"down q4k", kernels.Q4_K, 4096, 14336},
			{"down q6k", kernels.Q6_K, 4096, 14336},
		} {
			const ntok = 512
			sub, _, _, _ := kernels.Layout(sh.q)
			rng := rand.New(rand.NewSource(1))
			g := newGPU(t, d)
			qs, dw, scw, _ := kernels.PackWeights(sh.q, rawWeights(sh.q, sh.rows, sh.k, rng), sh.rows, sh.k)
			if len(scw) == 0 {
				scw = []uint32{0}
			}
			bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
			out := g.up(make([]byte, 4*sh.rows*ntok*4))
			bB := g.up(make([]byte, ntok*sh.k*2))
			sync := func() { out.Read(make([]byte, 4)) }
			ship := kernels.VoltaTile{MT: 4, NT: 4, WM: 2, WN: 2}
			var arms []*arm
			addArm := func(name string, tl kernels.VoltaTile, depth, split int) {
				tl.KB, tl.F16K = max(depth/sub, 1), f16k
				s := kernels.MatVecShape{T: sh.q, K: sh.k, Rows: sh.rows, NTok: ntok, Split: split}
				kk, err := kernels.GemmVolta(s, tl)
				if err != nil {
					return
				}
				kern, err := d.Compile(kk)
				if err != nil {
					t.Logf("%s: %v", name, err)
					return
				}
				arms = append(arms, &arm{name: name, tl: tl, depth: depth, split: split, kern: kern, s: s})
			}
			addArm("ship s1", ship, 32, 1)
			addArm("ship s1 (A/A)", ship, 32, 1)
			addArm("ship s2", ship, 32, 2)
			for _, depth := range []int{32, 64, 128} {
				for _, single := range []bool{false, true} {
					for _, tl := range []kernels.VoltaTile{
						{MT: 4, NT: 4, WM: 2, WN: 2}, {MT: 4, NT: 2, WM: 2, WN: 2},
						{MT: 4, NT: 2, WM: 2, WN: 4}, {MT: 2, NT: 4, WM: 4, WN: 2}, {MT: 4, NT: 4, WM: 2, WN: 4},
					} {
						tl.Single = single
						for _, split := range []int{1, 2} {
							if depth == 32 && !single && tl.WM*tl.WN == 4 && tl.MT == 4 && tl.NT == 4 {
								continue // the shipping tile
							}
							addArm(fmt.Sprintf("d%d w%d %dx%d w%dx%d 1b=%v s%d", depth, tl.WM*tl.WN, tl.MT, tl.NT, tl.WM, tl.WN, single, split), tl, depth, split)
						}
					}
				}
			}
			launch := func(a *arm) error {
				return a.kern.Launch(kernels.GemmVoltaGroups(a.s, a.tl), a.tl.Threads(), bQS, bD, bSC, bB, out)
			}
			for _, a := range arms {
				if err := launch(a); err != nil {
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
						launch(a)
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
				r, _ := backend.KernelResourcesOf(a.kern)
				med, iqr := medIQR(a.ratios)
				umed, _ := medIQR(a.us)
				t.Logf("%-9s %v %5dx%-5d t%d f%d %-34s %8.1f us  ratio %.3f IQR/med %.3f  regs %d spill %d shared %d resident %d",
					sh.what, sh.q, sh.rows, sh.k, ntok, f16k, a.name, umed, med, iqr/med, r.Regs, r.SpillBytes, r.SharedBytes, r.Resident)
				a.kern.Close()
			}
			// Every arm here reads the same binary16 operand, written by the
			// norm or activation before it, so operand preparation is equal
			// across arms and is the per-op split's to report.
			g.free()
		}
	}
}

func medIQR(v []float64) (med, iqr float64) {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	return s[n/2], s[(3*n)/4] - s[n/4]
}

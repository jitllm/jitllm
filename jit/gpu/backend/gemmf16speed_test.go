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

// TestGemmF16Speed times GemmVolta's m16n8 form against MatVecMMA on
// Llama-3.2-1B's shapes at a 512-token chunk (JITLLM_GEMMF16_BENCH=1). A
// probe, not a gate.
func TestGemmF16Speed(t *testing.T) {
	if os.Getenv("JITLLM_GEMMF16_BENCH") == "" {
		t.Skip("set JITLLM_GEMMF16_BENCH=1")
	}
	gpuLock(t)
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		for _, sh := range [][3]int{{8192, 2048, 512}, {2048, 8192, 512}, {2048, 2048, 512}, {512, 2048, 512}} {
			rows, k, ntok := sh[0], sh[1], sh[2]
			rng := rand.New(rand.NewSource(1))
			g := newGPU(t, d)
			qs, dw, scw, _ := mainkernels.PackWeights(mainkernels.Q4_K, rawWeights(mainkernels.Q4_K, rows, k, rng), rows, k)
			bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
			out := g.up(make([]byte, 8*rows*ntok*4))
			bB := g.up(make([]byte, ntok*k*2))
			bAX := g.up(make([]byte, ntok*k))
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
				t.Logf("%5dx%-5d t%d %-34s %8.1f us  %.2f TFLOPS", rows, k, ntok, name, float64(el.Microseconds()),
					2*float64(rows)*float64(k)*float64(ntok)/el.Seconds()/1e12)
			}
			for _, tl := range [][2]int{{4, 4}, {2, 4}, {4, 2}} {
				kk, err := mainkernels.MatVecMMA(mainkernels.MatVecShape{T: mainkernels.Q4_K, K: k, Rows: rows, NTok: ntok, MT: tl[0], NT: tl[1], ActWin: 32})
				if err != nil {
					t.Log(err)
					continue
				}
				if kern, err := d.Compile(kk); err == nil {
					thr := rows / (16 * tl[0]) * (ntok / (8 * tl[1])) * 32
					time1(fmt.Sprintf("int8 mma mt%d nt%d", tl[0], tl[1]), func() error {
						return kern.Launch((thr+127)/128, 128, bQS, bD, bSC, bB, bAX, out)
					})
				}
			}
			for _, tl := range []mainkernels.Int8Tile{
				{MT: 4, NT: 4, WM: 2, WN: 2}, {MT: 4, NT: 2, WM: 2, WN: 2}, {MT: 2, NT: 4, WM: 2, WN: 2},
				{MT: 4, NT: 8, WM: 2, WN: 2}, {MT: 2, NT: 2, WM: 2, WN: 2},
			} {
				s := mainkernels.MatVecShape{T: mainkernels.Q4_K, K: k, Rows: rows, NTok: ntok}
				kk, err := mainkernels.GemmInt8(s, tl)
				if err != nil {
					t.Log(err)
					continue
				}
				kern, err := d.Compile(kk)
				if err != nil {
					t.Logf("%+v: %v", tl, err)
					continue
				}
				time1(fmt.Sprintf("int8 gemm %dx%d w%dx%d", tl.MT, tl.NT, tl.WM, tl.WN), func() error {
					return kern.Launch(mainkernels.GemmInt8Groups(s, tl), tl.Threads(), bQS, bD, bSC, bB, bAX, out)
				})
			}
			for _, tl := range []mainkernels.VoltaTile{
				{MT: 4, NT: 8, WM: 2, WN: 2}, {MT: 4, NT: 4, WM: 2, WN: 2}, {MT: 2, NT: 8, WM: 2, WN: 2},
				{MT: 8, NT: 4, WM: 2, WN: 2}, {MT: 2, NT: 4, WM: 2, WN: 2}, {MT: 4, NT: 4, WM: 1, WN: 4}, {MT: 8, NT: 2, WM: 1, WN: 4},
				// The deep stage: eight warps, 64 (two buffers) or 128 (one) of k a trip.
				{MT: 4, NT: 2, WM: 2, WN: 4, KB: 2}, {MT: 2, NT: 4, WM: 4, WN: 2, KB: 2},
				{MT: 4, NT: 2, WM: 2, WN: 4, KB: 4, Single: true}, {MT: 2, NT: 4, WM: 4, WN: 2, KB: 4, Single: true},
				{MT: 4, NT: 4, WM: 2, WN: 4, KB: 2, Single: true}, {MT: 2, NT: 4, WM: 4, WN: 2, KB: 2, Single: true},
			} {
				for _, f16k := range []int{16, 8} {
					tl.KB, tl.F16K = max(tl.KB, 1), f16k
					s := mainkernels.MatVecShape{T: mainkernels.Q4_K, K: k, Rows: rows, NTok: ntok}
					kk, err := mainkernels.GemmVolta(s, tl)
					if err != nil {
						continue
					}
					kern, err := d.Compile(kk)
					if err != nil {
						t.Logf("%+v: %v", tl, err)
						continue
					}
					time1(fmt.Sprintf("f16 %dx%d w%dx%d kb%d 1b%v k%d", tl.MT, tl.NT, tl.WM, tl.WN, tl.KB, tl.Single, f16k), func() error {
						return kern.Launch(mainkernels.GemmVoltaGroups(s, tl), tl.Threads(), bQS, bD, bSC, bB, out)
					})
				}
			}
			g.free()
		}
	}
}

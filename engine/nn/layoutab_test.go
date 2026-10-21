//go:build jitllmbench

package nn

import (
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestLayoutRate prices the same weight in the two host layouts: GGUF's
// row-contiguous blocks (MatVecHost, the row-major kernel) and the container's
// packed planes (MatVecPacked). JITLLM_LAYOUT_AB=1. Every copy is a distinct
// matrix, so each call streams from DRAM; reported per call and as GB/s of
// each layout's own bytes and of the GGUF bytes (the work).
func TestLayoutRate(t *testing.T) {
	if os.Getenv("JITLLM_LAYOUT_AB") != "1" {
		t.Skip("set JITLLM_LAYOUT_AB=1")
	}
	gt := quant.Q4_K
	q, _ := kernels.QuantOf(gt)
	for _, sh := range []struct{ rows, k int }{{4096, 4096}, {14336, 4096}, {4096, 14336}, {1024, 4096}, {256, 4096}, {64, 4096}} {
		nb := uint64(sh.rows*sh.k) / gt.BlockElems()
		gg := int(nb * gt.BlockBytes())
		copies := max(2, (512<<20)/gg)
		rng := rand.New(rand.NewSource(1))
		srcs := make([][]byte, copies)
		packs := make([]*Packed, copies)
		for c := range srcs {
			src := make([]byte, gg)
			rng.Read(src)
			quant.PlantScales(gt, src, 0)
			qs, d, sc, err := kernels.PackWeights(q, src, sh.rows, sh.k)
			if err != nil {
				t.Fatal(err)
			}
			srcs[c], packs[c] = src, &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
		}
		pb := len(packs[0].QS)*4 + len(packs[0].D)*4 + len(packs[0].SC)*4
		x := make([]float32, sh.k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		out := make([]float32, sh.rows)
		j := NewJIT(sh.k, sh.rows, []quant.Type{gt}, layoutOpts()...)
		t.Logf("pool %d workers", j.pool.N())
		arms := []struct {
			name  string
			bytes int
			run   func(c int) bool
		}{
			{"gguf rows", gg, func(c int) bool { j.NewInput(); return j.MatVecHost(out, gt, srcs[c], x, sh.rows, sh.k) }},
			{"packed   ", pb, func(c int) bool { j.NewInput(); return j.MatVecPacked(out, gt, packs[c], x, sh.rows, sh.k) }},
		}
		// The ceiling: the same GGUF bytes read once, in order, on the same
		// pool. A layout cannot stream faster than this.
		sink := make([]uint64, 64*j.pool.N())
		arms = append(arms, struct {
			name  string
			bytes int
			run   func(c int) bool
		}{"raw read ", gg, func(c int) bool {
			w := unsafe.Slice((*uint64)(unsafe.Pointer(&srcs[c][0])), gg/8)
			j.pool.Do(len(w)/512, 64, func(id, lo, hi int) {
				var a0, a1, a2, a3 uint64
				for i := lo * 512; i < hi*512; i += 4 {
					a0 += w[i]
					a1 += w[i+1]
					a2 += w[i+2]
					a3 += w[i+3]
				}
				sink[id*64] += a0 ^ a1 ^ a2 ^ a3
			})
			return true
		}})
		for i := 0; i < len(arms); i++ {
			if !arms[i].run(0) {
				t.Logf("%s declined on this host", arms[i].name)
				arms = append(arms[:i], arms[i+1:]...)
				i--
			}
		}
		for w := 0; w < 400; w++ { // let the per-shape picker settle, as a decode does
			arms[0].run(w % len(srcs))
		}
		for round := 0; round < 3; round++ {
			for _, a := range arms {
				t0 := time.Now()
				n := 0
				for rep := 0; rep < 4; rep++ {
					for c := range srcs {
						a.run(c)
						n++
					}
				}
				per := time.Since(t0).Seconds() / float64(n)
				t.Logf("%5d x %-5d %s %7.1f us/call  %6.2f GB/s own  %6.2f GB/s of gguf bytes",
					sh.rows, sh.k, a.name, per*1e6, float64(a.bytes)/per/1e9, float64(gg)/per/1e9)
			}
		}
		j.Close()
	}
}

func layoutOpts() []Option {
	var o []Option
	if os.Getenv("JITLLM_LAYOUT_TUNE") != "1" {
		o = append(o, WithTune(TuneOff))
	}
	if n, err := strconv.Atoi(os.Getenv("JITLLM_LAYOUT_SLICES")); err == nil {
		o = append(o, WithFusedKSlices(n))
	}
	return o
}

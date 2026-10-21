//go:build amd64 && jitllmtest

package nn

import (
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestFamily2RunsOnTheForcedSSETier drives every nn entry point of the SSE
// tier's family 2 -- RMSNorm32JIT, LayerNorm32JIT, RoPE32JIT, RowPacked32JIT
// and Row32JIT -- with the tier forced to SSE, against internal/oracle, from
// sixteen goroutines at once. Under -race a fill that mutated the live
// snapshot fails with a concurrent map access, and a cache key without the
// tier would hand this run an AVX2 kernel.
//
// The configuration is checked, not assumed: HostTier must read SSE, no AVX2
// kernel may be mapped while forced, and every SSE-keyed cache entry this test
// creates must have mapped an SSE-tier kernel (counted per new entry, so the
// check does not depend on test order or -count).
func TestFamily2RunsOnTheForcedSSETier(t *testing.T) {
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("forcing the SSE tier left the host at %v", cpu.HostTier())
	}
	const sse = cpu.TierSSE

	type ln struct {
		n    int
		bias bool
	}
	type rope struct {
		hd, nrot int
		neox     bool
	}
	rmsWidths := []int{5, 67, 263, 2048}
	lnShapes := []ln{{7, false}, {67, true}, {768, true}, {771, false}}
	ropeShapes := []rope{{64, 64, false}, {64, 32, true}, {80, 80, true}, {128, 128, false}, {96, 38, false}}

	// The entries this test will create, counted before it creates them.
	fresh := 0
	for _, n := range rmsWidths {
		if _, ok := norms.snap.Load().code[tierN{sse, n}]; !ok {
			fresh++
		}
	}
	for _, s := range lnShapes {
		if _, ok := lns.snap.Load().code[lnKey{sse, s.n, s.bias, cpu.RowLayerNorm}]; !ok {
			fresh++
		}
	}
	for _, r := range ropeShapes {
		if _, ok := ropes.Load(ropeKey{sse, r.hd, r.nrot, r.neox}); !ok {
			fresh++
		}
	}
	for _, gt := range quant.PackedTypes {
		q, _ := kernels.QuantOf(gt)
		if _, ok := rowKernels.Load(rowKey{sse, q}); !ok {
			fresh++
		}
	}
	if !widenOnce.done[sse].Load() {
		fresh += 2
	}
	avx2Before := cpu.MappedByTier()[cpu.TierAVX2]
	sseBefore := cpu.MappedByTier()[sse]

	// The packed tables, built once and read by every goroutine.
	const nrows, k = 9, 512
	type table struct {
		gt quant.Type
		q  kernels.Quant
		p  *Packed
	}
	var tabs []table
	for _, gt := range quant.PackedTypes {
		q, _ := kernels.QuantOf(gt)
		tabs = append(tabs, table{gt, q, packOne(t, gt, nrows, k)})
	}

	var wg sync.WaitGroup
	errs := make(chan string, 64)
	fail := func(f string, a ...any) {
		select {
		case errs <- fmt.Sprintf(f, a...):
		default:
		}
	}
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for rep := 0; rep < 4; rep++ {
				i := g + rep
				eps := 1e-5 + float64(g%2)*1e-6

				n := rmsWidths[i%len(rmsWidths)]
				x, w := vec(n, float32(g)), vec(n, 1+float32(rep))
				y, want := make([]float32, n), make([]float32, n)
				RMSNorm32JIT(y, x, w, eps)
				oracle.RMSNorm32(want, x, w, eps)
				if e := f2NMSE32(y, want); !(e <= 1e-10) {
					fail("rmsnorm n=%d: NMSE %.3e against the oracle", n, e)
				}

				s := lnShapes[i%len(lnShapes)]
				x, w = vec(s.n, 3+float32(g)), vec(s.n, 5)
				for j := range x {
					// A large mean, TestEmitLayerNormMatchesReference's reason
					// and its input: 40 +- 3. At +-1 the f32 mean's own
					// rounding is 3e-10 of NMSE at n=7 on either tier.
					x[j] = 40 + 3*x[j]
				}
				var b []float32
				if s.bias {
					b = vec(s.n, 7)
				}
				y, want = make([]float32, s.n), make([]float32, s.n)
				LayerNorm32JIT(y, x, w, b, eps)
				oracle.LayerNorm32(want, x, w, b, eps)
				if e := f2NMSE32(y, want); !(e <= 1e-10) {
					fail("layernorm n=%d bias=%v: NMSE %.3e against the oracle", s.n, s.bias, e)
				}

				r := ropeShapes[i%len(ropeShapes)]
				heads := 1 + g%3
				xs := vec(r.hd*heads, float32(g))
				cs := vec(r.nrot, 9)
				got := append([]float32(nil), xs...)
				RoPE32JIT(got, r.hd, cs, r.neox)
				for h := 0; h < heads; h++ {
					ref := append([]float32(nil), xs[h*r.hd:(h+1)*r.hd]...)
					oracle.RopeApply(ref[:r.nrot], cs, r.neox)
					for d := 0; d < r.hd; d++ {
						gv, wv := got[h*r.hd+d], ref[d]
						if d >= r.nrot && math.Float32bits(gv) != math.Float32bits(wv) {
							fail("rope %+v head %d dim %d: partial-rotary dimension moved", r, h, d)
						} else if math.Abs(float64(gv-wv)) > 2e-6*(1+math.Abs(float64(wv))) {
							fail("rope %+v head %d dim %d: %v, oracle %v", r, h, d, gv, wv)
						}
					}
				}

				tb := tabs[i%len(tabs)]
				row := (g*7 + rep) % nrows
				dst, ref := make([]float32, k), make([]float32, k)
				RowPacked32JIT(dst, tb.gt, tb.p, row, nrows, k)
				if err := oracle.RowPacked(ref, tb.q, tb.p.QS, tb.p.D, tb.p.SC, row, nrows, k); err != nil {
					fail("%s: oracle: %v", tb.gt, err)
				}
				for j := range ref {
					if math.Abs(float64(dst[j]-ref[j])) > 1e-6*(1+math.Abs(float64(ref[j]))) {
						fail("%s row %d element %d: %v, oracle %v", tb.gt, row, j, dst[j], ref[j])
						break
					}
				}

				for _, ft := range []quant.Type{quant.F16, quant.BF16} {
					kk := 37 + g
					src := make([]byte, 3*kk*2)
					for j := range src {
						src[j] = byte(j*31 + g*7 + 1)
					}
					for j := 1; j < len(src); j += 2 {
						src[j] &= 0x3F // exponents that stay finite in both formats
					}
					got, want := make([]float32, kk), make([]float32, kk)
					Row32JIT(got, ft, src, 2, kk)
					if err := oracle.Row(want, ft, src, 2, kk); err != nil {
						fail("%s: oracle: %v", ft, err)
					}
					for j := range want {
						if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
							fail("%s k=%d element %d: %#08x, oracle %#08x", ft, kk, j,
								math.Float32bits(got[j]), math.Float32bits(want[j]))
							break
						}
					}
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}

	if d := cpu.MappedByTier()[cpu.TierAVX2] - avx2Before; d != 0 {
		t.Errorf("%d AVX2 kernels were mapped while the SSE tier was forced", d)
	}
	// A lost LoadOrStore race may map one kernel twice (rope.go, row.go), so
	// this is a floor, not an equality.
	if d := cpu.MappedByTier()[sse] - sseBefore; d < int64(fresh) {
		t.Errorf("%d SSE-tier kernels were mapped for %d cache entries this test created -- "+
			"an entry was filled from somewhere other than the SSE table", d, fresh)
	}
	t.Logf("%d SSE-keyed kernel entries created by this test, %d SSE-tier maps",
		fresh, cpu.MappedByTier()[sse]-sseBefore)
}

// f2NMSE32 is the normalised squared error of got against want, NaN when the
// reference is degenerate -- which !(e <= bound) then reports as a failure.
func f2NMSE32(got, want []float32) float64 {
	var se, s2 float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		se, s2 = se+d*d, s2+float64(want[i])*float64(want[i])
	}
	if s2 == 0 {
		return math.NaN()
	}
	return se / s2
}

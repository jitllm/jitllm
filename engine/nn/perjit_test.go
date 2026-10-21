//go:build amd64 || arm64

package nn

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestPrefetchIsPerJIT: two JITs in one process, one with the emit-time
// prefetch distances set and one without, built in both orders. Each must
// emit with its own, whichever was built last: the amd64 k-quant kernel
// grows a PREFETCHT0 and the arm64 packed matvec loses its PRFMs. The tuners
// are off: this test must not load the on-disk tune cache for the tests after
// it.
func TestPrefetchIsPerJIT(t *testing.T) {
	if cpu.HostTier() == cpu.TierSSE {
		t.Skip("the SSE tier has neither prefetch form")
	}
	size := func(f *JIT) [2]int {
		row, err := f.em.RowMajor(cpu.Spec{W: quant.Q4_K, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatalf("RowMajor: %v", err)
		}
		packed, err := f.em.PackedMatVec(quant.Q4_K, cpu.PackedRows)
		if err != nil {
			t.Fatalf("PackedMatVec: %v", err)
		}
		return [2]int{len(row), len(packed)}
	}
	for _, pfFirst := range []bool{true, false} {
		mk := func(pf bool) *JIT {
			o := []Option{WithTune(TuneOff)}
			if pf {
				o = append(o, WithPrefetch(4), WithA64Prefetch(-1))
			}
			f := NewJIT(256, 256, []quant.Type{quant.Q4_K}, o...)
			t.Cleanup(func() { f.Close() })
			return f
		}
		var pf, plain *JIT
		if pfFirst {
			pf, plain = mk(true), mk(false)
		} else {
			plain, pf = mk(false), mk(true)
		}
		a, b := size(pf), size(plain)
		if a == b {
			t.Errorf("prefetch first=%v: both JITs emit %v bytes (row-major, packed); "+
				"the prefetching one must differ", pfFirst, a)
		}
		def := NewJIT(256, 256, []quant.Type{quant.Q4_K}, WithTune(TuneOff))
		t.Cleanup(func() { def.Close() })
		if b != size(def) {
			t.Errorf("prefetch first=%v: the plain JIT emits %v, not the default's", pfFirst, b)
		}
	}
}

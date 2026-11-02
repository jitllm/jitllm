//go:build amd64

package cpu

import (
	"math"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"
)

// TestSSEPackedCallTime measures, and asserts nothing about speed: how
// long one SSE fused-kernel call takes at the chunk nn hands a worker
// (cpu.RowsPerCall, which prices a block at the AVX2 tier's 16 cycles at 3 GHz),
// against maxCallNanos -- the GC-preemption budget a call into generated code
// must stay inside, because a goroutine in JIT code cannot be preempted.
//
// It prints the per-call time for each format at k=2048 (Llama-3.2-1B's width)
// with RowsPerCall's rows and with RowsPerCallFor(TierSSE)'s, each on planted
// scales (three of sixteen are f16 subnormals) and on normal ones, so the SSE
// budget constant (sse_budget.go) can be read against the host that runs it.
// The minimum over repeated calls is reported: it is the uncontended cost,
// which is what the constant models (a loaded host is slower for reasons no
// chunk size fixes).
func TestSSEPackedCallTime(t *testing.T) {
	const k = 2048
	for _, g := range quant.PackedTypes {
		fused, err := EmitPackedMatVecFusedSSE(g)
		if err != nil {
			t.Fatal(err)
		}
		c := mustMap(t, fused)
		grp := PackedFusedGroupOf(g)
		bpr := k / int(g.BlockElems())
		for _, arm := range []struct {
			name  string
			rows  int
			exact bool
		}{
			{"RowsPerCall", RowsPerCall(bpr) / grp * grp, false},
			{"RowsPerCall,normal-d", RowsPerCall(bpr) / grp * grp, true},
			{"RowsPerCallFor", RowsPerCallFor(TierSSE, g, k) / grp * grp, false},
			{"RowsPerCallFor,normal-d", RowsPerCallFor(TierSSE, g, k) / grp * grp, true},
		} {
			rows := max(arm.rows, grp)
			p := newSSEPackFix(t, g, rows, k, 9, arm.exact)
			out := make([]float32, rows)
			scratch, q32 := make([]float32, rows), make([]float32, rows)
			a := p.args(out, 0, rows/grp, scratch, q32)
			best := time.Duration(math.MaxInt64)
			for i := 0; i < 30; i++ {
				clear(out)
				t0 := time.Now()
				c.Call(&a)
				if d := time.Since(t0); d < best {
					best = d
				}
			}
			blocks := rows * bpr
			t.Logf("%-6s %-24s %5d rows x %4d blocks: %8.1f us a call (%.1f ns/block, budget %d us)",
				g, arm.name, rows, bpr, float64(best.Nanoseconds())/1e3,
				float64(best.Nanoseconds())/float64(blocks), maxCallNanos/1000)
		}
		c.Close()
	}
}

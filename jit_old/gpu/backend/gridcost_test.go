package backend_test

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// TestGridBarrierCost measures what one grid barrier costs on each device, at
// a few grid sizes: a kernel of 256 barriers against the same kernel with
// none, the median of 50 launches each, the difference over 256. It asserts
// nothing about time (RULE 4); it reports, so a persistent kernel's phase
// count can be priced before it is built. -short skips it.
func TestGridBarrierCost(t *testing.T) {
	if testing.Short() {
		t.Skip("a measurement")
	}
	const nb = 256
	eachGridDevice(t, func(d backend.Device, width, limit int) {
		if width != 128 {
			return
		}
		for _, groups := range dedupe(2, limit/16, limit/8, limit/2, limit) {
			sync, err := d.Alloc(4 * ir.GridSyncWords)
			if err != nil {
				t.Fatal(err)
			}
			if err := sync.Write(make([]byte, 4*ir.GridSyncWords)); err != nil {
				t.Fatal(err)
			}
			sink, err := d.Alloc(4 * groups * width)
			if err != nil {
				t.Fatal(err)
			}
			med := func(n int) time.Duration {
				b := ir.New(fmt.Sprintf("gridcost%d", n), [3]int{width, 1, 1})
				out := b.Param("out", ir.U32)
				s := b.Param("sync", ir.U32)
				i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.Const(ir.U32, int64(width))), b.TID())
				for j := 0; j < n; j++ {
					b.GridBarrier(s)
				}
				b.Store(out, i, i, 0)
				k, err := d.Compile(b.Done())
				if err != nil {
					t.Fatal(err)
				}
				defer k.Close()
				var ts []time.Duration
				for r := 0; r < 51; r++ {
					var err error
					t0 := time.Now()
					d.Session(func(se backend.Session) {
						if err = se.Launch(k, groups, width, sink, sync); err == nil {
							err = se.Sync()
						}
					})
					if err != nil {
						t.Fatal(err)
					}
					if r > 0 {
						ts = append(ts, time.Since(t0))
					}
				}
				sort.Slice(ts, func(a, b int) bool { return ts[a] < ts[b] })
				return ts[len(ts)/2]
			}
			// A kernel with no barrier has no sync parameter to spare: n=0
			// still declares it and never touches it.
			t0, tn := med(0), med(nb)
			t.Logf("%s %s: %4d workgroups of %d: %.2f us a grid barrier (launch %.1f us, %d barriers %.1f us)",
				d.API(), d.Name(), groups, width, float64(tn-t0)/float64(nb)/1e3,
				float64(t0)/1e3, nb, float64(tn)/1e3)
			sync.Free()
			sink.Free()
		}
	})
}

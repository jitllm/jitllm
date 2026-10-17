package jlm

import (
	"sync/atomic"
	"time"
)

// HoldPages is Hold of every block in bs, each over its whole page, with
// every missing chunk of every page in flight at once through the File's
// page readers: one plan for a layer's routed experts rather than a read
// (and a goroutine) per page. out[i] and errs[i] are block bs[i]'s lease and
// error, as Hold returns them; a failed page holds nothing. A warm call
// allocates nothing: the plan is the File's (getPlan).
func (f *File) HoldPages(bs []int, out []Lease, errs []error) {
	for i, b := range bs {
		out[i], errs[i] = f.Hold(b, nil)
	}
	pl := f.getPlan()
	defer f.putPlan(pl)
	f.io.once.Do(f.startIO)
	for i, b := range bs {
		if errs[i] != nil {
			continue
		}
		base, psize := f.pageAt(b)
		pl.rng[0] = Range{Off: base, N: psize}
		r0 := len(pl.runs)
		page, _, runs, err := f.planRuns(b, pl.rng[:], pl.runs[r0:r0])
		pl.runs = append(pl.runs[:r0], runs...)
		f.flushRecycled()
		if err != nil {
			errs[i] = err
			out[i].Release()
			continue
		}
		if len(runs) == 0 {
			continue
		}
		pl.held = append(pl.held, heldRuns{i: i, b: b, r0: r0, r1: len(pl.runs)})
		f.cutRuns(pl, page, base, psize, runs)
	}
	if len(pl.held) == 0 {
		return
	}
	t0 := time.Now()
	es := f.sendRuns(pl)
	atomic.AddInt64(&f.waitNs, int64(time.Since(t0)))
	for _, h := range pl.held {
		err := f.markFilled(h.b, pl.runs[h.r0:h.r1], es[h.r0:h.r1])
		f.unpin(h.b)
		if err != nil {
			errs[h.i] = err
			out[h.i].Release()
		}
	}
}

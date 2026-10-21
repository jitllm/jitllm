package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// Key runs (nn.KeyRunDevice): every mask this tier runs is, per row, an
// interval of keys, handed over once for the calls that follow.
//
// A full run widens a causal row's key count to the run's end -- Gemma 3's
// image tokens see each other in both directions -- which is the per-row key
// count the batched attention already reads, so no kernel changes. A
// non-causal call (nn.LayerPlan.NonCausal) is one whole run by construction:
// every row counts the call's rows. On its Windowed blocks each row's scores
// outside its window run go to -inf before the softmax
// (mainkernels.WindowMaskRows, as llama.cpp adds its window_mask): Qwen2.5-VL's
// tower. The windows are a property of the picture, not of a block, so they
// are handed over once per picture and every windowed block reads them.

// SetKeyRuns records the runs for the Layers calls that follow on every device:
// a run crossing a device seam is honoured on both sides.
func (g *GPU) SetKeyRuns(full, windowed []nn.KeyRun) bool {
	return g.setKeyRuns(0, full, windowed)
}

// SetRowWindows records one window a row for the NonCausal calls that follow
// on every device (nn.RowWindowDevice): row r of a call attends inside
// wins[r] on its Windowed blocks. A symmetric sliding window is this
// (ModernBERT's local layers: the keys within half the window either side),
// and its rows' windows overlap, which SetKeyRuns' partition cannot say.
func (g *GPU) SetRowWindows(wins []nn.KeyRun) bool {
	return g.setRowWindows(0, wins)
}

// bidirCount is row r's key count under the runs, at position pos+r: the end
// of its run, or 0 when it is in none.
func bidirCount(runs []nn.KeyRun, at int) int {
	for _, r := range runs {
		if at >= r.Lo && at < r.Hi {
			return r.Hi
		}
	}
	return 0
}

// bidirCheck refuses a call whose rows [pos, pos+nrow) cut a run, and, on the
// contiguous cache, one whose windowed layers would mask a run's row from the
// wrong anchor: those window kernels place a row's window behind its key
// COUNT, which a run moves to its end, where the reference anchors it at the
// row's own position. The two agree while the count is inside the window,
// which is every image in the first window's worth of context. The paged
// cache's row descriptor carries the window's start apart from the end
// (pagedStage), so it has no such limit.
func bidirCheck(runs []nn.KeyRun, pos, nrow int, p *nn.LayerPlan, paged bool) string {
	for _, r := range runs {
		if r.Lo >= pos+nrow || r.Hi <= pos {
			continue
		}
		if r.Lo < pos || r.Hi > pos+nrow {
			return "a bidirectional run is cut by this call's rows"
		}
		if !paged && p.SWAWindow > 0 && r.Hi > p.SWAWindow {
			return "a bidirectional run ends past the sliding window, where the window kernels anchor it at the run's end"
		}
	}
	return ""
}

// winOf is the windows of query chunk c for a windowed block l, the packed
// sequences' runs for an unwindowed non-causal block when sequences are
// packed (SetSeqRuns: a windowed block's windows are inside them already),
// and nil for every other.
func (bs *blockScratch) winOf(l *layer, c int) backend.Buf {
	switch {
	case l.windowed:
		return bs.vwins[c]
	case l.nonCausal && bs.vsegRows > 0:
		return bs.vsegs[c]
	}
	return nil
}

// SetSeqRuns records the packed sequences for the non-causal calls that
// follow on every device (nn.SeqRunsDevice).
func (g *GPU) SetSeqRuns(runs []nn.KeyRun) bool {
	return g.setSeqRuns(0, runs)
}

// SetSeqRuns forwards under the session's call locks.
func (s *gpuSession) SetSeqRuns(runs []nn.KeyRun) bool {
	defer s.leave(s.step())
	return s.g.setSeqRuns(s.sid, runs)
}

// setKeyRuns is SetKeyRuns for session sid. The full runs are the session's:
// its calls' rows are its own sequence's. The windows are staged in the
// tower's set, which is one per device (visrows.go) and runs one picture at a
// time.
func (g *GPU) setKeyRuns(sid uint64, full, windowed []nn.KeyRun) bool {
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	for _, d := range ds {
		d.mu.Lock()
		ss := d.sessOf(sid)
		ss.bidir = append(ss.bidir[:0], full...)
		ss.winRuns = append(ss.winRuns[:0], windowed...)
		ss.winPerRow = false
		d.mu.Unlock()
		// Staged now, so a picture whose windows the device cannot hold is
		// refused here; staged again at each of the session's tower calls
		// (layersCall), since another session's picture may be staged in
		// between.
		if !d.setWindowsFor(sid) {
			return false
		}
	}
	return true
}

// setRowWindows is SetRowWindows for session sid.
func (g *GPU) setRowWindows(sid uint64, wins []nn.KeyRun) bool {
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	for _, d := range ds {
		d.mu.Lock()
		ss := d.sessOf(sid)
		ss.bidir = ss.bidir[:0]
		ss.winRuns = append(ss.winRuns[:0], wins...)
		ss.winPerRow = true
		d.mu.Unlock()
		if !d.setWindowsFor(sid) {
			return false
		}
	}
	return true
}

// setWindowsFor stages session sid's windows (devSess.winRuns) in lane0's
// tower set, waiting for lane0 if another call holds it; see
// setWindowsLocked. Callers must not hold g.mu.
func (g *devTier) setWindowsFor(sid uint64) bool { return g.stageFor(sid, true, false) }

// stageFor stages session sid's windows, its packed sequences, or both, in
// lane0's set: each setter stages its own list, the other being the last
// call's and maybe for other rows; a call stages both (layersCall).
func (g *devTier) stageFor(sid uint64, wins, seqs bool) bool {
	v := g.bareView(sid)
	g.mu.Lock()
	defer g.mu.Unlock()
	// A device with no tower has no set to stage windows in.
	if !g.holdsTower() {
		return true
	}
	v0 := v.borrowLane0()
	defer v.returnLane0(v0)
	return (!wins || v0.setWindowsLocked(v0.winRuns, v0.winPerRow)) && (!seqs || v0.setSeqsLocked(v0.seqRuns))
}

// setWindowsLocked stages runs -- [lo, hi) row pairs partitioning a
// non-causal call's rows, or with perRow one window a row -- as each row's
// window, in the non-causal geometry's scratch set, building the mask and its
// buffers on the first call. A device with no non-causal block takes nothing; nil forgets the
// windows, so a windowed block is refused rather than run under the last
// picture's. Callers hold g.mu, in a view over the lane holding the tower's
// set (lane0).
func (g *devTier) setWindowsLocked(runs []nn.KeyRun, perRow bool) bool {
	home := g.geoCur
	defer g.useGeom(home)
	g.useGeom(geoNonCausal)
	bs := g.bs
	if bs == nil {
		return true
	}
	if len(runs) == 0 {
		bs.vwinRows = 0
		return true
	}
	fail := func(f string, a ...any) bool {
		g.LastErr = fmt.Sprintf(f, a...)
		return false
	}
	// n is the call's rows: the partition's end, or one window a row.
	n := len(runs)
	segs := g.winSegs[:0]
	if !perRow {
		for _, r := range runs {
			segs = append(segs, r.Lo, r.Hi)
		}
		g.winSegs = segs
		if len(segs)%2 != 0 || len(segs) == 0 {
			return fail("windowed runs: %d bounds, want [lo, hi) pairs", len(segs))
		}
		n = segs[len(segs)-1]
	}
	if n > bs.rows {
		return fail("windowed runs over %d rows on a scratch of %d", n, bs.rows)
	}
	// The mask runs once per attention pass, which is a query chunk of
	// bs.arows rows when the block's score planes are too large for one
	// (visionattn.go), so it is built for a chunk and each chunk has its rows'
	// windows in a buffer of its own.
	ar := bs.arows
	if ar <= 0 {
		ar = bs.rows
	}
	g.LastErr = "the window mask has no composed form"
	return false
}

// setSeqRuns is SetSeqRuns for session sid: kept with the session's windows
// and staged with them (setWindowsFor, and again at each of its calls).
func (g *GPU) setSeqRuns(sid uint64, runs []nn.KeyRun) bool {
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	for _, d := range ds {
		d.mu.Lock()
		ss := d.sessOf(sid)
		ss.seqRuns = append(ss.seqRuns[:0], runs...)
		d.mu.Unlock()
		if !d.stageFor(sid, false, true) {
			return false
		}
	}
	return true
}

// setSeqsLocked stages runs -- [lo, hi) pairs partitioning a non-causal
// call's rows, one a packed sequence -- as each row's window for the
// unwindowed blocks (vsegs). One run or none is one sequence: no mask.
// Callers hold g.mu, in a view over the lane holding the set.
func (g *devTier) setSeqsLocked(runs []nn.KeyRun) bool {
	home := g.geoCur
	defer g.useGeom(home)
	g.useGeom(geoNonCausal)
	bs := g.bs
	if bs == nil {
		return true
	}
	if len(runs) <= 1 {
		bs.vsegRows = 0
		return true
	}
	fail := func(f string, a ...any) bool {
		g.LastErr = fmt.Sprintf(f, a...)
		return false
	}
	n := runs[len(runs)-1].Hi
	if n > bs.rows {
		return fail("packed sequences over %d rows on a scratch of %d", n, bs.rows)
	}
	g.LastErr = "the window mask has no composed form"
	return false
}

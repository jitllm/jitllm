package gpu

import (
	"github.com/jitllm/jitllm/engine/nn"
)

// Key runs (nn.KeyRunDevice): every mask a block runs is, per row, an
// interval of keys, handed over once for the calls that follow and kept with
// the session.
//
// A full run widens a causal row's key count to the run's end -- Gemma 3's
// image tokens see each other in both directions. A non-causal call
// (nn.LayerPlan.NonCausal) is one whole run by construction: every row counts
// the call's rows. On its Windowed blocks each row attends inside its window
// run: Qwen2.5-VL's tower. The windows are a property of the picture, not of
// a block, so they are handed over once per picture and every windowed block
// reads them.

// SetKeyRuns records the runs for the Layers calls that follow on every device:
// a run crossing a device seam is honoured on both sides.
func (g *GPU) SetKeyRuns(full, windowed []nn.KeyRun) bool {
	return g.setKeyRuns(0, full, windowed)
}

// setKeyRuns is SetKeyRuns for session sid. The full runs are the session's:
// its calls' rows are its own sequence's, and the composed tower reads its
// windows from them (picWinDesc).
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
	}
	return true
}

// SetRowWindows records one window a row for the NonCausal calls that follow
// on every device (nn.RowWindowDevice): row r of a call attends inside
// wins[r] on its Windowed blocks. A symmetric sliding window is this
// (ModernBERT's local layers: the keys within half the window either side),
// and its rows' windows overlap, which SetKeyRuns' partition cannot say.
func (g *GPU) SetRowWindows(wins []nn.KeyRun) bool {
	return g.setRowWindows(0, wins)
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
	}
	return true
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

// setSeqRuns is SetSeqRuns for session sid, kept with the session's windows.
func (g *GPU) setSeqRuns(sid uint64, runs []nn.KeyRun) bool {
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	for _, d := range ds {
		d.mu.Lock()
		ss := d.sessOf(sid)
		ss.seqRuns = append(ss.seqRuns[:0], runs...)
		d.mu.Unlock()
	}
	return true
}

package gpu

import (
	"github.com/jitllm/jitllm/nn"
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
func (g *GPU) SetKeyRuns(full, windowed []nn.KeyRun) {
	g.setKeyRuns(0, full, windowed)
}

// setKeyRuns is SetKeyRuns for session sid. The full runs are the session's:
// its calls' rows are its own sequence's, and the composed tower reads its
// windows from them (picWinDesc).
func (g *GPU) setKeyRuns(sid uint64, full, windowed []nn.KeyRun) {
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
}

// SetRowWindows records one window a row for the calls that follow on every
// device (nn.RowWindowDevice): row r of a call attends inside wins[r] on the
// blocks whose graph reads the windows, and inside its full run (SetKeyRuns)
// on every other; the rows' windows may overlap, which SetKeyRuns' partition
// cannot say.
func (g *GPU) SetRowWindows(wins []nn.KeyRun) {
	g.setRowWindows(0, wins)
}

// setRowWindows is SetRowWindows for session sid.
func (g *GPU) setRowWindows(sid uint64, wins []nn.KeyRun) {
	g.mu.Lock()
	ds := g.devs
	g.mu.Unlock()
	for _, d := range ds {
		d.mu.Lock()
		ss := d.sessOf(sid)
		ss.winRuns = append(ss.winRuns[:0], wins...)
		ss.winPerRow = true
		d.mu.Unlock()
	}
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

// runEnd is the key count of row at, in a call whose rows end at end, under
// the session's runs: its own window's end where the call's blocks read the
// windows (wins, rangeRule) and the session has one a row (SetRowWindows),
// else its run's end, 0 where it reads to itself -- in no run, or in a run's
// first phase, which reads only keys written (GPU.phased). The "runcut"
// fault ends a second phase's runs at the call's rows: the run cut at the
// submission.
func (g *devTier) runEnd(at, end int, wins bool) int {
	if g.runPhase == 1 {
		return 0
	}
	e := bidirCount(g.bidir, at)
	if w, ok := g.rowWindow(at); ok && wins {
		e = w.Hi
	}
	if g.runPhase == 2 && e > end && g.rowsFaulted("runcut") {
		e = end
	}
	return e
}

// rowWindow is the window the session handed row at (SetRowWindows), false
// where it handed none for it. A causal block's row reads inside it in place
// of a run, as the CPU device's does (cpudev.Lanes.Rows).
func (g *devTier) rowWindow(at int) (nn.KeyRun, bool) {
	if !g.winPerRow || at < 0 || at >= len(g.winRuns) {
		return nn.KeyRun{}, false
	}
	return g.winRuns[at], true
}

// windowCheck refuses a call of a causal block whose rows [pos, pos+nrow)
// read keys of a window past them, which no call has written yet: such rows
// go in phases, as a cut run's do (GPU.phased).
func windowCheck(wins []nn.KeyRun, perRow bool, pos, nrow int) string {
	if !perRow {
		return ""
	}
	for at := pos; at < min(pos+nrow, len(wins)); at++ {
		if wins[at].Hi > pos+nrow {
			return "a row's window reads past this call's rows"
		}
	}
	return ""
}

// bidirCheck refuses a call whose rows [pos, pos+nrow) cut a run: the engine
// never cuts one (a prefill chunk ends at a run's end), so a cut is a caller's
// contract broken. A windowed layer's row descriptor carries the window's
// start apart from the run's end (pagedStage), so a run past the window is
// masked from the reference's anchor.
func bidirCheck(runs []nn.KeyRun, pos, nrow int) string {
	for _, r := range runs {
		if r.Lo >= pos+nrow || r.Hi <= pos {
			continue
		}
		if r.Lo < pos || r.Hi > pos+nrow {
			return "a bidirectional run is cut by this call's rows"
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

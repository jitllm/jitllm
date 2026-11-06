package engine

import (
	"time"

	"github.com/jitllm/jitllm/model"
)

// PhaseRow is one phase's share of the pager's work.
//
// jlm.File counts bytes from Open, so a whole-run total mixes one-time
// admission with per-token reads; bytes per decode token need the split.
// Phases are snapshot deltas of the same counters taken where the caller says
// a phase ends, so they always sum to the totals.
type PhaseRow struct {
	Name   string
	Bytes  uint64 // container bytes read during the phase
	Reads  int64  // readAt calls issued
	Faults int64  // page-ins
	Dur    time.Duration
	// Read is how much of Dur the pager spent reading, when the caller can say.
	// Zero means not measured; Rate then falls back to the wall.
	Read time.Duration
}

// PerToken is Bytes divided by n, or zero when n is not positive.
func (r PhaseRow) PerToken(n int) uint64 {
	if n <= 0 {
		return 0
	}
	return r.Bytes / uint64(n)
}

// Rate is the phase's container read rate in bytes a second, over the read time
// when the caller supplied one and over the phase's wall otherwise. A rate over
// the wall includes compute and is only a floor, not a bandwidth; OverRead says
// which one this is.
func (r PhaseRow) Rate() float64 {
	d := r.Dur
	if r.Read > 0 {
		d = r.Read
	}
	if d <= 0 {
		return 0
	}
	return float64(r.Bytes) / d.Seconds()
}

// OverRead reports whether Rate is a bandwidth (true) or a floor over the wall.
func (r PhaseRow) OverRead() bool { return r.Read > 0 }

// ReadTime supplies the phase's read time to the next Mark, from a cumulative
// counter the caller owns such as gpu.Stats.TStreamRead.
func (p *PhaseMeter) ReadTime(total time.Duration) {
	p.rd = total
	if !p.rdSet {
		p.rd0, p.rdSet = total, true
	}
}

// PhaseMeter splits the pager's counters across the phases of one run.
type PhaseMeter struct {
	m     *model.Model
	t     time.Time
	b     uint64
	r     int64
	f     int64
	rd    time.Duration // the caller's cumulative read time, latest value
	rd0   time.Duration // its value when this phase began
	rdSet bool
	rows  []PhaseRow
}

// NewPhaseMeter starts the accounting of m's pages, read by every device
// that pages it, from now; whatever the caller has already done lands in the
// first Mark.
func NewPhaseMeter(m *model.Model) *PhaseMeter {
	p := &PhaseMeter{m: m}
	p.reset()
	return p
}

func (p *PhaseMeter) reset() {
	p.t = time.Now()
	p.rd0 = p.rd
	if p.m != nil {
		p.b, p.r, p.f = p.m.PagedBytes(), p.m.PagerReads(), p.m.PageFaults()
	}
}

// Mark closes the phase that has been running since the last Mark and returns
// it. A meter with a nil model returns zero rows rather than refusing, so a
// caller need not branch on whether a container exists.
func (p *PhaseMeter) Mark(name string) PhaseRow {
	row := PhaseRow{Name: name, Dur: time.Since(p.t)}
	if p.m != nil {
		b, r, f := p.m.PagedBytes(), p.m.PagerReads(), p.m.PageFaults()
		// Saturating: a counter reset under us must not print as an enormous
		// unsigned number.
		if b > p.b {
			row.Bytes = b - p.b
		}
		if r > p.r {
			row.Reads = r - p.r
		}
		if f > p.f {
			row.Faults = f - p.f
		}
	}
	if p.rdSet && p.rd > p.rd0 {
		row.Read = p.rd - p.rd0
	}
	p.rows = append(p.rows, row)
	p.reset()
	return row
}

// Rows is every phase closed so far, in order.
func (p *PhaseMeter) Rows() []PhaseRow { return p.rows }

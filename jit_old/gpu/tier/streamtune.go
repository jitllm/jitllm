package tier

import (
	"sort"
	"time"
)

// The streamed fill's knobs are measured, not tabulated: how many groups a
// selection is split into (more reads in flight behind the transfers, more
// pipeline restarts), the size of each page-locked half (the first half of
// every group is copied before any transfer starts), and the plane size from
// which a sheet is sent from its frame rather than gathered. On Kimi-K3 over
// the V100s 4 groups beat 1 and 16 lost to both, and a 12 MiB half beat 32
// (placement.md 16c); where each knee sits is the disk's, the link's and the
// sheet size's.
//
// Each knob is a ladderTuner, run one after another so a run measures one
// knob: engine/nn/tune.go's pack tuner, paired duels of the incumbent against
// the next rung, interleaved in ABBA quads so drift cancels, climbing while
// the challenger wins by more than the margin and settling otherwise. A run is
// one token's fills on this device, so both arms see the same blocks.
type ladderTuner struct {
	vals         []int
	best, chal   int // rungs; chal -1 once settled
	cur          int
	per, n, turn int
	t            time.Duration
	quad         [4]time.Duration
	rat          []float64
}

const (
	ladderRound  = 6    // paired ratios per duel
	ladderMargin = 0.98 // a challenger must be >2% faster
	ladderMaxIQR = 0.10
)

// newLadder starts at the highest rung not above start.
func newLadder(vals []int, start int) *ladderTuner {
	b := 0
	for i, v := range vals {
		if v <= start {
			b = i
		}
	}
	t := &ladderTuner{vals: vals, best: b, chal: b + 1, cur: b}
	if t.chal >= len(vals) {
		t.chal = -1
	}
	return t
}

func (t *ladderTuner) value() int     { return t.vals[t.cur] }
func (t *ladderTuner) settled() bool  { return t.chal < 0 }
func (t *ladderTuner) bestValue() int { return t.vals[t.best] }

// done charges one fill's wall and moves the tuner on at the end of a run.
func (t *ladderTuner) done(d time.Duration) {
	if t.settled() {
		return
	}
	t.t += d
	if t.n++; t.n < t.per {
		return
	}
	q := t.turn % 4
	t.quad[q], t.n, t.t = t.t, 0, 0
	t.turn++
	if q == 3 {
		// Incumbent at 0 and 3, challenger at 1 and 2: time ratios, so below
		// one is the challenger faster.
		t.rat = append(t.rat, float64(t.quad[1])/float64(t.quad[0]), float64(t.quad[2])/float64(t.quad[3]))
		if len(t.rat) >= ladderRound {
			med, iqr := medianIQRf(t.rat)
			t.rat = t.rat[:0]
			switch {
			case iqr/med > ladderMaxIQR || med >= ladderMargin:
				t.chal = -1
			case t.chal+1 >= len(t.vals):
				t.best, t.chal = t.chal, -1
			default:
				t.best, t.chal = t.chal, t.chal+1
			}
		}
	}
	if t.settled() {
		t.cur = t.best
	} else if p := t.turn % 4; p == 1 || p == 2 {
		t.cur = t.chal
	} else {
		t.cur = t.best
	}
}

// fillTune is the device's three ladders, tuned in turn.
type fillTune struct {
	groups, half, direct *ladderTuner
}

// tune returns the fill tuner, made on first use; per is the device's
// streamed blocks, one run's worth of fills.
func (g *devTier) tune(per int) *fillTune {
	if g.ftune == nil {
		g.ftune = &fillTune{
			groups: newLadder([]int{1, 2, 4, 8, 16}, max(g.StreamGroupsStart, 1)),
			half:   newLadder([]int{4 << 20, 8 << 20, 12 << 20, 16 << 20, 32 << 20}, pinHalf),
			direct: newLadder([]int{256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20}, directSheet),
		}
	}
	f := g.ftune
	for _, l := range []*ladderTuner{f.groups, f.half, f.direct} {
		l.per = max(per, 1)
	}
	return f
}

// streamGroups is the group count for this fill: Config.StreamGroups when
// set, else the tuner's.
func (g *devTier) streamGroups(per int) int {
	if g.StreamGroups != 0 {
		return g.StreamGroups
	}
	return g.tune(per).groups.value()
}

// pinHalfBytes is the page-locked half's size: Config.StreamPinHalf when set,
// else the tuner's.
func (g *devTier) pinHalfBytes() int {
	if g.StreamPinHalf > 0 {
		return g.StreamPinHalf
	}
	if f := g.ftune; f != nil {
		return f.half.value()
	}
	return pinHalf
}

// directBytes is the direct-send threshold: Config.StreamDirectBytes when set,
// else the tuner's.
func (g *devTier) directBytes() int {
	if g.StreamDirectBytes > 0 {
		return g.StreamDirectBytes
	}
	if f := g.ftune; f != nil {
		return f.direct.value()
	}
	return directSheet
}

// fillDone charges one fill's wall to whichever ladder is tuning: the group
// count, then the half, then the threshold, each unless stated.
func (g *devTier) fillDone(d time.Duration) {
	f := g.ftune
	if f == nil {
		return
	}
	switch {
	case g.StreamGroups == 0 && !f.groups.settled():
		f.groups.done(d)
	case g.StreamPinHalf == 0 && !f.half.settled():
		f.half.done(d)
	case g.StreamDirectBytes == 0 && !f.direct.settled():
		f.direct.done(d)
	}
	if g.StreamGroups == 0 {
		g.StreamGroupsTuned = f.groups.bestValue()
	}
	g.StreamPinHalfTuned = f.half.bestValue()
	g.StreamDirectTuned = f.direct.bestValue()
}

func medianIQRf(v []float64) (med, iqr float64) {
	c := append([]float64(nil), v...)
	sort.Float64s(c)
	at := func(f float64) float64 { return c[int(f*float64(len(c)-1))] }
	return at(0.5), at(0.75) - at(0.25)
}

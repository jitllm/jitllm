package model

import (
	"fmt"
	"github.com/jitllm/jitllm/engine/nn"
	"os"
	"sort"
	"time"
)

// The seam tuner: how many blocks belong on the device, decided by measurement.
//
// Unlike engine/nn/tune.go it cannot alternate per token: switching a placement costs
// a migration far larger than a token, so arms are runs of tokens with the
// migration between them, outside the timed window.
//
// It descends from the load-time placement ("as many as the card would take"),
// asking only whether fewer blocks would be faster, e.g. on a contended or slow
// device. It is off by default; see docs/engineering-history/placement.md.
type seamTuner struct {
	on      bool
	settled bool

	cands []int // block counts to try, best first
	best  int
	chal  int

	warmup  int // tokens to skip before starting
	perRun  int // tokens in one timed run
	rounds  int // ABBA quads before deciding
	tok     int
	turn    int
	quad    [4]float64
	ratios  []float64
	runN    int
	runFrom time.Time
	verbose bool
	why     string // what settled it
	trial   bool   // the stream trial (initStreamTrial), not the seam tuner
	// arms and mode are the trial's candidates and the expert choice now in
	// force ("" for the device's defaults).
	arms []trialArm
	mode string
	at   int // the arm in force
	// warming and warmLeft skip the tokens just after a migration.
	warming  bool
	warmLeft int
	counting bool // a timed run is under way
}

// SeamTuneMargin is how much faster a challenger must be to be adopted. It is
// larger than engine/nn/tune.go's because acting on the answer costs a migration, and
// a placement must not flap between two nearly-equal answers.
const SeamTuneMargin = 1.05

func (s *State) initSeamTuner() {
	if s.ld == nil || s.gpuLayers <= s.lo {
		return
	}
	full := s.gpuLayers
	cands := []int{full}
	for _, d := range []int{2, 4} {
		if n := full / d; n > 0 && n != cands[len(cands)-1] {
			cands = append(cands, n)
		}
	}
	cands = append(cands, 0)
	s.seam = &seamTuner{
		on:      true,
		cands:   cands,
		best:    full,
		chal:    cands[1],
		warmup:  s.m.opt.seamWarmup,
		perRun:  s.m.opt.seamRun,
		rounds:  s.m.opt.seamRounds,
		verbose: s.m.opt.seamVerbose,
	}
}

// initStreamTrial measures a placement that streamed blocks no card could
// hold (the incumbent) against the host, and then against the same blocks
// with their experts on the other side of the bus: the seam
// tuner's ABBA runs with the migration outside the clock, and the host is
// adopted only past SeamTuneMargin, so two near-equal answers cannot flap.
// On Kimi-K3 over eight V100s the two decoded within a few percent of each
// other while the streamed one prefilled 1.5x faster (placement.md 16c), so
// which wins is a property of the box, the disk and the link, not a rule.
func (s *State) initStreamTrial() {
	// Two of its arms run on the CPU, a destination only a caller that
	// listed it lets the engine try.
	if s.ld == nil || s.gpuLayers <= s.lo || !s.cpuListed() {
		return
	}
	full := s.gpuLayers
	// The arms: the placement made (its experts where the device put them),
	// the host, and the streamed blocks with their experts on the other side
	// -- sent to the card if they ran on the host, and the other way round.
	// Indices into arms; the incumbent is 0.
	other := "card"
	if s.m.opt.experts == "card" {
		other = "host"
	}
	arms := []trialArm{{blocks: full, experts: s.m.opt.experts}, {blocks: 0, experts: s.m.opt.experts}, {blocks: full, experts: other}}
	if s.m.opt.trialAA {
		// The self-control: the incumbent against itself, migrations and
		// all, so the harness's own bias and spread are measured before any
		// ratio between placements is believed (RULE 2).
		arms = []trialArm{arms[0], {blocks: 0, experts: s.m.opt.experts}, arms[0]}
		arms[1] = arms[0]
	}
	s.seam = &seamTuner{
		on:      true,
		trial:   true,
		arms:    arms,
		mode:    s.m.opt.experts,
		cands:   []int{0, 1, 2},
		best:    0,
		chal:    1,
		warmup:  s.m.opt.seamWarmup,
		perRun:  s.m.opt.seamRun,
		rounds:  s.m.opt.seamRounds,
		verbose: s.m.opt.seamVerbose,
	}
}

// SetSeamTuning turns automatic relocation on or off.
//
// Deciding takes about warmup + rounds*4*perRun tokens and four migrations.
// Off is the default and leaves the load-time placement as it is.
func (s *State) SetSeamTuning(on bool) {
	if !on {
		// The stream trial is not the seam tuner and has its own switch
		// (WithStreamTrial): turning seam tuning off leaves it running.
		if s.seam != nil && !s.seam.trial {
			s.seam = nil
		}
		return
	}
	// The tuner's seams leave blocks on the CPU: with the CPU unlisted there
	// is no seam to tune.
	if s.seam == nil && s.cpuListed() {
		s.initSeamTuner()
	}
}

// SetSeamSchedule sets this session's seam tuner's ABBA shape, as
// WithSeamTuneSchedule does for every session of the model; a non-positive
// value keeps what the tuner has. It shapes the tuner SetSeamTuning armed and
// does nothing to the stream trial, whose schedule is the load's.
func (s *State) SetSeamSchedule(warmup, perRun, rounds int) {
	t := s.seam
	if t == nil || t.trial {
		return
	}
	if warmup > 0 {
		t.warmup = warmup
	}
	if perRun > 0 {
		t.perRun = perRun
	}
	if rounds > 0 {
		t.rounds = rounds
	}
}

// SeamTuned reports the placement the tuner settled on, and whether it settled.
func (s *State) SeamTuned() (blocks int, settled bool) {
	if s.seam == nil {
		return s.gpuLayers, false
	}
	return s.seam.best, s.seam.settled
}

// seamStep runs one token's worth of the tuner. Called once per token, after
// the token is complete.
func (s *State) seamStep() {
	t := s.seam
	if t == nil || !t.on || t.settled {
		return
	}
	t.tok++
	if t.tok <= t.warmup {
		return
	}
	// The clock starts after the migration: counting it would bias every arm
	// switched to, on the same side of every switch, which ABBA cannot cancel.
	if !t.counting && !t.warming {
		if !t.place(s, t.armWant()) {
			// The card would not give us this arm; drop it and settle.
			t.finish(s, "the device refused the candidate placement")
			return
		}
		// A migrated placement starts cold -- pages to re-read, an expert
		// cache to refill -- so its first tokens are not its rate: on
		// Kimi-K3 one arm read 0.38 tok/s warm and 0.11 just after moving.
		t.warming, t.warmLeft = true, max(1, t.perRun/3)
	}
	if t.warming {
		if t.warmLeft--; t.warmLeft > 0 {
			return
		}
		// This token is the warm-up's last, so the clock starts after it.
		t.warming, t.counting = false, true
		t.runFrom, t.runN = time.Now(), 0
		return
	}
	t.runN++
	if t.runN < t.perRun {
		return
	}
	rate := float64(t.runN) / time.Since(t.runFrom).Seconds()
	t.runN, t.counting = 0, false
	t.observe(s, rate)
}

// armWant is the placement this quad position measures: ABBA, so the incumbent
// is not always the earlier and cheaper arm.
func (t *seamTuner) armWant() int {
	if q := t.turn % 4; q == 0 || q == 3 {
		return t.best
	}
	return t.chal
}

func (t *seamTuner) observe(s *State, rate float64) {
	q := t.turn % 4
	t.quad[q] = rate
	if t.verbose {
		fmt.Fprintf(os.Stderr, "seam: %s -> %.2f tok/s\n", t.name(t.armWant()), rate)
	}
	t.turn++
	if q != 3 {
		return
	}
	if t.quad[0] > 0 && t.quad[1] > 0 && t.quad[2] > 0 && t.quad[3] > 0 {
		// Two ratios per quad, incumbent first in one and second in the other,
		// so drift that is linear across the four runs cancels.
		t.ratios = append(t.ratios, t.quad[1]/t.quad[0], t.quad[2]/t.quad[3])
	}
	if len(t.ratios) < t.rounds*2 {
		return
	}
	med, iqr := medianIQR(t.ratios)
	t.ratios = t.ratios[:0]
	// A dispersed measurement keeps what it has rather than guessing.
	if iqr/med > 0.10 {
		t.finish(s, fmt.Sprintf("IQR/median %.3f is too wide to decide", iqr/med))
		return
	}
	if med <= SeamTuneMargin {
		why := fmt.Sprintf("%s is not %.0f%% better (%.3f)", t.name(t.chal), (SeamTuneMargin-1)*100, med)
		// The trial's candidates are not a descent: a losing one leaves the
		// next to be tried against the same incumbent.
		if t.trial {
			if n := t.next(t.chal); n >= 0 {
				t.chal = n
				return
			}
		}
		t.finish(s, why)
		return
	}
	t.best = t.chal
	if n := t.next(t.chal); n >= 0 {
		t.chal = n
		return
	}
	t.finish(s, "no candidate left below the winner")
}

// next is the candidate after c, skipping the incumbent, or -1.
func (t *seamTuner) next(c int) int {
	for i, v := range t.cands {
		if v != c {
			continue
		}
		for _, w := range t.cands[i+1:] {
			if w != t.best {
				return w
			}
		}
	}
	return -1
}

// name says what candidate c is: a block count for the seam tuner, an arm
// for the trial.
func (t *seamTuner) name(c int) string {
	if t.trial && c >= 0 && c < len(t.arms) {
		a := t.arms[c]
		if a.experts == "" {
			return fmt.Sprintf("%d blocks", a.blocks)
		}
		return fmt.Sprintf("%d blocks, experts on the %s", a.blocks, a.experts)
	}
	return fmt.Sprintf("%d blocks", c)
}

// trialArm is one of the stream trial's placements: how many blocks on the
// device, and where a streamed block's experts run ("host", "card").
type trialArm struct {
	blocks  int
	experts string
}

// place puts candidate c in force and reports whether the device gave it:
// the seam tuner's block count, or the trial's arm -- whose expert choice
// takes a fresh placement, since a block takes it when it is placed.
func (t *seamTuner) place(s *State, c int) bool {
	if !t.trial {
		return s.SetGPULayers(c) == c
	}
	a := t.arms[c]
	// Every change of arm is a migration, even between two arms that are the
	// same placement (the A/A self-control), so both sides of every ratio pay
	// the same move.
	moved := c != t.at
	t.at = c
	if a.experts != t.mode || moved && a.blocks > 0 && s.GPULayers() == a.blocks {
		if a.experts != t.mode {
			em, ok := s.ld.(nn.ExpertModer)
			if !ok {
				return false
			}
			em.SetExpertMode(a.experts)
			t.mode = a.experts
		}
		s.SetGPULayers(0)
	}
	return s.SetGPULayers(a.blocks) == a.blocks
}

func (t *seamTuner) finish(s *State, why string) {
	t.settled = true
	t.why = why
	if t.verbose {
		fmt.Fprintf(os.Stderr, "seam: settled on %s (%s)\n", t.name(t.best), why)
	}
	// nil only in the tuner's own gate, which drives observe without a model.
	if s != nil {
		t.place(s, t.best)
	}
}

func medianIQR(v []float64) (med, iqr float64) {
	c := append([]float64(nil), v...)
	sort.Float64s(c)
	at := func(f float64) float64 {
		i := int(f * float64(len(c)-1))
		return c[i]
	}
	return at(0.5), at(0.75) - at(0.25)
}

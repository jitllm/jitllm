package server

import (
	"time"

	"github.com/jitllm/jitllm/engine"
)

// Decode groups: a model whose blocks are on several cards runs a joint step
// one card after another (gpu.GPU.layersRows), so with one step in flight
// only one card works. The loop then splits the decoding rows into groups,
// one per card holding blocks (Config.DecodeGroups), and steps each group on
// its own long-lived worker: as group 0 moves on from card 1 to card 2,
// group 1 takes card 1, and the groups fill the pipeline the cards make.
// Each group is led by its own row's session, so it has its own scratch on
// every card (tier.devTier.as) and its own per-call state in the tier
// (tier.rowsCall); a card runs the groups that reach it one at a time, or in a
// lane of its own where one fits.
//
// What a group costs: each card reads its blocks' shared weights once per
// group per cycle where one group reads them once, and does so while every
// other card works too, so a dense short-context model at low load runs at
// about today's rate; what each row reads on its own -- the experts a
// mixture's row routes to, a long history -- is read on a card while the
// other cards read theirs, and the cycle shortens by up to the card count.
//
// Each group's step is the joint arm (model.StepJoint) whatever its rows, a
// group of one included, so a row's arithmetic is the same in one group or
// many wherever a step's kernels do not depend on its row count -- where the
// composer serves every block and the head in its reserved width
// (gpu.GPU.RowsComposed). The tier's own kernels are chosen per row count,
// so there the loop does not group by itself: a row's tokens would move with
// the rows that share its step. A count the caller forces groups anyway.
//
// Where it may, the loop still groups only where groups measure faster per
// row token than one group (groupChoice, the joint choice's ABBA probe with
// one group the incumbent): one group runs the measured arm of its own,
// often the one-at-a-time path at a few rows, while a group's step is always
// the joint arm, which on a mixture costs several Forwards at one row
// (Qwen3-30B-A3B on two V100s: 90 ms a token in two groups of one row, where
// one group read 24).
//
// Only decoding is grouped. Admission, the room check and a prompt's chunks
// stay the loop's, one per model: a request waiting for a row, a prompting
// row, a move a session asked for, a refused step or the loop stopping ends
// grouped decoding -- the groups that are stepping land, none starts again
// -- and the loop's next iteration runs as it always has.

// decGroup is one decode group: its rows, the step its worker runs, and what
// that step returned.
type decGroup struct {
	rows  []*row
	units []unit
	runs  []engine.Run
	out   [][]float32
	err   error
	t0    time.Time
	// steps is how many steps the group took in this round.
	steps int
	// start hands the worker a step; it is closed with the loop.
	start chan struct{}
}

// groupWorker steps g each time the loop hands it a step, until the loop
// closes.
func (lp *stepLoop) groupWorker(g *decGroup) {
	for range g.start {
		g.out, g.err = engine.StepJoint(g.runs)
		lp.landed <- g
	}
}

// groupCount is how many groups this iteration's decoding rows step in: 1,
// today's one joint step, unless the loop was asked to group (Config.DecodeGroups
// -1 or above 1), every row is decoding on a model whose blocks are on several
// cards and nothing waits for the loop.
func (lp *stepLoop) groupCount() int {
	if lp.groupsCfg == 0 || lp.groupsCfg == 1 || (lp.refused && lp.refusedFor == lp.shape) {
		return 1
	}
	gpu := lp.lm.gpuTier()
	if gpu == nil {
		return 1
	}
	decoding := 0
	for _, r := range lp.rows {
		if r.prompting() || !r.s.steppable() || r.s.movePending() {
			return 1
		}
		decoding++
	}
	if decoding < 2 || lp.admissionWaits() {
		return 1
	}
	n := lp.groupsCfg
	if n <= 0 {
		// By itself the loop splits rows only where the device's steps are
		// one kernel for every row count (gpu.GPU.RowsComposed): elsewhere a
		// row's tokens would move with the rows that share its step.
		if !gpu.RowsComposed() {
			return 1
		}
		n = gpu.PipelineDepth()
	}
	for _, r := range lp.rows {
		if r.groups > 0 {
			n = min(n, r.groups)
		}
	}
	return min(n, decoding)
}

// grouping says whether an iteration's rows decoding rows step in decode groups,
// given groupCount found more than one: a count the caller forced
// always does, and the loop's own count only where the measured choice
// (groupChoice) found groups faster per row token than one group, its
// incumbent -- a group's step on the tier can cost more than its share of one
// group's (a mixture's joint step at one row against the one-at-a-time path),
// and groups must not cost a request its speed.
func (lp *stepLoop) grouping(rows int) bool {
	if lp.groupsCfg > 1 {
		return true
	}
	return !lp.groupChoice.pick(rows)
}

// decodeGroupedTimed is decodeGrouped for the loop's own count, timed for
// groupChoice: while its row count is probed, each group takes a choice's run
// of steps (jointRun) and the time per row token is what it observes.
func (lp *stepLoop) decodeGroupedTimed(n, rows int) {
	limit := 0
	if lp.groupsCfg <= 1 && !lp.groupChoice.settled(rows) {
		limit = jointRun
	}
	r0, t0 := lp.stats.rowsStepped.Load(), time.Now()
	lp.decodeGrouped(n, limit)
	if stepped := lp.stats.rowsStepped.Load() - r0; stepped > 0 && lp.groupsCfg <= 1 {
		// observe takes a step's time and divides it by its rows.
		lp.groupChoice.observe(rows, false, time.Since(t0)*time.Duration(rows)/time.Duration(stepped))
	}
}

// admissionWaits reports a request waiting that the loop could admit now --
// a free row -- or by parking a row (a fairness level above 0).
func (lp *stepLoop) admissionWaits() bool {
	lp.mu.Lock()
	w := len(lp.waiting)
	lp.mu.Unlock()
	return w > 0 && (len(lp.rows) < lp.width || lp.fair > 0)
}

// movePending reports a move of blocks the session asked for and the loop has
// yet to commit (serveMove).
func (s *Session) movePending() bool {
	s.moveMu.Lock()
	defer s.moveMu.Unlock()
	return s.move != nil
}

// decodeGrouped deals the decoding rows round robin into n groups and steps
// each on its own until something ends grouped decoding, or each took limit
// steps (0 for no limit), then lets every group land.
func (lp *stepLoop) decodeGrouped(n, limit int) {
	if lp.landed == nil {
		lp.landed = make(chan *decGroup, engine.MaxStepRows)
	}
	for len(lp.groups) < n {
		g := &decGroup{start: make(chan struct{}, 1)}
		lp.groups = append(lp.groups, g)
		go lp.groupWorker(g)
	}
	gs := lp.groups[:n]
	for _, g := range gs {
		g.rows = g.rows[:0]
	}
	for i, r := range lp.rows {
		gs[i%n].rows = append(gs[i%n].rows, r)
	}
	lp.stats.groupRounds.Add(1)
	busy, drain := 0, false
	for _, g := range gs {
		g.steps = 0
		if lp.launch(g) {
			busy++
		}
	}
	for busy > 0 {
		g := <-lp.landed
		busy--
		if g.err != nil && !g.advanced() {
			// Its rows run alone, which is the one-at-a-time path, once no
			// other group's step is on the devices.
			lp.refusedGroups = append(lp.refusedGroups, g)
			drain = true
			continue
		}
		lp.land(g)
		drain = drain || lp.groupsDrain(g)
		if !drain && (limit == 0 || g.steps < limit) && lp.launch(g) {
			busy++
		}
	}
	for _, g := range lp.refusedGroups {
		lp.land(g)
	}
	clear(lp.refusedGroups)
	lp.refusedGroups = lp.refusedGroups[:0]
	lp.post.wait()
	for _, g := range gs {
		clear(g.rows)
		g.rows = g.rows[:0]
	}
}

// launch samples g's rows' next tokens, retiring those that end here, and
// hands the rest to g's worker. It reports false when no row is left.
func (lp *stepLoop) launch(g *decGroup) bool {
	lp.post.wait()
	live, units := g.rows[:0], g.units[:0]
	for _, r := range g.rows {
		if lp.nextToken(r) {
			live = append(live, r)
			units = append(units, unit{r: r, tokens: r.next[:], logits: true})
		}
	}
	clear(g.rows[len(live):])
	g.rows, g.units = live, units
	if len(units) == 0 {
		return false
	}
	lp.post.start()
	runs := g.runs[:0]
	for _, u := range units {
		runs = append(runs, engine.Run{Session: u.r.s.ms, Tokens: u.tokens, Logits: true})
	}
	g.runs = runs
	lp.stats.steps.Add(1)
	lp.stats.groupSteps.Add(1)
	lp.stats.rowsStepped.Add(int64(len(units)))
	lp.stats.maxRows.Store(max(lp.stats.maxRows.Load(), int32(len(units))))
	lp.stats.maxSessions.Store(max(lp.stats.maxSessions.Load(), int32(len(units))))
	g.t0 = time.Now()
	g.steps++
	if lp.serialGroups {
		g.out, g.err = engine.StepJoint(g.runs)
		lp.landed <- g
		return true
	}
	g.start <- struct{}{}
	return true
}

// land moves g's rows on by the step its worker ran. A step the device
// refused before any row advanced runs its rows alone, as stepJoint's does,
// and the refusal ends grouped decoding until the rows change. Alone a row
// runs the one-at-a-time path, which is no rows stage (gpu.GPU.stageBegin),
// so decodeGrouped lands such a group only once no other group steps.
func (lp *stepLoop) land(g *decGroup) {
	defer func() {
		clear(g.runs)
		g.runs, g.out, g.err = g.runs[:0], nil, nil
	}()
	if g.err != nil {
		if g.advanced() {
			for _, u := range g.units {
				lp.finish(u.r, FinishError, "", g.err)
			}
			g.rows = g.rows[:0]
			return
		}
		why := g.err.Error()
		lp.stats.refusals.Add(1)
		lp.stats.lastRefusal.Store(&why)
		lp.refused, lp.refusedFor = true, lp.shape
		live := g.rows[:0]
		for _, u := range g.units {
			out, err := engine.StepRuns([]engine.Run{{Session: u.r.s.ms, Tokens: u.tokens, Logits: true}})
			if err != nil {
				lp.finish(u.r, FinishError, "", err)
				continue
			}
			lp.stats.soloRows.Add(1)
			lp.ran(u, out[0])
			live = append(live, u.r)
		}
		g.rows = live
		return
	}
	lp.stepNs = ewma(lp.stepNs, float64(time.Since(g.t0)))
	lp.stats.jointSteps.Add(1)
	lp.stats.jointRows.Add(int64(len(g.units)))
	for i, u := range g.units {
		lp.ran(u, g.out[i])
	}
}

// advanced reports whether a row of g's failed step advanced its recurrent
// state, so running its tokens again would apply them twice.
func (g *decGroup) advanced() bool {
	for _, u := range g.units {
		if u.r.s.recurrenceAdvanced() {
			return true
		}
	}
	return false
}

// groupsDrain reports that grouped decoding ends: no group starts again once
// g, just landed, says so.
func (lp *stepLoop) groupsDrain(g *decGroup) bool {
	if lp.ctx.Err() != nil || (lp.refused && lp.refusedFor == lp.shape) || lp.admissionWaits() {
		return true
	}
	for _, r := range g.rows {
		if r.s.movePending() {
			return true
		}
	}
	return false
}

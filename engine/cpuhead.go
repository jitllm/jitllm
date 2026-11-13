package engine

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// The head and the sampler on the CPU device: where no card holds the head,
// the session's logits come from what the composer forms for the head's graph
// (graph.Head over the row the streams collapse to, the norm, the
// projection and what follows it), a step's several rows from its form of
// cpuRowsCap rows, a label pick from the head projecting the picked rows
// alone, and a greedy token or a top-k draw's token from what it forms for
// the sampler's (graph.Sample, graph.SampleWalk). Every one is composed
// once for the instance at its placement (cpuPlaceHead) and counted served or
// declined by name (CPUHeadSteps, CPUSampleSteps); a step forms nothing.

// partPlan is graph g as a part at rows rows: its weights read in the
// formats the container states for its block (the model's own,
// jlm.DenseBlock, for the head's), over's for the roles over names. A role
// the block reads from another's tensor (a prediction block's embedding, the
// trunk's table) takes that tensor's format.
func (s *Instance) partPlan(g *graph.Graph, rows int, over map[jlm.Role]jlm.Type) nn.PartPlan {
	if g == nil {
		return nn.PartPlan{Skip: true}
	}
	p := nn.PartPlan{Graph: g, Rows: rows, ActWindow: s.m.ActWindow}
	p.Formats = map[jlm.Role]jlm.Type{}
	f := s.m.Composing()
	var l *model.Layer
	if g.Block >= 0 && g.Block < len(s.m.Layers) {
		l = &s.m.Layers[g.Block]
	}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if t, ok := over[w.Role]; ok {
				p.Formats[w.Role] = t
			} else if e, ok := f.Find(w.Role, int32(g.Block), -1); ok {
				p.Formats[w.Role] = e.Type
			} else if t := s.ws.matrix(g.Block, l, w.Role); t != nil && t.E != nil {
				p.Formats[w.Role] = t.E.Type
			}
		}
	}
	return p
}

// Sample is smp's token over logits, a row of this instance's vocabulary,
// from the composed samplers on the CPU: the greedy token (graph.Sample), a
// top-k draw's (graph.SampleWalk) and the row's or its cut's
// (graph.SampleRow, SampleCut), at the uniform smp's stream draws.
func (s *Instance) Sample(smp *Sampler, logits []float32) (int32, error) {
	if len(logits) != s.c.NVocab {
		return 0, fmt.Errorf("model: a draw over %d logits, and the vocabulary is %d", len(logits), s.c.NVocab)
	}
	if smp == nil || smp.Temp <= 0 {
		if t, ok := s.greedy(logits); ok {
			return t, nil
		}
		return 0, s.sampleDeclined("the greedy token")
	}
	u := float32(smp.uniform())
	if t, ok := s.draw(logits, smp, u, smp.xtcDraw()); ok {
		return t, nil
	}
	return 0, s.sampleDeclined("the draw")
}

// headDeclined is the error of head rows no composed form serves.
func (s *Instance) headDeclined(why nn.Decline) error {
	return fmt.Errorf("model: the head has no composed form on this instance (%s): %v", nn.DeclineName(int(why)), s.prepErr())
}

// sampleDeclined is the error of a draw no composed form serves: a form the
// composer refused at the placement, which CPUFormErr names.
func (s *Instance) sampleDeclined(what string) error {
	return fmt.Errorf("model: %s has no composed form on this instance: %v", what, s.prepErr())
}

// dist fills d with the distribution smp draws from over logits (a row this
// instance's model produces), from the composed forms (distOf).
func (s *Instance) dist(smp *Sampler, logits []float32, d *specDist) error {
	if !s.distOf(logits, smp, d) {
		return s.sampleDeclined("the distribution")
	}
	return nil
}

// topProb is the largest probability of logits' softmax: the draft's
// confidence in its argmax, from the composed greedy sampler and the
// distribution of a draw at temperature 1 with no cut.
func (s *Instance) topProb(logits []float32, d *specDist) (float32, error) {
	one := Sampler{Temp: 1}
	t, ok := s.greedy(logits)
	if !ok || !s.distOf(logits, &one, d) {
		return 0, s.sampleDeclined("the draft's confidence")
	}
	return d.prob(t), nil
}

// headRow is the head over the row x into logits, and the norm's rows into
// hidden where it is given, on the device the instance places the head on
// (nn.HeadDevice.Head), or why not.
func (s *Instance) headRow(x, logits, hidden []float32) nn.Decline {
	if s.heads == nil {
		return nn.DeclineCaller
	}
	return s.heads.Head(x, logits, hidden)
}

// headRows is the head over the first n rows of x into the rows of logits,
// and the norm's rows into hidden where it is given (nn.HeadDevice.HeadRows),
// or why not.
func (s *Instance) headRows(x, logits, hidden []float32, n int) nn.Decline {
	if s.heads == nil {
		return nn.DeclineCaller
	}
	return s.heads.HeadRows(x, logits, hidden, n)
}

// outNormRows writes the output norm of n rows of x into dst
// (nn.HeadDevice.OutNorm), or why not.
func (s *Instance) outNormRows(dst, x []float32, n int) nn.Decline {
	if s.heads == nil {
		return nn.DeclineCaller
	}
	return s.heads.OutNorm(dst, x, n)
}

// pickLabels is the label pick: the head projecting the instance's picked
// ids (PickIDs) alone from the row x into PickVals, the logits left as
// they were (nn.HeadDevice.Pick), or why not.
func (s *Instance) pickLabels(x []float32) nn.Decline {
	s.Picked = false
	if s.heads == nil {
		return nn.DeclineCaller
	}
	why := s.heads.Pick(x, s.PickIDs, s.PickVals)
	s.Picked = why == nn.DeclineServes
	return why
}

// decideScores is a label readout over the picked logits vals, its
// distribution into probs and its statistics into stats
// (nn.HeadDevice.Decide), or false where it declines.
func (s *Instance) decideScores(vals, keep, params, prior, probs, stats []float32) bool {
	if s.heads == nil {
		return false
	}
	return s.heads.Decide(vals, keep, params, prior, probs, stats)
}

// greedy is the token of the largest of logits, the lower id of two equal
// (nn.HeadDevice.Greedy), or false where it declines.
func (s *Instance) greedy(logits []float32) (int32, bool) {
	if s.heads == nil {
		return 0, false
	}
	return s.heads.Greedy(logits)
}

// drawArgs is smp's draw as a device reads it, at XTC's threshold xtc (0
// for none) and its mixture mix (a distribution's).
func (s *Instance) drawArgs(smp *Sampler, xtc, mix float32) nn.DrawArgs {
	topp, minp := smp.cuts()
	rng, ex := smp.dyn()
	a := nn.DrawArgs{TopK: smp.TopK, DynRange: rng, DynExp: ex, InvTemp: float32(1 / smp.Temp), TopP: topp, MinP: minp, Typical: float32(smp.TypicalP),
		XTC: xtc, XTCMix: mix, Sigma: float32(smp.TopNSigma), Penalty: smp.penalties(), Infill: smp.Infill}
	if smp.Mirostat != 0 {
		a.Mirostat, a.Penalty = smp.Mirostat, nn.NoPenalties()
		var mu *float32
		a.MiroTau, a.MiroEta, mu = smp.miro()
		a.MiroMu, a.Mu = *mu, mu
	}
	if !a.Penalty.Off() {
		a.Window = smp.window(s.c.NVocab)
	}
	return a
}

// draw is smp's sampled token over logits at the uniform u smp drew for the
// call (nn.HeadDevice.Draw), or false where none serves it. The draw's
// arguments stay in smp for its distribution (Sampler.drew).
func (s *Instance) draw(logits []float32, smp *Sampler, u, xtc float32) (int32, bool) {
	if s.heads == nil {
		return 0, false
	}
	a := s.drawArgs(smp, xtc, 0)
	smp.drew = a
	smp.drew.Mu = nil
	if xtc > 0 {
		smp.drew.XTCMix = 1
	}
	if smp.staged() {
		row, ok := s.staged(logits, smp, a)
		if !ok {
			return 0, false
		}
		logits, a = row, plainArgs(a.Infill)
	}
	if smp.adaptive() {
		return s.adaptiveDraw(logits, a, smp, u)
	}
	return s.heads.Draw(logits, a, u)
}

// plainArgs is the row's draw at temperature 1 with no cut and no penalty,
// infill's where infill is set.
func plainArgs(infill bool) nn.DrawArgs {
	return nn.DrawArgs{InvTemp: 1, TopP: float32(math.Inf(1)), DynExp: 1, Penalty: nn.NoPenalties(), Infill: infill}
}

// stagedRows is a chain run in an order of its own on an instance
// (Instance.staged): the row each stage leaves and the next, the
// distribution a cut reads its survivors from, a row of zeros a top-k
// draw's candidates are put into, those candidates and their ids, and the
// ids of none for a whole row's.
type stagedRows struct {
	cur, next, p, zero, cand []float32
	ids, none                []uint32
}

// staged runs smp's stages in their order over logits (graph.SampleStage),
// each over the row the one before it leaves: a cut keeps the survivors the
// head's composed draw with that cut alone leaves at temperature 1
// (nn.HeadDevice.Dist), its candidates put in place; the penalties and the
// temperature keep every finite logit. a is the draw's arguments, XTC's at
// its chance as it fell. A stage that is off does not run. It returns the
// last row, the instance's until the next draw, or false where a form
// declines.
func (s *Instance) staged(logits []float32, smp *Sampler, a nn.DrawArgs) ([]float32, bool) {
	n, r := len(logits), &s.st
	if len(r.cur) != n {
		r.cur, r.next, r.p, r.zero = make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n)
		r.cand, r.ids, r.none = make([]float32, nn.SampleMaxK), make([]uint32, nn.SampleMaxK), make([]uint32, nn.SampleMaxK)
		for i := range r.none {
			r.none[i] = uint32(n)
		}
	}
	copy(r.cur, logits)
	for _, st := range smp.Stages {
		c, w, cut := plainArgs(false), [4]float32{1, 0, 0, 1}, true
		switch st {
		case StagePenalties:
			if a.Penalty.Off() {
				continue
			}
			c.Penalty, c.Window, w[1], cut = a.Penalty, a.Window, 1, false
		case StageTopNSigma:
			if a.Sigma <= 0 {
				continue
			}
			c.Sigma = a.Sigma
		case StageTopK:
			if a.TopK <= 0 || a.TopK >= n {
				continue
			}
			c.TopK = a.TopK
		case StageTypical:
			if a.Typical <= 0 || a.Typical >= 1 {
				continue
			}
			c.Typical = a.Typical
		case StageTopP:
			if a.TopP >= 1 {
				continue
			}
			c.TopP = a.TopP
		case StageMinP:
			if a.MinP <= 0 {
				continue
			}
			c.MinP = a.MinP
		case StageXTC:
			if a.XTC <= 0 {
				continue
			}
			c.XTC, c.XTCMix = a.XTC, 1
		case StageTemperature:
			if a.InvTemp == 1 && a.DynRange <= 0 {
				continue
			}
			w, cut = [4]float32{a.InvTemp, 1, a.DynRange, a.DynExp}, false
		}
		src, cids := r.zero, r.none
		if cut {
			dense, ids, vals, ok := s.heads.Dist(r.cur, c, r.p)
			if !ok {
				return nil, false
			}
			src = r.p
			if !dense {
				src, cids = r.zero, r.ids
				copy(r.ids, r.none)
				copy(r.ids, ids)
				clear(r.cand)
				copy(r.cand, vals)
			}
		}
		if !s.heads.Stage(r.cur, src, cids, r.cand, c, w, r.next) {
			return nil, false
		}
		r.cur, r.next = r.next, r.cur
	}
	return r.cur, true
}

// adaptiveRows is adaptive-p's rows on an instance (Instance.adaptive).
type adaptiveRows struct {
	// p is the chain's distribution where it is the whole row's; zero a
	// row of zeros a top-k draw's candidates are put into; ids and cand the
	// candidates, none the ids of none for a whole row's.
	p, zero, cand []float32
	ids, none     []uint32
	// logits, q and pd are the transform's: the logits the row's draw
	// walks, their softmax and the chain's whole distribution.
	logits, q, pd []float32
	sum, weight   float32
}

// adaptive runs adaptive-p's transform over the distribution the draw a
// reads over logits (nn.HeadDevice.Dist, its candidates put in place) at
// words w, into s.ad, or false where a form declines.
func (s *Instance) adaptive(logits []float32, a nn.DrawArgs, w [6]float32) bool {
	n, r := len(logits), &s.ad
	if len(r.p) != n {
		r.p, r.zero, r.logits, r.q, r.pd = make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n)
		r.cand, r.ids, r.none = make([]float32, nn.SampleMaxK), make([]uint32, nn.SampleMaxK), make([]uint32, nn.SampleMaxK)
		for i := range r.none {
			r.none[i] = uint32(n)
		}
	}
	dense, ids, vals, ok := s.heads.Dist(logits, a, r.p)
	if !ok {
		return false
	}
	src, cids := r.p, r.none
	if !dense {
		src, cids = r.zero, r.ids
		copy(r.ids, r.none)
		copy(r.ids, ids)
		clear(r.cand)
		copy(r.cand, vals)
	}
	r.sum, r.weight, ok = s.heads.Adaptive(src, cids, r.cand, w, r.logits, r.q, r.pd)
	return ok
}

// adaptiveDraw is smp's draw through adaptive-p at the uniform u: the
// transform of the distribution the draw a reads over logits, then the
// row's draw over the transformed logits at temperature 1. The drawn token's
// probability under the chain's distribution goes into the average at the
// next draw, as llama.cpp's accept takes it in.
func (s *Instance) adaptiveDraw(logits []float32, a nn.DrawArgs, smp *Sampler, u float32) (int32, bool) {
	smp.adDrew = smp.adaptiveWords()
	if !s.adaptive(logits, a, smp.adDrew) {
		return 0, false
	}
	t, ok := s.heads.Draw(s.ad.logits, plainArgs(false), u)
	if !ok {
		return 0, false
	}
	smp.ad[2], smp.ad[3], smp.ad[4], smp.ad[5] = s.ad.sum, s.ad.weight, s.ad.pd[t], 1
	return t, true
}

// distOf fills d with the distribution smp draws from over logits: a top-k
// draw's candidates and their share of the survivors, else every token's
// probability (nn.HeadDevice.Dist). False where it declines.
func (s *Instance) distOf(logits []float32, smp *Sampler, d *specDist) bool {
	n := len(logits)
	if s.heads == nil || n != s.c.NVocab {
		return false
	}
	bounded := smp.TopK > 0 && smp.TopK < n && smp.TopK <= nn.SampleMaxK && smp.TopNSigma <= 0
	if !bounded {
		d.pall = grow32(d.pall, n)
	}
	thr, mix := smp.xtc()
	dense, ids, vals, ok := s.heads.Dist(logits, s.drawArgs(smp, thr, mix), d.pall)
	if !ok {
		return false
	}
	d.dense = dense
	if !dense {
		d.ids, d.p = d.ids[:0], d.p[:0]
		for i, id := range ids {
			d.ids, d.p = append(d.ids, int32(id)), append(d.p, vals[i])
		}
	}
	return true
}

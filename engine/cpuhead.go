package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/cpudev"
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

// cpuPlaceHead composes the head's and the samplers' graphs for the CPU
// device at the instance's placement (cpudev.Head).
func (s *Instance) cpuPlaceHead() {
	if s.cpu.x.Device() == nil {
		return
	}
	fg := s.m.FrontGraphs()
	var p [15]cpudev.Part
	p[0] = s.cpuPart(fg.CpuHead, nn.DeclineNoGraph, 1)
	p[1] = s.cpuPart(fg.CpuHidden, nn.DeclineNoGraph, 1)
	p[2] = s.cpuPart(fg.CpuHidden, nn.DeclineNoGraph, cpuRowsCap)
	none := nn.DeclineNoGraph
	if fg.CpuHead != nil {
		none = nn.DeclineHeadBias
	}
	p[3] = s.cpuPart(fg.PickHead, none, 1)
	for i, g := range []*graph.Graph{fg.CpuSample, fg.CpuWalk, fg.CpuRow, fg.CpuCut, fg.CpuWalkProb, fg.CpuCutProb,
		fg.CpuCutK, fg.CpuCutKProb} {
		p[4+i] = s.cpuPart(g, nn.DeclineNoSample, 1)
	}
	if fg.OutNorm != nil {
		p[12] = s.cpuPart(fg.OutNorm, nn.DeclineNoGraph, cpuRowsCap)
	}
	p[13] = s.cpuPart(fg.CpuLogprobs, nn.DeclineNoSample, 1)
	p[14] = s.cpuPart(fg.CpuMask, nn.DeclineNoSample, 1)
	parts := cpudev.HeadParts{Head: p[0], Hidden: p[1], Rows: p[2], Pick: p[3], Sample: p[4], Walk: p[5], Row: p[6],
		Cut: p[7], WalkProb: p[8], CutProb: p[9], CutK: p[10], CutKProb: p[11], OutNorm: p[12], Logprobs: p[13], Mask: p[14]}
	s.heads = cpudev.NewHead(s.cpu.x.Device(), s.cpu.x.Steps(), parts, fg.OutNorm != nil, blockWeights{ws: &s.ws, li: -1},
		s.c.NEmbd, s.c.NVocab, model.MaxTopLogprobs)
}

// cpuPart is graph g composed on the CPU device as a part at rows rows
// (cpudev.Exec.FormPart), none where the front end states no graph.
func (s *Instance) cpuPart(g *graph.Graph, none nn.Decline, rows int) cpudev.Part {
	return s.cpu.x.FormPart(s.partPlan(g, none, rows, nil))
}

// partPlan is graph g as a part at rows rows: its weights read in the
// formats the container states for its block (the model's own,
// jlm.DenseBlock, for the head's), over's for the roles over names. A role
// the block reads from another's tensor (a prediction block's embedding, the
// trunk's table) takes that tensor's format.
func (s *Instance) partPlan(g *graph.Graph, none nn.Decline, rows int, over map[jlm.Role]jlm.Type) cpudev.PartPlan {
	p := cpudev.PartPlan{Graph: g, Rows: rows, ActWindow: s.m.ActWindow, None: none}
	if g == nil {
		return p
	}
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

// formCPUGraph is graph g on the CPU at rows rows (cpudev.Exec.FormGraph),
// over's formats in place of the container's for the roles it names.
func (s *Instance) formCPUGraph(g *graph.Graph, rows int, over map[jlm.Role]jlm.Type) (*cpudev.Form, error) {
	return s.cpu.x.FormGraph(s.partPlan(g, nn.DeclineNoGraph, rows, over))
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
	if t, ok := s.draw(logits, smp, u); ok {
		return t, nil
	}
	return 0, s.sampleDeclined("the draw")
}

// headDeclined is the error of head rows no composed form serves.
func (s *Instance) headDeclined(why nn.Decline) error {
	return fmt.Errorf("model: the head has no composed form on this instance (%s): %v", nn.DeclineName(int(why)), s.cpu.x.Composer().Err())
}

// sampleDeclined is the error of a draw no composed form serves: a form the
// composer refused at the placement, which CPUFormErr names.
func (s *Instance) sampleDeclined(what string) error {
	return fmt.Errorf("model: %s has no composed form on this instance: %v", what, s.cpu.x.Composer().Err())
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

// cpuFormCount is how many forms the model's CPU cache holds, refused ones
// among them: a step that forms nothing leaves it as it was.
func (s *Instance) cpuFormCount() int { return s.eng.cpu.Forms(s.m.Id).Count() }

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
// ids (HostPick) alone from the row x into HostPickVals, the logits left as
// they were (nn.HeadDevice.Pick), or why not.
func (s *Instance) pickLabels(x []float32) nn.Decline {
	s.HostPicked = false
	if s.heads == nil {
		return nn.DeclineCaller
	}
	why := s.heads.Pick(x, s.HostPick, s.HostPickVals)
	s.HostPicked = why == nn.DeclineServes
	return why
}

// greedy is the token of the largest of logits, the lower id of two equal
// (nn.HeadDevice.Greedy), or false where it declines.
func (s *Instance) greedy(logits []float32) (int32, bool) {
	if s.heads == nil {
		return 0, false
	}
	return s.heads.Greedy(logits)
}

// drawArgs is smp's draw as a device reads it.
func (s *Instance) drawArgs(smp *Sampler) nn.DrawArgs {
	topp, minp := smp.cuts()
	a := nn.DrawArgs{TopK: smp.TopK, InvTemp: float32(1 / smp.Temp), TopP: topp, MinP: minp, Penalty: smp.penalty()}
	if a.Penalty != 1 {
		a.Window = smp.window(s.c.NVocab)
	}
	return a
}

// draw is smp's sampled token over logits at the uniform u smp drew for the
// call (nn.HeadDevice.Draw), or false where none serves it.
func (s *Instance) draw(logits []float32, smp *Sampler, u float32) (int32, bool) {
	if s.heads == nil {
		return 0, false
	}
	return s.heads.Draw(logits, s.drawArgs(smp), u)
}

// distOf fills d with the distribution smp draws from over logits: a top-k
// draw's candidates and their share of the survivors, else every token's
// probability (nn.HeadDevice.Dist). False where it declines.
func (s *Instance) distOf(logits []float32, smp *Sampler, d *specDist) bool {
	n := len(logits)
	if s.heads == nil || n != s.c.NVocab {
		return false
	}
	bounded := smp.TopK > 0 && smp.TopK < n && smp.TopK <= nn.SampleMaxK
	if !bounded {
		d.pall = grow32(d.pall, n)
	}
	dense, ids, vals, ok := s.heads.Dist(logits, s.drawArgs(smp), d.pall)
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

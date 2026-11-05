package engine

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
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

// cpuHeadState is a instance's composed head and samplers on the CPU and
// their census.
type cpuHeadState struct {
	// head is the head's form, hidden the one also writing the norm's
	// rows, rows the latter at cpuRowsCap rows, pick the label pick's,
	// sample the greedy sampler's, walk a top-k draw's, row and cut a draw's
	// with no top-k.
	head, hidden, rows, pick, sample, walk, row, cut cpuHeadForm
	// walkProb and cutProb are the distributions the walk and the cut draw
	// from, probVals and probIDs the walk's; cutK and cutKProb a top-k
	// draw past the walk's bound and its distribution.
	walkProb, cutProb, cutK, cutKProb cpuHeadForm
	// outNorm is the output norm's alone (outNormGraph), onPad its rows
	// where a call is shorter than its unit, in then out.
	outNorm cpuHeadForm
	onPad   []float32
	// logprobs is the log-softmax and the most likely tokens' form, lpRow
	// and lpIDs what it writes; mask a constrained draw's masked logits'.
	logprobs, mask  cpuHeadForm
	lpRow           []float32
	lpIDs           []uint32
	cutKArgs        [5]float32
	probVals        []float32
	probIDs         []uint32
	served, sampled int
	declined        [composer.NumDeclines]int
	sampleDeclined  [composer.NumDeclines]int
	token           [1]uint32
	// noWin is a window that holds no token, for a sampler with no
	// penalty; pen is the penalty a draw reads.
	noWin []float32
	pen   [1]float32
	// walkArgs is the walk's words: k, 1/T, top-p, min-p and u; rowArgs
	// the row's (1/T, u) and cutArgs the cut's (1/T, top-p, min-p, u).
	walkArgs [5]float32
	rowArgs  [2]float32
	cutArgs  [4]float32
	// ids is the row's ids, 0 to the vocabulary's last, which the draws
	// with no top-k read.
	ids []uint32
	// hid is the rows' norm rows where the caller keeps none; xpad, lpad
	// and hpad a call's last rows where the rows' form's units reach past
	// them.
	hid, xpad, lpad, hpad []float32
	pickIDs               []uint32
	pickVals              []float32
}

// cpuHeadForm is one of the head's graphs composed for the CPU, or why
// not.
type cpuHeadForm struct {
	composed bool
	f        *cpudev.Form
	why      composer.Decline
	// err is the composer's, where it formed none.
	err error
}

// CPUHeadSteps is how many head rows the CPU's composed head served, and
// how many it declined, by reason. With the composed CPU off nothing is
// counted.
func (s *Instance) CPUHeadSteps() (served int, declined [composer.NumDeclines]int) {
	return s.cpu.head.served, s.cpu.head.declined
}

// CPUSampleSteps is CPUHeadSteps for the composed samplers: a greedy token
// or a top-k draw's token.
func (s *Instance) CPUSampleSteps() (served int, declined [composer.NumDeclines]int) {
	return s.cpu.head.sampled, s.cpu.head.sampleDeclined
}

// cpuPlaceHead composes the head's and the samplers' graphs for the CPU
// device, at the instance's placement, with the buffers their calls bind.
func (s *Instance) cpuPlaceHead() {
	h := &s.cpu.head
	fg := s.m.FrontGraphs()
	s.cpuComposeHead(&h.head, fg.CpuHead, composer.DeclineNoGraph, 1)
	s.cpuComposeHead(&h.hidden, fg.CpuHidden, composer.DeclineNoGraph, 1)
	s.cpuComposeHead(&h.rows, fg.CpuHidden, composer.DeclineNoGraph, cpuRowsCap)
	none := composer.DeclineNoGraph
	if fg.CpuHead != nil {
		none = composer.DeclineHeadBias
	}
	s.cpuComposeHead(&h.pick, fg.PickHead, none, 1)
	s.cpuComposeHead(&h.sample, fg.CpuSample, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.walk, fg.CpuWalk, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.row, fg.CpuRow, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.cut, fg.CpuCut, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.walkProb, fg.CpuWalkProb, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.cutProb, fg.CpuCutProb, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.cutK, fg.CpuCutK, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.cutKProb, fg.CpuCutKProb, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.logprobs, fg.CpuLogprobs, composer.DeclineNoSample, 1)
	s.cpuComposeHead(&h.mask, fg.CpuMask, composer.DeclineNoSample, 1)
	if fg.OutNorm != nil {
		s.cpuComposeHead(&h.outNorm, fg.OutNorm, composer.DeclineNoGraph, cpuRowsCap)
		if f := h.outNorm.f; f != nil {
			h.onPad = make([]float32, 2*max(f.Unit, 1)*s.c.NEmbd)
		}
	}
	d, nv := s.c.NEmbd, s.c.NVocab
	if f := h.rows.f; f != nil {
		h.hid = make([]float32, f.Rows*d)
		if u := f.Unit; u > 1 {
			h.xpad, h.lpad, h.hpad = make([]float32, u*d), make([]float32, u*nv), make([]float32, u*d)
		}
	}
	if h.pick.f != nil {
		h.pickIDs, h.pickVals = make([]uint32, nn.PickMax), make([]float32, nn.PickMax)
	}
	if h.walk.f != nil || h.row.f != nil || h.cut.f != nil || h.walkProb.f != nil || h.cutProb.f != nil || h.cutK.f != nil || h.cutKProb.f != nil {
		h.noWin = make([]float32, nv)
	}
	if h.walkProb.f != nil {
		h.probVals, h.probIDs = make([]float32, nn.SampleMaxK), make([]uint32, nn.SampleMaxK)
	}
	if h.logprobs.f != nil {
		h.lpRow, h.lpIDs = make([]float32, nv), make([]uint32, min(model.MaxTopLogprobs, nv))
	}
	if h.row.f != nil || h.cut.f != nil || h.cutProb.f != nil || h.cutK.f != nil || h.cutKProb.f != nil {
		h.ids = make([]uint32, nv)
		for i := range h.ids {
			h.ids[i] = uint32(i)
		}
	}
}

// cpuComposeHead composes graph g for the CPU at rows rows into hf, or
// says why not: none where the front end states no graph.
func (s *Instance) cpuComposeHead(hf *cpuHeadForm, g *graph.Graph, none composer.Decline, rows int) {
	s.cpuComposeHeadAs(hf, g, rows, nil)
	if g == nil {
		hf.why = none
	}
}

// cpuComposeHeadAs is cpuComposeHead with the formats over states the
// weights of its roles are held in, in place of the container's.
func (s *Instance) cpuComposeHeadAs(hf *cpuHeadForm, g *graph.Graph, rows int, over map[jlm.Role]jlm.Type) {
	*hf = cpuHeadForm{composed: true}
	switch {
	case !composer.LowersCPU(uint64(s.cpuISA())):
		hf.why = composer.DeclineISA
	case g == nil:
		hf.why = composer.DeclineNoGraph
	default:
		f, err := s.formCPUGraph(g, rows, over)
		if err != nil {
			hf.why, hf.err = s.cpuFormDecline(err), err
			return
		}
		hf.f = f
	}
}

// formOf is hf's form, or why it does not serve: not composed at the
// instance's placement, or declined there.
func (hf *cpuHeadForm) formOf() (*cpudev.Form, composer.Decline) {
	if !hf.composed {
		return nil, composer.DeclineUnplaced
	}
	return hf.f, hf.why
}

// formCPUGraph is graph g on the CPU at rows rows through the model's keyed
// cache: its weights its block's (the model's own, jlm.DenseBlock, for the
// head's), its inputs and outputs buffers of their own. A role the block
// reads from another's tensor (a prediction block's embedding, the trunk's
// table) takes that tensor's format.
func (s *Instance) formCPUGraph(g *graph.Graph, rows int, over map[jlm.Role]jlm.Type) (*cpudev.Form, error) {
	formats := map[jlm.Role]jlm.Type{}
	f := s.m.Composing()
	var l *model.Layer
	if g.Block >= 0 && g.Block < len(s.m.Layers) {
		l = &s.m.Layers[g.Block]
	}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if t, ok := over[w.Role]; ok {
				formats[w.Role] = t
			} else if e, ok := f.Find(w.Role, int32(g.Block), -1); ok {
				formats[w.Role] = e.Type
			} else if t := s.cpuRoleWeight(g.Block, l, w.Role); t != nil && t.E != nil {
				formats[w.Role] = t.E.Type
			}
		}
	}
	req := composer.Request{Graph: g, Formats: formats, Step: composer.Step{Rows: rows, Page: 1, Keys: 1, ActWindow: s.m.ActWindow},
		Device: s.composedCPU(), Apart: true}
	return s.cpuFormOf(req, g, rows, true, false, false)
}

// cpuHeadOn says the composed CPU path is on for this Instance.
func (s *Instance) cpuHeadOn() bool { return s.cpu.dev != nil }

// cpuHead runs the head on the CPU the composer's way: the norm of the row
// x, its projection and what follows it into logits, and the norm's rows
// into hidden where it is given, or says why not.
func (s *Instance) cpuHead(x, logits, hidden []float32) composer.Decline {
	if !s.cpuHeadOn() {
		return composer.DeclineCaller
	}
	h := &s.cpu.head
	hf := &h.head
	if hidden != nil {
		hf = &h.hidden
	}
	f, why := hf.formOf()
	if why == composer.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why != composer.DeclineServes {
		h.declined[why]++
		return why
	}
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpu.dev.Own(f), n: 1}
	b.io[0], b.io[1] = unsafe.Pointer(&x[0]), unsafe.Pointer(&logits[0])
	if s.cpu.fault == cpuFaultHead {
		b.io[0] = unsafe.Pointer(&s.h[0])
	}
	if hidden != nil {
		b.io[2] = unsafe.Pointer(&hidden[0])
	}
	s.cpuRunForm(b, f)
	h.served++
	return composer.DeclineServes
}

// cpuHeadRows is the head over the first n rows of x, NEmbd apart, into the
// rows of logits, and the norm's rows into hidden where it is given (n rows
// of it), through the rows' form a call of at most its rows, or says why
// not. A call's rows past a whole number of the form's units run from
// copies, so its units write no row past the caller's.
func (s *Instance) cpuHeadRows(x, logits, hidden []float32, n int) composer.Decline {
	if !s.cpuHeadOn() {
		return composer.DeclineCaller
	}
	h := &s.cpu.head
	f, why := h.rows.formOf()
	if why == composer.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why != composer.DeclineServes {
		h.declined[why] += n
		return why
	}
	d, nv := s.c.NEmbd, s.c.NVocab
	for r0 := 0; r0 < n; {
		k := min(f.Rows, n-r0)
		u := max(f.Unit, 1)
		if k >= u {
			k -= k % u
			var hd unsafe.Pointer
			if hidden != nil {
				hd = unsafe.Pointer(&hidden[r0*d])
			} else {
				hd = unsafe.Pointer(&h.hid[0])
			}
			s.cpuHeadCall(f, unsafe.Pointer(&x[r0*d]), unsafe.Pointer(&logits[r0*nv]), hd, k)
		} else {
			copy(h.xpad, x[r0*d:(r0+k)*d])
			s.cpuHeadCall(f, unsafe.Pointer(&h.xpad[0]), unsafe.Pointer(&h.lpad[0]), unsafe.Pointer(&h.hpad[0]), k)
			copy(logits[r0*nv:(r0+k)*nv], h.lpad)
			if hidden != nil {
				copy(hidden[r0*d:(r0+k)*d], h.hpad)
			}
		}
		r0 += k
	}
	h.served += n
	return composer.DeclineServes
}

// cpuOutNorm writes the output norm of n rows of x into dst through the
// composed norm (outNormGraph) -- a model with no output norm copies them
// -- or says why not.
func (s *Instance) cpuOutNorm(dst, x []float32, n int) composer.Decline {
	d := s.c.NEmbd
	if s.m.FrontGraphs().OutNorm == nil {
		copy(dst[:n*d], x[:n*d])
		return composer.DeclineServes
	}
	if !s.cpuHeadOn() {
		return composer.DeclineCaller
	}
	h := &s.cpu.head
	f, why := h.outNorm.formOf()
	if why == composer.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why != composer.DeclineServes {
		h.declined[why] += n
		return why
	}
	for r0 := 0; r0 < n; {
		k := min(f.Rows, n-r0)
		u := max(f.Unit, 1)
		in, out := unsafe.Pointer(&x[r0*d]), unsafe.Pointer(&dst[r0*d])
		padded := k < u
		if padded {
			copy(h.onPad[:k*d], x[r0*d:(r0+k)*d])
			in, out = unsafe.Pointer(&h.onPad[0]), unsafe.Pointer(&h.onPad[u*d])
		} else {
			k -= k % u
		}
		b := &s.cpu.bound
		*b = cpuBound{own: s.cpu.dev.Own(f), n: k}
		b.io[0], b.io[1] = in, out
		s.cpu.call[0] = uint32(k)
		s.cpuRunForm(b, f)
		if padded {
			copy(dst[r0*d:(r0+k)*d], h.onPad[u*d:(u+k)*d])
		}
		r0 += k
	}
	return composer.DeclineServes
}

// cpuHeadCall runs the rows' form f over k rows of x into logits and the
// norm's rows hidden.
func (s *Instance) cpuHeadCall(f *cpudev.Form, x, logits, hidden unsafe.Pointer, k int) {
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpu.dev.Own(f), n: k}
	b.io[0], b.io[1], b.io[2] = x, logits, hidden
	if s.cpu.fault == cpuFaultHead {
		b.io[0] = unsafe.Pointer(&s.cpu.head.hid[0])
	}
	s.cpu.call[0] = uint32(k)
	s.cpuRunForm(b, f)
}

// cpuPick is the label pick on the CPU: the head projecting the instance's
// picked ids (HostPick) alone from the row x into HostPickVals, the
// logits left as they were, or why not.
func (s *Instance) cpuPick(x []float32) composer.Decline {
	s.HostPicked = false
	if !s.cpuHeadOn() {
		return composer.DeclineCaller
	}
	h := &s.cpu.head
	f, why := h.pick.formOf()
	if why == composer.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why == composer.DeclineServes && (len(s.HostPick) > len(h.pickIDs) || len(s.HostPickVals) < len(s.HostPick)) {
		why = composer.DeclineCaller
	}
	if why != composer.DeclineServes {
		h.declined[why]++
		return why
	}
	clear(h.pickIDs)
	for i, id := range s.HostPick {
		h.pickIDs[i] = uint32(id)
	}
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpu.dev.Own(f), n: 1}
	b.io[0], b.io[1], b.io[2] = unsafe.Pointer(&x[0]), unsafe.Pointer(&h.pickIDs[0]), unsafe.Pointer(&h.pickVals[0])
	if s.cpu.fault == cpuFaultHead {
		b.io[0] = unsafe.Pointer(&s.h[0])
	}
	s.cpuRunForm(b, f)
	copy(s.HostPickVals, h.pickVals[:len(s.HostPick)])
	s.HostPicked = true
	h.served++
	return composer.DeclineServes
}

// cpuGreedy is the token of the largest of logits, the lower id of two
// equal, from the CPU's composed sampler, or false where it declines.
func (s *Instance) cpuGreedy(logits []float32) (int32, bool) {
	if !s.cpuHeadOn() {
		return 0, false
	}
	h := &s.cpu.head
	f, why := h.sample.formOf()
	if why != composer.DeclineServes {
		h.sampleDeclined[why]++
		return 0, false
	}
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpu.dev.Own(f), n: 1}
	b.io[0], b.io[1] = unsafe.Pointer(&logits[0]), unsafe.Pointer(&h.token[0])
	s.cpuRunForm(b, f)
	h.sampled++
	return int32(h.token[0]), true
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
		if t, ok := s.cpuGreedy(logits); ok {
			return t, nil
		}
		return 0, s.sampleDeclined("the greedy token")
	}
	u := float32(smp.uniform())
	if t, ok := s.cpuDraw(logits, smp, u); ok {
		return t, nil
	}
	return 0, s.sampleDeclined("the draw")
}

// headDeclined is the error of head rows no composed form serves.
func (s *Instance) headDeclined(why composer.Decline) error {
	return fmt.Errorf("model: the head has no composed form on this instance (%s): %v", composer.DeclineName(int(why)), s.cpu.formErr)
}

// sampleDeclined is the error of a draw no composed form serves: a form the
// composer refused at the placement, which CPUFormErr names.
func (s *Instance) sampleDeclined(what string) error {
	return fmt.Errorf("model: %s has no composed form on this instance: %v", what, s.cpu.formErr)
}

// cpuDraw is smp's sampled token over logits from the CPU's composed draws
// at the uniform u smp drew for the call: a top-k draw's walk, or with no
// top-k the row's or, where top-p or min-p is set, the cut's. False where
// none serves it: a top-k past the walk's bound, a history past theirs.
func (s *Instance) cpuDraw(logits []float32, smp *Sampler, u float32) (int32, bool) {
	n := len(logits)
	if smp.TopK > 0 && smp.TopK < n && smp.TopK <= nn.SampleMaxK {
		return s.cpuWalkToken(logits, smp, u)
	}
	return s.cpuWideToken(logits, smp, u)
}

// dist fills d with the distribution smp draws from over logits (a row this
// instance's model produces), from the CPU's composed forms (cpuDist).
func (s *Instance) dist(smp *Sampler, logits []float32, d *specDist) error {
	if !s.cpuDist(logits, smp, d) {
		return s.sampleDeclined("the distribution")
	}
	return nil
}

// topProb is the largest probability of logits' softmax: the draft's
// confidence in its argmax, from the composed greedy sampler and the
// distribution of a draw at temperature 1 with no cut.
func (s *Instance) topProb(logits []float32, d *specDist) (float32, error) {
	one := Sampler{Temp: 1}
	t, ok := s.cpuGreedy(logits)
	if !ok || !s.cpuDist(logits, &one, d) {
		return 0, s.sampleDeclined("the draft's confidence")
	}
	return d.prob(t), nil
}

// cpuDist fills d with the distribution smp draws from over logits, from
// the CPU's composed forms: a top-k draw's candidates and their share of the
// survivors (graph.SampleWalk with Prob), else every token's probability
// (graph.SampleCut with Prob, the row's where no cut fires). False where
// they decline.
func (s *Instance) cpuDist(logits []float32, smp *Sampler, d *specDist) bool {
	n := len(logits)
	if !s.cpuHeadOn() || n != s.c.NVocab {
		return false
	}
	h := &s.cpu.head
	topK := smp.TopK > 0 && smp.TopK < n
	bounded := topK && smp.TopK <= nn.SampleMaxK
	hf := &h.cutProb
	switch {
	case bounded:
		hf = &h.walkProb
	case topK:
		hf = &h.cutKProb
	}
	f, why := hf.formOf()
	if why != composer.DeclineServes {
		h.sampleDeclined[why]++
		return false
	}
	win := s.cpuStageSample(smp)
	topp, minp := smp.cuts()
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpu.dev.Own(f), n: 1}
	b.io[0], b.io[1], b.io[2] = unsafe.Pointer(&logits[0]), win, unsafe.Pointer(&h.pen[0])
	if bounded {
		h.walkArgs = [5]float32{float32(smp.TopK), float32(1 / smp.Temp), topp, minp, 0}
		b.io[3], b.io[4], b.io[5] = unsafe.Pointer(&h.walkArgs[0]), unsafe.Pointer(&h.probVals[0]), unsafe.Pointer(&h.probIDs[0])
	} else {
		d.pall = grow32(d.pall, n)
		words := unsafe.Pointer(&h.cutArgs[0])
		h.cutArgs = [4]float32{float32(1 / smp.Temp), topp, minp, 0}
		if topK {
			words = unsafe.Pointer(&h.cutKArgs[0])
			h.cutKArgs = [5]float32{float32(1 / smp.Temp), topp, minp, 0, float32(smp.TopK)}
		}
		b.io[3], b.io[4], b.io[5] = unsafe.Pointer(&h.ids[0]), words, unsafe.Pointer(&d.pall[0])
	}
	s.cpuRunForm(b, f)
	h.sampled++
	d.dense = !bounded
	if bounded {
		d.ids, d.p = d.ids[:0], d.p[:0]
		for i, id := range h.probIDs {
			d.ids, d.p = append(d.ids, int32(id)), append(d.p, h.probVals[i])
		}
	}
	return true
}

// cpuWideToken is smp's draw with no top-k over logits from the CPU's
// composed row (graph.SampleRow) or, where top-p or min-p is set, its cut
// (graph.SampleCut), or with a top-k past the walk's bound the cut's top-k
// variant, at the uniform u, or false where it declines.
func (s *Instance) cpuWideToken(logits []float32, smp *Sampler, u float32) (int32, bool) {
	if !s.cpuHeadOn() {
		return 0, false
	}
	h := &s.cpu.head
	topp, minp := smp.cuts()
	hf, words := &h.row, unsafe.Pointer(&h.rowArgs[0])
	h.rowArgs = [2]float32{float32(1 / smp.Temp), u}
	switch n := len(logits); {
	case smp.TopK > 0 && smp.TopK < n:
		hf, words = &h.cutK, unsafe.Pointer(&h.cutKArgs[0])
		h.cutKArgs = [5]float32{float32(1 / smp.Temp), topp, minp, u, float32(smp.TopK)}
	case minp > 0 || topp < 1:
		hf, words = &h.cut, unsafe.Pointer(&h.cutArgs[0])
		h.cutArgs = [4]float32{float32(1 / smp.Temp), topp, minp, u}
	}
	f, why := hf.formOf()
	if why != composer.DeclineServes {
		h.sampleDeclined[why]++
		return 0, false
	}
	win := s.cpuStageSample(smp)
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpu.dev.Own(f), n: 1}
	b.io[0], b.io[1], b.io[2] = unsafe.Pointer(&logits[0]), win, unsafe.Pointer(&h.pen[0])
	b.io[3], b.io[4], b.io[5] = unsafe.Pointer(&h.ids[0]), words, unsafe.Pointer(&h.token[0])
	s.cpuRunForm(b, f)
	h.sampled++
	return int32(h.token[0]), true
}

// cpuWalkToken is smp's top-k draw over logits from the CPU's composed walk
// (graph.SampleWalk) at the uniform u smp drew for the call, or false where
// it declines: a k or a history past the bounds the walk was formed for.
func (s *Instance) cpuWalkToken(logits []float32, smp *Sampler, u float32) (int32, bool) {
	if !s.cpuHeadOn() {
		return 0, false
	}
	h := &s.cpu.head
	f, why := h.walk.formOf()
	if why == composer.DeclineServes && smp.TopK > nn.SampleMaxK {
		why = composer.DeclineCaller
	}
	if why != composer.DeclineServes {
		h.sampleDeclined[why]++
		return 0, false
	}
	win := s.cpuStageSample(smp)
	topp, minp := smp.cuts()
	h.walkArgs = [5]float32{float32(smp.TopK), float32(1 / smp.Temp), topp, minp, u}
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpu.dev.Own(f), n: 1}
	b.io[0], b.io[1], b.io[2] = unsafe.Pointer(&logits[0]), win, unsafe.Pointer(&h.pen[0])
	b.io[3], b.io[4] = unsafe.Pointer(&h.walkArgs[0]), unsafe.Pointer(&h.token[0])
	s.cpuRunForm(b, f)
	h.sampled++
	return int32(h.token[0]), true
}

// cpuStageSample writes smp's penalty where the composed samplers read it
// and is the window row they read: smp's count row, or a row holding no
// token where the penalty is off.
func (s *Instance) cpuStageSample(smp *Sampler) unsafe.Pointer {
	h := &s.cpu.head
	h.pen[0] = smp.penalty()
	if h.pen[0] == 1 {
		return unsafe.Pointer(&h.noWin[0])
	}
	return unsafe.Pointer(&smp.window(s.c.NVocab)[0])
}

// cpuFormCount is how many forms the model's CPU cache holds, refused ones
// among them: a step that forms nothing leaves it as it was.
func (s *Instance) cpuFormCount() int { return s.eng.cpu.Forms(s.m.Id).Count() }

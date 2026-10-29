package model

import (
	"math"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// The head and the sampler on the CPU device: where no card holds the head,
// the State's logits come from what the composer forms for the head's graph
// (graph.Head over the row the streams collapse to, the norm, the
// projection and what follows it), a step's several rows from its form of
// cpuRowsCap rows, a label pick from the head projecting the picked rows
// alone, and a greedy token or a top-k draw's candidates from what it forms
// for the sampler's (graph.Sample, graph.SampleTopK). Every one is composed
// once for the State at its placement (cpuPlaceHead) and counted served or
// declined by name (CPUHeadSteps, CPUSampleSteps); a step forms nothing.

// cpuHeadState is a State's composed head and samplers on the CPU and
// their census.
type cpuHeadState struct {
	// head is the head's form, hidden the one also writing the norm's
	// rows, rows the latter at cpuRowsCap rows, pick the label pick's,
	// sample the greedy sampler's and topK the candidates'.
	head, hidden, rows, pick, sample, topK cpuHeadForm
	served, sampled                        int
	declined, sampleDeclined               [jit.NumDeclines]int
	token                                  [1]uint32
	args                                   []uint32
	pen                                    [1]uint32
	vals                                   []float32
	ids                                    []uint32
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
	f        *cpuBlockForm
	why      jit.Decline
}

// CPUHeadSteps is how many head rows the CPU's composed head served, and
// how many it declined, by reason. With the composed CPU off nothing is
// counted.
func (s *State) CPUHeadSteps() (served int, declined [jit.NumDeclines]int) {
	return s.cpu.head.served, s.cpu.head.declined
}

// CPUSampleSteps is CPUHeadSteps for the composed samplers: a greedy token
// or a top-k draw's candidates.
func (s *State) CPUSampleSteps() (served int, declined [jit.NumDeclines]int) {
	return s.cpu.head.sampled, s.cpu.head.sampleDeclined
}

// cpuPlaceHead composes the head's and the samplers' graphs for the CPU
// device, at the State's placement, with the buffers their calls bind.
func (s *State) cpuPlaceHead() {
	h := &s.cpu.head
	fg := s.m.frontGraphs()
	s.cpuComposeHead(&h.head, fg.cpuHead, jit.DeclineNoGraph, 1)
	s.cpuComposeHead(&h.hidden, fg.cpuHidden, jit.DeclineNoGraph, 1)
	s.cpuComposeHead(&h.rows, fg.cpuHidden, jit.DeclineNoGraph, cpuRowsCap)
	none := jit.DeclineNoGraph
	if fg.cpuHead != nil {
		none = jit.DeclineHeadBias
	}
	s.cpuComposeHead(&h.pick, fg.pickHead, none, 1)
	s.cpuComposeHead(&h.sample, fg.cpuSample, jit.DeclineNoSample, 1)
	s.cpuComposeHead(&h.topK, fg.cpuTopK, jit.DeclineNoSample, 1)
	d, nv := s.c.NEmbd, s.c.NVocab
	if f := h.rows.f; f != nil {
		h.hid = make([]float32, f.rows*d)
		if u := f.unit; u > 1 {
			h.xpad, h.lpad, h.hpad = make([]float32, u*d), make([]float32, u*nv), make([]float32, u*d)
		}
	}
	if h.pick.f != nil {
		h.pickIDs, h.pickVals = make([]uint32, nn.PickMax), make([]float32, nn.PickMax)
	}
	if h.topK.f != nil {
		h.args = make([]uint32, 2+nn.SampleMaxHist)
		h.vals, h.ids = make([]float32, nn.SampleMaxK), make([]uint32, nn.SampleMaxK)
	}
}

// cpuComposeHead composes graph g for the CPU at rows rows into hf, or
// says why not: none where the front end states no graph.
func (s *State) cpuComposeHead(hf *cpuHeadForm, g *graph.Graph, none jit.Decline, rows int) {
	s.cpuComposeHeadAs(hf, g, rows, nil)
	if g == nil {
		hf.why = none
	}
}

// cpuComposeHeadAs is cpuComposeHead with the formats over states the
// weights of its roles are held in, in place of the container's.
func (s *State) cpuComposeHeadAs(hf *cpuHeadForm, g *graph.Graph, rows int, over map[jlm.Role]jlm.Type) {
	*hf = cpuHeadForm{composed: true}
	switch {
	case !jit.HostCPULowers():
		hf.why = jit.DeclineISA
	case g == nil:
		hf.why = jit.DeclineNoGraph
	default:
		f, err := s.m.formCPUGraph(s, g, rows, over)
		if err != nil {
			hf.why = s.cpuFormDecline(err)
			return
		}
		hf.f = f
	}
}

// formOf is hf's form, or why it does not serve: not composed at the
// State's placement, or declined there.
func (hf *cpuHeadForm) formOf() (*cpuBlockForm, jit.Decline) {
	if !hf.composed {
		return nil, jit.DeclineUnplaced
	}
	return hf.f, hf.why
}

// formCPUGraph is graph g of the model's head on the CPU at rows rows
// through the model's keyed cache: its weights the model's own
// (jlm.DenseBlock), its inputs and outputs buffers of their own.
func (m *Model) formCPUGraph(s *State, g *graph.Graph, rows int, over map[jlm.Role]jlm.Type) (*cpuBlockForm, error) {
	formats := map[jlm.Role]jlm.Type{}
	f := m.Composing()
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if t, ok := over[w.Role]; ok {
				formats[w.Role] = t
			} else if e, ok := f.Find(w.Role, jlm.DenseBlock, -1); ok {
				formats[w.Role] = e.Type
			}
		}
	}
	req := jit.Request{Graph: g, Formats: formats, Step: jit.Step{Rows: rows, Page: 1, Keys: 1, ActWindow: 32},
		Device: s.composedCPU(), Apart: true}
	return m.cpuFormOf(req, g, rows, true, false, false)
}

// cpuHeadOn says the composed CPU path is on for this State.
func (s *State) cpuHeadOn() bool { return s.m.opt.composedCPU && s.cpu.args != nil }

// cpuHead runs the head on the CPU the composer's way: the norm of the row
// x, its projection and what follows it into logits, and the norm's rows
// into hidden where it is given, or says why not.
func (s *State) cpuHead(x, logits, hidden []float32) jit.Decline {
	if !s.cpuHeadOn() {
		return jit.DeclineCaller
	}
	h := &s.cpu.head
	hf := &h.head
	if hidden != nil {
		hf = &h.hidden
	}
	f, why := hf.formOf()
	if why == jit.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why != jit.DeclineServes {
		h.declined[why]++
		return why
	}
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpuOwn(f), n: 1}
	b.io[0], b.io[1] = unsafe.Pointer(&x[0]), unsafe.Pointer(&logits[0])
	if s.cpu.fault == cpuFaultHead {
		b.io[0] = unsafe.Pointer(&s.h[0])
	}
	if hidden != nil {
		b.io[2] = unsafe.Pointer(&hidden[0])
	}
	for i := range f.codes {
		s.cpuCall(b, &f.codes[i])
	}
	h.served++
	return jit.DeclineServes
}

// cpuHeadRows is the head over the first n rows of x, NEmbd apart, into the
// rows of logits, and the norm's rows into hidden where it is given (n rows
// of it), through the rows' form a call of at most its rows, or says why
// not. A call's rows past a whole number of the form's units run from
// copies, so its units write no row past the caller's.
func (s *State) cpuHeadRows(x, logits, hidden []float32, n int) jit.Decline {
	if !s.cpuHeadOn() {
		return jit.DeclineCaller
	}
	h := &s.cpu.head
	f, why := h.rows.formOf()
	if why == jit.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why != jit.DeclineServes {
		h.declined[why] += n
		return why
	}
	d, nv := s.c.NEmbd, s.c.NVocab
	for r0 := 0; r0 < n; {
		k := min(f.rows, n-r0)
		u := max(f.unit, 1)
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
	return jit.DeclineServes
}

// cpuHeadCall runs the rows' form f over k rows of x into logits and the
// norm's rows hidden.
func (s *State) cpuHeadCall(f *cpuBlockForm, x, logits, hidden unsafe.Pointer, k int) {
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpuOwn(f), n: k}
	b.io[0], b.io[1], b.io[2] = x, logits, hidden
	if s.cpu.fault == cpuFaultHead {
		b.io[0] = unsafe.Pointer(&s.cpu.head.hid[0])
	}
	s.cpu.call[0] = uint32(k)
	for i := range f.codes {
		s.cpuCall(b, &f.codes[i])
	}
}

// cpuPick is the label pick on the CPU: the head projecting the State's
// picked ids (HostPick) alone from the row x into HostPickVals, the
// logits left as they were, or why not.
func (s *State) cpuPick(x []float32) jit.Decline {
	s.HostPicked = false
	if !s.cpuHeadOn() {
		return jit.DeclineCaller
	}
	h := &s.cpu.head
	f, why := h.pick.formOf()
	if why == jit.DeclineServes {
		why = s.cpuWeights(f, -1, nil)
	}
	if why == jit.DeclineServes && (len(s.HostPick) > len(h.pickIDs) || len(s.HostPickVals) < len(s.HostPick)) {
		why = jit.DeclineCaller
	}
	if why != jit.DeclineServes {
		h.declined[why]++
		return why
	}
	clear(h.pickIDs)
	for i, id := range s.HostPick {
		h.pickIDs[i] = uint32(id)
	}
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpuOwn(f), n: 1}
	b.io[0], b.io[1], b.io[2] = unsafe.Pointer(&x[0]), unsafe.Pointer(&h.pickIDs[0]), unsafe.Pointer(&h.pickVals[0])
	if s.cpu.fault == cpuFaultHead {
		b.io[0] = unsafe.Pointer(&s.h[0])
	}
	for i := range f.codes {
		s.cpuCall(b, &f.codes[i])
	}
	copy(s.HostPickVals, h.pickVals[:len(s.HostPick)])
	s.HostPicked = true
	h.served++
	return jit.DeclineServes
}

// cpuGreedy is the token of the largest of logits, the lower id of two
// equal, from the CPU's composed sampler, or false where it declines.
func (s *State) cpuGreedy(logits []float32) (int32, bool) {
	if !s.cpuHeadOn() {
		return 0, false
	}
	h := &s.cpu.head
	f, why := h.sample.formOf()
	if why != jit.DeclineServes {
		h.sampleDeclined[why]++
		return 0, false
	}
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpuOwn(f), n: 1}
	b.io[0], b.io[1] = unsafe.Pointer(&logits[0]), unsafe.Pointer(&h.token[0])
	for i := range f.codes {
		s.cpuCall(b, &f.codes[i])
	}
	h.sampled++
	return int32(h.token[0]), true
}

// cpuCandidates is smp's k best candidates of logits after its repeat
// penalty, best first, the lower id on a tie, from the CPU's composed
// sampler, or false where it declines.
func (s *State) cpuCandidates(logits []float32, smp *Sampler, k int) ([]float32, []uint32, bool) {
	if !s.cpuHeadOn() {
		return nil, nil, false
	}
	h := &s.cpu.head
	f, why := h.topK.formOf()
	if why == jit.DeclineServes && (k > len(h.vals) || len(smp.hist) > nn.SampleMaxHist) {
		why = jit.DeclineCaller
	}
	if why != jit.DeclineServes {
		h.sampleDeclined[why]++
		return nil, nil, false
	}
	s.sampleArgs = smp.DeviceArgs(s.sampleArgs)
	for i := range h.args {
		h.args[i] = math.MaxUint32
		if i < len(s.sampleArgs) {
			h.args[i] = s.sampleArgs[i]
		}
	}
	h.pen[0] = h.args[1]
	b := &s.cpu.bound
	*b = cpuBound{own: s.cpuOwn(f), n: 1}
	b.io[0], b.io[1], b.io[2] = unsafe.Pointer(&logits[0]), unsafe.Pointer(&h.args[0]), unsafe.Pointer(&h.pen[0])
	b.io[3], b.io[4] = unsafe.Pointer(&h.vals[0]), unsafe.Pointer(&h.ids[0])
	for i := range f.codes {
		s.cpuCall(b, &f.codes[i])
	}
	h.sampled++
	return h.vals[:k], h.ids[:k], true
}

// cpuFormCount is how many forms the model's CPU cache holds, refused ones
// among them: a step that forms nothing leaves it as it was.
func (m *Model) cpuFormCount() int {
	c := &m.cpuForms
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.m) + len(c.err)
}

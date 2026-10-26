package model

import (
	"math"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/cpu"
)

// The head and the sampler on the CPU device: where no card holds the head,
// the State's logits come from what the composer forms for the head's graph
// (graph.Head, the norm, the projection and what follows it), and a greedy
// token or a top-k draw's candidates from what it forms for the sampler's
// (graph.Sample, graph.SampleTopK), each composed once for the State and
// counted served or declined by name (CPUHeadSteps, CPUSampleSteps).
// Each graph is composed at the first step that asks for it: a State that
// never draws a top-k sample never forms its sampler.

// cpuHeadState is a State's composed head and samplers on the CPU and
// their census.
type cpuHeadState struct {
	// head is the head's form, hidden the one also writing the norm's rows,
	// sample the greedy sampler's and topK the candidates', each composed
	// at the first step that asks for it.
	head, hidden, sample, topK cpuHeadForm
	served, sampled            int
	declined, sampleDeclined   [jit.NumDeclines]int
	token                      [1]uint32
	args                       []uint32
	pen                        [1]uint32
	vals                       []float32
	ids                        []uint32
}

// cpuHeadForm is one of the head's graphs composed for the CPU, or why
// not.
type cpuHeadForm struct {
	composed bool
	f        *cpuBlockForm
	why      jit.Decline
}

// CPUHeadSteps is how many head steps the CPU's composed head served, and
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

// cpuComposeHead is graph g composed for the CPU device into hf, once for the
// State, or why not: none where the front end states no graph.
func (s *State) cpuComposeHead(hf *cpuHeadForm, g *graph.Graph, none jit.Decline) (*cpuBlockForm, jit.Decline) {
	if hf.composed {
		return hf.f, hf.why
	}
	hf.composed = true
	switch {
	case cpu.CPU().ISA()&cpu.ISATierAVX2 != cpu.ISATierAVX2:
		hf.why = jit.DeclineISA
	case g == nil:
		hf.why = none
	default:
		f, err := s.m.formCPUGraph(s, g)
		if err != nil {
			hf.why = s.cpuFormDecline(err)
			break
		}
		hf.f = f
	}
	return hf.f, hf.why
}

// formCPUGraph is graph g of the model's head on the CPU through the
// model's keyed cache: its weights the model's own (jlm.DenseBlock), its
// inputs and outputs buffers of their own.
func (m *Model) formCPUGraph(s *State, g *graph.Graph) (*cpuBlockForm, error) {
	formats := map[jlm.Role]jlm.Type{}
	f := m.Composing()
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if e, ok := f.Find(w.Role, jlm.DenseBlock, -1); ok {
				formats[w.Role] = e.Type
			}
		}
	}
	workers := 1
	if s.jit != nil {
		workers = s.jit.Workers()
	}
	req := jit.Request{Graph: g, Formats: formats, Step: jit.Step{Rows: 1, Page: 1, Keys: 1, ActWindow: 32},
		Device: composedCPUDevice(workers)}
	return m.cpuFormOf(req, g, 1, true)
}

// cpuHead runs the head on the CPU the composer's way: the norm of the row
// x, its projection and what follows it into logits, and the norm's rows
// into hidden where it is given, or says why not.
func (s *State) cpuHead(x, logits, hidden []float32) jit.Decline {
	if !s.m.opt.composedCPU || s.cpu.args == nil {
		return jit.DeclineCaller
	}
	h := &s.cpu.head
	fg := s.m.frontGraphs()
	var f *cpuBlockForm
	var why jit.Decline
	if hidden != nil {
		f, why = s.cpuComposeHead(&h.hidden, fg.headHidden, jit.DeclineNoGraph)
	} else {
		f, why = s.cpuComposeHead(&h.head, fg.head, jit.DeclineNoGraph)
	}
	if why == jit.DeclineServes {
		why = s.cpuWeights(f, nil)
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

// cpuHeads is cpuHead over the first n rows of the residual x into the
// rows of logits, a row a call, reporting whether the CPU served them: the
// first row's decline leaves every row to main's head.
func (s *State) cpuHeads(x, logits []float32, n int) bool {
	w, nv := s.c.ResidW(), s.c.NVocab
	for r := range n {
		if s.cpuHead(x[r*w:(r+1)*w], logits[r*nv:(r+1)*nv], nil) != jit.DeclineServes {
			return false
		}
	}
	return true
}

// cpuGreedy is the token of the largest of logits, the lower id of two
// equal, from the CPU's composed sampler, or false where it declines.
func (s *State) cpuGreedy(logits []float32) (int32, bool) {
	if !s.m.opt.composedCPU || s.cpu.args == nil {
		return 0, false
	}
	h := &s.cpu.head
	f, why := s.cpuComposeHead(&h.sample, s.m.frontGraphs().sample, jit.DeclineNoSample)
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
	if !s.m.opt.composedCPU || s.cpu.args == nil {
		return nil, nil, false
	}
	h := &s.cpu.head
	f, why := s.cpuComposeHead(&h.topK, s.m.frontGraphs().topK, jit.DeclineNoSample)
	if h.args == nil {
		h.args = make([]uint32, 2+nn.SampleMaxHist)
		h.vals, h.ids = make([]float32, nn.SampleMaxK), make([]uint32, nn.SampleMaxK)
	}
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

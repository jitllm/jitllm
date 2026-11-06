package engine

// stepBuf is a ragged step's scratch across sessions (stepRouted): the
// residual rows, their rotary tables and the wanted rows' logits. The model
// holds one, and whichever instance leads a step borrows it for the step and
// hands it back, so the scratch is one step's whatever instance leads -- not
// one step's per instance that ever led, which a pool of instances kept at
// its widest.
type stepBuf struct {
	bx, bcs, bcsSWA, blogits []float32
}

func (b *stepBuf) bytes() uint64 {
	return 4 * uint64(cap(b.bx)+cap(b.bcs)+cap(b.bcsSWA)+cap(b.blogits))
}

// borrowStep gives s its maker's step scratch for a step it leads. What s
// holds of its own goes to the spare prompt buffers, as a closing
// instance's does, so an instance that prefilled alone does not keep a second set.
func (s *Instance) borrowStep() {
	s.eng.host.mu.Lock()
	b := s.eng.host.step
	s.eng.host.step = nil
	s.eng.host.mu.Unlock()
	s.keepSpare()
	if b == nil {
		s.blogits = nil
		s.stepb = &stepBuf{}
		return
	}
	s.bx, s.bcs, s.bcsSWA, s.blogits = b.bx, b.bcs, b.bcsSWA, b.blogits
	*b = stepBuf{}
	s.stepb = b
}

// returnStep hands the step scratch s borrowed back.
func (s *Instance) returnStep() {
	// The holder borrowStep emptied, refilled: a step allocates nothing.
	b := s.stepb
	s.stepb = nil
	*b = stepBuf{bx: s.bx, bcs: s.bcs, bcsSWA: s.bcsSWA, blogits: s.blogits}
	s.bx, s.bcs, s.bcsSWA, s.blogits = nil, nil, nil, nil
	s.eng.host.mu.Lock()
	s.eng.host.step = b
	s.eng.host.mu.Unlock()
}

// ScratchBytes is the scratch this instance holds beyond its single-token
// decode: prompt and batch rows, their rotary tables and batch logits. An
// instance that only decodes, or that leads routed steps, holds none of it
// between calls.
func (s *Instance) ScratchBytes() uint64 {
	b := stepBuf{bx: s.bx, bcs: s.bcs, bcsSWA: s.bcsSWA, blogits: s.blogits}
	return b.bytes()
}

// StepScratchBytes is the scratch the model holds for routed steps across
// sessions and for the next session's prompt (stepBuf, promptBuf).
func (s *Instance) StepScratchBytes() uint64 {
	s.eng.host.mu.Lock()
	defer s.eng.host.mu.Unlock()
	var n uint64
	if s.eng.host.step != nil {
		n += s.eng.host.step.bytes()
	}
	if p := s.eng.host.spare; p != nil {
		n += 4 * uint64(cap(p.bx)+cap(p.bcs)+cap(p.bcsSWA))
	}
	return n
}

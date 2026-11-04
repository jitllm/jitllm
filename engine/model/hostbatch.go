package model

// hostBatch is a prompt chunk's host scratch: what growBatch and growMoEBatch
// size for a block the host runs. A closing instance hands it to the model with
// its promptBuf, and the next instance's first prompt takes it back.
//
// It is the largest thing a request allocates: tens of MiB at a 512-row
// chunk, against a decode token's few KiB, so a server opening an instance per
// request was handing the collector that much per request -- and under a memory limit near the weight budget, that is
// a cycle per request (AGENTS.md RULE 2f).
//
// Nothing here is read before a chunk writes it, so it is reused as it is,
// as promptBuf's residual rows always were. TestReusedPromptScratchIsNotRead
// poisons a handed-back set with NaN and demands the next session's logits are
// bit-identical to a fresh model's.
type hostBatch struct {
	bh, battf               []float32
	qgate, ogate, bqf, bxbf []float32
	bmRow                   []int32
	bmW, bmX, bmY, bmG, bmU []float32
}

// scratch is n float32s with room for c, from b when it is big enough.
func scratch(b []float32, n, c int) []float32 {
	c = max(c, n)
	if cap(b) >= c {
		return b[:n]
	}
	return make([]float32, n, c)
}

// takeHostBatch moves this instance's host scratch out, leaving it to regrow.
func (s *Instance) takeHostBatch() hostBatch {
	h := hostBatch{
		bh: s.bh, battf: s.battf, qgate: s.qgate, ogate: s.ogate, bqf: s.bqf, bxbf: s.bxbf,
		bmRow: s.bmRow, bmW: s.bmW, bmX: s.bmX, bmY: s.bmY, bmG: s.bmG, bmU: s.bmU,
	}
	// One the instance took from the model and never grew into is still worth
	// handing on.
	if h.bh == nil {
		h = s.spareHost
	}
	if h.bmRow == nil {
		h.bmRow, h.bmW, h.bmX, h.bmY, h.bmG, h.bmU = s.spareHost.bmRow, s.spareHost.bmW,
			s.spareHost.bmX, s.spareHost.bmY, s.spareHost.bmG, s.spareHost.bmU
	}
	s.bh, s.battf, s.qgate, s.ogate, s.bqf, s.bxbf = nil, nil, nil, nil, nil, nil
	s.bmRow, s.bmW, s.bmX, s.bmY, s.bmG, s.bmU = nil, nil, nil, nil, nil, nil
	s.spareHost = hostBatch{}
	return h
}

// bytes is the host scratch's size.
func (h *hostBatch) bytes() uint64 {
	n := 0
	for _, b := range [][]float32{h.bh, h.battf, h.qgate, h.ogate, h.bqf, h.bxbf, h.bmW, h.bmX, h.bmY, h.bmG, h.bmU} {
		n += cap(b)
	}
	return 4 * uint64(n+cap(h.bmRow))
}

// stepBuf is a ragged step's scratch across host sessions (stepHost): the
// residual rows, their rotary tables, the host batch scratch, the scores
// rows and the wanted rows' logits. The model holds one, and whichever instance
// leads a step borrows it for the step and hands it back, so the scratch is
// one step's whatever instance leads -- not one step's per instance that ever led,
// which a pool of instances kept at its widest.
type stepBuf struct {
	bx, bcs, bcsSWA, bathf, blogits []float32
	bsel                            []int32
	bw                              []float32
	host                            hostBatch
}

func (b *stepBuf) bytes() uint64 {
	n := cap(b.bx) + cap(b.bcs) + cap(b.bcsSWA) + cap(b.bathf) + cap(b.blogits) + cap(b.bsel) + cap(b.bw)
	return 4*uint64(n) + b.host.bytes()
}

// borrowStep gives s the model's step scratch for a step it leads. What s
// holds of its own goes to the model's spare prompt buffers, as a closing
// instance's does, so an instance that prefilled alone does not keep a second set.
func (s *Instance) borrowStep() {
	m := s.m
	m.eng.spareMu.Lock()
	b := m.eng.step
	m.eng.step = nil
	m.eng.spareMu.Unlock()
	s.keepSpare()
	h := s.takeHostBatch()
	if b == nil {
		// Start from what s held, which is as good as nothing.
		s.restoreHostBatch(h)
		s.bathf, s.blogits, s.bsel, s.bw = nil, nil, nil, nil
		s.stepb = &stepBuf{}
		return
	}
	s.bx, s.bcs, s.bcsSWA, s.bathf, s.blogits, s.bsel, s.bw = b.bx, b.bcs, b.bcsSWA, b.bathf, b.blogits, b.bsel, b.bw
	s.restoreHostBatch(b.host)
	*b = stepBuf{}
	s.stepb = b
}

// returnStep hands the step scratch s borrowed back to the model.
func (s *Instance) returnStep() {
	// The holder borrowStep emptied, refilled: a step allocates nothing.
	b := s.stepb
	s.stepb = nil
	*b = stepBuf{bx: s.bx, bcs: s.bcs, bcsSWA: s.bcsSWA, bathf: s.bathf, blogits: s.blogits,
		bsel: s.bsel, bw: s.bw, host: s.takeHostBatch()}
	s.bx, s.bcs, s.bcsSWA, s.bathf, s.blogits, s.bsel, s.bw = nil, nil, nil, nil, nil, nil, nil
	m := s.m
	m.eng.spareMu.Lock()
	m.eng.step = b
	m.eng.spareMu.Unlock()
}

// restoreHostBatch puts h back as s's host scratch (takeHostBatch's inverse).
func (s *Instance) restoreHostBatch(h hostBatch) {
	s.bh, s.battf, s.qgate, s.ogate, s.bqf, s.bxbf = h.bh, h.battf, h.qgate, h.ogate, h.bqf, h.bxbf
	s.bmRow, s.bmW, s.bmX, s.bmY, s.bmG, s.bmU = h.bmRow, h.bmW, h.bmX, h.bmY, h.bmG, h.bmU
}

// ScratchBytes is the host scratch this instance holds beyond its single-token
// decode: prompt and batch rows, scores rows and batch logits. An instance that
// only decodes, or that leads host steps, holds none of it between calls.
func (s *Instance) ScratchBytes() uint64 {
	h := hostBatch{bh: s.bh, battf: s.battf, qgate: s.qgate, ogate: s.ogate, bqf: s.bqf, bxbf: s.bxbf,
		bmRow: s.bmRow, bmW: s.bmW, bmX: s.bmX, bmY: s.bmY, bmG: s.bmG, bmU: s.bmU}
	b := stepBuf{bx: s.bx, bcs: s.bcs, bcsSWA: s.bcsSWA, bathf: s.bathf, blogits: s.blogits,
		bsel: s.bsel, bw: s.bw, host: h}
	return b.bytes() + s.spareHost.bytes()
}

// StepScratchBytes is the scratch the model holds for host steps across
// sessions and for the next session's prompt (stepBuf, promptBuf).
func (m *Model) StepScratchBytes() uint64 {
	m.eng.spareMu.Lock()
	defer m.eng.spareMu.Unlock()
	var n uint64
	if m.eng.step != nil {
		n += m.eng.step.bytes()
	}
	if p := m.eng.spare; p != nil {
		n += 4*uint64(cap(p.bx)+cap(p.bcs)+cap(p.bcsSWA)) + p.host.bytes()
	}
	return n
}

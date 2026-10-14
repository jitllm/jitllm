package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit/cpu"
)

// An encoder on a device. An encoder's blocks (BERT, nomic-bert, ModernBERT
// and a decision head after it) are non-causal blocks: a sequence goes in
// whole, every row attends over the call's rows, and the keys die with the
// block -- what a vision tower's blocks are. So they reach a device the way a
// tower's do, through the one placement path: a State over the encoder's
// blocks (the segment, as Tower.newState builds one over the tower's) whose
// SetDeviceLayers offers each block (offerRange), with the plan and weights
// planFor and layerWeightsAt read here, and whose devAt says where each runs.
//
// What runs a block stays where it was: a block the device holds runs there
// in one Layers call per run of placed blocks, the sequence's rows as one
// submission (nn.LayerPlan.NonCausal); a block at home runs on the Embedder's
// generated host kernels as before. A ModernBERT local block attends inside a
// symmetric window of its own per row (nn.RowWindowDevice), its rotary the
// local table handed as Layers' csSWA (nn.LayerPlan.LocalRope).

// encRun marks a State as an encoder segment's. violation is the Embedder's
// (a feature of the graph a test removes), read when the blocks are offered
// so the device runs without it too; empty everywhere but those tests.
type encRun struct {
	violation string
}

// encBlocks is how many blocks an encoder's segment holds: the encoder's, and
// a decision head's after them.
func (m *Model) encBlocks() int {
	if m.enc == nil {
		return 0
	}
	if mb := m.enc.mb; mb != nil {
		return len(mb.blocks) + len(mb.head)
	}
	return len(m.enc.blocks)
}

// encMaxRows is the longest sequence an encoder's blocks take on a device:
// BERT's position table, or the context it was trained at.
func (m *Model) encMaxRows() int {
	if m.enc.mb == nil && !m.enc.swiglu {
		return m.enc.pos.rows
	}
	return m.Cfg.NCtx
}

// encState is the segment State an Embedder places its encoder through, on
// the Embedder's JIT, which it borrows: the device's single matvecs are not
// the encoder's, whose host half runs on the Embedder's own kernels.
func (m *Model) encState(j *nn.JIT) *State {
	c := m.Cfg
	n := m.encMaxRows()
	s := &State{m: m, c: c, lo: 0, hi: m.encBlocks(), maxSeq: n, reqSeq: n, nseq: 1,
		prof: m.opt.profile, outW: &tensor{}, relocate: true}
	s.holdHostRuns()
	s.enc = &encRun{}
	s.jit = j
	// The segment keeps no history: a non-causal block's keys live one call
	// deep, on the device. The cache is the shape every State has, empty.
	s.kvl = kvLayout{maxSeq: 1, nKV: c.NKVHead, headDim: c.HeadDim}
	s.kv = newKVCacheRange(c, nextCacheID(), 1, s.kvl, nil, align(1, kvKeyTile), s.lo, s.hi)
	s.attnG = s.jit.AttnSetFor(c.HeadDim, c.HeadDim, s.kvl.Stride(), cpu.KVF32)
	s.attnL = s.attnG
	s.attStride = attStride(1)
	s.memBudget = sched.MemBudget()
	return s
}

// borrowsJIT reports whether this State runs on another's JIT (a vision
// segment's on its text State's, an encoder segment's on its Embedder's),
// which that owner attaches to a device and closes.
func (s *State) borrowsJIT() bool {
	return (s.vis != nil && s.vis.borrowed) || s.enc != nil
}

// SetDeviceLayers places the encoder's blocks on d through its segment State
// (State.SetDeviceLayers): max caps how many are offered, -1 offers every one.
// A decoder embedding model's session is placed the same way.
func (e *Embedder) SetDeviceLayers(d nn.Device, max int) error {
	if e.st != nil {
		// A bidirectional decoder (EmbeddingGemma) is not offered: its
		// blocks would run under the decoder path's causal mask on a device
		// (embedPrefill refuses that), so it embeds on the host until they
		// are planned non-causal.
		if e.m.Cfg.NonCausal && d != nil {
			e.st.noteDecline(fmt.Sprintf("%s's decoder blocks attend bidirectionally, which the "+
				"device's decoder path does not plan; it embeds on the host", e.m.Cfg.Arch))
			return nil
		}
		return e.st.SetDeviceLayers(d, max)
	}
	if e.seg == nil {
		if d == nil {
			return nil
		}
		e.seg = e.m.encState(e.jit)
	}
	e.seg.enc.violation = e.violation
	return e.seg.SetDeviceLayers(d, max)
}

// DeviceBlocks is how many of the encoder's blocks run on a device.
func (e *Embedder) DeviceBlocks() int {
	switch {
	case e.st != nil:
		return e.st.GPULayers()
	case e.seg != nil:
		return e.seg.GPULayers()
	}
	return 0
}

// DeviceDeclines is why the device refused the blocks it did not take.
func (e *Embedder) DeviceDeclines() []DeviceDecline {
	switch {
	case e.st != nil:
		return e.st.DeviceDeclines()
	case e.seg != nil:
		return e.seg.DeviceDeclines()
	}
	return nil
}

// onDevice reports whether encoder block li runs on the device.
func (e *Embedder) onDevice(li int) bool {
	return e.seg != nil && e.seg.ld != nil && e.seg.devAt(li)
}

// devRun runs the placed blocks from li on, up to stop, over x's n rows in
// one Layers call, and returns the block after the last it ran. cs and csSWA
// are the rows' rotary tables (nil for none).
func (e *Embedder) devRun(li, stop, n int, x, cs, csSWA []float32) (int, error) {
	s := e.seg
	hi := li + 1
	for hi < stop && s.devAt(hi) {
		hi++
	}
	if !s.ld.Layers(li, hi, 0, n, x, cs, csSWA, nil) {
		// A device that fails mid-sequence has left the residual in an
		// unknown state, so this is an error rather than a fallback.
		why := "no reason given"
		if er, ok := s.ld.(nn.ErrReporter); ok && er.Err() != "" {
			why = er.Err()
		}
		return li, fmt.Errorf("model: the device failed encoder block(s) [%d,%d): %s", li, hi, why)
	}
	return hi, nil
}

// devPrepare readies the device for a sequence of n rows before any of its
// blocks run there: the non-causal set sized for the rows (nn.RowsReserver),
// a placement it cannot hold sending blocks home, highest first, as a tower's
// picture does; and a local block's windows, one a row (nn.RowWindowDevice).
// window is the keys either side a local row sees, 0 for none.
func (e *Embedder) devPrepare(n, window int) error {
	s := e.seg
	if s == nil || s.ld == nil || s.devCount() == 0 {
		return nil
	}
	if rr, ok := s.ld.(nn.RowsReserver); ok {
		for !rr.ReserveRows(n) {
			hi := -1
			for li := s.lo; li < s.hi; li++ {
				if s.devAt(li) {
					hi = li
				}
			}
			if hi < 0 {
				break
			}
			s.SetGPULayers(hi)
		}
	}
	if window <= 0 || s.devCount() == 0 {
		return nil
	}
	rw, ok := s.ld.(nn.RowWindowDevice)
	if !ok {
		return fmt.Errorf("model: the device holds a windowed encoder block and takes no row windows")
	}
	wins := e.wins[:0]
	for i := 0; i < n; i++ {
		wins = append(wins, nn.KeyRun{Lo: max(0, i-window), Hi: min(n, i+window+1)})
	}
	e.wins = wins
	if !rw.SetRowWindows(wins) {
		why := ""
		if er, ok := s.ld.(nn.ErrReporter); ok {
			why = er.Err()
		}
		return fmt.Errorf("model: the device refused the encoder's %d row windows: %s", n, why)
	}
	return nil
}

// zeros is a zero row of n, the bias a LayerNorm without one is handed on a
// device (whose LayerNorm kernel reads one), kept so a plan asks no
// allocation per offer.
func (m *Model) encZeros(n int) []float32 {
	if len(m.enc.zeros) < n {
		m.enc.zeros = make([]float32, n)
	}
	return m.enc.zeros[:n]
}

// encPlan is encoder block li's plan for a device.
func (s *State) encPlan(li int) *nn.LayerPlan {
	m, c := s.m, s.c
	p := &nn.LayerPlan{
		Model:     m.id,
		NonCausal: true,
		NEmbd:     c.NEmbd, NHead: c.NHead, NKVHead: c.NHead, HeadDim: c.HeadDim,
		NFFN: c.NFFN, MaxSeq: s.maxSeq,
		ActWin:    s.jit.ActWindow(),
		RMSEps:    c.RMSEps,
		LayerNorm: true,
	}
	enc := m.enc
	if mb := enc.mb; mb != nil {
		v := s.enc.violation
		if li >= len(mb.blocks) {
			// The decision head's: a pre-norm torch TransformerEncoderLayer,
			// no positions, an ungated ReLU MLP of its own width.
			p.NHead, p.NKVHead = mb.headHeads, mb.headHeads
			if v == "encoder-heads" {
				p.NHead, p.NKVHead = c.NHead, c.NHead
			}
			p.HeadDim = c.NEmbd / p.NHead
			p.NFFN, p.UngatedFFN, p.Act = mb.headFFN, true, nn.ActReLU
			p.NoPosEnc = true
			return p
		}
		p.Act = nn.ActGELUErf
		p.NRot, p.RopeNeox = c.HeadDim, true
		// The local blocks rotate by the local table (Layers' csSWA) and
		// attend inside their row windows; block 0 has no attention norm.
		p.SWAPeriod = int(c.SWAPeriod)
		p.SWALocal = mb.local
		p.Windowed = mb.local(li)
		p.NoPreNorm = true
		switch v {
		case "tanh-gelu":
			p.Act = nn.ActGELU
		case "no-window":
			p.Windowed = false
		case "one-base":
			p.SWALocal = func(int) bool { return false }
		}
		return p
	}
	// BERT and nomic-bert: the norms come after each residual add, and the
	// attention reads the block input as it is.
	p.PostResidNorm, p.NoPreNorm = true, true
	p.Act, p.UngatedFFN = nn.ActGELU, true
	if enc.swiglu {
		p.Act, p.UngatedFFN = nn.ActSiLU, false
		p.NRot, p.RopeNeox = c.NRot, c.RopeNeox
	} else {
		p.NoPosEnc = true
	}
	return p
}

// encWeights is encoder block li's weights for a device.
func (s *State) encWeights(li int) nn.LayerWeights {
	m, d := s.m, s.c.NEmbd
	enc := m.enc
	if mb := enc.mb; mb != nil {
		if li >= len(mb.blocks) {
			b := &mb.head[li-len(mb.blocks)]
			return nn.LayerWeights{
				AttnNorm: b.ln1W, AttnNormB: b.ln1B, FFNNorm: b.ln2W, FFNNormB: b.ln2B,
				Wq: wt(b.wq), Wk: wt(b.wk), Wv: wt(b.wv), Wo: wt(b.wo),
				Bq: b.bq, Bk: b.bk, Bv: b.bv, Bo: b.bo,
				Up: wt(b.up), Down: wt(b.down), BUp: b.bUp, BDown: b.bDown,
			}
		}
		b := &mb.blocks[li]
		w := nn.LayerWeights{
			FFNNorm: b.ffnNorm, FFNNormB: m.encZeros(d),
			Wq: wt(b.wq), Wk: wt(b.wk), Wv: wt(b.wv), Wo: wt(b.wo),
			Gate: wt(b.gate), Up: wt(b.up), Down: wt(b.down),
		}
		if b.attnNorm != nil {
			w.AttnNorm, w.AttnNormB = b.attnNorm, m.encZeros(d)
		}
		if s.enc.violation == "up-gated" {
			w.Gate, w.Up = w.Up, w.Gate
		}
		return w
	}
	b := &enc.blocks[li]
	w := nn.LayerWeights{
		// BERT's two norms, each after its residual add: FFNNorm over x +
		// attention, ResidNorm over x + MLP (nn.LayerPlan.PostResidNorm).
		FFNNorm: b.ln1W, FFNNormB: b.ln1B, ResidNorm: b.ln2W, ResidNormB: b.ln2B,
		Wq: wt(b.wq), Wk: wt(b.wk), Wv: wt(b.wv), Wo: wt(b.wo),
		Bq: b.bq, Bk: b.bk, Bv: b.bv, Bo: b.bo,
		Up: wt(b.up), Down: wt(b.down), BUp: b.bUp, BDown: b.bDown,
	}
	if b.gate.e != nil {
		w.Gate = wt(b.gate)
	}
	return w
}

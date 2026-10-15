package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
)

// The audio tower's blocks on a device: each FastConformer block is one
// convolutional-tower block (nn.ConvConformer), a program of generated
// kernels over a scratch of the longest clip's rows, placed through an
// encoder segment State over the audio blocks (the one placement path,
// State.offerRange) and run by ConvLayers, the clip's rows on the card from
// block to block. The front end, the subsampling, the adapter and the
// residual correction stay on the host.

// audioMaxRows is the most prefix rows a clip encodes to: the reference's 30
// seconds.
func (a *audioTower) audioMaxRows() int {
	return a.audioPrefixRows(30 * int(a.cfg.SampleRate))
}

// audioFirst is the model's block index of audio block 0: past the text
// model's blocks and the tower's vision blocks.
func (m *Model) audioFirst() int { return int(m.container.H.NBlocks) + m.audio.base }

// audioPE is the relative positions' sinusoids for a scratch of r rows:
// positions r-1 down to -(r-1), 2r-1 rows of the width, built once (audio.py's
// pos_emb, as a table of the sizes).
func (a *audioTower) audioPE(r int) []float32 {
	if len(a.pe) == (2*r-1)*int(a.cfg.NEmbd) {
		return a.pe
	}
	d := int(a.cfg.NEmbd)
	pe := make([]float32, (2*r-1)*d)
	for i := 0; i < 2*r-1; i++ {
		pos := float64(r - 1 - i)
		for c := 0; c < d/2; c++ {
			div := math.Exp(float64(2*c) * -(math.Log(10000.0) / float64(d)))
			pe[i*d+2*c] = float32(math.Sin(pos * div))
			pe[i*d+2*c+1] = float32(math.Cos(pos * div))
		}
	}
	a.pe = pe
	return pe
}

// audState is the segment State the audio blocks are placed through, on the
// Embedder's JIT.
func (m *Model) audState(j *nn.JIT) *State {
	s := m.encState(j)
	s.lo, s.hi = m.audioFirst(), m.audioFirst()+len(m.audio.blocks)
	// The (empty) cache over this segment's blocks, as encState makes the
	// encoder's: a history no block keeps.
	m.kvPool.put(s.kv)
	s.kv = newKVCacheRange(m.Cfg, nextCacheID(), 1, s.kvl, nil, align(1, kvKeyTile), s.lo, s.hi)
	s.maxSeq, s.reqSeq = m.audio.audioMaxRows(), m.audio.audioMaxRows()
	s.enc.audio = true
	return s
}

// audPlan is audio block li's plan for a device.
func (s *State) audPlan(li int) *nn.LayerPlan {
	a := s.m.audio
	cf := &a.cfg
	d := int(cf.NEmbd)
	r := a.audioMaxRows()
	eps := float64(cf.Eps)
	return &nn.LayerPlan{
		Model: s.m.id, NonCausal: true, NEmbd: d, NHead: int(cf.NHead), NKVHead: int(cf.NHead),
		HeadDim: d / int(cf.NHead), NFFN: int(cf.NFFN), MaxSeq: r, ActWin: s.jit.ActWindow(),
		RMSEps: float64(cf.Eps), LayerNorm: true,
		Conv: &nn.ConvPlan{Kind: nn.ConvConformer, HIn: r, WIn: 1, CIn: d, HOut: r, WOut: 1, COut: d,
			CExp: int(cf.NFFN), Heads: int(cf.NHead), KD: d / int(cf.NHead), KMid: int(cf.ConvKernel),
			Residual: true, Eps: eps, ActWin: s.jit.ActWindow()},
	}
}

// audWeights is audio block li's weights for a device.
func (s *State) audWeights(li int) nn.LayerWeights {
	a := s.m.audio
	b := &a.blocks[li-s.m.audioFirst()]
	w := nn.LayerWeights{
		Wq: wt(b.q), Wk: wt(b.k), Wv: wt(b.v), Wo: wt(b.o), Gate: wt(b.pos),
		Up: wt(b.ff1Up), Down: wt(b.ff1Down), ShGate: wt(b.gluV), ShUp: wt(b.ff2Up), ShDown: wt(b.ff2Down),
	}
	w.SSM.Gate, w.SSM.BA = wt(b.gluG), wt(b.pw2)
	w.Conv = &nn.ConvWeights{Conformer: &nn.ConformerVecs{
		FF1Norm: b.ff1N, FF1NormB: b.ff1NB, FF1UpB: b.ff1UB, FF1DownB: b.ff1DB,
		AttnNorm: b.attN, AttnNormB: b.attNB, QB: b.qB, KB: b.kB, VB: b.vB, OB: b.oB, U: b.u, V: b.vbias,
		ConvNorm: b.convN, ConvNormB: b.convNB, GLUValB: b.gluVB, GLUGateB: b.gluGB,
		DW: b.dw, BNScale: b.bnS, BNShift: b.bnB, PW2B: b.pw2B,
		FF2Norm: b.ff2N, FF2NormB: b.ff2NB, FF2UpB: b.ff2UB, FF2DownB: b.ff2DB,
		OutNorm: b.outN, OutNormB: b.outNB, PE: a.audioPE(a.audioMaxRows()),
	}}
	if s.enc.violation == "audio-u-v-swapped" {
		// The two attention biases change places: the device reads which is
		// the content term's and which the position term's off the vectors.
		cv := w.Conv.Conformer
		cv.U, cv.V = cv.V, cv.U
	}
	return w
}

// setAudioDevice places the audio blocks on d through their segment State,
// every one the device takes; nil takes them home.
func (e *Embedder) setAudioDevice(d nn.Device) error {
	if e.m.audio == nil {
		return nil
	}
	if e.audSeg == nil {
		if d == nil {
			return nil
		}
		e.audSeg = e.m.audState(e.jit)
	}
	e.audSeg.enc.violation = e.violation
	return e.audSeg.SetDeviceLayers(d, -1)
}

// AudioDeviceBlocks is how many of the audio tower's blocks run on a device.
func (e *Embedder) AudioDeviceBlocks() int {
	if e.audSeg == nil {
		return 0
	}
	return e.audSeg.GPULayers()
}

// audioDevRun runs the placed audio blocks from j on over x's t rows in one
// ConvLayers call, and returns the block after the last it ran.
func (e *Embedder) audioDevRun(j int, x []float32, t int) (int, error) {
	s, a := e.audSeg, e.m.audio
	first := e.m.audioFirst()
	lo, hi := first+j, first+j+1
	for hi < first+len(a.blocks) && s.devAt(hi) {
		hi++
	}
	cd, ok := s.ld.(nn.ConvDevice)
	if !ok {
		return j, fmt.Errorf("model: the device holds audio blocks and cannot run a convolutional block")
	}
	d := int(a.cfg.NEmbd)
	out := growA(&e.aud.devOut, t*d)
	if !cd.ConvLayers(lo, hi, x[:t*d], out, -1, nil) {
		why := "no reason given"
		if er, ok := s.ld.(nn.ErrReporter); ok && er.Err() != "" {
			why = er.Err()
		}
		return j, fmt.Errorf("model: the device failed audio block(s) [%d,%d): %s", j, hi-first, why)
	}
	copy(x[:t*d], out)
	return hi - first, nil
}

// audioOnDevice reports whether audio block j runs on the device.
func (e *Embedder) audioOnDevice(j int) bool {
	s := e.audSeg
	return s != nil && s.ld != nil && s.devAt(e.m.audioFirst()+j)
}

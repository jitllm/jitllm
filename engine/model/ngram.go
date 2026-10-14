package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The n-gram embedding: Qwen3.8-Flash-Next's PLE (transformers'
// Qwen4ExpTextPLELayer, llama.cpp's build_ple), on one linear block, before
// its attention mix.
//
// Each position's token and its NGram-1 predecessors are hashed per head
// (Config.NGramMul, NGramOff, NGramVoc; a predecessor at or before an
// end-of-sequence token, or before the sequence, reads as that token) into a
// row of a shared table, which is read a row at a time from the container
// (jlm.RowTable): it is hundreds of millions of rows, of which a position
// reads (NGram-1)*NGramHeadsPer. The rows, concatenated, project to a key
// per stream and one value; each stream's normed key and normed residual give
// a gate, sigmoid of the signed square root of their scaled dot product; the
// gated value is added to the stream, and so is a depthwise convolution over
// it (normed), dilated by NGram, through silu.
//
// The predecessors and the convolution's history are the block's recurrent
// state, after the delta rule's (Config.ngramStateLen), so they page,
// relocate and are prefix-cached with it.

// ngramTable is the container's n-gram table.
type ngramTable struct {
	f        *jlm.File
	e        *jlm.Entry
	typ      quant.Type
	rowBytes int
}

// ngramScratch is a State's n-gram buffers.
type ngramScratch struct {
	rows int
	// per row: the hashed table rows, the gathered embedding, its key and
	// value, the normed residual, the gate per stream and the gated value
	// (normed in place for the convolution once added).
	idx, ids                   []uint64
	konst                      []byte
	hwin, hnext                []uint32
	emb, key, val, q, gate, gv []float32
	gvn, conv, tap             []float32
	raw, scr                   []byte
	dot                        *nn.AttnSet
	// undo and undoPrev are what ngramToDevice moved on, for ngramUndoRows.
	undo     []ngramUndo
	undoPrev []float32
}

// allocNGram sizes the n-gram buffers for one row.
func (s *State) allocNGram() {
	c := s.m.Cfg
	if c.NGramBlock == 0 {
		return
	}
	g := &ngramScratch{}
	s.q4.ng = g
	g.idx = make([]uint64, c.ngramHeads())
	g.ids = make([]uint64, c.NGram)
	g.hwin, g.hnext = make([]uint32, c.NGram), make([]uint32, c.NGram)
	g.konst = nn.NGramConsts(uint64(c.NGramEOS), c.NGramMul, c.NGramOff, c.NGramVoc)
	g.raw = make([]byte, s.m.ngram.rowBytes)
	g.scr = jlm.RowScratch()
	hd := c.HCMult * c.NEmbd
	g.conv, g.tap = make([]float32, hd), make([]float32, hd)
	g.dot = s.attnSetF32(c.NEmbd)
}

// growNGram sizes the per-row buffers for n rows.
func (s *State) growNGram(n int) {
	c := s.m.Cfg
	g := s.q4.ng
	if g == nil || g.rows >= n {
		return
	}
	g.rows = n
	hd := c.HCMult * c.NEmbd
	g.emb = make([]float32, n*c.NGramWidth())
	g.key, g.q, g.gv, g.gvn = make([]float32, n*hd), make([]float32, n*hd), make([]float32, n*hd), make([]float32, n*hd)
	g.val = make([]float32, n*c.NEmbd)
	g.undo = make([]ngramUndo, 0, n)
	g.undoPrev = make([]float32, n*(c.NGram-1))
	g.gate = make([]float32, n*c.HCMult)
}

// ngramState is sequence slot's n-gram state in block li: its predecessors
// and its convolution history, after every slot's delta-rule state.
func (s *State) ngramState(li, slot int) (prev, hist []float32) {
	c := s.m.Cfg
	st := s.recTail(li, slot)
	return st[:c.NGram-1], st[c.NGram-1:]
}

// ngramApply adds the n-gram embedding into n rows of x in block li. Row r is
// the next position of sequence slot of the State owner(r) returns, with
// token tok(r) (-1 for a row with no token, which hashes as the picture
// placeholder). order is the rows in each sequence's position order (nil for
// 0..n-1), since each row moves its sequence's state on.
func (s *State) ngramApply(li int, l *layer, x []float32, n int, order []int,
	owner func(r int) (*State, int), tok func(r int) int32) error {
	c, g, d := s.m.Cfg, s.q4.ng, l.q4e
	if c.q4eFault == q4eFaultNoNGram {
		return nil
	}
	h, e := c.HCMult, c.NEmbd
	hd, w := h*e, c.NGramWidth()
	if err := s.ngramRows(li, n, order, owner, tok); err != nil {
		return err
	}
	s.jit.NewInput()
	if err := s.mm(g.key[:n*hd], d.ngKey, g.emb[:n*w], n); err != nil {
		return err
	}
	if err := s.mm(g.val[:n*e], d.ngVal, g.emb[:n*w], n); err != nil {
		return err
	}
	// The key per stream, normed in place, and the residual's streams normed.
	for i := 0; i < n*h; i++ {
		k := g.key[i*e : (i+1)*e]
		s.rmsnorm(k, k, d.ngNormK[(i%h)*e:(i%h+1)*e], c.RMSEps)
	}
	s.hcNorm(g.q[:n*hd], x, n, d.ngNormQ)
	s.jit.NewInput()
	for i := 0; i < n*h; i++ {
		g.dot.AttnScores(g.gate[i:i+1], g.key[i*e:(i+1)*e], g.q[i*e:(i+1)*e], 1)
	}
	gate := g.gate[:n*h]
	s.scale(gate, invSqrt(e))
	nn.Act32JIT(gate, nn.ActSqrtSigmoid)
	s.jit.NewInput()
	for i := 0; i < n*h; i++ {
		v := g.gv[i*e : (i+1)*e]
		copy(v, g.val[(i/h)*e:(i/h+1)*e])
		s.scale(v, gate[i])
		s.rmsnorm(g.gvn[i*e:(i+1)*e], v, d.ngNormC[(i%h)*e:(i%h+1)*e], c.RMSEps)
	}
	s.jit.NewInput()
	hist := c.ngramHist()
	for oi := 0; oi < n; oi++ {
		r := oi
		if order != nil {
			r = order[oi]
		}
		o, sl := owner(r)
		_, hs := o.ngramState(li, sl)
		cur := g.gvn[r*hd : (r+1)*hd]
		// Tap t reads (NGramConv-1-t)*NGram positions back: the last tap is
		// the row itself, the others the history, oldest first.
		clear(g.conv)
		for t := 0; t < c.NGramConv; t++ {
			back := (c.NGramConv - 1 - t) * c.NGram
			src := cur
			if back > 0 {
				p := hist - back
				src = hs[p*hd : (p+1)*hd]
			}
			copy(g.tap, src)
			nn.ActMul32JIT(g.tap, d.ngCv[t*hd:(t+1)*hd], nn.ActIdentity)
			s.axpy(g.conv, g.tap, 1)
		}
		nn.Act32JIT(g.conv, nn.ActSiLU)
		s.jit.NewInput()
		if hist > 0 {
			copy(hs, hs[hd:])
			copy(hs[(hist-1)*hd:], cur)
		}
		for k := 0; k < h; k++ {
			xk := c.stream(x, k, n, r)
			s.axpy(xk, g.gv[(r*h+k)*e:(r*h+k+1)*e], 1)
			if c.q4eFault != q4eFaultNoNGramConv {
				s.axpy(xk, g.conv[k*e:(k+1)*e], 1)
			}
		}
	}
	s.jit.NewInput()
	return nil
}

// ngramRows hashes each of n rows' n-gram and gathers its table rows into
// g.emb, NGramWidth a row, moving each sequence's predecessors on: row r is
// the next position of sequence slot of the State owner(r) returns, with
// token tok(r) (-1 for none, which hashes as the picture placeholder), and
// order is the rows in each sequence's position order (nil for 0..n-1).
func (s *State) ngramRows(li, n int, order []int, owner func(r int) (*State, int), tok func(r int) int32) error {
	c, g := s.m.Cfg, s.q4.ng
	w, rd := c.NGramWidth(), c.NGramRowDim
	for oi := 0; oi < n; oi++ {
		r := oi
		if order != nil {
			r = order[oi]
		}
		o, sl := owner(r)
		prev, _ := o.ngramState(li, sl)
		t := tok(r)
		if t < 0 {
			t = c.NGramImage
		}
		// The window is the token and its predecessors, whose bits the state
		// holds (id+1, zero for none); the kernel hashes it and writes the
		// next position's predecessors back.
		g.hwin[0] = uint32(t)
		for i, p := range prev {
			g.hwin[i+1] = math.Float32bits(p)
		}
		nn.NGramHashJIT(g.idx, g.hnext, g.hwin, g.konst, g.ids, c.NGramHeadsPer, c.q4eFault != q4eFaultNoEOSReset)
		for i := range prev {
			prev[i] = math.Float32frombits(g.hnext[i])
		}
		for hh, row := range g.idx {
			if err := s.m.ngram.f.ReadRow(s.m.ngram.e, row, g.raw, g.scr); err != nil {
				return err
			}
			if err := ngramRow(g.emb[r*w+hh*rd:r*w+(hh+1)*rd], s.m.ngram.typ, g.raw, g.scr); err != nil {
				return err
			}
		}
	}
	s.jit.NewInput()
	return nil
}

// ngramRow decodes one table row into dst.
func ngramRow(dst []float32, t quant.Type, raw, scratch []byte) error {
	switch t {
	case quant.F32, quant.F16, quant.BF16:
		nn.Row32JIT(dst, t, raw, 0, len(dst))
		return nil
	case quant.IQ4_NL:
		nn.RowIQ4NL32JIT(dst, raw, scratch)
		return nil
	}
	return fmt.Errorf("model: an n-gram table row of %v", t)
}

package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/graph"
)

// marksHeld is the device's failure to hold the marks it was last sent, once.
func (s *Session) marksHeld() error {
	err := s.markErr
	s.markErr = nil
	return err
}

// sendMarks tells the session's device every slot's marks that changed
// since it last was (nn.RopeMarksDevice).
func (s *Session) sendMarks() {
	r := &s.rmarks
	if s.ldMarks == nil {
		return
	}
	for len(r.Sent) < len(r.At) {
		r.Sent = append(r.Sent, 0)
	}
	for i := range r.At {
		if r.At[i] < 0 || r.Sent[i] == r.Gen[i]+1 {
			continue
		}
		if err := s.ldMarks.SetSeqMarks(i, r.Block(i)); err != nil {
			s.markErr = err
			continue
		}
		r.Sent[i] = r.Gen[i] + 1
	}
}

// ropeOf is the rotary position of a row of slot at cache position p that
// advances like text: p itself until an image has moved them apart.
func (s *Session) ropeOf(slot, p int) int { return s.rmarks.TextAt(slot, p) }

// markRope records that slot's text-like rows from cache position at onwards
// turn at their position plus delta, replacing every mark at or past at: the
// sequence is being written from there.
func (s *Instance) markRope(slot, at, delta int) {
	s.rewindRope(slot, at-1)
	if s.ropeOf(slot, at) == at+delta {
		return
	}
	s.rmarks.Add(slot, textMark(at, at+delta))
	s.sendMarks()
}

// textMark is a mark whose rows from cache position at turn at base onwards,
// one a row, on every axis.
func textMark(at, base int) (m [graph.MarkWords]uint32) {
	m[graph.MarkAt] = uint32(at)
	for a := range 4 {
		m[graph.MarkAxis(a, graph.MarkBase)], m[graph.MarkAxis(a, graph.MarkMul)] = uint32(base), 1
	}
	return m
}

// pictureMark is the mark of a picture's grid g whose rows start at cache
// position at, laid out as pr states (jlm.PictureRope) from rotary position
// start, the sequence's picture ord: every frame's rows, the grid's ends
// before and after them turning at their positions.
func pictureMark(pr *jlm.PictureRope, at, start, ord int, g model.ImageGrid) (m [graph.MarkWords]uint32) {
	cols := g.W + g.Line
	term := func(t jlm.PicTerm) uint32 {
		switch t {
		case jlm.PicOne:
			return 1
		case jlm.PicStart:
			return uint32(start)
		case jlm.PicOrdinal:
			return uint32(ord)
		case jlm.PicStride:
			return uint32(g.Stride)
		case jlm.PicCols:
			return uint32(cols)
		case jlm.PicRows:
			return uint32(g.H)
		case jlm.PicCells:
			return uint32(g.H * cols)
		}
		return 0
	}
	m[graph.MarkAt], m[graph.MarkSkip], m[graph.MarkCount] = uint32(at), uint32(g.Ends), uint32(g.T*g.H*cols)
	for a, x := range pr.Axes {
		m[graph.MarkAxis(a, graph.MarkBase)] = term(x.Base)
		if x.Pos {
			m[graph.MarkAxis(a, graph.MarkPos)] = 1
		}
		m[graph.MarkAxis(a, graph.MarkMul)] = term(x.Mul)
		m[graph.MarkAxis(a, graph.MarkDiv)] = term(x.Div)
		m[graph.MarkAxis(a, graph.MarkMod)] = term(x.Mod)
	}
	return m
}

// laysFramesApart reports whether some axis of pr turns by a grid's frame:
// its frames' stride, or its rows counted in whole frames.
func laysFramesApart(pr *jlm.PictureRope) bool {
	for _, x := range pr.Axes {
		if x.Mul == jlm.PicStride || x.Div == jlm.PicCells {
			return true
		}
	}
	return false
}

// markTop is the highest rotary position a bounded mark's rows take on any
// axis, its ends' cache positions among them.
func markTop(m [graph.MarkWords]uint32) int {
	skip, n := m[graph.MarkSkip], m[graph.MarkCount]
	if n == 0 {
		return 0
	}
	last := m[graph.MarkAt] + skip + n - 1
	top := 0
	if skip > 0 {
		top = int(last + skip)
	}
	for a := range 4 {
		w := func(k int) uint32 { return m[graph.MarkAxis(a, k)] }
		q := (n - 1) / max(w(graph.MarkDiv), 1)
		if mod := w(graph.MarkMod); mod != 0 {
			q = min(q, mod-1)
		}
		top = max(top, int(w(graph.MarkBase)+w(graph.MarkPos)*last+w(graph.MarkMul)*q))
	}
	return top
}

// rewindRope forgets slot's marks past cache position p.
func (s *Instance) rewindRope(slot, p int) {
	s.rmarks.Rewind(slot, p)
	s.sendMarks()
}

// spanRope lays a mixed prompt's rows out on the rotary axes as slot's
// marks, from cache position p0 where the slot's next text row would turn,
// and says the rotary position the row after the prompt takes: an image's
// rows a grid from the position its span starts at, as the container lays
// a picture out (jlm.PictureRope, pictureMark), the text after it at that
// position plus the grid's longer side or its rows. A video whose pairs are
// spans of their own is one grid over them, its frames Stride apart on the
// temporal axis. The row after the prompt takes one past the highest
// position the prompt holds where a grid reaches past the text
// (transformers' rope deltas: max + 1 - the prompt's length). A
// single-axis model's rows all advance like text, the image's too.
//
// An embedding span with no grid is rows that advance like text: the
// embedding seam carries more than images (a vocabulary row looked up
// elsewhere, TestPrefillMixedMatchesPrefill). An IMAGE without its grid is
// therefore laid out sequentially, which is fluent and wrong -- the
// violation model.TestMRopeMatchesTransformers runs -- so every caller that
// holds an image passes its grid (ChatSpans does).
func (s *Instance) spanRope(slot, p0 int, spans []model.Span) (next int, err error) {
	c := s.c
	r := s.ropeOf(slot, p0)
	if len(s.m.Rope.Runs) == 0 {
		for _, sp := range spans {
			r += sp.N(c.NEmbd)
		}
		return r, nil
	}
	pr := &s.m.Container.Config().PictureRope
	s.rewindRope(slot, p0-1)
	ord := 0
	for k := range s.rmarks.Count(slot) {
		if s.rmarks.Mark(slot, k)[graph.MarkCount] != 0 {
			ord++
		}
	}
	p, hi := p0, r-1
	// open is the grid a video's next pair continues, frame its next frame,
	// from cache position gp.
	var open model.ImageGrid
	frame, gp := 0, 0
	for i, sp := range spans {
		n := sp.N(c.NEmbd)
		if !sp.Rows() || sp.Grid.Zero() {
			if open.T > 0 {
				return 0, fmt.Errorf("model: span %d interrupts a video's grid at frame %d of %d", i, frame, open.T)
			}
			hi = max(hi, r+n-1)
			r, p = r+n, p+n
			continue
		}
		g := sp.Grid
		if g.Rows() != n {
			return 0, fmt.Errorf("model: span %d is %d embedding rows and its grid %dx%dx%d (+%d, %d ends) is %d",
				i, n, g.T, g.H, g.W, g.Line, g.Ends, g.Rows())
		}
		switch {
		case open.T > 0 && (g.Frame != frame || g.T != open.T || g.H != open.H || g.W != open.W || g.Stride != open.Stride):
			return 0, fmt.Errorf("model: span %d is frame %d of a %dx%dx%d grid where frame %d of a %dx%dx%d one is next",
				i, g.Frame, g.T, g.H, g.W, frame, open.T, open.H, open.W)
		case open.T == 0 && g.Frame != 0:
			return 0, fmt.Errorf("model: span %d starts a grid at frame %d", i, g.Frame)
		case open.T == 0 && g.T > 1 && !laysFramesApart(pr):
			return 0, fmt.Errorf("model: span %d is a video of %d frames and the container's picture layout "+
				"turns no axis by a frame", i, g.T)
		case open.T == 0:
			m := pictureMark(pr, p, r, ord, g)
			s.rmarks.Add(slot, m)
			hi = max(hi, markTop(m))
			ord, gp = ord+1, p
		}
		p += n
		if g.Last() {
			open, frame = model.ImageGrid{}, 0
			if pr.Side {
				r += max(g.H, g.W)
			} else {
				r += p - gp
			}
			if s.rmarks.TextAt(slot, p) != r {
				s.rmarks.Add(slot, textMark(p, r))
			}
			continue
		}
		open, frame = g, g.Frame+g.Frames
	}
	if open.T > 0 {
		return 0, fmt.Errorf("model: the prompt ends at frame %d of a video's %d", frame, open.T)
	}
	if hi+1 > r {
		r = hi + 1
		s.rmarks.Add(slot, textMark(p, r))
	}
	s.sendMarks()
	return r, nil
}

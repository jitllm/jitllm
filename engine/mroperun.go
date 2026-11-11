package engine

import (
	"fmt"

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
	s.rmarks.Add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(at), graph.MarkBase: uint32(at + delta), graph.MarkKind: graph.MarkText})
	s.sendMarks()
}

// rewindRope forgets slot's marks past cache position p.
func (s *Instance) rewindRope(slot, p int) {
	s.rmarks.Rewind(slot, p)
	s.sendMarks()
}

// xdRope is spanRope under XD-RoPE (HunyuanVL, transformers' get_rope_index):
// every row's first coordinate is its sequence position, nothing after an
// image is shifted, and a picture's grid rows -- the newline column included
// -- take (position, column, row, picture) on the four axes, the picture's
// ordinal in the sequence; its begin and end rows are text-like. A picture
// is a mark where it starts (graph.MarkGridXD), its ordinal the slot's
// pictures before it.
func (s *Instance) xdRope(slot, p0 int, spans []model.Span) (next int, err error) {
	c := s.c
	s.rewindRope(slot, p0-1)
	pics := 0
	for k := range s.rmarks.Count(slot) {
		if s.rmarks.Mark(slot, k)[graph.MarkKind] == graph.MarkGridXD {
			pics++
		}
	}
	r := p0
	for i, sp := range spans {
		n := sp.N(c.NEmbd)
		if !sp.Rows() || sp.Grid.Zero() {
			r += n
			continue
		}
		g := sp.Grid
		if g.Rows() != n || g.T != 1 {
			return 0, fmt.Errorf("model: span %d is %d embedding rows and its grid %dx%dx%d (+%d, %d ends) is %d",
				i, n, g.T, g.H, g.W, g.Line, g.Ends, g.Rows())
		}
		s.rmarks.Add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(r), graph.MarkKind: graph.MarkGridXD, graph.MarkT: 1,
			graph.MarkH: uint32(g.H), graph.MarkW: uint32(g.W + g.Line), graph.MarkEnds: uint32(g.Ends), graph.MarkIndex: uint32(pics)})
		pics++
		r += n
	}
	s.sendMarks()
	return r, nil
}

// spanRope lays a mixed prompt's rows out on the rotary axes as slot's
// marks, from cache position p0 where the slot's next text row would turn,
// and says the rotary position the row after the prompt takes: an image's
// rows a grid from the position its span starts at (graph.MarkGrid), the
// text after it at that position plus max(H, W) (transformers'
// get_rope_index: current_pos += max(h, w), whatever T). A video whose
// pairs are spans of their own is one grid over them, its frames Stride
// apart on the temporal axis. The row after the prompt takes one past the
// highest position the prompt holds where a grid's frames reach past the
// text (transformers' rope deltas: max + 1 - the prompt's length). A
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
	if c.RopeXD {
		return s.xdRope(slot, p0, spans)
	}
	r := s.ropeOf(slot, p0)
	if len(s.m.Rope.Runs) == 0 {
		for _, sp := range spans {
			r += sp.N(c.NEmbd)
		}
		return r, nil
	}
	s.rewindRope(slot, p0-1)
	p, hi := p0, r-1
	// open is the grid a video's next pair continues, frame its next frame.
	var open model.ImageGrid
	frame := 0
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
			return 0, fmt.Errorf("model: span %d is %d embedding rows and its grid %dx%dx%d is %d",
				i, n, g.T, g.H, g.W, g.Rows())
		}
		switch {
		case open.T > 0 && (g.Frame != frame || g.T != open.T || g.H != open.H || g.W != open.W || g.Stride != open.Stride):
			return 0, fmt.Errorf("model: span %d is frame %d of a %dx%dx%d grid where frame %d of a %dx%dx%d one is next",
				i, g.Frame, g.T, g.H, g.W, frame, open.T, open.H, open.W)
		case open.T == 0 && g.Frame != 0:
			return 0, fmt.Errorf("model: span %d starts a grid at frame %d", i, g.Frame)
		case open.T == 0:
			s.rmarks.Add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(p), graph.MarkBase: uint32(r), graph.MarkKind: graph.MarkGrid,
				graph.MarkT: uint32(g.T), graph.MarkH: uint32(g.H), graph.MarkW: uint32(g.W), graph.MarkStride: uint32(g.Stride)})
			hi = max(hi, r+(g.T-1)*g.Stride, r+max(g.H, g.W)-1)
		}
		p += n
		if g.Last() {
			open, frame = model.ImageGrid{}, 0
			r += max(g.H, g.W)
			s.rmarks.Add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(p), graph.MarkBase: uint32(r), graph.MarkKind: graph.MarkText})
			continue
		}
		open, frame = g, g.Frame+g.Frames
	}
	if open.T > 0 {
		return 0, fmt.Errorf("model: the prompt ends at frame %d of a video's %d", frame, open.T)
	}
	if hi+1 > r {
		r = hi + 1
		s.rmarks.Add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(p), graph.MarkBase: uint32(r), graph.MarkKind: graph.MarkText})
	}
	s.sendMarks()
	return r, nil
}

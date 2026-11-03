package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// M-RoPE: a row's rotary position is a property of the ROW, with a coordinate
// per axis, and it is not its place in the cache.
//
// Qwen2-VL's text rotary splits a head's pairs between three coordinates
// (t, h, w). A text row has all three equal, so for a text-only prompt it is a
// plain rotary at the row's index. An image span's rows are a grid: row (y, x)
// of a still image turns at (s, s+y, s+x), where s is the rotary position the
// span starts at, and the text after it resumes at s + max(H, W) rather than at
// s + H*W -- so from the first image on, a row's rotary position and its cache
// position part (transformers' get_rope_index, llama.cpp's mtmd n_pos).
//
// Nothing about that is an image pipeline's: a sequence carries its rotary
// marks (graph.OpRopePos), written where a prompt's spans enter and dropped
// where the sequence is rewound -- a picture's grid, the text after it -- and
// every device computes a row's coordinates from its cache position and its
// sequence's marks in the step's rotary table (composer.TableRopeMarks), the CPU
// as a card does. No call hands a device a row's position on the axes.

// ImageGrid is the shape of an embedding span's rows: T frames of H x W, in
// raster order (t slowest, x fastest), after the projector's merge. A span with
// a grid is an image; one without is rows that advance like text.
type ImageGrid struct {
	T, H, W int
	// Line is extra columns at the end of every grid row, and Ends rows
	// before and after the grid: HunyuanVL's merger appends a newline row to
	// each row of the merged grid and wraps the picture in a begin and an end
	// row (Line 1, Ends 1). Zero for every other projector.
	Line, Ends int
}

// Rows is how many embedding rows the grid covers.
func (g ImageGrid) Rows() int { return g.T*g.H*(g.W+g.Line) + 2*g.Ends }

func (g ImageGrid) zero() bool { return g == ImageGrid{} }

// mropeSections is the container's M-RoPE split: contiguous runs of pairs for
// Qwen2-VL, Qwen2.5-VL and GLM-4.xV, an interleaved split for Qwen3-VL and
// Qwen3.5 (jlm.Arch.InterleavedMRope, Config.RopeInterleaved). A container
// with no sections has none.
func mropeSections(c *jlm.Config) (out [4]int) {
	for i, v := range c.RopeSections {
		out[i] = int(v)
	}
	return out
}

// ropeMarks is a session's rotary marks (graph.OpRopePos) as the composed
// rotary tables read them (composer.TableRopeMarks): a block a slot -- its count,
// then its marks of graph.MarkWords words in order of first position -- at
// at[slot] in buf, room[slot] marks large. A slot outgrowing its block moves
// to one twice as large at buf's end; gen[slot] counts its changes, so a
// device holding a copy is sent it again (sendMarks).
type ropeMarks struct {
	buf      []uint32
	at, room []int
	gen      []uint64
	// sent is, per slot, one past the gen its device was last told.
	sent []uint64
}

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
	for len(r.sent) < len(r.at) {
		r.sent = append(r.sent, 0)
	}
	for i := range r.at {
		if r.at[i] < 0 || r.sent[i] == r.gen[i]+1 {
			continue
		}
		if err := s.ldMarks.SetSeqMarks(i, r.block(i)); err != nil {
			s.markErr = err
			continue
		}
		r.sent[i] = r.gen[i] + 1
	}
}

// count is how many marks slot i holds.
func (r *ropeMarks) count(i int) int {
	if i >= len(r.at) || r.at[i] < 0 {
		return 0
	}
	return int(r.buf[r.at[i]])
}

// mark is slot i's mark k.
func (r *ropeMarks) mark(i, k int) []uint32 {
	o := r.at[i] + 1 + k*graph.MarkWords
	return r.buf[o : o+graph.MarkWords]
}

// block is slot i's count and marks, as a table reads them; nil for a slot
// that holds no block.
func (r *ropeMarks) block(i int) []uint32 {
	if i >= len(r.at) || r.at[i] < 0 {
		return nil
	}
	return r.buf[r.at[i] : r.at[i]+1+r.count(i)*graph.MarkWords]
}

// offset is where slot i's block starts in buf, its room made first.
func (r *ropeMarks) offset(i int) int {
	r.grow(i, 0)
	return r.at[i]
}

// grow makes room for slot i to hold n marks.
func (r *ropeMarks) grow(i, n int) {
	for len(r.at) <= i {
		r.at, r.room, r.gen = append(r.at, -1), append(r.room, 0), append(r.gen, 0)
	}
	if r.at[i] >= 0 && r.room[i] >= n {
		return
	}
	room := max(4, 2*n)
	at := len(r.buf)
	r.buf = append(r.buf, make([]uint32, 1+room*graph.MarkWords)...)
	if r.at[i] >= 0 {
		copy(r.buf[at:], r.block(i))
	}
	r.at[i], r.room[i] = at, room
}

// add appends m to slot i's marks.
func (r *ropeMarks) add(i int, m [graph.MarkWords]uint32) {
	n := r.count(i)
	r.grow(i, n+1)
	copy(r.mark(i, n), m[:])
	r.buf[r.at[i]] = uint32(n + 1)
	r.gen[i]++
}

// rewind drops slot i's marks past cache position p, for a sequence rolled
// back to p (its next row is written at p+1, or at p for a mark at p).
func (r *ropeMarks) rewind(i, p int) {
	n := r.count(i)
	k := n
	for k > 0 && int(r.mark(i, k-1)[graph.MarkAt]) > p {
		k--
	}
	if k != n {
		r.buf[r.at[i]] = uint32(k)
		r.gen[i]++
	}
}

// textAt is the rotary position a row of slot i at cache position p that
// advances like text turns at, from the slot's marks: where a prompt's
// spans start laying out.
func (r *ropeMarks) textAt(i, p int) int {
	for k := r.count(i) - 1; k >= 0; k-- {
		m := r.mark(i, k)
		if int(m[graph.MarkAt]) > p {
			continue
		}
		if m[graph.MarkKind] == graph.MarkText {
			return p - int(m[graph.MarkAt]) + int(m[graph.MarkBase])
		}
		return p
	}
	return p
}

// ropeOf is the rotary position of a row of slot at cache position p that
// advances like text: p itself until an image has moved them apart.
func (s *Session) ropeOf(slot, p int) int { return s.rmarks.textAt(slot, p) }

// markRope records that slot's text-like rows from cache position at onwards
// turn at their position plus delta, replacing every mark at or past at: the
// sequence is being written from there.
func (s *Instance) markRope(slot, at, delta int) {
	s.rewindRope(slot, at-1)
	if s.ropeOf(slot, at) == at+delta {
		return
	}
	s.rmarks.add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(at), graph.MarkBase: uint32(at + delta), graph.MarkKind: graph.MarkText})
	s.sendMarks()
}

// rewindRope forgets slot's marks past cache position p.
func (s *Instance) rewindRope(slot, p int) {
	s.rmarks.rewind(slot, p)
	s.sendMarks()
}

// xdRope is spanRope under XD-RoPE (HunyuanVL, transformers' get_rope_index):
// every row's first coordinate is its sequence position, nothing after an
// image is shifted, and a picture's grid rows -- the newline column included
// -- take (position, column, row, picture) on the four axes, the picture's
// ordinal in the sequence; its begin and end rows are text-like. A picture
// is a mark where it starts (graph.MarkGridXD), its ordinal the slot's
// pictures before it.
func (s *Instance) xdRope(slot, p0 int, spans []Span) (next int, err error) {
	c := s.c
	s.rewindRope(slot, p0-1)
	pics := 0
	for k := range s.rmarks.count(slot) {
		if s.rmarks.mark(slot, k)[graph.MarkKind] == graph.MarkGridXD {
			pics++
		}
	}
	r := p0
	for i, sp := range spans {
		n := sp.n(c.NEmbd)
		if !sp.rows() || sp.Grid.zero() {
			r += n
			continue
		}
		g := sp.Grid
		if g.Rows() != n || g.T != 1 {
			return 0, fmt.Errorf("model: span %d is %d embedding rows and its grid %dx%dx%d (+%d, %d ends) is %d",
				i, n, g.T, g.H, g.W, g.Line, g.Ends, g.Rows())
		}
		s.rmarks.add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(r), graph.MarkKind: graph.MarkGridXD, graph.MarkT: 1,
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
// get_rope_index: current_pos += max(h, w), whatever T). A single-axis
// model's rows all advance like text, the image's too.
//
// An embedding span with no grid is rows that advance like text: the
// embedding seam carries more than images (a vocabulary row looked up
// elsewhere, TestPrefillMixedMatchesPrefill). An IMAGE without its grid is
// therefore laid out sequentially, which is fluent and wrong -- the
// violation model.TestMRopeMatchesTransformers runs -- so every caller that
// holds an image passes its grid (ChatSpans does).
func (s *Instance) spanRope(slot, p0 int, spans []Span) (next int, err error) {
	c := s.c
	if c.RopeXD {
		return s.xdRope(slot, p0, spans)
	}
	r := s.ropeOf(slot, p0)
	if len(s.m.rope.Runs) == 0 {
		for _, sp := range spans {
			r += sp.n(c.NEmbd)
		}
		return r, nil
	}
	s.rewindRope(slot, p0-1)
	p := p0
	for i, sp := range spans {
		n := sp.n(c.NEmbd)
		if !sp.rows() || sp.Grid.zero() {
			r, p = r+n, p+n
			continue
		}
		g := sp.Grid
		if g.Rows() != n {
			return 0, fmt.Errorf("model: span %d is %d embedding rows and its grid %dx%dx%d is %d",
				i, n, g.T, g.H, g.W, g.Rows())
		}
		s.rmarks.add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(p), graph.MarkBase: uint32(r), graph.MarkKind: graph.MarkGrid,
			graph.MarkT: uint32(g.T), graph.MarkH: uint32(g.H), graph.MarkW: uint32(g.W)})
		r, p = r+max(g.H, g.W), p+n
		s.rmarks.add(slot, [graph.MarkWords]uint32{graph.MarkAt: uint32(p), graph.MarkBase: uint32(r), graph.MarkKind: graph.MarkText})
	}
	s.sendMarks()
	return r, nil
}

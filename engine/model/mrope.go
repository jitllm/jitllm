package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
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
// Nothing about that is an image pipeline's: the positions ride the rows
// through the one sequence. A prefill computes every row's coordinates from its
// spans; the rows after it -- decode, a batch step, a ragged step, a
// speculation -- take theirs from ropeOf, which reads the offset the prompt
// left behind (a ropeMark). The rotation itself is every tier's own kernel over
// a per-row table (nn.Rope.TableAt), so a device needs no knowledge of M-RoPE:
// it is handed the host's table (Model.ropeTabPlanes), as every tier already
// can be.

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

// ropeMark says the rows of a slot from cache position at onwards that advance
// like text turn at their cache position plus delta, until the next mark.
type ropeMark struct{ at, delta int }

// ropeOf is the rotary position of a row of slot at cache position p that
// advances like text: p itself until an image has moved them apart.
func (s *State) ropeOf(slot, p int) int {
	if slot >= len(s.rmarks) {
		return p
	}
	ms := s.rmarks[slot]
	for i := len(ms) - 1; i >= 0; i-- {
		if ms[i].at <= p {
			return p + ms[i].delta
		}
	}
	return p
}

// markRope records that slot's text-like rows from cache position at onwards
// turn at their position plus delta, replacing every mark at or past at: the
// sequence is being written from there.
func (s *State) markRope(slot, at, delta int) {
	s.rewindRope(slot, at-1)
	if s.ropeOf(slot, at) == at+delta {
		return
	}
	s.rmarks[slot] = append(s.rmarks[slot], ropeMark{at, delta})
}

// rewindRope forgets slot's marks past cache position p, for a sequence rolled
// back to p (its next row is written at p+1, or at p for a mark at p).
func (s *State) rewindRope(slot, p int) {
	if slot >= len(s.rmarks) {
		return
	}
	ms := s.rmarks[slot]
	for len(ms) > 0 && ms[len(ms)-1].at > p {
		ms = ms[:len(ms)-1]
	}
	s.rmarks[slot] = ms
	if slot < len(s.ximg) {
		at := s.ximg[slot]
		for len(at) > 0 && at[len(at)-1] > p {
			at = at[:len(at)-1]
		}
		s.ximg[slot] = at
	}
}

// xdRope is spanRope under XD-RoPE (HunyuanVL, transformers' get_rope_index):
// every row's first coordinate is its sequence position, nothing after an
// image is shifted, and a picture's grid rows -- the newline column included
// -- take (position, column, row, picture) on the four axes, the picture's
// ordinal in the sequence; its begin and end rows are text-like. The slot's
// pictures are recorded where they start, so a later prompt counts them.
func (s *State) xdRope(slot, p0 int, spans []Span) (rows [][4]int, next int, err error) {
	c := s.c
	for len(s.ximg) <= slot {
		s.ximg = append(s.ximg, nil)
	}
	s.rewindRope(slot, p0-1)
	r := p0
	for i, sp := range spans {
		n := sp.n(c.NEmbd)
		if !sp.rows() || sp.Grid.zero() {
			for j := 0; j < n; j++ {
				rows = append(rows, [4]int{r + j, r + j, r + j, r + j})
			}
			r += n
			continue
		}
		g := sp.Grid
		if g.Rows() != n || g.T != 1 {
			return nil, 0, fmt.Errorf("model: span %d is %d embedding rows and its grid %dx%dx%d (+%d, %d ends) is %d",
				i, n, g.T, g.H, g.W, g.Line, g.Ends, g.Rows())
		}
		idx := len(s.ximg[slot])
		s.ximg[slot] = append(s.ximg[slot], r)
		j := 0
		text := func() {
			rows = append(rows, [4]int{r + j, r + j, r + j, r + j})
			j++
		}
		for e := 0; e < g.Ends; e++ {
			text()
		}
		for y := 0; y < g.H; y++ {
			for x := 0; x < g.W+g.Line; x++ {
				rows = append(rows, [4]int{r + j, x, y, idx})
				j++
			}
		}
		for e := 0; e < g.Ends; e++ {
			text()
		}
		r += n
	}
	return rows, r, nil
}

// spanRope lays a mixed prompt's rows out on the rotary axes: rows[i] is row
// i's (t, h, w), starting where the slot's next text row would turn (at cache
// position p0), and next is the rotary position the row after the prompt
// takes. nil rows means the model's rotary is single-axis and every row is
// ropeOf's.
//
// An embedding span with no grid is rows that advance like text: the
// embedding seam carries more than images (a vocabulary row looked up
// elsewhere, TestPrefillMixedMatchesPrefill). An IMAGE without its grid is
// therefore laid out sequentially, which is fluent and wrong -- the state this
// replaced, and the violation model.TestMRopeMatchesTransformers runs -- so
// every caller that holds an image passes its grid (ChatSpans does).
func (s *State) spanRope(slot, p0 int, spans []Span) (rows [][4]int, next int, err error) {
	c := s.c
	if c.RopeXD {
		return s.xdRope(slot, p0, spans)
	}
	r := s.ropeOf(slot, p0)
	if len(s.m.rope.Runs) == 0 {
		// Every row advances like text, the image's too.
		for _, sp := range spans {
			r += sp.n(c.NEmbd)
		}
		return nil, r, nil
	}
	for i, sp := range spans {
		n := sp.n(c.NEmbd)
		if !sp.rows() || sp.Grid.zero() {
			for j := 0; j < n; j++ {
				rows = append(rows, [4]int{r + j, r + j, r + j, r + j})
			}
			r += n
			continue
		}
		g := sp.Grid
		if g.Rows() != n {
			return nil, 0, fmt.Errorf("model: span %d is %d embedding rows and its grid %dx%dx%d is %d",
				i, n, g.T, g.H, g.W, g.Rows())
		}
		for t := 0; t < g.T; t++ {
			for y := 0; y < g.H; y++ {
				for x := 0; x < g.W; x++ {
					rows = append(rows, [4]int{r + t, r + y, r + x, r + t})
				}
			}
		}
		// transformers' get_rope_index: current_pos += max(h, w), whatever T.
		r += max(g.H, g.W)
	}
	return rows, r, nil
}

// ropeTableRow writes the rotary rows of a row with an explicit multi-axis
// position into row j of the batch tables: the plain table when its
// coordinates agree (a text row, to the bit), the multi-axis one otherwise.
func (s *State) ropeTableRow(j int, at [4]int) {
	nrot, nrotSWA := s.c.RopeW(), s.c.NRotSWA
	if at[0] == at[1] && at[1] == at[2] && at[2] == at[3] {
		s.ropeRow(j, at[0])
		return
	}
	cs := s.bcs[j*nrot : (j+1)*nrot]
	s.jit.RopeTableAt(s.m.rope, cs[:s.c.NRot], at[:])
	if s.m.ropeB != nil {
		s.jit.RopeTableAt(*s.m.ropeB, cs[s.c.NRot:], at[:])
	}
	if s.m.ropeSWA != nil {
		s.jit.RopeTableAt(*s.m.ropeSWA, s.bcsSWA[j*nrotSWA:(j+1)*nrotSWA], at[:])
	}
}

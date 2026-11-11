package model

import (
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
	// Frame and Frames, where Frames is set, are the frames of the grid a
	// span holds, from frame Frame: a video whose pairs are spans of their
	// own and one grid together (jlm.Video.Joined). Zero Frames is all T.
	Frame, Frames int
	// Stride is the positions the temporal axis steps from one frame to the
	// next (jlm.Video.PerSecond); a grid of one frame reads none.
	Stride int
}

// Rows is how many embedding rows the span of the grid covers.
func (g ImageGrid) Rows() int {
	f := g.T
	if g.Frames > 0 {
		f = g.Frames
	}
	return f*g.H*(g.W+g.Line) + 2*g.Ends
}

// Last says the span holds the grid's last frame.
func (g ImageGrid) Last() bool { return g.Frames == 0 || g.Frame+g.Frames == g.T }

func (g ImageGrid) Zero() bool { return g == ImageGrid{} }

// mropeSections is the container's M-RoPE split: contiguous runs of pairs for
// Qwen2-VL, Qwen2.5-VL and GLM-4.xV, an interleaved split for Qwen3-VL and
// Qwen3.5 (jlm.FlagRopeInterleaved, Config.RopeInterleaved). A container
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
type RopeMarks struct {
	Buf      []uint32
	At, room []int
	Gen      []uint64
	// sent is, per slot, one past the gen its device was last told.
	Sent []uint64
}

// count is how many marks slot i holds.
func (r *RopeMarks) Count(i int) int {
	if i >= len(r.At) || r.At[i] < 0 {
		return 0
	}
	return int(r.Buf[r.At[i]])
}

// mark is slot i's mark k.
func (r *RopeMarks) Mark(i, k int) []uint32 {
	o := r.At[i] + 1 + k*graph.MarkWords
	return r.Buf[o : o+graph.MarkWords]
}

// block is slot i's count and marks, as a table reads them; nil for a slot
// that holds no block.
func (r *RopeMarks) Block(i int) []uint32 {
	if i >= len(r.At) || r.At[i] < 0 {
		return nil
	}
	return r.Buf[r.At[i] : r.At[i]+1+r.Count(i)*graph.MarkWords]
}

// offset is where slot i's block starts in buf, its room made first.
func (r *RopeMarks) Offset(i int) int {
	r.grow(i, 0)
	return r.At[i]
}

// grow makes room for slot i to hold n marks.
func (r *RopeMarks) grow(i, n int) {
	for len(r.At) <= i {
		r.At, r.room, r.Gen = append(r.At, -1), append(r.room, 0), append(r.Gen, 0)
	}
	if r.At[i] >= 0 && r.room[i] >= n {
		return
	}
	room := max(4, 2*n)
	at := len(r.Buf)
	r.Buf = append(r.Buf, make([]uint32, 1+room*graph.MarkWords)...)
	if r.At[i] >= 0 {
		copy(r.Buf[at:], r.Block(i))
	}
	r.At[i], r.room[i] = at, room
}

// add appends m to slot i's marks.
func (r *RopeMarks) Add(i int, m [graph.MarkWords]uint32) {
	n := r.Count(i)
	r.grow(i, n+1)
	copy(r.Mark(i, n), m[:])
	r.Buf[r.At[i]] = uint32(n + 1)
	r.Gen[i]++
}

// rewind drops slot i's marks past cache position p, for a sequence rolled
// back to p (its next row is written at p+1, or at p for a mark at p).
func (r *RopeMarks) Rewind(i, p int) {
	n := r.Count(i)
	k := n
	for k > 0 && int(r.Mark(i, k-1)[graph.MarkAt]) > p {
		k--
	}
	if k != n {
		r.Buf[r.At[i]] = uint32(k)
		r.Gen[i]++
	}
}

// textAt is the rotary position a row of slot i at cache position p that
// advances like text turns at, from the slot's marks: where a prompt's
// spans start laying out.
func (r *RopeMarks) TextAt(i, p int) int {
	for k := r.Count(i) - 1; k >= 0; k-- {
		m := r.Mark(i, k)
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

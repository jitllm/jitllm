package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// readSide reads off the input rows' graph whether they write each row's
// side inputs (nn.InputsWriteSide), as every card reads its own: the CPU's
// rows (one stream) and a card's (every stream) must agree.
func (s *Instance) readSide() error {
	f := s.m.FrontGraphs()
	s.sideIn = s.c.SideWidth() != 0 && nn.InputsWriteSide(f.CpuEmbed, 1)
	if f.CpuEmbed != nil && f.Embed != nil && s.c.SideWidth() != 0 && s.sideIn != nn.InputsWriteSide(f.Embed, s.c.Streams()) {
		return fmt.Errorf("model: the input rows write side inputs on the CPU (%v) and not alike on a card", s.sideIn)
	}
	return nil
}

// inputsSide says the input rows write each row's side inputs (their graph's
// output past the residual streams), which the blocks read their slices of.
func (s *Instance) inputsSide() bool { return s.sideIn }

// inputSideW is the side row the input rows write: the whole row, or none.
func (s *Instance) inputSideW() int {
	if !s.sideIn {
		return 0
	}
	return s.c.SideWidth()
}

// givenSide says given rows carry the head of each row's side inputs: the
// rows the tower's taps write (jlm.TowerTap), over the input rows' zeros
// (fillGivenSide).
func (s *Instance) givenSide() bool { return s.c.TapsWidth() != 0 }

// growPLE sizes a decode row's side inputs. A model without them gets none.
func (s *Instance) growPLE() {
	c := s.c
	if c.SideWidth() == 0 || s.ple != nil {
		return
	}
	// A decode row's tap rows are zeros, which nothing writes over.
	s.ple = make([]float32, c.SideWidth())
}

// historySources is, for each block below hi, the block whose histories it
// binds at their indices as its graph reads them: its own, or the block its
// attention names (graph.Graph.ReadsHistoryOf); and the other blocks'
// histories it reads past them (graph.Graph.ReadsMore).
func historySources(m *model.Model, hi int) ([]int, [][]graph.HistoryRead) {
	src, more := make([]int, max(hi, 0)), make([][]graph.HistoryRead, max(hi, 0))
	for li := range src {
		g := m.Composing().BlockGraph(li)
		src[li], more[li] = li, g.ReadsMore()
		if n := g.ReadsHistoryOf(); n >= 0 {
			src[li] = n
		}
	}
	return src, more
}

// historyGroups is each block's group of blocks whose histories are read
// together, as its least member: a block, every block whose history it
// reads, and every block reading one of theirs, placed on one side.
func historyGroups(src []int, more [][]graph.HistoryRead) []int {
	g := make([]int, len(src))
	for li := range g {
		g[li] = li
	}
	root := func(b int) int {
		for g[b] != b {
			b = g[b]
		}
		return b
	}
	join := func(a, b int) {
		if b < 0 || b >= len(g) {
			return
		}
		ra, rb := root(a), root(b)
		g[max(ra, rb)] = min(ra, rb)
	}
	for li := range src {
		join(li, src[li])
		for _, r := range more[li] {
			join(li, r.Block)
		}
	}
	for li := range g {
		g[li] = root(li)
	}
	return g
}

// blockWindow is a block's attention window as its graph states it
// (graph.Graph.AttendWindow): w keys back from a row, or the aligned chunk
// of w keys it is in where chunked; w 0 for a block that sees its whole
// history.
type blockWindow struct {
	w       int
	chunked bool
}

// attendWindows is each block's window in [lo, hi), as its graph states it. A
// tower's blocks attend over one picture's rows and slide over none.
func attendWindows(m *model.Model, lo, hi int) []blockWindow {
	out := make([]blockWindow, max(hi, 0))
	for li := max(lo, 0); li < len(out); li++ {
		if t := m.Tower; t != nil && li >= m.TowerBase && li < m.TowerBase+len(t.Blocks) {
			continue
		}
		out[li].w, out[li].chunked = m.Composing().BlockGraph(li).AttendWindow()
	}
	return out
}

// modelWindow is the window the model's sliding blocks state, 0 where none
// slides: a model states one.
func modelWindow(wins []blockWindow) int {
	w := 0
	for _, b := range wins {
		w = max(w, b.w)
	}
	return w
}

// histSrc is the block whose history block li attends to.
func (s *Instance) histSrc(li int) int {
	if li < 0 || li >= len(s.histOf) {
		return li
	}
	return s.histOf[li]
}

// readsOther reports that block li keeps no history of its own and attends
// to another block's.
func (s *Instance) readsOther(li int) bool { return s.histSrc(li) != li }

// historyGroup reports whether block b is in block src's history group
// (historyGroups): it reads, or is, a history a member reads.
func (s *Instance) historyGroup(b, src int) bool {
	if b < 0 || src < 0 || b >= len(s.kvGroup) || src >= len(s.kvGroup) {
		return b == src
	}
	return s.kvGroup[b] == s.kvGroup[src]
}

// kvGroupOffered reports whether block li may be offered by offerRange(lo,
// hi): every block of its history group is on a device already or offered
// by the same call.
func (s *Instance) kvGroupOffered(li, lo, hi int) bool {
	for b := range s.histOf {
		if s.historyGroup(b, li) && !s.devAt(b) && (b < lo || b >= hi) {
			return false
		}
	}
	return true
}

// kvSpan is the blocks that must be offered with block li: the span of its
// history group.
func (s *Instance) kvSpan(li int) (lo, hi int) {
	lo, hi = li, li+1
	for b := range s.histOf {
		if s.historyGroup(b, li) {
			lo, hi = min(lo, b), max(hi, b+1)
		}
	}
	return lo, hi
}

// splitFromItsHistory reports that block li reads a history of a block on
// the other side of the seam.
func (s *Instance) splitFromItsHistory(li int) bool {
	if src := s.histSrc(li); src != li && s.devAt(li) != s.devAt(src) {
		return true
	}
	if li < len(s.histMore) {
		for _, r := range s.histMore[li] {
			if s.devAt(li) != s.devAt(r.Block) {
				return true
			}
		}
	}
	return false
}

// fixKVGroups keeps every block that attends to another block's history on
// that block's device: the history lives on exactly one side, so its block
// and every block reading it must be placed together. A group split by the
// placement -- a device that took the source and declined a reader, or a
// relocation that took one member to the CPU -- moves to the CPU whole,
// with every history its members keep. The device declines a reader whose
// sources it does not hold (tier) and offerRange offers a group only whole,
// so this is the third half.
func (s *Instance) fixKVGroups() {
	if s.ld == nil {
		return
	}
	for li := range s.histOf {
		if !s.splitFromItsHistory(li) {
			continue
		}
		for b := range s.histOf {
			if !s.historyGroup(b, li) || !s.devAt(b) {
				continue
			}
			if s.histSrc(b) == b && !s.migrateKV(b, s.pos, false) {
				continue // the history cannot move; the device keeps the block
			}
			s.ld.ReleaseLayers(b, b+1)
			s.unmarkOnDev(b)
		}
		s.gpuTarget = min(s.gpuTarget, s.gpuLayers)
	}
}

// growBPLE sizes a batched chunk's side inputs for n rows. Grows, never
// shrinks.
func (s *Instance) growBPLE(n int) {
	if w := n * s.c.SideWidth(); len(s.bple) < w {
		s.bple = make([]float32, w)
	}
}

// tokenSide is n token rows' side inputs, zeros where the input rows write
// none (a token carries no given side rows); input rows that write them
// write over these.
func (s *Instance) tokenSide(n int) []float32 {
	w := s.c.SideWidth()
	if w == 0 {
		return nil
	}
	s.growBPLE(n)
	side := s.bple[:n*w]
	if !s.inputsSide() {
		clear(side)
	}
	return side
}

// inputSide is n token rows' side inputs as their input rows write them, nil
// where they write none (tokenSide's zeros then stand).
func (s *Instance) inputSide(n int) []float32 {
	side := s.tokenSide(n)
	if !s.inputsSide() {
		return nil
	}
	return side
}

// fillGivenSide writes rows base..base+n-1 of src's given side rows into
// side (SideWidth a row): a given run's where its span carries them, zeros
// for every other row.
func (s *Instance) fillGivenSide(src rowSrc, base, n int, side []float32) {
	if !s.givenSide() {
		return
	}
	E, W, T := s.c.NEmbd, s.c.SideWidth(), s.c.TapsWidth()
	for i := range n {
		clear(side[i*W : i*W+T])
	}
	for _, g := range src.given {
		if g.side == nil {
			continue
		}
		lo, hi := max(g.at, base), min(g.at+len(g.rows)/E, base+n)
		for i := lo; i < hi; i++ {
			at := g.n0 + i - g.at
			for k := range T / E {
				copy(side[(i-base)*W+k*E:(i-base)*W+(k+1)*E], g.side[(k*g.n+at)*E:])
			}
		}
	}
}

package engine

import (
	"fmt"
	"github.com/jitllm/jitllm/nn/history"
	"runtime"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// The CPU is a device like a card: a block placed on it runs what the
// composer forms for it from the block's graph (Model.Composing), through
// the one keyed cache, on the instance's own buffers -- the residual, the
// block's weights and vectors by role, its rotary table and its history in
// the host's KV pages (history.Arena) -- each kernel mapped once for the model and
// called through the trampoline, the calls of a kernel over heads spread
// over the instance's pool. Every block step on the CPU is counted served or
// declined by name (CPUBlockSteps), as the tier counts a card's.

// cpuHeadRefused is the composer's decline of the head on the CPU as the
// step's error, as blockRefusal is a block's.
func cpuHeadRefused(why nn.Decline) error {
	return fmt.Errorf("model: the composer does not serve the head on the CPU: %s", nn.DeclineName(int(why)))
}

// cpuRowsCap is the rows the CPU's form of several rows is composed for
// (cpudev.RowsCap).
const cpuRowsCap = cpudev.RowsCap

// CPUForm is how the composer forms this host's CPU device
// (composer.CPUForm).
func CPUForm() string {
	return composer.CPUForm(cpudev.Target(cpu.Probe().ISA(), 1), runtime.GOARCH == "arm64")
}

// rowsOf is rows [a, b) of a step's rows' tokens, nil for a step of none.
func rowsOf(tok []int32, a, b int) []int32 {
	if tok == nil {
		return nil
	}
	return tok[a:b]
}

// chunkRows is the sequences and positions of a chunk's n rows of slot
// from position p0, in the instance's own rows.
func (s *Instance) chunkRows(slot, p0, n int) (seq, pos []int) {
	if cap(s.promptPos) < n {
		s.promptSeq, s.promptPos = make([]int, n), make([]int, n)
	}
	seq, pos = s.promptSeq[:n], s.promptPos[:n]
	for i := range n {
		seq[i], pos[i] = slot, p0+i
	}
	return seq, pos
}

// addr is x's first element's address, nil for a nil slice.
func addr[T any](x []T) unsafe.Pointer { return unsafe.Pointer(unsafe.SliceData(x)) }

// unmapOwn lets the instance's execution on the CPU device go, as a card's
// session view goes (nn.Session): its own buffers and the n-gram table's
// view held off the heap, its workers among what it made.
func (s *Instance) unmapOwn() {
	if d, ok := s.places[routeLanes].(nn.Session); ok {
		d.Detach()
	}
}

// placeLanes composes a new instance's blocks on the CPU: each block of the
// instance's range no device holds, once, at its placement (cpuCompose).
func (s *Instance) placeLanes() {
	// A segment's blocks are past the text model's in the one block space.
	n := max(s.c.NLayer, s.hi)
	planes := [2][]float32{s.tablePlanes(0), s.tablePlanes(1)}
	s.lease = blockLease{m: s.m, fr: s.fr, mu: &s.bind, hold: &s.expHold, res: s.res, hot: s.hot, layers: s.layers,
		lo: s.lo, trace: &s.opt.tracer}
	s.places[routeLanes].(nn.LanePlacer).PlaceLanes(n, &s.lease, planes, s.c.NExpert, s.prof)
	s.bindLanes(s.Session)
	if !s.prepLater {
		s.prepFrames()
	}
	s.placeParts()
}

// prepFrames prepares each block on its route's device where that device runs
// it from the frames; one with a copy of its own took it at its offer.
func (s *Instance) prepFrames() {
	n := max(s.c.NLayer, s.hi)
	for li := s.lo; li < min(s.hi, n); li++ {
		if r := s.slotAt(li); r != slotNone && s.fromFrames[r] {
			s.places[r].PrepLayer(li, s.planFor(li), nil)
		}
	}
}

// sideAtBinding moves the offset of every slice of gr's side input (its
// input past its residual streams) into the input's binding: where each
// slice reads from one offset, the slices read from 0 and the offset is
// returned for the call to bind the input at. A graph whose slices of the
// side differ, or that reads it otherwise, keeps its offsets (0).
func sideAtBinding(gr *graph.Graph) int {
	if len(gr.In) != len(gr.Out)+1 {
		return 0
	}
	side := gr.In[len(gr.Out)]
	off := -1
	for _, o := range gr.Ops {
		if !slices.Contains(o.In, side) {
			continue
		}
		if o.Kind != graph.OpSlice || o.Attr.HeadDim != 0 || off >= 0 && o.Attr.Offset != off {
			return 0
		}
		off = o.Attr.Offset
	}
	if off <= 0 {
		return 0
	}
	gr.Ops = slices.Clone(gr.Ops)
	for i := range gr.Ops {
		if slices.Contains(gr.Ops[i].In, side) {
			gr.Ops[i].Attr.Offset = 0
		}
	}
	return off
}

// blockSpec is what block li is composed from (nn.BlockSpec): the graph its
// front end states, the formats its weights are read in, their strides, the
// step's history geometry and the instance's rotaries -- or what keeps the
// block off the composer's CPU path before it is asked (cpuWhy).
func (s *Instance) blockSpec(li int) nn.BlockSpec {
	return s.specOf(li, s.m.Composing().BlockGraph(li))
}

// specOf is blockSpec over g: block li's graph, or a half of it cut at its
// routed experts (cutBlock).
func (s *Instance) specOf(li int, g *graph.Graph) nn.BlockSpec {
	p := nn.BlockSpec{Why: s.blockWhy(li), Rec: s.c.LayerKind(li).Recurrent()}
	if p.Why != nn.DeclineServes {
		return p
	}
	gr := *g
	gr.Block = 0
	p.Source, p.Graph, p.SideOff = g, &gr, sideAtBinding(&gr)
	if s.m.Ngram != nil {
		p.Table, p.TableRow = s.m.Ngram.E, s.m.Ngram.RowBytes
	}
	formats := map[jlm.Role]jlm.Type{}
	f := s.m.Composing()
	if li >= s.m.TowerBase {
		formats = s.m.TowerFormats(g)
	}
	for _, o := range g.Ops {
		if li >= s.m.TowerBase {
			break
		}
		for _, w := range o.Weight {
			if e, ok := f.Find(w.Role, jlm.DenseBlock, -1); ok && jlm.RowTable(e) {
				formats[w.Role] = e.Type
			} else if e, ok := f.Find(w.Role, int32(li), -1); ok {
				formats[w.Role] = e.Type
				// A table the block lays out as float32 at load is read so.
				if t := s.ws.matrix(li, s.layer(li), w.Role); w.Rows > 0 && t != nil && t.E == nil && t.Typ == quant.F32 {
					formats[w.Role] = jlm.TypeF32
				}
			} else if w.Rows == 0 {
				// A vector a restatement adds is held as every vector is.
				formats[w.Role] = jlm.TypeF32
			}
		}
	}
	pg := &s.kv.layers[li]
	page := max(pg.p, 1)
	p.Formats, p.Strides = formats, s.m.CpuStrides(s.fr, li, g)
	p.Page, p.Keys, p.ActWindow, p.Seqs = page, max(pg.arenaSlots(), 1)*page, s.m.ActWindow, 1
	p.ValuesInKeys = pg.latent
	// A block turns by the rotary its rope ops state, of the
	// instance's (rotaries): each its own table, bound by instance.
	p.Rotaries = s.rotaries()
	p.F32 = s.kvl.fmt != nn.KVF16
	if s.kvl.fmt == nn.KVQ8 {
		p.KVQ8 = s.kvlAt(li).headDim
	}
	// A non-causal model's recurrent block reads both ways too: its
	// convolution is centred over the whole run.
	k := s.c.LayerKind(li)
	p.Bidir = k.Attends() && s.c.BidirAt(li) || k.Recurrent() && s.c.NonCausal
	if k.Recurrent() && s.c.NonCausal {
		p.Run = s.maxSeq
	}
	p.Whole = s.wholePicture(li)
	return p
}

// arena is the table of the layer's arena, nil for a layer with none (or no
// layer).
// kvAddrs is where history pg's keys and values are read, nil for none.
func kvAddrs(pg *kvPages) (keys, values unsafe.Pointer) {
	if t := pg.arena(); t != nil {
		return addr(t.Arena().Keys()), addr(t.Arena().Values())
	}
	return nil, nil
}

func (s *kvPages) arena() *history.Table {
	if s == nil || s.mem == nil {
		return nil
	}
	return s.mem.Table()
}

// arenaSlots is how many pages the layer's arena holds, 0 for none.
func (s *kvPages) arenaSlots() int {
	if t := s.arena(); t != nil {
		return t.Slots()
	}
	return 0
}

// blockWhy is what keeps block li off the composer's CPU path before it is
// asked, as the model and its history state it: no graph, an input the
// device is handed none of, a bank with no expert pages (the forms read a
// bank as one view of its experts in the frames' expert pages,
// jlm.Frames.BankView), a recurrent block with no state, a history format
// the forms do not read (the CPU keeps a history binary16 itself,
// composer.Halves, or at f32 without that pass) or pages outside an arena.
func (s *Instance) blockWhy(li int) nn.Decline {
	c := s.c
	g := s.m.Composing().BlockGraph(li)
	// A block sharing another's history reads its source's pages.
	pg := &s.kv.layers[c.KVSource(li)]
	l := &s.m.Layers[li]
	switch {
	case g == nil:
		return nn.DeclineNoGraph
	case len(g.Out) != c.Streams() || len(g.In) > len(g.Out)+1 || len(g.In) > len(g.Out) && c.SideWidth() == 0:
		return nn.DeclineFeeds
	case c.MoEAt(li) && (s.m.Container == nil ||
		!s.places[routeLanes].(nn.BankViewer).BankViews(li, len(s.m.Layers), [3]*jlm.Entry{l.Gate.E, l.Up.E, l.Down.E}) || l.Up.E == nil || l.Down.E == nil):
		return nn.DeclineExperts
	case c.LayerKind(li).Recurrent() && !s.keepsRec(li):
		return nn.DeclineRecurrent
	case pg.p > 0 && pg.arena() == nil:
		// A paged history with no arena is one with no room for its pages.
		return nn.DeclineNoRoom
	}
	return nn.DeclineServes
}

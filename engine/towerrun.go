package engine

import (
	"fmt"
	"image"
	"slices"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// towerRun is an instance's tower: an instance of its own over the tower's
// blocks (towerConfig), whose history holds one picture's rows at a time,
// and the entry's forms (one a sample format) and the head's on it, all
// composed when the instance is made.
type towerRun struct {
	// parts is the entry, the head and the taps on the device that runs
	// them (nn.TowerParts).
	parts nn.TowerParts
	s     *Instance
	// tok and given are the picture's rows as a prompt's source (prefillSrc).
	tok   []int32
	given [1]givenRun
	// tapRows is the rows each tap's block leaves (tapBuf their buffers,
	// grown with the square) and mode's taps the card's run keeping them
	// (embedMode.taps).
	tapRows  [][]float32
	tapBuf   [][]float32
	modeTaps []embedTap
	// rows is a square's patch rows at the tower's width, as the entry
	// writes them, and blk as the last block leaves them; seq and pos the
	// rows' sequence and positions -- row i at position i, the patch gather
	// and the rotary reading its cell in the grid's nested order -- posTab
	// the step's live rows and positions as the entry reads them, mode the
	// blocks' run on a card (embedMode.rows). Each is as long as the square
	// being encoded, over buffers grown to the largest one yet (size).
	rows, blk  []float32
	seq, pos   []int
	posTab     []uint32
	mode       embedMode
	req        []uint32
	bufE, bufB []float32
	bufS, bufP []int
	bufT       []uint32
	// wins is a square's windows as runs of its rows, rowWins each row's
	// own (a card's row windows).
	wins, rowWins []nn.KeyRun
	// out is the rows EncodePicture hands back, at the language model's
	// width, and deep its taps' (EncodedTaps); encodes counts the pictures
	// encoded.
	out, deep []float32
	encodes   int
	// pad holds a picture's planes where they are not read in place, ent a
	// palette's entries; tapsW and tapsH the source size the request's taps
	// are for.
	pad          []uint32
	ent          [4 * 256]byte
	tapsW, tapsH int
	// pics is a prompt's pictures' rows, one buffer a picture, deeps their
	// taps' rows, and spans the prompt's spans with them in place
	// (encodePictures).
	pics, deeps [][]float32
	spans       []model.Span
	// chain is a chained tower's blocks' own layers, each bound at its page
	// by lease (chainRun); act the two buffers the blocks write their rows
	// in by turns, kept the rows the head reads beside the last block's,
	// and headIn the head's inputs.
	chain []model.Layer
	lease blockLease
	// chainOn is the device each chained block runs on where it is not the
	// CPU device, chainParts its part there (nn.RowsPart), both nil for a
	// block on the CPU device.
	chainOn    []nn.PictureRowsDevice
	chainParts []*nn.RowsPart
	act        [2][]float32
	kept       [][]float32
	headIn     [][]float32
	// base is the tower's first block of the model's: the vision tower's
	// TowerBase, the audio tower's the block after the vision tower's last.
	base int
	// clip is an audio run's clip as the entry reads it, cut and padded,
	// over a buffer as long as the longest clip.
	clip []float32
}

// newTowerRun is a run of the model's tower tg, stated as t, its blocks the
// model's from base on.
func (in *Instance) newTowerRun(tg *model.TowerGraphs, base int, t *jlm.Tower) (*towerRun, error) {
	v := in.m.Container.Vision()
	n := tg.Rows
	c := model.TowerConfig(base, n, v, t)
	hi := base + len(tg.Blocks)
	if tg.Chain {
		// A chain's blocks run as forms over their own grids (chainRun),
		// not as the instance's blocks over one residual.
		hi = base
	}
	ts, err := newInstanceOver(in.m, in.eng, in.fr, in.opt, c, n, base, hi, nil)
	if err != nil {
		return nil, err
	}
	if err := ts.prepErr(); err != nil {
		ts.Close()
		return nil, err
	}
	words := 0
	for _, cl := range tg.Classes {
		words = max(words, (cl.Pic.PicW+cl.Pic.PicH)*(cl.Pic.Taps+1))
	}
	tr := &towerRun{s: ts, base: base, req: make([]uint32, composer.RequestPicTaps+words)}
	hn := n
	var chain []nn.PartPlan
	if tg.Chain {
		hn = tg.Head.Values[tg.Head.In[0]].Rows
		for _, g := range tg.Blocks {
			chain = append(chain, ts.partPlan(g, nn.DeclineNoGraph, g.Values[g.In[0]].Rows, in.m.TowerFormats(g)))
		}
	}
	plans := nn.TowerPlans{Head: ts.partPlan(tg.Head, nn.DeclineNoGraph, hn, nil), Chain: chain}
	for _, cl := range tg.Classes {
		e := map[uint8]nn.PartPlan{}
		for px, g := range cl.Entry {
			e[px] = ts.partPlan(g, nn.DeclineNoGraph, cl.Rows, nil)
		}
		plans.Entry = append(plans.Entry, e)
	}
	tapsW := make([]nn.Tensors, len(tg.Taps))
	for j, g := range tg.Taps {
		plans.Taps = append(plans.Taps, in.partPlan(g, nn.DeclineNoGraph, n, in.m.TowerTapFormats(int32(tg.TapBlocks[j]), g)))
		tapsW[j] = &ts.dense.TowerTaps[j]
	}
	parts, err := ts.places[routeLanes].(nn.TowerPlacer).PlaceTower(plans, &ts.dense.TowerDense, tapsW)
	if err != nil {
		ts.Close()
		return nil, fmt.Errorf("model: %w", err)
	}
	tr.parts = parts
	if tg.Chain {
		if err := tr.chainSetup(in.m, ts, tg); err != nil {
			ts.Close()
			return nil, err
		}
	}
	tr.tapRows, tr.tapBuf = make([][]float32, len(tg.Taps)), make([][]float32, len(tg.Taps))
	for li := ts.lo; li < ts.hi; li++ {
		if why := ts.places[routeLanes].(nn.BlockRows).RowsServed(li); why != nn.DeclineServes {
			ts.Close()
			return nil, fmt.Errorf("model: the tower's block %d does not run on the CPU: %v", li-in.m.TowerBase, why)
		}
	}
	return tr, nil
}

// squareRequest writes square sq's words of a picture of squares squares
// into the request, its picture op's attributes a: its grid's width, the
// size the op resizes to, and where Pillow's filters resize it the taps of
// the square's window of the resize -- one picture size's whole-picture taps
// kept from one picture to the next.
func (tr *towerRun) squareRequest(pl model.PictureLayout, sq model.Square, a graph.Attrs, squares int) {
	tr.req[composer.RequestGrid], tr.req[composer.RequestGridH] = uint32(sq.Gw), uint32(sq.Gh)
	tr.req[composer.RequestMark], tr.req[composer.RequestTrail] = uint32(sq.Mark), uint32(sq.Trail)
	tr.req[composer.RequestMarkN], tr.req[composer.RequestTrailN] = uint32(sq.MarkN), uint32(sq.TrailN)
	tr.req[composer.RequestPicOW], tr.req[composer.RequestPicOH] = uint32(sq.Pw), uint32(sq.Ph)
	if a.Chain != graph.ChainPIL || squares == 1 && pl.W == tr.tapsW && pl.H == tr.tapsH {
		return
	}
	composer.PILTapsAt(a.Filter, a.PicTorch, pl.W, sq.Ow, sq.X0, sq.Pw, a.Taps, tr.req[composer.RequestPicTaps:])
	composer.PILTapsAt(a.Filter, a.PicTorch, pl.H, sq.Oh, sq.Y0, sq.Ph, a.Taps, tr.req[composer.RequestPicTaps+a.PicW*(a.Taps+1):])
	tr.tapsW, tr.tapsH = pl.W, pl.H
	if squares > 1 {
		tr.tapsW = 0
	}
}

// size is the run's buffers over n rows of width e, row i at position i:
// grown, their positions written, where the square is the largest yet.
func (tr *towerRun) size(n, e int) {
	if len(tr.bufS) < n {
		tr.bufE, tr.bufB = make([]float32, n*e), make([]float32, n*e)
		tr.bufS, tr.bufP, tr.bufT = make([]int, n), make([]int, n), make([]uint32, 1+n)
		for i := range n {
			tr.bufP[i], tr.bufT[1+i] = i, uint32(i)
		}
		for j := range tr.tapBuf {
			tr.tapBuf[j] = make([]float32, n*e)
		}
	}
	for j := range tr.tapRows {
		tr.tapRows[j] = tr.tapBuf[j][:n*e]
	}
	tr.rows, tr.blk = tr.bufE[:n*e], tr.bufB[:n*e]
	tr.seq, tr.pos, tr.posTab = tr.bufS[:n], tr.bufP[:n], tr.bufT[:1+n]
	tr.posTab[0] = uint32(n)
}

// windows is a gw x gh patch grid's windows as runs of its rows in their
// nested order (towerTiles): the lead rows one run, then window by window;
// nil for a tower whose blocks attend the whole picture.
func (tr *towerRun) windows(tg *model.TowerGraphs, gw, gh int) []nn.KeyRun {
	if tg.Window < 1 {
		return nil
	}
	m, w := tg.Merge, tg.Window
	gw2, gh2 := gw/m, gh/m
	tr.wins = tr.wins[:0]
	at := tg.Lead
	if at > 0 {
		tr.wins = append(tr.wins, nn.KeyRun{Lo: 0, Hi: at})
	}
	for top := 0; top < gh2; top += w {
		for left := 0; left < gw2; left += w {
			n := min(w, gh2-top) * min(w, gw2-left) * m * m
			tr.wins = append(tr.wins, nn.KeyRun{Lo: at, Hi: at + n})
			at += n
		}
	}
	return tr.wins
}

// close lets the tower's instance go.
func (tr *towerRun) close() {
	if tr != nil {
		tr.s.Close()
	}
}

// EncodePicture is the rows a picture's span puts in the prompt: the picture
// through the tower's entry, its blocks over the patch rows as one run
// attending both ways, and its head, every step on the CPU device's composed
// forms. The rows are the instance's until its next encode.
func (s *Instance) EncodePicture(img image.Image) ([]float32, error) {
	if err := s.towerReady(); err != nil {
		return nil, err
	}
	defer s.fr.Enter()()
	b := img.Bounds()
	n := s.m.Tower.PicRows(b.Dx(), b.Dy()) * int(s.m.Container.Vision().ProjDim)
	k := len(s.m.Tower.Taps)
	if cap(s.tower.out) < n {
		s.tower.out, s.tower.deep = make([]float32, n), make([]float32, k*n)
	}
	return s.encodePicture(img, s.tower.out[:n], s.tower.deep[:k*n])
}

// EncodedTaps is the last picture EncodePicture encoded's taps' rows
// (jlm.TowerTap), [taps][rows][width]: Qwen3-VL's deepstack rows. A tower
// with no taps has none.
func (s *Instance) EncodedTaps() []float32 {
	if s.tower == nil {
		return nil
	}
	return s.tower.deep
}

// towerReady is why the instance cannot encode a picture, nil when it can.
func (s *Instance) towerReady() error {
	switch {
	case s.tower != nil:
		return nil
	case s.towerErr != nil:
		return fmt.Errorf("model: the tower did not compose: %w", s.towerErr)
	}
	return fmt.Errorf("model: the model has no vision tower")
}

// encodePicture is EncodePicture into dst, the head's rows, and deep, its
// taps' ([taps][rows][width]), under the pager the caller holds.
func (s *Instance) encodePicture(img image.Image, dst, deep []float32) ([]float32, error) {
	tr := s.tower
	tr.encodes++
	words, pl, err := model.PictureWords(img, &tr.pad, &tr.ent)
	if err != nil {
		return nil, err
	}
	ts, m := tr.s, s.m
	tg := m.Tower
	pl.Request(tr.req)
	if a := tg.Classes[0].Pic; a.Chain == graph.ChainPIL && (pl.W > a.PicSide || pl.H > a.PicSide) {
		return nil, fmt.Errorf("model: a %dx%d picture is past the %d pixels a side the tower's resize reaches", pl.W, pl.H, a.PicSide)
	}
	D, E := int(m.Container.Vision().ProjDim), int(m.Container.Vision().NEmbd)
	if want := tg.PicRows(pl.W, pl.H); len(dst) != want*D || len(deep) != len(tg.Taps)*want*D {
		return nil, fmt.Errorf("model: %d values and %d for its taps' for a picture's %d rows of %d", len(dst), len(deep), want, D)
	}
	sq, squares := tg.Square(pl.W, pl.H, 0)
	if tg.Chain {
		tr.squareRequest(pl, sq, tg.Classes[sq.Class].Pic, squares)
		if err := ts.chainRun(tr, tg, words, pl.Px, dst, tg.Classes[0].Rows, tg.Classes[0].Rows); err != nil {
			return nil, fmt.Errorf("model: %w", err)
		}
		return dst, nil
	}
	at := 0
	for i := range squares {
		if i > 0 {
			sq, _ = tg.Square(pl.W, pl.H, i)
		}
		tr.squareRequest(pl, sq, tg.Classes[sq.Class].Pic, squares)
		n := tg.Lead + sq.Gw*sq.Gh + tg.Tail
		tr.size(n, E)
		if err := tr.parts.Entry(sq.Class, pl.Px, words, tr.req, tr.posTab, n, tr.rows); err != nil {
			return nil, fmt.Errorf("model: %w", err)
		}
		if err := ts.towerBlocks(tr, sq.Gw, tr.windows(tg, sq.Gw, sq.Gh)); err != nil {
			return nil, fmt.Errorf("model: the tower's blocks: %w", err)
		}
		if err := tr.parts.Head(n, tr.req, tr.posTab, tr.blk, dst[at*D:]); err != nil {
			return nil, fmt.Errorf("model: %w", err)
		}
		// Each tap's rows at the same rows of its plane of deep.
		for j := range tg.Taps {
			if err := tr.parts.Tap(j, n, tr.req, tr.posTab, tr.tapRows[j], deep[(j*len(dst)/D+at)*D:]); err != nil {
				return nil, fmt.Errorf("model: %w", err)
			}
		}
		at += tg.SquareOut(sq)
	}
	return dst, nil
}

// wholePicture says tower block li attends the whole picture though its
// rows carry windows (TowerBlock.Full).
func (s *Instance) wholePicture(li int) bool {
	return s.towered() && s.m.Tower.Full[li-s.lo]
}

// towerBlocks runs the tower's blocks over the patch rows of a gw x gh grid,
// one run in which every row's keys are every row's, leaving the last
// block's rows in blk: the tower instance's prefill (embedMode.rows), each
// run of blocks on the device that holds it. Blocks that turn
// q and k read each row's cell of the grid from its position and the
// picture's request (the tower's grid table, on a card the card's own from
// the rows' cells and the grid's width gw); a windowed block attends within
// its row's run of wins, every other block the whole picture.
func (ts *Instance) towerBlocks(tr *towerRun, gw int, wins []nn.KeyRun) error {
	n := len(tr.pos)
	ts.bidir = append(ts.bidir[:0], nn.KeyRun{Lo: 0, Hi: n})
	defer func() { ts.bidir = ts.bidir[:0] }()
	if ts.tables != nil {
		ts.tables.SetGrid(tr.req, tr.posTab)
	}
	ts.Reset()
	if ts.m.Tower.Table != nil && ts.devCount() > 0 {
		gc, ok := ts.ld.(nn.GridCellsDevice)
		if !ok || !gc.SetGridCells(tr.posTab, gw) {
			return fmt.Errorf("the tower's blocks turn by the picture's grid on a device that holds no grid table")
		}
	}
	tr.mode = embedMode{bidir: true, rows: true, total: n, out: tr.blk, wins: tr.mode.wins}
	// The rows each tapped block leaves, which a run ending at that block
	// writes back.
	tr.modeTaps = tr.modeTaps[:0]
	for j, blk := range ts.m.Tower.TapBlocks {
		tr.modeTaps = append(tr.modeTaps, embedTap{li: ts.lo + blk, rows: tr.tapRows[j]})
	}
	tr.mode.taps = tr.modeTaps
	if wins != nil {
		// A windowed block reads each row's window (graph.Attrs.RowWindows),
		// handed with the call (embedTellOn), on whichever device holds it.
		tr.rowWins = tr.rowWins[:0]
		for _, w := range wins {
			for range w.Hi - w.Lo {
				tr.rowWins = append(tr.rowWins, w)
			}
		}
		tr.mode.rowWins = tr.rowWins
	}
	ts.emb = &tr.mode
	defer func() { ts.emb = nil }()
	// The picture's rows as the prompt's given rows, from the run's own
	// scratch: a warm encode allocates nothing for them.
	tr.tok = slices.Grow(tr.tok[:0], n)[:n]
	for i := range tr.tok {
		tr.tok[i] = -1
	}
	tr.given[0] = givenRun{rows: tr.rows[:n*ts.c.NEmbd], n: n}
	_, err := ts.prefillSrc(0, rowSrc{tok: tr.tok, given: tr.given[:]})
	return err
}

// encodePictures is spans with every picture span's rows in place: encoded
// into a buffer of the instance's (one a picture of the prompt), or, for a
// picture whose rows all lie before row from -- restored by the prefix
// cache -- a buffer nothing reads, with no tower block run. The caller's
// spans are left as they were.
func (s *Instance) encodePictures(spans []model.Span, from int) ([]model.Span, error) {
	n := 0
	for _, sp := range spans {
		if sp.Picture != nil && sp.Embd == nil {
			n++
		}
	}
	if n == 0 {
		return spans, nil
	}
	if err := s.towerReady(); err != nil {
		return nil, err
	}
	tr := s.tower
	D := int(s.m.Container.Vision().ProjDim)
	tr.spans = append(tr.spans[:0], spans...)
	k, row := 0, 0
	for i := range tr.spans {
		sp := &tr.spans[i]
		r := sp.N(s.c.NEmbd)
		if sp.Picture != nil && sp.Embd == nil {
			b := sp.Picture.Bounds()
			if want := s.m.Tower.PicRows(b.Dx(), b.Dy()); sp.PicRows != want {
				return nil, fmt.Errorf("model: span %d is a picture of %d rows and the tower writes %d",
					i, sp.PicRows, want)
			}
			if k == len(tr.pics) {
				tr.pics = append(tr.pics, nil)
			}
			if k == len(tr.deeps) {
				tr.deeps = append(tr.deeps, nil)
			}
			if cap(tr.pics[k]) < sp.PicRows*D {
				tr.pics[k], tr.deeps[k] = make([]float32, sp.PicRows*D), make([]float32, len(s.m.Tower.Taps)*sp.PicRows*D)
			}
			sp.Embd, sp.Deep = tr.pics[k][:sp.PicRows*D], tr.deeps[k][:len(s.m.Tower.Taps)*sp.PicRows*D]
			if len(s.m.Tower.Taps) == 0 {
				sp.Deep = nil
			}
			k++
			if row+r > from {
				if _, err := s.encodePicture(sp.Picture, sp.Embd, tr.deeps[k-1][:len(s.m.Tower.Taps)*sp.PicRows*D]); err != nil {
					return nil, fmt.Errorf("model: span %d: %w", i, err)
				}
			}
		}
		row += r
	}
	return tr.spans, nil
}

// chainSetup is a chained tower's run: each block's own copy of its layer,
// which the lease binds at the block's page as the instance's pager reads
// it in, and the buffers the blocks' rows are written in, as large as the
// largest grid of rows any block or the entry writes.
func (tr *towerRun) chainSetup(m *model.Model, ts *Instance, tg *model.TowerGraphs) error {
	if _, ok := tr.parts.(nn.TowerChain); !ok {
		return fmt.Errorf("model: the device holding the tower runs no chained blocks")
	}
	e := tg.Classes[0].Entry[graph.PixRGBA8]
	o := e.Values[e.Out[0]]
	most := o.Rows * o.Width
	tr.chain = make([]model.Layer, len(tg.Blocks))
	for i, g := range tg.Blocks {
		m.Layers[tr.base+i].CloneTo(&tr.chain[i])
		o := g.Values[g.Out[0]]
		most = max(most, o.Rows*o.Width)
	}
	tr.lease = blockLease{m: m, fr: ts.fr, mu: &ts.bind, layers: tr.chain, lo: tr.base}
	tr.act = [2][]float32{make([]float32, most), make([]float32, most)}
	for _, j := range tg.HeadFrom {
		o := tg.Blocks[j].Values[tg.Blocks[j].Out[0]]
		tr.kept = append(tr.kept, make([]float32, o.Rows*o.Width))
	}
	tr.headIn = make([][]float32, 1+len(tg.HeadFrom))
	return nil
}

// chainRun runs a chained tower over a picture's or a clip's words, its
// request in the run's: the entry into the first block's rows over live of
// its rows, each block in turn over blockLive of the rows the block before
// it left -- its page read in and its form bound there -- the rows the head
// reads beside the last block's kept as their blocks leave them, then the
// head into dst.
func (ts *Instance) chainRun(tr *towerRun, tg *model.TowerGraphs, words []uint32, px uint8, dst []float32, live, blockLive int) error {
	ch := tr.parts.(nn.TowerChain)
	cl := tg.Classes[0]
	tr.size(cl.Rows, 0)
	tr.posTab[0] = uint32(live)
	e := cl.Entry[px]
	o := e.Values[e.Out[0]]
	cur := tr.act[0][:o.Rows*o.Width]
	if err := tr.parts.Entry(0, px, words, tr.req, tr.posTab, cl.Rows, cur); err != nil {
		return err
	}
	tr.posTab[0] = uint32(blockLive)
	k := 0
	for i, g := range tg.Blocks {
		li := tr.base + i
		o := g.Values[g.Out[0]]
		out := tr.act[(i+1)%2][:o.Rows*o.Width]
		if dev := tr.chainDev(i); dev != nil {
			n := g.Values[g.In[0]].Rows
			if !dev.RunRows(tr.chainParts[i], n, tr.req, tr.posTab[:n+1], cur, out) {
				return fmt.Errorf("the tower's block %d did not run on its device", i)
			}
		} else {
			w, err := tr.lease.Ensure(li)
			if err != nil {
				return err
			}
			if err := ch.ChainBlock(i, li, w, tr.req, tr.posTab, cur, out); err != nil {
				return err
			}
		}
		if k < len(tg.HeadFrom) && tg.HeadFrom[k] == i {
			copy(tr.kept[k], out)
			k++
		}
		cur = out
	}
	tr.headIn[0] = cur
	copy(tr.headIn[1:], tr.kept)
	return ch.ChainHead(tr.req, tr.posTab, dst, tr.headIn...)
}

// chainDev is the device chained block i runs on, nil for the CPU device.
func (tr *towerRun) chainDev(i int) nn.PictureRowsDevice {
	if i < len(tr.chainOn) {
		return tr.chainOn[i]
	}
	return nil
}

// PlaceTowerChain puts a chained tower's first n blocks on dev and the rest
// on the CPU device: each placed block's part formed and its tensors held
// there now, so no picture forms one. n 0 brings every block back.
func (s *Instance) PlaceTowerChain(dev nn.PictureRowsDevice, n int) error {
	if err := s.towerReady(); err != nil {
		return err
	}
	tr, tg := s.tower, s.m.Tower
	if !tg.Chain || n < 0 || n > len(tg.Blocks) {
		return fmt.Errorf("model: %d of a tower's %d chained blocks (chained %v)", n, len(tg.Blocks), tg.Chain)
	}
	if tr.chainOn == nil {
		tr.chainOn, tr.chainParts = make([]nn.PictureRowsDevice, len(tg.Blocks)), make([]*nn.RowsPart, len(tg.Blocks))
	}
	for i, g := range tg.Blocks {
		tr.chainOn[i], tr.chainParts[i] = nil, nil
		if i >= n {
			continue
		}
		p, err := chainPart(s.m, i, g)
		if err != nil {
			return err
		}
		if !dev.PrepRows(p) {
			return fmt.Errorf("model: the tower's block %d does not form on the device", i)
		}
		tr.chainOn[i], tr.chainParts[i] = dev, p
	}
	return nil
}

// chainPart is chained block i as a part of a device of its own: its graph
// over its input's rows, its matrices the container's entries, which the
// device reads into frames of its own, and its vectors.
func chainPart(m *model.Model, i int, g *graph.Graph) (*nn.RowsPart, error) {
	l := &m.Layers[m.TowerBase+i]
	p := &nn.RowsPart{Graph: g, Rows: g.Values[g.In[0]].Rows, Weights: map[jlm.Role]nn.Weight{}, Vectors: map[jlm.Role][]float32{},
		File: m.Container}
	for _, o := range g.Ops {
		for _, b := range o.Weight {
			if v := l.Vector(b.Role); len(v) > 0 {
				p.Vectors[b.Role] = v
				continue
			}
			t := l.Matrix(b.Role)
			if t == nil || t.E == nil {
				return nil, fmt.Errorf("model: the tower's block %d binds %v, which it does not hold", i, b.Role)
			}
			p.Weights[b.Role] = model.Wt(*t)
		}
	}
	return p, nil
}

package engine

import (
	"fmt"
	"image"
	"unsafe"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// towerRun is an instance's tower: an instance of its own over the tower's
// blocks (towerConfig), whose history holds one picture's rows at a time,
// and the entry's forms (one a sample format) and the head's on it, all
// composed when the instance is made.
type towerRun struct {
	s *Instance
	// entry is a class's entry forms, one a sample format.
	entry []map[uint8]*cpudev.Form
	head  *cpudev.Form
	// taps is each tap's form, tapRows the rows its block leaves (tapBuf
	// their buffers, grown with the square) and mode's taps the card's run
	// keeping them (embedMode.taps).
	taps     []*cpudev.Form
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
}

// newTowerRun is the model's tower run, nil for a model with no tower.
func (in *Instance) newTowerRun() (*towerRun, error) {
	tg := in.m.Tower
	if tg == nil {
		return nil, nil
	}
	v := in.m.Container.Vision()
	n := tg.Rows
	c := model.TowerConfig(in.m.TowerBase, n, v, v.Tower)
	ts, err := newInstanceOver(in.m, in.eng, in.fr, in.opt, c, 1, n, in.m.TowerBase, in.m.TowerBase+len(tg.Blocks))
	if err != nil {
		return nil, err
	}
	if ts.cpu.formErr != nil {
		ts.Close()
		return nil, ts.cpu.formErr
	}
	words := 0
	for _, cl := range tg.Classes {
		words = max(words, (cl.Pic.PicW+cl.Pic.PicH)*(cl.Pic.Taps+1))
	}
	tr := &towerRun{s: ts, req: make([]uint32, composer.RequestPicTaps+words)}
	for _, cl := range tg.Classes {
		forms := map[uint8]*cpudev.Form{}
		for px, g := range cl.Entry {
			f, err := ts.formCPUGraph(g, cl.Rows, nil)
			if err != nil {
				ts.Close()
				return nil, fmt.Errorf("model: the tower's entry: %w", err)
			}
			forms[px] = f
		}
		tr.entry = append(tr.entry, forms)
	}
	f, err := ts.formCPUGraph(tg.Head, n, nil)
	if err != nil {
		ts.Close()
		return nil, fmt.Errorf("model: the tower's head: %w", err)
	}
	tr.head = f
	for j, g := range tg.Taps {
		f, err := in.formCPUGraph(g, n, in.m.TowerTapFormats(int32(tg.TapBlocks[j]), g))
		if err != nil {
			ts.Close()
			return nil, fmt.Errorf("model: the tower's tap %d: %w", j, err)
		}
		tr.taps = append(tr.taps, f)
	}
	tr.tapRows, tr.tapBuf = make([][]float32, len(tg.Taps)), make([][]float32, len(tg.Taps))
	for li := ts.lo; li < ts.hi; li++ {
		if cb := ts.cpu.blocks[li]; cb == nil || cb.rows == nil {
			why := nn.DeclineNoGraph
			if cb != nil {
				why = max(cb.why, cb.rowsWhy)
			}
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
	tr.req[composer.RequestGrid] = uint32(sq.Gw)
	tr.req[composer.RequestPicOW], tr.req[composer.RequestPicOH] = uint32(sq.Pw), uint32(sq.Ph)
	if a.Chain != graph.ChainPIL || squares == 1 && pl.W == tr.tapsW && pl.H == tr.tapsH {
		return
	}
	composer.PILTaps(a.Filter, pl.W, sq.Ow, sq.X0, sq.Pw, a.Taps, tr.req[composer.RequestPicTaps:])
	composer.PILTaps(a.Filter, pl.H, sq.Oh, sq.Y0, sq.Ph, a.Taps, tr.req[composer.RequestPicTaps+a.PicW*(a.Taps+1):])
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
	head := tr.head
	if _, why := ts.ws.bind(head, -1, &ts.dense.TowerDense); why != nn.DeclineServes {
		return nil, fmt.Errorf("model: the tower's head: %v", why)
	}
	D, E := int(m.Container.Vision().ProjDim), int(m.Container.Vision().NEmbd)
	if want := tg.PicRows(pl.W, pl.H); len(dst) != want*D || len(deep) != len(tr.taps)*want*D {
		return nil, fmt.Errorf("model: %d values and %d for its taps' for a picture's %d rows of %d", len(dst), len(deep), want, D)
	}
	for j, f := range tr.taps {
		if _, why := ts.ws.bind(f, -1, &ts.dense.TowerTaps[j]); why != nn.DeclineServes {
			return nil, fmt.Errorf("model: the tower's tap %d: %v", j, why)
		}
	}
	b := &ts.cpu.bound
	sq, squares := tg.Square(pl.W, pl.H, 0)
	at := 0
	for i := range squares {
		if i > 0 {
			sq, _ = tg.Square(pl.W, pl.H, i)
		}
		f := tr.entry[sq.Class][pl.Px]
		if _, why := ts.ws.bind(f, -1, &ts.dense.TowerDense); why != nn.DeclineServes {
			return nil, fmt.Errorf("model: the tower's entry: %v", why)
		}
		tr.squareRequest(pl, sq, tg.Classes[sq.Class].Pic, squares)
		n := tg.Lead + sq.Gw*sq.Gh
		tr.size(n, E)
		*b = cpuBound{li: -1, l: &ts.dense.TowerDense, own: ts.cpu.dev.Own(f), n: n, req: tr.req, pos: tr.posTab, pic: words}
		b.io[0] = unsafe.Pointer(&tr.rows[0])
		ts.cpuRunForm(b, f)
		if err := ts.towerBlocks(tr, sq.Gw, tr.windows(tg, sq.Gw, sq.Gh)); err != nil {
			return nil, fmt.Errorf("model: the tower's blocks: %w", err)
		}
		*b = cpuBound{li: -1, l: &ts.dense.TowerDense, own: ts.cpu.dev.Own(head), n: n, req: tr.req, pos: tr.posTab}
		b.io[0], b.io[1] = unsafe.Pointer(&tr.blk[0]), unsafe.Pointer(&dst[at*D])
		ts.cpuRunForm(b, head)
		// Each tap's rows at the same rows of its plane of deep.
		for j, f := range tr.taps {
			*b = cpuBound{li: -1, l: &ts.dense.TowerTaps[j], own: ts.cpu.dev.Own(f), n: n, req: tr.req, pos: tr.posTab}
			b.io[0], b.io[1] = unsafe.Pointer(&tr.tapRows[j][0]), unsafe.Pointer(&deep[(j*len(dst)/D+at)*D])
			ts.cpuRunForm(b, f)
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
// block's rows in blk: on the CPU device, the runner's CPU rows over the
// run, block by block; with blocks on a card, the tower instance's prefill
// (embedMode.rows), each block on the device that holds it. Blocks that turn
// q and k read each row's cell of the grid from its position and the
// picture's request (the tower's grid table, on a card the card's own from
// the rows' cells and the grid's width gw); a windowed block attends within
// its row's run of wins, every other block the whole picture.
func (ts *Instance) towerBlocks(tr *towerRun, gw int, wins []nn.KeyRun) error {
	n := len(tr.pos)
	ts.bidir = append(ts.bidir[:0], nn.KeyRun{Lo: 0, Hi: n})
	defer func() { ts.bidir, ts.cpu.runWindows, ts.cpu.rowWins = ts.bidir[:0], false, nil }()
	ts.cpu.tables.req, ts.cpu.tables.pos = tr.req, tr.posTab
	if ts.devCount() > 0 {
		ts.Reset()
		if ts.m.Tower.Table != nil {
			gc, ok := ts.ld.(nn.GridCellsDevice)
			if !ok || !gc.SetGridCells(tr.posTab, gw) {
				return fmt.Errorf("the tower's blocks turn by the picture's grid on a device that holds no grid table")
			}
		}
		tr.mode = embedMode{bidir: true, rows: true, total: n, out: tr.blk, wins: tr.mode.wins}
		// The rows each tapped block leaves, which a card writes back where
		// its run ends at that block.
		tr.modeTaps = tr.modeTaps[:0]
		for j, blk := range ts.m.Tower.TapBlocks {
			tr.modeTaps = append(tr.modeTaps, embedTap{li: ts.lo + blk, rows: tr.tapRows[j]})
		}
		tr.mode.taps = tr.modeTaps
		if wins != nil {
			// A card's windowed blocks read each row's window (graph.Attrs
			// .RowWindows), handed with the call (embedTell); the blocks the
			// CPU device keeps read them per row as well.
			tr.rowWins = tr.rowWins[:0]
			for _, w := range wins {
				for range w.Hi - w.Lo {
					tr.rowWins = append(tr.rowWins, w)
				}
			}
			tr.mode.rowWins, ts.cpu.rowWins = tr.rowWins, tr.rowWins
		}
		ts.emb = &tr.mode
		defer func() { ts.emb = nil }()
		_, err := ts.prefillSpans(0, model.Span{Embd: tr.rows})
		return err
	}
	if w := ts.c.RopeW(); len(ts.bcs) < n*w {
		ts.bcs = make([]float32, n*w)
	}
	ts.cpu.tabsFresh = false
	copy(tr.blk, tr.rows)
	for li := ts.lo; li < ts.hi; li++ {
		l, err := ts.pageIn(li)
		if err != nil {
			return err
		}
		// Each run of a windowed block is a window its rows attend inside.
		windowed := wins != nil && !ts.m.Tower.Full[li-ts.lo]
		if windowed {
			ts.bidir = append(ts.bidir[:0], wins...)
		} else {
			ts.bidir = append(ts.bidir[:0], nn.KeyRun{Lo: 0, Hi: n})
		}
		ts.cpu.runWindows = windowed
		why, err := ts.cpuRows(li, l, tr.blk, nil, nil, tr.seq, tr.pos, nil)
		if err != nil {
			return err
		}
		if why != nn.DeclineServes {
			return fmt.Errorf("block %d: %v", li-ts.m.TowerBase, why)
		}
		for j, blk := range ts.m.Tower.TapBlocks {
			if ts.lo+blk == li {
				copy(tr.tapRows[j], tr.blk)
			}
		}
	}
	return nil
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
				tr.pics[k], tr.deeps[k] = make([]float32, sp.PicRows*D), make([]float32, len(tr.taps)*sp.PicRows*D)
			}
			sp.Embd, sp.Deep = tr.pics[k][:sp.PicRows*D], tr.deeps[k][:len(tr.taps)*sp.PicRows*D]
			if len(tr.taps) == 0 {
				sp.Deep = nil
			}
			k++
			if row+r > from {
				if _, err := s.encodePicture(sp.Picture, sp.Embd, tr.deeps[k-1][:len(tr.taps)*sp.PicRows*D]); err != nil {
					return nil, fmt.Errorf("model: span %d: %w", i, err)
				}
			}
		}
		row += r
	}
	return tr.spans, nil
}

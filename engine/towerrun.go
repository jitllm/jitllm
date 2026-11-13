package engine

import (
	"fmt"
	"image"
	"slices"
	"unsafe"

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
	// cols says the square squareRequest wrote runs in column bands, its
	// windows outrunning its taps, and colsPl, colsSq and colsPic are its
	// picture, square and class's picture op (columnEntry).
	cols    bool
	colsPl  model.PictureLayout
	colsSq  model.Square
	colsPic graph.Attrs
	s       *Instance
	// tok and given are the picture's rows as a prompt's source (prefillSrc).
	tok   []int32
	given [1]givenRun
	// squares evaluates the picture being encoded's squares from the
	// tower's geometry, its scratch kept from one picture to the next.
	squares model.PictureSquares
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
	// palette's entries; tapsW and tapsH the source size, tapsOw and tapsOh the
	// size it resizes to, the request's taps are for.
	pad            []uint32
	ent, ent2      [4 * 256]byte
	tapsW, tapsH   int
	tapsOw, tapsOh int
	// tapsVideo says the taps in the request are a video class's, whose
	// count is not a picture class's.
	tapsVideo bool
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
	// clip is an audio run's step picture table, a clip's PCM and its plan
	// (composer.SoundTable), over a buffer as long as the longest clip's; swr
	// is the resampler from the last clip's rate, mix its channels' weights
	// (for mixOf, its channels and mask).
	clip  []uint32
	swr   composer.SWR
	mix   []float32
	mixOf [2]uint64
	// stageOut is the picture's bytes after each whole-picture resize
	// (model.TowerGraphs.Stages), a word a pixel, the stages taking turns:
	// each reads the last one's.
	stageOut [2][]float32
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
	// A tower block's keys are read by its own call alone: the run holds
	// one block's history at a time.
	ts.kv.spent = true
	// A video class's entry is class len(tg.Classes)+i of the parts', and a
	// picture stage's past them.
	classes := append(append(tg.Classes[:len(tg.Classes):len(tg.Classes)], tg.Videos...), tg.Stages...)
	words, stage := 0, 0
	for _, cl := range classes {
		words = max(words, composer.PicEntries(cl.Pic)*composer.PILEntryWords(cl.Pic.Taps))
	}
	for _, cl := range tg.Stages {
		stage = max(stage, cl.Rows*model.StageWidth)
	}
	tr := &towerRun{s: ts, base: base, req: make([]uint32, composer.RequestPicTaps+words)}
	if stage > 0 {
		tr.stageOut = [2][]float32{make([]float32, stage), make([]float32, stage)}
	}
	hn := n
	var chain []nn.PartPlan
	if tg.Chain {
		hn = tg.Head.Values[tg.Head.In[0]].Rows
		for _, g := range tg.Blocks {
			chain = append(chain, ts.partPlan(g, g.Values[g.In[0]].Rows, in.m.TowerFormats(g)))
		}
	}
	plans := nn.TowerPlans{Head: ts.partPlan(tg.Head, hn, nil), Chain: chain}
	for _, cl := range classes {
		e := map[uint8]nn.PartPlan{}
		for px, g := range cl.Entry {
			e[px] = ts.partPlan(g, cl.Rows, nil)
		}
		plans.Entry = append(plans.Entry, e)
		var band map[uint8]nn.PartPlan
		for px, g := range cl.Band {
			if band == nil {
				band = map[uint8]nn.PartPlan{}
			}
			band[px] = ts.partPlan(g, cl.Rows, nil)
		}
		if band != nil {
			plans.Acc = max(plans.Acc, cl.Pic.Area()*3)
			plans.BandOut = max(plans.BandOut, cl.Rows*3)
			plans.HAcc = max(plans.HAcc, composer.PicHCap(model.ColumnPic(cl.Pic))*3)
		}
		plans.Band = append(plans.Band, band)
	}
	tapsW := make([]nn.Tensors, len(tg.Taps))
	for j, g := range tg.Taps {
		plans.Taps = append(plans.Taps, in.partPlan(g, n, in.m.TowerTapFormats(int32(tg.TapBlocks[j]), g)))
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
	tr.req[composer.RequestPicCW], tr.req[composer.RequestPicCH] = uint32(sq.Cw), uint32(sq.Ch)
	tr.req[composer.RequestPicCX], tr.req[composer.RequestPicCY] = uint32(sq.Cx), uint32(sq.Cy)
	tr.req[composer.RequestPicBandRows] = uint32(composer.PicBandRows(a, sq.Pw))
	// A window wider than the taps on either axis runs in column bands, its
	// entries written band by band (entry).
	tr.cols = a.Chain == graph.ChainPIL &&
		(composer.PILTapCount(a.Filter, pl.W, sq.Ow) > a.Taps || composer.PILTapCount(a.Filter, pl.H, sq.Oh) > a.Taps)
	if tr.cols {
		tr.colsPl, tr.colsSq, tr.colsPic, tr.tapsW = pl, sq, a, 0
		tr.req[composer.RequestPicBandRows] = uint32(min(composer.PicBandRows(model.ColumnPic(a), sq.Pw), a.Taps))
		return
	}
	if a.Chain != graph.ChainPIL || squares == 1 && pl.W == tr.tapsW && pl.H == tr.tapsH && sq.Ow == tr.tapsOw && sq.Oh == tr.tapsOh {
		return
	}
	composer.PILTapsAt(a.Filter, a.PicTorch, pl.W, sq.Ow, sq.X0, sq.Pw, a.Taps, tr.req[composer.RequestPicTaps:])
	composer.PILTapsAt(a.Filter, a.PicTorch, pl.H, sq.Oh, sq.Y0, sq.Ph, a.Taps, tr.req[composer.RequestPicTaps+sq.Pw*composer.PILEntryWords(a.Taps):])
	tr.tapsW, tr.tapsH, tr.tapsOw, tr.tapsOh = pl.W, pl.H, sq.Ow, sq.Oh
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
	if _, err := s.tower.squares.Of(s.m.Tower, b.Dx(), b.Dy()); err != nil {
		return nil, err
	}
	rows, err := s.tower.squares.Rows()
	if err != nil {
		return nil, err
	}
	n := rows * int(s.m.Container.Vision().ProjDim)
	k := len(s.m.Tower.Taps)
	if cap(s.tower.out) < n {
		s.tower.out, s.tower.deep = make([]float32, n), make([]float32, k*n)
	}
	return s.encodePicture(img, false, s.tower.out[:n], s.tower.deep[:k*n])
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
// taps' ([taps][rows][width]), under the pager the caller holds; whole, a
// video's frame read whole, its last square alone (model.Span.Whole).
func (s *Instance) encodePicture(img image.Image, whole bool, dst, deep []float32) ([]float32, error) {
	tr := s.tower
	tr.encodes++
	words, pl, err := model.PictureWords(img, &tr.pad, &tr.ent)
	if err != nil {
		return nil, err
	}
	ts, m := tr.s, s.m
	tg := m.Tower
	pl.Request(tr.req)
	if tr.tapsVideo {
		tr.tapsW, tr.tapsVideo = 0, false
	}
	D := int(m.Container.Vision().ProjDim)
	ps := &tr.squares
	var sq model.Square
	var squares, want int
	if whole {
		if _, err := ps.FrameOf(tg, pl.W, pl.H); err != nil {
			return nil, err
		}
		if sq, err = ps.Last(); err != nil {
			return nil, err
		}
		want, squares = tg.SquareOut(sq), 1
		if words, pl, err = tr.stages(tg, ps, words, pl); err != nil {
			return nil, err
		}
	} else {
		if squares, err = ps.Of(tg, pl.W, pl.H); err != nil {
			return nil, err
		}
		if want, err = ps.Rows(); err != nil {
			return nil, err
		}
		if sq, err = ps.At(0); err != nil {
			return nil, err
		}
		if words, pl, err = tr.stages(tg, ps, words, pl); err != nil {
			return nil, err
		}
	}
	if len(dst) != want*D || len(deep) != len(tg.Taps)*want*D {
		return nil, fmt.Errorf("model: %d values and %d for its taps' for a picture's %d rows of %d", len(dst), len(deep), want, D)
	}
	if err := squareFits(sq, tg.Classes[sq.Class].Pic); err != nil {
		return nil, err
	}
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
			if sq, err = ps.At(i); err != nil {
				return nil, err
			}
			if err := squareFits(sq, tg.Classes[sq.Class].Pic); err != nil {
				return nil, err
			}
		}
		tr.squareRequest(pl, sq, tg.Classes[sq.Class].Pic, squares)
		if err := ts.towerSquare(tr, sq, sq.Class, pl.Px, words, dst, deep, at); err != nil {
			return nil, err
		}
		at += tg.SquareOut(sq)
	}
	return dst, nil
}

// entry runs class class's entry of sample format px over the picture's
// words into n rows: a banded resize's source rows a band at a time into the
// accumulator, each of its band forms runs in order, then the entry
// finishing it.
func (tr *towerRun) entry(runs []uint8, class int, px uint8, words []uint32, n int, rows []float32) error {
	if tr.cols {
		return tr.columnEntry(runs, class, px, words, n, rows)
	}
	h, side := int(tr.req[composer.RequestPicH]), max(int(tr.req[composer.RequestPicBandRows]), 1)
	for _, k := range runs {
		for b := 0; b == 0 || b < h; b += side {
			tr.req[composer.RequestPicBand] = uint32(b)
			if err := tr.parts.Band(class, k, words, tr.req, tr.posTab, n); err != nil {
				return err
			}
		}
	}
	return tr.parts.Entry(class, px, words, tr.req, tr.posTab, n, rows)
}

// columnEntry is entry for a picture whose windows outrun its taps
// (squareRequest): each band of source rows, its row taps clipped to the
// band, runs its bands of Taps source columns, each with its own share of
// every output's column taps (composer.PILChunk), their sums added into the
// horizontal accumulator; then the row band's pass over those sums into the
// vertical one; then the entry finishes it.
func (tr *towerRun) columnEntry(runs []uint8, class int, px uint8, words []uint32, n int, rows []float32) error {
	a, pl, sq := tr.colsPic, tr.colsPl, tr.colsSq
	K, E := a.Taps, composer.PILEntryWords(a.Taps)
	side := max(int(tr.req[composer.RequestPicBandRows]), 1)
	colTaps, rowTaps := tr.req[composer.RequestPicTaps:], tr.req[composer.RequestPicTaps+sq.Pw*E:]
	colPrec, rowPrec := composer.PILPrec(a.Filter, a.PicTorch, pl.W, sq.Ow), composer.PILPrec(a.Filter, a.PicTorch, pl.H, sq.Oh)
	for _, k := range runs {
		for rb := 0; rb < pl.H; rb += side {
			tr.req[composer.RequestPicBand] = uint32(rb)
			composer.PILChunk(a.Filter, rowPrec, pl.H, sq.Oh, sq.Y0, sq.Ph, K, rb, min(rb+side, pl.H), rowTaps)
			for cb := 0; cb < pl.W; cb += K {
				tr.req[composer.RequestPicCol] = uint32(cb)
				composer.PILChunk(a.Filter, colPrec, pl.W, sq.Ow, sq.X0, sq.Pw, K, cb, min(cb+K, pl.W), colTaps)
				if err := tr.parts.Band(class, model.ColsKey(k), words, tr.req, tr.posTab, n); err != nil {
					return err
				}
			}
			if err := tr.parts.Band(class, model.ColBandKey(k), words, tr.req, tr.posTab, n); err != nil {
				return err
			}
		}
	}
	return tr.parts.Entry(class, px, words, tr.req, tr.posTab, n, rows)
}

// stages runs the bound picture's whole-picture resizes (ps.Stages) over its
// words laid out as pl, each a stage class's entry writing its bytes a word a
// pixel, which the next reads as an RGBA8 picture; it is the last one's words
// and layout, the picture's own where it states none.
func (tr *towerRun) stages(tg *model.TowerGraphs, ps *model.PictureSquares, words []uint32, pl model.PictureLayout) ([]uint32, model.PictureLayout, error) {
	base := len(tg.Classes) + len(tg.Videos)
	for i := range ps.Stages() {
		w, h, err := ps.Stage(i)
		if err != nil {
			return nil, pl, err
		}
		// The smallest stage class that holds it (model.TowerGraphs.Stages).
		c := -1
		for j := range tg.Stages {
			if tg.Stages[j].Pic.Holds(w, h) {
				c = j
				break
			}
		}
		if c < 0 {
			return nil, pl, fmt.Errorf("model: picture stage %d resizes to %dx%d, past every stage class", i, w, h)
		}
		cl := tg.Stages[c]
		out := tr.stageOut[i%2]
		pl.Request(tr.req)
		tr.squareRequest(pl, model.Square{Ow: w, Oh: h, Pw: w, Ph: h, Cw: w, Ch: h}, cl.Pic, 2)
		if err := tr.entry(bandRuns(tg, base+c, pl.Px), base+c, pl.Px, words, cl.Rows, out); err != nil {
			return nil, pl, fmt.Errorf("model: picture stage %d: %w", i, err)
		}
		words, pl = unsafe.Slice((*uint32)(unsafe.Pointer(&out[0])), w*h), model.RGBA8Layout(w, h)
	}
	pl.Request(tr.req)
	return words, pl, nil
}

// bandRuns is the band forms part class class's entry of sample format px
// runs first: a picture class's, or past them a video class's.
func bandRuns(tg *model.TowerGraphs, class int, px uint8) []uint8 {
	switch {
	case class < len(tg.Classes):
		return tg.Classes[class].BandRuns[px]
	case class < len(tg.Classes)+len(tg.Videos):
		return tg.Videos[class-len(tg.Classes)].BandRuns[px]
	}
	return tg.Stages[class-len(tg.Classes)-len(tg.Videos)].BandRuns[px]
}

// squareFits says whether square sq's resized picture fits its class
// (graph.Attrs.Holds).
func squareFits(sq model.Square, a graph.Attrs) error {
	if !a.Holds(sq.Pw, sq.Ph) {
		return fmt.Errorf("model: a picture resized to %dx%d is past what the tower's largest class holds, %d pixels at most %dx%d",
			sq.Pw, sq.Ph, a.Area(), a.PicW, a.PicH)
	}
	if a.PicCanvas && (!a.Holds(sq.Cw, sq.Ch) || sq.Cx+sq.Pw > sq.Cw || sq.Cy+sq.Ph > sq.Ch) {
		return fmt.Errorf("model: a picture of %dx%d at (%d, %d) on a canvas of %dx%d, the class's at most %dx%d of %d pixels",
			sq.Pw, sq.Ph, sq.Cx, sq.Cy, sq.Cw, sq.Ch, a.PicW, a.PicH, a.Area())
	}
	return nil
}

// towerSquare runs square sq, its request written, through the tower: the
// entry's class class over the step's picture words, the blocks, the head
// into dst's rows from at, and each tap into deep's.
func (ts *Instance) towerSquare(tr *towerRun, sq model.Square, class int, px uint8, words []uint32, dst, deep []float32, at int) error {
	tg := ts.m.Tower
	D, E := int(ts.m.Container.Vision().ProjDim), int(ts.m.Container.Vision().NEmbd)
	n := tg.Lead + sq.Gw*sq.Gh + tg.Tail
	tr.size(n, E)
	if err := tr.entry(bandRuns(tg, class, px), class, px, words, n, tr.rows); err != nil {
		return fmt.Errorf("model: %w", err)
	}
	if err := ts.towerBlocks(tr, sq.Gw, tr.windows(tg, sq.Gw, sq.Gh)); err != nil {
		return fmt.Errorf("model: the tower's blocks: %w", err)
	}
	if err := tr.parts.Head(n, tr.req, tr.posTab, tr.blk, dst[at*D:]); err != nil {
		return fmt.Errorf("model: %w", err)
	}
	// Each tap's rows at the same rows of its plane of deep.
	for j := range tg.Taps {
		if err := tr.parts.Tap(j, n, tr.req, tr.posTab, tr.tapRows[j], deep[(j*len(dst)/D+at)*D:]); err != nil {
			return fmt.Errorf("model: %w", err)
		}
	}
	return nil
}

// encodePair is a video pair's rows into dst and its taps' into deep: both
// frames resized to sq's size, the video class's entry over the two, then
// the tower's blocks and head as a picture's square runs them.
func (s *Instance) encodePair(f0, f1 image.Image, sq model.Square, dst, deep []float32) error {
	tr := s.tower
	tr.encodes++
	words, pl, pl1, off, err := model.PairWords(f0, f1, &tr.pad, &tr.ent, &tr.ent2)
	if err != nil {
		return err
	}
	tg := s.m.Tower
	a := tg.Videos[sq.Class].Pic
	if want := tg.SquareOut(sq) * int(s.m.Container.Vision().ProjDim); len(dst) != want {
		return fmt.Errorf("model: %d values for a video pair's %d", len(dst), want)
	}
	pl.Request(tr.req)
	pl1.RequestFrame(tr.req)
	tr.req[composer.RequestPicFrame] = uint32(off)
	if !tr.tapsVideo {
		tr.tapsW, tr.tapsVideo = 0, true
	}
	if err := squareFits(sq, a); err != nil {
		return err
	}
	tr.squareRequest(pl, sq, a, 1)
	return tr.s.towerSquare(tr, sq, len(tg.Classes)+sq.Class, model.PairKey(pl.Px, pl1.Px), words, dst, deep, 0)
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
	// No call reads the run's history past it: the instance holds none, so
	// a move of its blocks between pictures carries nothing.
	ts.Reset()
	return err
}

// encodePictures is spans with every picture span's rows in place: encoded
// into a buffer of the instance's (one a picture of the prompt), or, for a
// picture whose rows all lie before row from -- restored by the prefix
// cache -- a buffer nothing reads, with no tower block run. The caller's
// spans are left as they were.
func (s *Instance) encodePictures(spans []model.Span, from int) ([]model.Span, error) {
	n, clips := 0, 0
	for _, sp := range spans {
		switch {
		case sp.Embd != nil:
		case sp.Picture != nil:
			n++
		case sp.Clip != nil:
			clips++
		}
	}
	if clips > 0 {
		var err error
		if spans, err = s.encodeClips(spans, from); err != nil {
			return nil, err
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
			var want int
			var err error
			switch {
			case sp.Frame != nil:
				want = s.m.Tower.SquareOut(s.m.Tower.VideoSquare(sp.FrameW, sp.FrameH))
			case sp.Whole:
				var sq model.Square
				if _, err = tr.squares.FrameOf(s.m.Tower, b.Dx(), b.Dy()); err == nil {
					sq, err = tr.squares.Last()
				}
				want = s.m.Tower.SquareOut(sq)
			default:
				if _, err = tr.squares.Of(s.m.Tower, b.Dx(), b.Dy()); err == nil {
					want, err = tr.squares.Rows()
				}
			}
			if err != nil {
				return nil, fmt.Errorf("model: span %d: %w", i, err)
			}
			if sp.PicRows != want {
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
				deep := tr.deeps[k-1][:len(s.m.Tower.Taps)*sp.PicRows*D]
				var err error
				if sp.Frame != nil {
					err = s.encodePair(sp.Picture, sp.Frame, s.m.Tower.VideoSquare(sp.FrameW, sp.FrameH), sp.Embd, deep)
				} else {
					_, err = s.encodePicture(sp.Picture, sp.Whole, sp.Embd, deep)
				}
				if err != nil {
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
	// The entry's rows as long as its widest class's: a clip's sample
	// formats', a picture's.
	most := 0
	for _, e := range tg.Classes[0].Entry {
		o := e.Values[e.Out[0]]
		most = max(most, o.Rows*o.Width)
	}
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
	if err := tr.entry(cl.BandRuns[px], 0, px, words, cl.Rows, cur); err != nil {
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
			// A block reads the request's words before the picture op's
			// taps, which only the entry reads.
			if !dev.RunRows(tr.chainParts[i], n, tr.req[:composer.RequestPicTaps], tr.posTab[:n+1], cur, out) {
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
	return s.placeChain(s.tower, s.m.Tower, dev, n)
}

// PlaceAudioChain is PlaceTowerChain for the audio tower's blocks.
func (s *Instance) PlaceAudioChain(dev nn.PictureRowsDevice, n int) error {
	if err := s.audioReady(); err != nil {
		return err
	}
	return s.placeChain(s.audio, s.m.Audio, dev, n)
}

// placeChain puts chained tower tg's first n blocks, run by tr, on dev and
// the rest on the CPU device.
func (s *Instance) placeChain(tr *towerRun, tg *model.TowerGraphs, dev nn.PictureRowsDevice, n int) error {
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
		p, err := chainPart(s.m, tr.base+i, g)
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

// chainPart is the model's block li, a chained tower's, as a part of a
// device of its own: its graph over its input's rows, its matrices the
// container's entries, which the device reads into frames of its own, and
// its vectors.
func chainPart(m *model.Model, li int, g *graph.Graph) (*nn.RowsPart, error) {
	l := &m.Layers[li]
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
				return nil, fmt.Errorf("model: block %d binds %v, which it does not hold", li, b.Role)
			}
			p.Weights[b.Role] = model.Wt(*t)
		}
	}
	return p, nil
}

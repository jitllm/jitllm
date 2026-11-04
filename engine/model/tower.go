package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"math"
	"slices"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/tok/jinja"
)

// The vision tower as its container states it (jlm.Tower), lowered into the
// engine's graphs: an entry from the picture's samples to the first block's
// rows, the repeated block, and a head from the last block's rows to the
// language model's. Nothing here knows a family: each graph is the record's
// steps and ops, op for op, and the tensors its inventory holds.

// towerGraphs is a stated tower's graphs: an entry for each sample format a
// decoded picture arrives in, a graph a block, and the head.
type towerGraphs struct {
	// classes is every size the entry is formed for, smallest first: a
	// fixed tower's one, or squares of growing sides for a picture resized
	// to a size of its own (jlm.PreSmartResize).
	classes []towerClass
	blocks  []*graph.Graph
	head    *graph.Graph
	// rows is the most rows the entry writes and every block reads, lead the
	// class rows before the patches; side a fixed picture's patch grid's
	// width and patch a patch's side in pixels; out the rows the head writes
	// for rows rows, one a token of the picture's span.
	rows, lead, side, patch, out int
	// gridDiv is the language model's grid over the patches'
	// (jlm.TowerSpan.GridDiv); merge the side of the merge windows the patch
	// rows are ordered in and window that of the windows of them, in merge
	// windows (towerTiles); full says block i attends the whole picture.
	gridDiv, merge, window int
	full                   []bool
	// fit is how a picture becomes the tower's squares.
	fit pictureFit
	// table and planes are the blocks' rotary table and its planes, nil for
	// blocks that turn nothing.
	table  *graph.Graph
	planes []float32
	// holders is what each chat template writes for a picture, by the
	// template's source (picturePlaceholder).
	holders sync.Map
}

// towerClass is one size the entry is formed for: its graph for each sample
// format, its picture op's attributes and its rows.
type towerClass struct {
	entry map[uint8]*graph.Graph
	pic   graph.Attrs
	rows  int
}

// placeholder is what the template src writes for an image part, read off it
// once.
func (tg *towerGraphs) placeholder(tpl *jinja.Template, src, bos, eos, mark string) (string, error) {
	if ph, ok := tg.holders.Load(src); ok {
		return ph.(string), nil
	}
	ph, err := picturePlaceholder(tpl, src, bos, eos, mark)
	if err != nil {
		return "", err
	}
	tg.holders.Store(src, ph)
	return ph, nil
}

// bindTower makes the tower's blocks layers of the model from towerBase on,
// the page each is read from the container's vision block of that index, and
// its entry's and head's tensors towerDense: every vector read once, every
// matrix bound by its role (layer.mats) for its page-in to point again.
func (m *Model) bindTower(c *jlm.File) error {
	if m.towerBase != int(c.H.NBlocks) {
		return fmt.Errorf("model: %d text blocks over a page array of %d: the tower's pages would be another block's",
			m.towerBase, c.H.NBlocks)
	}
	v := c.Vision()
	// The tower's blocks are bound where they live: a matrix is recorded by a
	// pointer into its layer (layer.mats), which a copy would leave behind.
	n := len(m.layers)
	if cap(m.layers)-n < int(v.NLayer) {
		return fmt.Errorf("model: room for %d tower blocks after the text's, %d made", v.NLayer, cap(m.layers)-n)
	}
	m.layers = m.layers[:n+int(v.NLayer)]
	ls := m.layers[n:]
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if !e.Role.Vision() || e.Index >= 0 {
			continue
		}
		l := &m.towerDense
		switch {
		case e.Block >= 0 && int(e.Block) < len(ls):
			l = &ls[e.Block]
		case e.Block != jlm.DenseBlock:
			return fmt.Errorf("model: the tower holds %v in block %d of %d", e.Role, e.Block, len(ls))
		}
		if e.Role.Expanded() || e.NDim < 2 || squeezed(e) {
			x, err := entryVector(c, e)
			if err != nil {
				return err
			}
			l.vecs = append(l.vecs, nn.RoleVector{Role: e.Role, V: x})
			continue
		}
		t, err := entryTensor(c, e)
		if err != nil {
			return err
		}
		dst := l.towerSlot(e.Role)
		if dst == nil {
			dst = &t
		}
		*dst = t
		l.mats = append(l.mats, roleTensor{e.Role, dst})
	}
	return bindPacked(c, m.Cfg, &m.towerDense)
}

// squeezed reports whether e is a vector stored with trailing dimensions
// of 1 (InternViT's class embedding, [1024, 1, 1]).
func squeezed(e *jlm.Entry) bool {
	for d := 1; d < int(e.NDim); d++ {
		if e.Dims[d] != 1 {
			return false
		}
	}
	return true
}

// towerSlot is the field of a tower block's layer matrix r is held in: the
// one a text block's matrix of the same place in the block is, which a card
// reads a block's weights from (Instance.layerWeightsAt); nil for a matrix
// bound by its role alone.
func (l *layer) towerSlot(r jlm.Role) *tensor {
	switch r {
	case jlm.RoleVAttnQ:
		return &l.wq
	case jlm.RoleVAttnK:
		return &l.wk
	case jlm.RoleVAttnV:
		return &l.wv
	case jlm.RoleVAttnOut:
		return &l.wo
	case jlm.RoleVFFNGate:
		return &l.gate
	case jlm.RoleVFC1:
		return &l.up
	case jlm.RoleVFC2:
		return &l.down
	}
	return nil
}

// towerFacts is the shapes of the tower's tensors in its block b (the
// tower's own index), jlm.DenseBlock the entry's and the head's.
func (m *Model) towerFacts(b int32) graph.Facts {
	inv := map[jlm.Role]graph.Shape{}
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		if e.Block != b || !e.Role.Vision() || e.Index >= 0 {
			continue
		}
		s := graph.Shape{K: e.K()}
		if e.NDim >= 2 && !squeezed(e) {
			s.Rows = e.Rows()
		}
		inv[e.Role] = s
	}
	return graph.Facts{Inventory: inv}
}

// towerFormats is the container's type of every tensor g binds: a tower
// block's in its own index (its block less the model's other blocks), the
// entry's and the head's outside any.
func (m *Model) towerFormats(g *graph.Graph) map[jlm.Role]jlm.Type {
	b := int32(jlm.DenseBlock)
	if g.Block >= 0 {
		b = int32(g.Block - m.towerBase)
	}
	out := map[jlm.Role]jlm.Type{}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if e, ok := m.container.Find(w.Role, b, -1); ok {
				out[w.Role] = e.Type
			}
		}
	}
	return out
}

// buildTowerGraphs lowers the container's stated tower, verifying each graph
// against the tensors it binds. A container whose tower is not stated has
// none to lower, and says which family it is so its converter can be told.
func (m *Model) buildTowerGraphs() (*towerGraphs, error) {
	v := m.container.Vision()
	t := v.Tower
	if t == nil {
		return nil, fmt.Errorf("model: the container's vision tower (%v) is not stated: convert it again "+
			"with a converter that states this family (jlm.Tower)", v.Projector)
	}
	dense := m.towerFacts(jlm.DenseBlock)
	ops := t.Entry.Ops
	if len(ops) == 0 || ops[0].Code != jlm.TOpPatchify || t.Entry.NIn != 1 {
		return nil, fmt.Errorf("model: the tower's entry does not begin with the picture's patches")
	}
	fit, err := towerFit(t)
	if err != nil {
		return nil, fmt.Errorf("model: the tower's entry: %w", err)
	}
	P := int(ops[0].Span)
	merge, window := towerTiles(t)
	tg := &towerGraphs{lead: int(ops[0].Offset), gridDiv: int(t.Span.GridDiv), merge: merge, window: window,
		side: int(v.ImageSize / v.PatchSize), patch: P, fit: fit}
	sides := []int{0}
	if fit.factor > 0 {
		sides = fit.classSides()
	}
	for _, sd := range sides {
		grid := tg.side * tg.side
		if sd > 0 {
			grid = min((sd/P)*(sd/P), fit.maxPix/(P*P))
		}
		cl := towerClass{entry: map[uint8]*graph.Graph{}, rows: grid + tg.lead}
		for px := graph.PixRGBA8; px < graph.PixFormats; px++ {
			g, err := towerEntry(t, px, grid, tg.lead, sd, fit, dense)
			if err != nil {
				return nil, fmt.Errorf("model: the tower's entry: %w", err)
			}
			cl.pic, cl.entry[px] = g.Ops[0].Attr, g
		}
		tg.classes = append(tg.classes, cl)
		tg.rows = max(tg.rows, cl.rows)
	}
	if r := t.Block.Rope; len(r.Runs) > 0 {
		tg.table, tg.planes = towerRotary(t)
	}
	for i := range int(v.NLayer) {
		tg.full = append(tg.full, window == 0 || slices.Contains(t.Block.Full, int32(i)))
		g, err := towerBlock(m.towerBase+i, i, t, v, m.towerFacts(int32(i)))
		if err != nil {
			return nil, fmt.Errorf("model: the tower's block %d: %w", i, err)
		}
		tg.blocks = append(tg.blocks, g)
	}
	h, out, err := towerHead(t, v, tg.rows, dense)
	if err != nil {
		return nil, fmt.Errorf("model: the tower's head: %w", err)
	}
	tg.head, tg.out = h, out
	return tg, nil
}

// pictureFit is how a picture becomes the tower's squares: resized as a
// whole to the picture op's size; its shorter side resized to short and its
// centre cropped to it; or cut into between minTiles and maxTiles tiles of
// it, a thumbnail of the whole after them where thumb says (jlm.PreTile).
type pictureFit struct {
	short              int
	minTiles, maxTiles int
	thumb              bool
	// factor, minPix and maxPix are a resize to the picture's own size
	// (jlm.PreSmartResize); factor 0 where there is none.
	factor, minPix, maxPix int
}

// smartSize is a w x h picture's own size (Qwen2-VL's smart_resize): each
// side rounded half to even to a multiple of factor, at least one; then,
// past maxPix pixels, both scaled down by sqrt(area/maxPix) and floored to
// one, or short of minPix, scaled up by sqrt(minPix/area) and ceiled -- in
// float64, as Python computes them.
func (f pictureFit) smartSize(w, h int) (ow, oh int) {
	fa := f.factor
	round := func(v int) int {
		q, r := v/fa, v%fa
		if 2*r > fa || 2*r == fa && q%2 == 1 {
			q++
		}
		return max(q, 1) * fa
	}
	ow, oh = round(w), round(h)
	area := float64(w) * float64(h)
	switch {
	case ow*oh > f.maxPix:
		beta := math.Sqrt(area / float64(f.maxPix))
		oh = max(fa, int(math.Floor(float64(h)/beta/float64(fa)))*fa)
		ow = max(fa, int(math.Floor(float64(w)/beta/float64(fa)))*fa)
	case ow*oh < f.minPix:
		beta := math.Sqrt(float64(f.minPix) / area)
		oh = int(math.Ceil(float64(h)*beta/float64(fa))) * fa
		ow = int(math.Ceil(float64(w)*beta/float64(fa))) * fa
	}
	return ow, oh
}

// classSides is the sides of the squares a picture of its own size is
// encoded in, smallest first: 16 factors, doubling, and last the most a
// side reaches, a picture's at most composer.PictureSide rounded up a
// factor.
func (f pictureFit) classSides() []int {
	most := (composer.PictureSide/f.factor + 1) * f.factor
	var out []int
	for s := 16 * f.factor; s < most; s *= 2 {
		out = append(out, s)
	}
	return append(out, most)
}

// towerFit is how the record's steps make a picture the tower's squares,
// or why they cannot.
func towerFit(t *jlm.Tower) (pictureFit, error) {
	var fit pictureFit
	cropped, pil, sized := false, false, false
	for _, s := range t.Pre {
		switch s.Kind {
		case jlm.PreResize:
			sized, pil = true, jlm.Filter(s.Ints[2]) != jlm.FilterTwoTap
		case jlm.PreResizeShort:
			fit.short, sized, pil = int(s.Ints[0]), true, jlm.Filter(s.Ints[1]) != jlm.FilterTwoTap
		case jlm.PreSmartResize:
			fit.factor, fit.minPix, fit.maxPix = int(s.Ints[0]), int(s.Ints[1]), int(s.Ints[2])
			sized, pil = true, jlm.Filter(s.Ints[3]) != jlm.FilterTwoTap
		case jlm.PreCenterCrop:
			cropped = true
		case jlm.PreTile:
			fit.minTiles, fit.maxTiles, fit.thumb = int(s.Ints[0]), int(s.Ints[1]), s.Ints[2] == 1
		}
	}
	ways := 0
	for _, w := range []bool{fit.short > 0, fit.maxTiles > 0, fit.factor > 0} {
		if w {
			ways++
		}
	}
	switch {
	case !sized:
		return fit, fmt.Errorf("the picture is never resized to the tower's size")
	case fit.short > 0 && !cropped:
		return fit, fmt.Errorf("a picture resized by its shorter side and never cropped has no one size")
	case ways > 1:
		return fit, fmt.Errorf("a picture cropped, tiled or sized its own way at once")
	case ways > 0 && !pil:
		// A crop, a tile and a size of the picture's own are a resize's
		// windows and live sizes, which only Pillow's taps (composer.PILTaps)
		// take.
		return fit, fmt.Errorf("a window or a size of its own of a picture resized by the two-tap filter")
	}
	return fit, nil
}

// towerEntry is the entry for pictures of sample format px: the record's
// preprocessing as one op (graph.OpPicture), then its entry program over
// the grid's patch rows after lead rows (a class row's); for a picture of a
// size of its own, at most side pixels a side.
func towerEntry(t *jlm.Tower, px uint8, grid, lead, side int, fit pictureFit, f graph.Facts) (*graph.Graph, error) {
	a := graph.Attrs{Pixels: px, Std: [3]float64{1, 1, 1}}
	for _, s := range t.Pre {
		switch s.Kind {
		case jlm.PreResize:
			a.PicW, a.PicH = int(s.Ints[0]), int(s.Ints[1])
			if err := resizeChain(&a, jlm.Filter(s.Ints[2])); err != nil {
				return nil, err
			}
		case jlm.PreResizeShort:
			a.PicW, a.PicH = int(s.Ints[0]), int(s.Ints[0])
			if err := resizeChain(&a, jlm.Filter(s.Ints[1])); err != nil {
				return nil, err
			}
		case jlm.PreSmartResize:
			a.PicW, a.PicH, a.PicLive = side, side, true
			if err := resizeChain(&a, jlm.Filter(s.Ints[3])); err != nil {
				return nil, err
			}
		case jlm.PreCenterCrop:
			a.PicW, a.PicH = int(s.Ints[0]), int(s.Ints[1])
		case jlm.PreNormalize:
			for c := range 3 {
				a.Mean[c], a.Std[c] = float64(s.Floats[c]), float64(s.Floats[3+c])
			}
		}
	}
	switch {
	case a.Chain == graph.ChainPIL && a.PicLive:
		// A picture's own size is its source's but for the rounding to the
		// factor and the area's bound: a side under twice smaller than the
		// source's times sqrt(area / maxPix), the source at most PicSide a
		// side (pictureFit.smartSize).
		root := int(math.Sqrt(float64(fit.maxPix)))
		a.Taps = composer.PILTapCount(a.Filter, 2*max(a.PicSide, root), root)
	case a.Chain == graph.ChainPIL:
		// A crop's taps are a window of the resize's, whose scale is at most
		// the crop's own.
		a.Taps = max(composer.PILTapCount(a.Filter, a.PicSide, a.PicW), composer.PILTapCount(a.Filter, a.PicSide, a.PicH))
	}
	p := &t.Entry
	b := graph.NewBlock(-1)
	P := int(p.Ops[0].Span)
	pix := b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: 3 * P * P}}, nil, a, nil, nil)[0]
	if a.PicLive && grid*P*P > a.PicW*a.PicH || !a.PicLive && a.PicW*a.PicH != grid*P*P {
		return nil, fmt.Errorf("a %dx%d picture is not %d patches of %d", a.PicW, a.PicH, grid, P)
	}
	out, err := lowerProgram(b, t, p, []graph.ValueID{pix}, grid+lead, f)
	if err != nil {
		return nil, err
	}
	b.Out(out)
	return b.G, graph.Verify(b.G, f)
}

// towerTiles is the nested order the tower's patch rows are in: merge x
// merge merge windows (TOpPatchify's Width, 1 for none) in windows of window
// x window of them (TowerBlock.Window over the merge, 0 for none).
func towerTiles(t *jlm.Tower) (merge, window int) {
	merge = 1
	if ops := t.Entry.Ops; len(ops) > 0 && ops[0].Code == jlm.TOpPatchify && ops[0].Width > 1 {
		merge = int(ops[0].Width)
	}
	return merge, int(t.Block.Window) / merge
}

// towerRotary is the blocks' rotary as the tower's table turns it: the
// table graph over each row's cell of the picture's grid (graph.RopeTable
// .Grid), and its planes -- every pair's frequency, then its axis, 0 a
// cell's row and 1 its column.
func towerRotary(t *jlm.Tower) (*graph.Graph, []float32) {
	rp := t.Block.Rope
	r := nn.Rope{NRot: int(rp.Dim), Base: float64(rp.Base), Neox: rp.Neox}
	var axes []float32
	for _, run := range rp.Runs {
		r.Runs = append(r.Runs, nn.RopeRun{Pairs: int(run.Pairs), Axis: int(run.Axis), Freq: int(run.Freq)})
		for range run.Pairs {
			axes = append(axes, float32(run.Axis))
		}
	}
	return composer.TableGraph(towerRotaryOf(t), 0), append(r.TabPlanes(len(axes)), axes...)
}

// towerRotaryOf is the blocks' rotary as their graphs state it: every turned
// pair's element, each row's position the cell of the grid in the order the
// patch gather lays the rows out (TOpPatchify's Width and Offset).
func towerRotaryOf(t *jlm.Tower) graph.Rotary {
	g := graph.Rotary{Base: float64(t.Block.Rope.Base), Grid: true}
	for _, run := range t.Block.Rope.Runs {
		g.NRot += 2 * int(run.Pairs)
	}
	if ops := t.Entry.Ops; len(ops) > 0 && ops[0].Code == jlm.TOpPatchify {
		g.Lead = int(ops[0].Offset)
	}
	g.Merge, g.Window = towerTiles(t)
	return g
}

// tiles is a w x h picture's tile grid in squares of side s and how many
// squares it becomes, the thumbnail among them: one square for a tower
// that does not tile.
func (f pictureFit) tiles(w, h, s int) (cols, rows, n int) {
	if f.maxTiles == 0 {
		return 1, 1, 1
	}
	cols, rows = tileGrid(w, h, s, f.minTiles, f.maxTiles)
	n = cols * rows
	if f.thumb && n > 1 {
		n++
	}
	return cols, rows, n
}

// window is square t of a w x h picture as the picture op's taps take it:
// the size the picture is resized to and the square's corner in it.
func (f pictureFit) window(w, h, pw, ph, t int) (ow, oh, x0, y0 int) {
	switch {
	case f.short > 0:
		// transformers' center_crop: the offsets floored.
		ow, oh = shortSide(w, h, f.short)
		return ow, oh, (ow - pw) / 2, (oh - ph) / 2
	case f.maxTiles > 0:
		cols, rows, _ := f.tiles(w, h, pw)
		if t < cols*rows {
			return cols * pw, rows * ph, (t % cols) * pw, (t / cols) * ph
		}
	}
	return pw, ph, 0, 0
}

// tileGrid is the (columns, rows) grid of between minT and maxT tiles of
// side s whose aspect is closest to a w x h picture's: a tie goes to the
// later (larger) grid while the picture's area is more than half the grid's,
// the grids in the reference's own order -- every (columns, rows) pair,
// columns outer, stably sorted by tile count (InternVL's
// find_closest_aspect_ratio, transformers' get_optimal_tiled_canvas).
func tileGrid(w, h, s, minT, maxT int) (cols, rows int) {
	aspect, area := float64(w)/float64(h), float64(w*h)
	bc, br, best := 1, 1, -1.0
	// By tile count, then in generation order: the stable sort, walked.
	for n := minT; n <= maxT; n++ {
		for c := 1; c <= maxT; c++ {
			if n%c != 0 {
				continue
			}
			r := n / c
			d := math.Abs(aspect - float64(c)/float64(r))
			switch {
			case best < 0 || d < best:
				bc, br, best = c, r, d
			case d == best && area > 0.5*float64(s*s*c*r):
				bc, br = c, r
			}
		}
	}
	return bc, br
}

// square is one square of a picture as the tower encodes it: the class its
// entry is formed for, the size the picture is resized to and the window of
// it the square is -- its corner and its size -- and its patch grid.
type square struct {
	class                  int
	ow, oh, x0, y0, pw, ph int
	gw, gh                 int
}

// square is square i of a w x h picture and how many the picture has: one
// at its own size in the smallest class that holds it, or a fixed square,
// a crop or a tile at the entry's size.
func (tg *towerGraphs) square(w, h, i int) (square, int) {
	if tg.fit.factor > 0 {
		ow, oh := tg.fit.smartSize(w, h)
		c := 0
		for c < len(tg.classes)-1 && max(ow, oh) > tg.classes[c].pic.PicW {
			c++
		}
		return square{class: c, ow: ow, oh: oh, pw: ow, ph: oh, gw: ow / tg.patch, gh: oh / tg.patch}, 1
	}
	a := tg.classes[0].pic
	_, _, n := tg.fit.tiles(w, h, a.PicW)
	ow, oh, x0, y0 := tg.fit.window(w, h, a.PicW, a.PicH, i)
	return square{ow: ow, oh: oh, x0: x0, y0: y0, pw: a.PicW, ph: a.PicH, gw: tg.side, gh: tg.side}, n
}

// squareOut is the rows the head writes for square sq: its patches over
// the head's ratio of rows out to patch rows in.
func (tg *towerGraphs) squareOut(sq square) int {
	return sq.gw * sq.gh * tg.out / (tg.rows - tg.lead)
}

// picRows is the rows a w x h picture becomes: its squares' every row.
func (tg *towerGraphs) picRows(w, h int) int {
	sq, n := tg.square(w, h, 0)
	rows := tg.squareOut(sq)
	for i := 1; i < n; i++ {
		sq, _ = tg.square(w, h, i)
		rows += tg.squareOut(sq)
	}
	return rows
}

// resizeChain sets a's resample to filter f's.
func resizeChain(a *graph.Attrs, f jlm.Filter) error {
	switch f {
	case jlm.FilterTwoTap:
		a.Chain, a.Filter = graph.ChainHalfPixel, graph.FilterLinear
	case jlm.FilterPILBilinear:
		a.Chain, a.Filter, a.PicSide = graph.ChainPIL, graph.FilterBilinear, composer.PictureSide
	case jlm.FilterPILBicubic:
		a.Chain, a.Filter, a.PicSide = graph.ChainPIL, graph.FilterCubic, composer.PictureSide
	default:
		return fmt.Errorf("a resize by filter %d, which no picture op runs", f)
	}
	return nil
}

// shortSide is a w x h picture's size with its shorter side resized to s
// and its longer to int(s * long / short), as transformers'
// get_resize_output_image_size sizes a shortest_edge resize.
func shortSide(w, h, s int) (int, int) {
	if w <= h {
		return s, int(float64(s) * float64(h) / float64(w))
	}
	return int(float64(s) * float64(w) / float64(h)), s
}

// towerHead is the head over the last block's rows, and the rows it writes.
func towerHead(t *jlm.Tower, v *jlm.Vision, rows int, f graph.Facts) (*graph.Graph, int, error) {
	b := graph.NewBlock(-1)
	x := b.In("rows", int(v.NEmbd))
	out, err := lowerProgram(b, t, &t.Head, []graph.ValueID{x}, rows, f)
	if err != nil {
		return nil, 0, err
	}
	b.Out(out)
	n := b.G.Values[out].Rows
	if n == 0 {
		n = rows
	}
	return b.G, n, graph.Verify(b.G, f)
}

// lowerProgram adds p's ops to b over its inputs in, each op the graph's of
// the same meaning, and is p's result. A value's width is its op's: a
// matrix's rows, a regroup's input times its group, and a patch gather's the
// width the matrix reading it takes.
func lowerProgram(b *graph.Builder, t *jlm.Tower, p *jlm.Program, in []graph.ValueID, rows int, f graph.Facts) (graph.ValueID, error) {
	merge, window := towerTiles(t)
	if len(in) != int(p.NIn) {
		return 0, fmt.Errorf("a program of %d inputs over %d", p.NIn, len(in))
	}
	vals := append([]graph.ValueID(nil), in...)
	shape := func(r jlm.Role) (graph.Shape, error) {
		s, ok := f.Inventory[r]
		if !ok {
			return s, fmt.Errorf("binds %v, which the container does not hold", r)
		}
		return s, nil
	}
	// The width a value is read at by the matrix that reads it.
	readAt := map[int]int{}
	for _, op := range p.Ops {
		if op.Code == jlm.TOpLinear && len(op.Bind) > 0 {
			if s, ok := f.Inventory[op.Bind[0]]; ok {
				readAt[int(op.In[0])] = s.K
			}
		}
	}
	width := func(v graph.ValueID) int { return b.G.Values[v].Width }
	// rowsOf is a value's rows: its own, where an op that changes the count
	// made it, else the program's.
	rowsOf := func(v graph.ValueID) int {
		if r := b.G.Values[v].Rows; r > 0 {
			return r
		}
		return rows
	}
	for i := range p.Ops {
		op := &p.Ops[i]
		ins := make([]graph.ValueID, len(op.In))
		for j, v := range op.In {
			ins[j] = vals[v]
		}
		at := len(in) + i
		name := fmt.Sprintf("op%d", i)
		var out graph.ValueID
		switch op.Code {
		case jlm.TOpPatchify:
			w := 3 * int(op.Span*op.Span)
			if k := readAt[at]; k > w {
				w = k
			}
			out = b.Emit(graph.OpPatchify, ins, []graph.Result{{Name: "patches", Width: w}}, nil,
				graph.Attrs{Span: int(op.Span), Offset: int(op.Offset), Width: int(op.Width), Window: window}, nil, nil)[0]
		case jlm.TOpLinear:
			s, err := shape(op.Bind[0])
			if err != nil {
				return 0, err
			}
			out = b.MatVec(ins[0], op.Bind[0], s.Rows, op.Bind[1:]...)
		case jlm.TOpPosTable:
			s, err := shape(op.Bind[0])
			if err != nil {
				return 0, err
			}
			out = b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "pos", Width: s.K}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, graph.Attrs{Positional: true, Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpAdd:
			out = b.Add(ins[0], ins[1])
		case jlm.TOpLayerNorm:
			out = b.LayerNorm(ins[0], op.Bind[0], float64(op.Eps), op.Bind[1:]...)
		case jlm.TOpRMSNorm:
			out = b.Norm(ins[0], op.Bind[0], float64(op.Eps))
		case jlm.TOpRegroup:
			S := int(op.Span)
			out = b.Emit(graph.OpRegroup, ins, []graph.Result{{Name: "grouped", Width: S * S * width(ins[0]),
				Rows: rowsOf(ins[0]) / (S * S)}}, nil, graph.Attrs{Span: S}, nil, nil)[0]
		case jlm.TOpPool:
			// The mean of each block's S*S rows: the block regrouped into one
			// row, its slots summed and scaled by 1/S^2.
			S, E := int(op.Span), width(ins[0])
			g := b.Emit(graph.OpRegroup, ins, []graph.Result{{Name: "pool.grouped", Width: S * S * E,
				Rows: rowsOf(ins[0]) / (S * S)}}, nil, graph.Attrs{Span: S}, nil, nil)[0]
			slot := func(k int) graph.ValueID {
				return b.Emit(graph.OpSlice, []graph.ValueID{g}, []graph.Result{{Name: "pool.slot", Width: E}}, nil,
					graph.Attrs{Offset: k * E, Width: E}, nil, nil)[0]
			}
			acc := slot(0)
			for k := 1; k < S*S; k++ {
				acc = b.Add(acc, slot(k))
			}
			var e jlm.ExprBuilder
			e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(1/float32(S*S)))
			mean := jlm.Expr{Terms: e.T}
			var err error
			if out, err = b.Stated("pooled", &mean, []graph.ValueID{acc}, E, f); err != nil {
				return 0, fmt.Errorf("the pool's mean: %w", err)
			}
		case jlm.TOpFold:
			S := int(op.Span)
			out = b.Emit(graph.OpFold, ins, []graph.Result{{Name: "folded", Width: S * width(ins[0]),
				Rows: rowsOf(ins[0]) / S}}, nil, graph.Attrs{Span: S}, nil, nil)[0]
		case jlm.TOpUntile:
			out = b.Emit(graph.OpUntile, ins, []graph.Result{{Name: "raster", Width: width(ins[0]), Rows: rowsOf(ins[0])}}, nil,
				graph.Attrs{Width: merge, Window: window}, nil, nil)[0]
		case jlm.TOpLeadRow:
			out = b.Emit(graph.OpLeadRow, ins, []graph.Result{{Name: "class", Width: width(ins[0])}},
				[]graph.Binding{{Role: op.Bind[0], K: width(ins[0])}}, graph.Attrs{Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpDropRows:
			out = b.Emit(graph.OpDropRows, ins, []graph.Result{{Name: "rows", Width: width(ins[0]),
				Rows: rowsOf(ins[0]) - int(op.Offset)}}, nil, graph.Attrs{Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpStated:
			v, err := b.Stated(name, op.Expr, ins, int(op.Width), f)
			if err != nil {
				return 0, err
			}
			out = v
		default:
			return 0, fmt.Errorf("op %d is code %d, which this engine does not lower", i, op.Code)
		}
		vals = append(vals, out)
	}
	return vals[p.Out], nil
}

// towerBlock is block li of the model's block space (the tower's block
// li-NLayer): the stated block over the tensors its inventory holds -- each
// norm before a sublayer where held, biases where held, a gate where held,
// post-norms where held -- attending both ways over the picture's rows.
func towerBlock(li, i int, t *jlm.Tower, v *jlm.Vision, f graph.Facts) (*graph.Graph, error) {
	rot := towerRotaryOf(t)
	bl := &t.Block
	has := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	stored := func(r jlm.Role) []jlm.Role {
		if has(r) {
			return []jlm.Role{r}
		}
		return nil
	}
	for _, r := range []jlm.Role{jlm.RoleVAttnQ, jlm.RoleVAttnK, jlm.RoleVAttnV, jlm.RoleVAttnOut, jlm.RoleVFC1, jlm.RoleVFC2} {
		if !has(r) {
			return nil, fmt.Errorf("the block holds no %v", r)
		}
	}
	d, hd := int(v.NEmbd), int(bl.HeadDim)
	qdim, kvDim := int(bl.NHead)*hd, int(bl.NKVHead)*hd
	eps := float64(bl.Eps)
	b := graph.NewBlock(li)
	norm := func(x graph.ValueID, r, bias jlm.Role) graph.ValueID {
		switch {
		case !has(r):
			return x
		case bl.Norm == jlm.NormLayer:
			return b.LayerNorm(x, r, eps, stored(bias)...)
		}
		return b.Norm(x, r, eps)
	}
	x := b.In("x", d)
	n := norm(x, jlm.RoleVAttnNorm, jlm.RoleVAttnNormBias)
	q := b.MatVec(n, jlm.RoleVAttnQ, qdim, stored(jlm.RoleVAttnQBias)...)
	k := b.MatVec(n, jlm.RoleVAttnK, kvDim, stored(jlm.RoleVAttnKBias)...)
	vv := b.MatVec(n, jlm.RoleVAttnV, kvDim, stored(jlm.RoleVAttnVBias)...)
	if len(bl.Rope.Runs) > 0 {
		layout := graph.RopeNorm
		if bl.Rope.Neox {
			layout = graph.RopeNeox
		}
		a := graph.Attrs{HeadDim: hd, NRot: rot.NRot, Base: rot.Base, Layout: layout}
		q = b.Emit(graph.OpRope, []graph.ValueID{q}, []graph.Result{{Name: "q.rope", Width: qdim}}, nil, a, nil, nil)[0]
		k = b.Emit(graph.OpRope, []graph.ValueID{k}, []graph.Result{{Name: "k.rope", Width: kvDim}}, nil, a, nil, nil)[0]
	}
	hist := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, vv, 0)}
	scale := float64(bl.AttnScale)
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(hd))
	}
	// A windowed block reads each row's keys from its window (the step's
	// row windows); a whole-picture block every row's.
	windowed := bl.Window > 0 && !slices.Contains(bl.Full, int32(i))
	att := b.AttendOf(q, hist, graph.Attrs{NHead: int(bl.NHead), NKVHead: int(bl.NKVHead), HeadDim: hd, Scale: scale, Bidir: true,
		RowWindows: windowed})
	x2 := b.Add(x, b.MatVec(att, jlm.RoleVAttnOut, d, stored(jlm.RoleVAttnOutBias)...))
	n2 := norm(x2, jlm.RoleVFFNNorm, jlm.RoleVFFNNormBias)
	ffn := f.Inventory[jlm.RoleVFC1].Rows
	var acts []graph.ValueID
	if has(jlm.RoleVFFNGate) {
		acts = append(acts, b.MatVec(n2, jlm.RoleVFFNGate, ffn, stored(jlm.RoleVFFNGateBias)...))
	}
	acts = append(acts, b.MatVec(n2, jlm.RoleVFC1, ffn, stored(jlm.RoleVFC1Bias)...))
	h, err := b.Stated("act", &bl.Act, acts, ffn, f)
	if err != nil {
		return nil, fmt.Errorf("the stated activation: %w", err)
	}
	b.Out(b.Add(x2, b.MatVec(h, jlm.RoleVFC2, d, stored(jlm.RoleVFC2Bias)...)))
	return b.G, graph.Verify(b.G, f)
}

// towerConfig is the tower's segment as a Config: its blocks' geometry,
// attending both ways, with its own rotary where it turns q and k and no
// vocabulary. Its block indices are the model's (base on), so a block is
// asked by the index it runs at.
func towerConfig(base, rows int, v *jlm.Vision, t *jlm.Tower) *Config {
	b := &t.Block
	r := towerRotaryOf(t)
	return &Config{Arch: "tower", NLayer: base + int(v.NLayer), NEmbd: int(v.NEmbd), NHead: int(b.NHead),
		NKVHead: int(b.NKVHead), HeadDim: int(b.HeadDim), NFFN: int(v.NFFN), NCtx: rows,
		RMSEps: float64(b.Eps), EmbdScale: 1, AttnFactor: 1, NonCausal: true, NoPosEnc: r.NRot == 0,
		NRot: r.NRot, RopeBase: r.Base, RopeNeox: b.Rope.Neox, LayerNorm: b.Norm == jlm.NormLayer}
}

// towerRun is an instance's tower: an instance of its own over the tower's
// blocks (towerConfig), whose history holds one picture's rows at a time,
// and the entry's forms (one a sample format) and the head's on it, all
// composed when the instance is made.
type towerRun struct {
	s *Instance
	// entry is a class's entry forms, one a sample format.
	entry []map[uint8]*cpudev.Form
	head  *cpudev.Form
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
	// width; encodes counts the pictures encoded.
	out     []float32
	encodes int
	// pad holds a picture's planes where they are not read in place, ent a
	// palette's entries; tapsW and tapsH the source size the request's taps
	// are for.
	pad          []uint32
	ent          [4 * 256]byte
	tapsW, tapsH int
	// pics is a prompt's pictures' rows, one buffer a picture, and spans the
	// prompt's spans with them in place (encodePictures).
	pics  [][]float32
	spans []Span
}

// newTowerRun is the model's tower run, nil for a model with no tower.
func (m *Model) newTowerRun() (*towerRun, error) {
	tg := m.tower
	if tg == nil {
		return nil, nil
	}
	v := m.container.Vision()
	n := tg.rows
	c := towerConfig(m.towerBase, n, v, v.Tower)
	ts := m.newInstanceOver(c, 1, n, m.towerBase, m.towerBase+len(tg.blocks))
	if ts.cpu.formErr != nil {
		ts.Close()
		return nil, ts.cpu.formErr
	}
	words := 0
	for _, cl := range tg.classes {
		words = max(words, (cl.pic.PicW+cl.pic.PicH)*(cl.pic.Taps+1))
	}
	tr := &towerRun{s: ts, req: make([]uint32, composer.RequestPicTaps+words)}
	for _, cl := range tg.classes {
		forms := map[uint8]*cpudev.Form{}
		for px, g := range cl.entry {
			f, err := m.formCPUGraph(ts, g, cl.rows, nil)
			if err != nil {
				ts.Close()
				return nil, fmt.Errorf("model: the tower's entry: %w", err)
			}
			forms[px] = f
		}
		tr.entry = append(tr.entry, forms)
	}
	f, err := m.formCPUGraph(ts, tg.head, n, nil)
	if err != nil {
		ts.Close()
		return nil, fmt.Errorf("model: the tower's head: %w", err)
	}
	tr.head = f
	for li := ts.lo; li < ts.hi; li++ {
		if cb := ts.cpu.blocks[li]; cb == nil || cb.rows == nil {
			why := composer.DeclineNoGraph
			if cb != nil {
				why = max(cb.why, cb.rowsWhy)
			}
			ts.Close()
			return nil, fmt.Errorf("model: the tower's block %d does not run on the CPU: %v", li-m.towerBase, why)
		}
	}
	return tr, nil
}

// squareRequest writes square sq's words of a picture of squares squares
// into the request, its picture op's attributes a: its grid's width, the
// size the op resizes to, and where Pillow's filters resize it the taps of
// the square's window of the resize -- one picture size's whole-picture taps
// kept from one picture to the next.
func (tr *towerRun) squareRequest(pl pictureLayout, sq square, a graph.Attrs, squares int) {
	tr.req[composer.RequestGrid] = uint32(sq.gw)
	tr.req[composer.RequestPicOW], tr.req[composer.RequestPicOH] = uint32(sq.pw), uint32(sq.ph)
	if a.Chain != graph.ChainPIL || squares == 1 && pl.w == tr.tapsW && pl.h == tr.tapsH {
		return
	}
	composer.PILTaps(a.Filter, pl.w, sq.ow, sq.x0, sq.pw, a.Taps, tr.req[composer.RequestPicTaps:])
	composer.PILTaps(a.Filter, pl.h, sq.oh, sq.y0, sq.ph, a.Taps, tr.req[composer.RequestPicTaps+a.PicW*(a.Taps+1):])
	tr.tapsW, tr.tapsH = pl.w, pl.h
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
	}
	tr.rows, tr.blk = tr.bufE[:n*e], tr.bufB[:n*e]
	tr.seq, tr.pos, tr.posTab = tr.bufS[:n], tr.bufP[:n], tr.bufT[:1+n]
	tr.posTab[0] = uint32(n)
}

// windows is a gw x gh patch grid's windows as runs of its rows in their
// nested order (towerTiles): the lead rows one run, then window by window;
// nil for a tower whose blocks attend the whole picture.
func (tr *towerRun) windows(tg *towerGraphs, gw, gh int) []nn.KeyRun {
	if tg.window < 1 {
		return nil
	}
	m, w := tg.merge, tg.window
	gw2, gh2 := gw/m, gh/m
	tr.wins = tr.wins[:0]
	at := tg.lead
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

// pictureLayout is how a decoded picture's samples sit in the step's picture
// table: their format, the picture's size and its rows' stride in bytes, and
// for a second and third plane (YCbCr's chroma, a palette) their stride, byte
// offsets and the chroma subsampling (composer.RequestPicSub).
type pictureLayout struct {
	px                   uint8
	w, h, stride         int
	cstride, p2, p3, sub int
}

// request writes the layout's words into the step's request.
func (l pictureLayout) request(req []uint32) {
	req[composer.RequestPicW], req[composer.RequestPicH], req[composer.RequestPicStride] = uint32(l.w), uint32(l.h), uint32(l.stride)
	req[composer.RequestPicCStride], req[composer.RequestPicP2], req[composer.RequestPicP3] = uint32(l.cstride), uint32(l.p2), uint32(l.p3)
	req[composer.RequestPicSub] = uint32(l.sub)
}

// picturePlanes is a decoded picture's layout and the planes the step's
// picture table holds for it, each from a word boundary after the last: its
// samples' bytes as the decoder left them, and a palette's entries -- four
// bytes each, the colour's straight red, green and blue and its alpha --
// written into ent.
func picturePlanes(img image.Image, ent *[4 * 256]byte) (pictureLayout, [3][]byte, error) {
	b := img.Bounds()
	l := pictureLayout{w: b.Dx(), h: b.Dy()}
	var planes [3][]byte
	if l.w <= 0 || l.h <= 0 {
		return l, planes, fmt.Errorf("model: an empty picture")
	}
	rows := func(pix []byte, stride, bpp int) []byte { return pix[:(l.h-1)*stride+l.w*bpp] }
	switch p := img.(type) {
	case *image.NRGBA:
		l.px, l.stride, planes[0] = graph.PixRGBA8, p.Stride, rows(p.Pix, p.Stride, 4)
	case *image.RGBA:
		l.px, l.stride, planes[0] = graph.PixRGBA8, p.Stride, rows(p.Pix, p.Stride, 4)
	case *image.Gray:
		l.px, l.stride, planes[0] = graph.PixGray8, p.Stride, rows(p.Pix, p.Stride, 1)
	case *image.YCbCr:
		return ycbcrPlanes(l, p)
	case *image.NYCbCrA:
		return ycbcrPlanes(l, &p.YCbCr)
	case *image.Paletted:
		clear(ent[:])
		for i, c := range p.Palette[:min(len(p.Palette), 256)] {
			e := ent[4*i : 4*i+4]
			if n, ok := c.(color.NRGBA); ok {
				e[0], e[1], e[2], e[3] = n.R, n.G, n.B, n.A
				continue
			}
			// color.NRGBAModel's arithmetic, without its boxed result.
			r, g, bb, a := c.RGBA()
			if a != 0 && a != 0xffff {
				r, g, bb = r*0xffff/a, g*0xffff/a, bb*0xffff/a
			}
			if a == 0 {
				r, g, bb = 0, 0, 0
			}
			e[0], e[1], e[2], e[3] = uint8(r>>8), uint8(g>>8), uint8(bb>>8), uint8(a>>8)
		}
		l.px, l.stride = graph.PixPaletted8, p.Stride
		planes[0], planes[1] = rows(p.Pix, p.Stride, 1), ent[:]
	default:
		return l, planes, fmt.Errorf("model: a %T picture has no picture op reading its samples", img)
	}
	l.place(planes)
	return l, planes, nil
}

// ycbcrPlanes is picturePlanes for a YCbCr picture: its luma and chroma
// planes, and the chroma's shifts and phase on each axis read off the
// picture's own COffset, so a sub-picture's chroma is the decoder's.
func ycbcrPlanes(l pictureLayout, p *image.YCbCr) (pictureLayout, [3][]byte, error) {
	r := p.Rect
	ylen := p.YOffset(r.Max.X-1, r.Max.Y-1) + 1
	clen := p.COffset(r.Max.X-1, r.Max.Y-1) + 1
	planes := [3][]byte{p.Y[:ylen], p.Cb[:clen], p.Cr[:clen]}
	hs, mx, okx := chromaShift(r.Dx(), 1, func(k int) int { return p.COffset(r.Min.X+k, r.Min.Y) })
	vs, my, oky := chromaShift(r.Dy(), p.CStride, func(k int) int { return p.COffset(r.Min.X, r.Min.Y+k) })
	if !okx || !oky {
		return l, planes, fmt.Errorf("model: a %v YCbCr picture's chroma follows no shift", p.SubsampleRatio)
	}
	l.px, l.stride, l.cstride, l.sub = graph.PixYCbCr8, p.YStride, p.CStride, hs|vs<<8|mx<<16|my<<24
	l.place(planes)
	return l, planes, nil
}

// chromaShift is the shift and phase taking an axis's first pixels, n of
// them, to their chroma samples as off places them, step bytes apart.
func chromaShift(n, step int, off func(k int) int) (shift, phase int, ok bool) {
	for s := 0; s < 3; s++ {
		for m := 0; m < 1<<s; m++ {
			ok = true
			for k := 0; k < min(n, 8) && ok; k++ {
				ok = off(k) == ((k+m)>>s)*step
			}
			if ok {
				return s, m, true
			}
		}
	}
	return 0, 0, false
}

// place sets the second and third planes' offsets, each from a word
// boundary after the last plane.
func (l *pictureLayout) place(planes [3][]byte) {
	at := len(planes[0])
	for i := 1; i < 3; i++ {
		if planes[i] == nil {
			continue
		}
		at = (at + 3) &^ 3
		if i == 1 {
			l.p2 = at
		} else {
			l.p3 = at
		}
		at += len(planes[i])
	}
}

// pictureWords is a decoded picture's samples as the step's picture table
// reads them, and their layout: one plane whose bytes are whole aligned
// words is read in place, and any other picture's planes are copied into
// pad, which grows to hold them.
func pictureWords(img image.Image, pad *[]uint32, ent *[4 * 256]byte) ([]uint32, pictureLayout, error) {
	l, planes, err := picturePlanes(img, ent)
	if err != nil {
		return nil, l, err
	}
	p := planes[0]
	if planes[1] == nil && len(p)%4 == 0 && uintptr(unsafe.Pointer(&p[0]))%4 == 0 {
		return unsafe.Slice((*uint32)(unsafe.Pointer(&p[0])), len(p)/4), l, nil
	}
	end := len(p)
	if planes[2] != nil {
		end = l.p3 + len(planes[2])
	} else if planes[1] != nil {
		end = l.p2 + len(planes[1])
	}
	n := (end + 3) / 4
	if n > composer.PictureWords {
		return nil, l, fmt.Errorf("model: a picture of %d bytes is past the %d the picture table holds", end, 4*composer.PictureWords)
	}
	if cap(*pad) < n {
		*pad = make([]uint32, n)
	}
	words := (*pad)[:n]
	b := unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), 4*n)
	copy(b, p)
	copy(b[l.p2:], planes[1])
	copy(b[l.p3:], planes[2])
	return words, l, nil
}

// EncodePicture is the rows a picture's span puts in the prompt: the picture
// through the tower's entry, its blocks over the patch rows as one run
// attending both ways, and its head, every step on the CPU device's composed
// forms. The rows are the instance's until its next encode.
func (s *Instance) EncodePicture(img image.Image) ([]float32, error) {
	if err := s.towerReady(); err != nil {
		return nil, err
	}
	defer s.m.enterPager()()
	b := img.Bounds()
	n := s.m.tower.picRows(b.Dx(), b.Dy()) * int(s.m.container.Vision().ProjDim)
	if cap(s.tower.out) < n {
		s.tower.out = make([]float32, n)
	}
	return s.encodePicture(img, s.tower.out[:n])
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

// encodePicture is EncodePicture into dst, the head's rows, under the pager
// the caller holds.
func (s *Instance) encodePicture(img image.Image, dst []float32) ([]float32, error) {
	tr := s.tower
	tr.encodes++
	words, pl, err := pictureWords(img, &tr.pad, &tr.ent)
	if err != nil {
		return nil, err
	}
	ts, m := tr.s, s.m
	tg := m.tower
	pl.request(tr.req)
	if a := tg.classes[0].pic; a.Chain == graph.ChainPIL && (pl.w > a.PicSide || pl.h > a.PicSide) {
		return nil, fmt.Errorf("model: a %dx%d picture is past the %d pixels a side the tower's resize reaches", pl.w, pl.h, a.PicSide)
	}
	head := tr.head
	if why := ts.cpuWeights(head, -1, &m.towerDense); why != composer.DeclineServes {
		return nil, fmt.Errorf("model: the tower's head: %v", why)
	}
	D, E := int(m.container.Vision().ProjDim), int(m.container.Vision().NEmbd)
	if want := tg.picRows(pl.w, pl.h); len(dst) != want*D {
		return nil, fmt.Errorf("model: %d values for a picture's %d rows of %d", len(dst), want, D)
	}
	b := &ts.cpu.bound
	sq, squares := tg.square(pl.w, pl.h, 0)
	at := 0
	for i := range squares {
		if i > 0 {
			sq, _ = tg.square(pl.w, pl.h, i)
		}
		f := tr.entry[sq.class][pl.px]
		if why := ts.cpuWeights(f, -1, &m.towerDense); why != composer.DeclineServes {
			return nil, fmt.Errorf("model: the tower's entry: %v", why)
		}
		tr.squareRequest(pl, sq, tg.classes[sq.class].pic, squares)
		n := tg.lead + sq.gw*sq.gh
		tr.size(n, E)
		*b = cpuBound{li: -1, l: &m.towerDense, own: ts.cpu.dev.Own(f), n: n, req: tr.req, pos: tr.posTab, pic: words}
		b.io[0] = unsafe.Pointer(&tr.rows[0])
		ts.cpuRunForm(b, f)
		if err := ts.towerBlocks(tr, tr.windows(tg, sq.gw, sq.gh)); err != nil {
			return nil, fmt.Errorf("model: the tower's blocks: %w", err)
		}
		*b = cpuBound{li: -1, l: &m.towerDense, own: ts.cpu.dev.Own(head), n: n, req: tr.req, pos: tr.posTab}
		b.io[0], b.io[1] = unsafe.Pointer(&tr.blk[0]), unsafe.Pointer(&dst[at*D])
		ts.cpuRunForm(b, head)
		at += tg.squareOut(sq)
	}
	return dst, nil
}

// towerBlocks runs the tower's blocks over the patch rows of a gw x gh grid,
// one run in which every row's keys are every row's, leaving the last
// block's rows in blk: on the CPU device, the runner's CPU rows over the
// run, block by block; with blocks on a card, the tower instance's prefill
// (embedMode.rows), each block on the device that holds it. Blocks that turn
// q and k read each row's cell of the grid from its position and the
// picture's request (the tower's grid table); a windowed block attends
// within its row's run of wins, every other block the whole picture.
func (ts *Instance) towerBlocks(tr *towerRun, wins []nn.KeyRun) error {
	n := len(tr.pos)
	ts.bidir = append(ts.bidir[:0], nn.KeyRun{Lo: 0, Hi: n})
	defer func() { ts.bidir, ts.cpu.runWindows = ts.bidir[:0], false }()
	ts.cpu.tables.req, ts.cpu.tables.pos = tr.req, tr.posTab
	if ts.devCount() > 0 {
		ts.Reset()
		if wins != nil {
			// A card's windowed blocks read each row's window (graph.Attrs
			// .RowWindows).
			rw, ok := ts.ld.(nn.RowWindowDevice)
			if !ok {
				return fmt.Errorf("the tower's windowed blocks are on a device that holds no row windows")
			}
			tr.rowWins = tr.rowWins[:0]
			for _, w := range wins {
				for range w.Hi - w.Lo {
					tr.rowWins = append(tr.rowWins, w)
				}
			}
			if !rw.SetRowWindows(tr.rowWins) {
				return fmt.Errorf("the device would not hold the tower's %d row windows", len(tr.rowWins))
			}
		}
		tr.mode = embedMode{bidir: true, rows: true, total: n, out: tr.blk}
		ts.emb = &tr.mode
		defer func() { ts.emb = nil }()
		_, err := ts.prefillSpans(0, Span{Embd: tr.rows})
		return err
	}
	if w := ts.c.RopeW(); len(ts.bcs) < n*w {
		ts.bcs = make([]float32, n*w)
	}
	ts.cpu.tabsFresh = false
	copy(tr.blk, tr.rows)
	for li := ts.lo; li < ts.hi; li++ {
		if err := ts.m.pageIn(li); err != nil {
			return err
		}
		// Each run of a windowed block is a window its rows attend inside.
		windowed := wins != nil && !ts.m.tower.full[li-ts.lo]
		if windowed {
			ts.bidir = append(ts.bidir[:0], wins...)
		} else {
			ts.bidir = append(ts.bidir[:0], nn.KeyRun{Lo: 0, Hi: n})
		}
		ts.cpu.runWindows = windowed
		why, err := ts.cpuRows(li, &ts.m.layers[li], tr.blk, nil, nil, tr.seq, tr.pos, nil)
		if err != nil {
			return err
		}
		if why != composer.DeclineServes {
			return fmt.Errorf("block %d: %v", li-ts.m.towerBase, why)
		}
	}
	return nil
}

// encodePictures is spans with every picture span's rows in place: encoded
// into a buffer of the instance's (one a picture of the prompt), or, for a
// picture whose rows all lie before row from -- restored by the prefix
// cache -- a buffer nothing reads, with no tower block run. The caller's
// spans are left as they were.
func (s *Instance) encodePictures(spans []Span, from int) ([]Span, error) {
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
	D := int(s.m.container.Vision().ProjDim)
	tr.spans = append(tr.spans[:0], spans...)
	k, row := 0, 0
	for i := range tr.spans {
		sp := &tr.spans[i]
		r := sp.n(s.c.NEmbd)
		if sp.Picture != nil && sp.Embd == nil {
			b := sp.Picture.Bounds()
			if want := s.m.tower.picRows(b.Dx(), b.Dy()); sp.PicRows != want {
				return nil, fmt.Errorf("model: span %d is a picture of %d rows and the tower writes %d",
					i, sp.PicRows, want)
			}
			if k == len(tr.pics) {
				tr.pics = append(tr.pics, nil)
			}
			if cap(tr.pics[k]) < sp.PicRows*D {
				tr.pics[k] = make([]float32, sp.PicRows*D)
			}
			sp.Embd = tr.pics[k][:sp.PicRows*D]
			k++
			if row+r > from {
				if _, err := s.encodePicture(sp.Picture, sp.Embd); err != nil {
					return nil, fmt.Errorf("model: span %d: %w", i, err)
				}
			}
		}
		row += r
	}
	return tr.spans, nil
}

// SeesPictures reports whether the model carries a tower, so a prompt may hold
// a picture.
func (m *Model) SeesPictures() bool { return m.tower != nil }

// PictureSpan is the span a picture stands in a prompt as, for ChatSpans or
// PrefillMixed: the picture, which the prefill encodes through the tower,
// the rows it takes, and the key naming them (pictureKey).
func (m *Model) PictureSpan(img image.Image) (Span, error) {
	if m.tower == nil {
		return Span{}, fmt.Errorf("model: the model has no vision tower")
	}
	k, err := pictureKey(img)
	if err != nil {
		return Span{}, err
	}
	b := img.Bounds()
	return Span{Picture: img, PicRows: m.tower.picRows(b.Dx(), b.Dy()), Grid: m.tower.spanGrid(b.Dx(), b.Dy()), Key: k}, nil
}

// spanGrid is the grid a w x h picture's rows are for the language model's
// rotary (jlm.TowerSpan.GridDiv): its one square's patch grid with each side
// divided; none where the rows advance as text.
func (tg *towerGraphs) spanGrid(w, h int) ImageGrid {
	if tg.gridDiv == 0 {
		return ImageGrid{}
	}
	sq, _ := tg.square(w, h, 0)
	return ImageGrid{T: 1, H: sq.gh / tg.gridDiv, W: sq.gw / tg.gridDiv}
}

// pictureKey names a picture by everything the tower reads of it: its
// layout and its planes as the step's picture table holds them.
func pictureKey(img image.Image) (*RowsKey, error) {
	var ent [4 * 256]byte
	l, planes, err := picturePlanes(img, &ent)
	if err != nil {
		return nil, err
	}
	hs := sha256.New()
	var hdr [29]byte
	hdr[0] = l.px
	for i, v := range [7]int{l.w, l.h, l.stride, l.cstride, l.p2, l.p3, l.sub} {
		binary.LittleEndian.PutUint32(hdr[1+4*i:], uint32(v))
	}
	hs.Write(hdr[:])
	for _, p := range planes {
		hs.Write(p)
	}
	var k RowsKey
	hs.Sum(k[:0])
	return &k, nil
}

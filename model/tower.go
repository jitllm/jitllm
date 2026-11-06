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
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
	"github.com/jitllm/jitllm/tok/jinja"
)

// The vision tower as its container states it (jlm.Tower), lowered into the
// engine's graphs: an entry from the picture's samples to the first block's
// rows, the repeated block, and a head from the last block's rows to the
// language model's. Nothing here knows a family: each graph is the record's
// steps and ops, op for op, and the tensors its inventory holds.

// towerGraphs is a stated tower's graphs: an entry for each sample format a
// decoded picture arrives in, a graph a block, and the head.
type TowerGraphs struct {
	// classes is every size the entry is formed for, smallest first: a
	// fixed tower's one, or squares of growing sides for a picture resized
	// to a size of its own (jlm.PreSmartResize).
	Classes []towerClass
	Blocks  []*graph.Graph
	Head    *graph.Graph
	// taps is each tap's graph (jlm.TowerTap), tapBlocks the tower block
	// whose rows it reads.
	Taps      []*graph.Graph
	TapBlocks []int
	// rows is the most rows the entry writes and every block reads, lead the
	// class rows before the patches; side a fixed picture's patch grid's
	// width and patch a patch's side in pixels; out the rows the head writes
	// for rows rows, one a token of the picture's span.
	Rows, Lead, Side, Patch, Out int
	// gridDiv is the language model's grid over the patches'
	// (jlm.TowerSpan.GridDiv); merge the side of the merge windows the patch
	// rows are ordered in and window that of the windows of them, in merge
	// windows (towerTiles); full says block i attends the whole picture.
	GridDiv, Merge, Window int
	Full                   []bool
	// fit is how a picture becomes the tower's squares.
	Fit PictureFit
	// table and planes are the blocks' rotary table and its planes, nil for
	// blocks that turn nothing.
	Table  *graph.Graph
	Planes []float32
	// holders is what each chat template writes for a picture, by the
	// template's source (picturePlaceholder).
	holders sync.Map
}

// towerClass is one size the entry is formed for: its graph for each sample
// format, its picture op's attributes and its rows.
type towerClass struct {
	Entry map[uint8]*graph.Graph
	Pic   graph.Attrs
	Rows  int
}

// placeholder is what the template src writes for an image part, read off it
// once.
func (tg *TowerGraphs) placeholder(tpl *jinja.Template, src, bos, eos, mark string) (string, error) {
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
// matrix a descriptor recorded by its role (layer.mats), which an instance's
// copy binds (BindDense, BindBlock).
func (m *Model) bindTower(c *jlm.File) error {
	if m.TowerBase != int(c.H.NBlocks) {
		return fmt.Errorf("model: %d text blocks over a page array of %d: the tower's pages would be another block's",
			m.TowerBase, c.H.NBlocks)
	}
	v := c.Vision()
	// The tower's blocks are bound where they live: a matrix is recorded by a
	// pointer into its layer (layer.mats), which a copy would leave behind.
	n := len(m.Layers)
	if cap(m.Layers)-n < int(v.NLayer) {
		return fmt.Errorf("model: room for %d tower blocks after the text's, %d made", v.NLayer, cap(m.Layers)-n)
	}
	m.Layers = m.Layers[:n+int(v.NLayer)]
	ls := m.Layers[n:]
	// A tap's tensors sit at its block's index of the dense region, each tap
	// a layer of its own.
	var taps []jlm.TowerTap
	if v.Tower != nil {
		taps = v.Tower.Taps
	}
	m.TowerTaps = make([]Layer, len(taps))
	tapAt := map[int32]int{}
	for j, tp := range taps {
		tapAt[tp.Block] = j
	}
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if !e.Role.Vision() {
			continue
		}
		l := &m.TowerDense
		if e.Index >= 0 {
			j, ok := tapAt[e.Index]
			if e.Block != jlm.DenseBlock || !ok || !tapBinds(taps[j], e.Role) {
				continue
			}
			l = &m.TowerTaps[j]
		}
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
			l.Vecs = append(l.Vecs, nn.RoleVector{Role: e.Role, V: x})
			continue
		}
		t, err := entryTensor(e)
		if err != nil {
			return err
		}
		dst := l.towerSlot(e.Role)
		if dst == nil {
			dst = &t
		}
		*dst = t
		l.Mats = append(l.Mats, roleTensor{e.Role, dst})
	}
	return nil
}

// tapBinds says tap tp's program binds role r.
func tapBinds(tp jlm.TowerTap, r jlm.Role) bool {
	for _, op := range tp.Prog.Ops {
		if slices.Contains(op.Bind, r) {
			return true
		}
	}
	return false
}

// towerTapFacts is the inventory of the tensors a tap of tower block b binds:
// the dense region's at index b.
func (m *Model) towerTapFacts(b int32) graph.Facts {
	inv := map[jlm.Role]graph.Shape{}
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
		if e.Block != jlm.DenseBlock || e.Index != b || !e.Role.Vision() {
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

// towerTapFormats is the container's type of every tensor tap graph g of
// tower block b binds.
func (m *Model) TowerTapFormats(b int32, g *graph.Graph) map[jlm.Role]jlm.Type {
	out := map[jlm.Role]jlm.Type{}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if e, ok := m.Container.Find(w.Role, jlm.DenseBlock, b); ok {
				out[w.Role] = e.Type
			}
		}
	}
	return out
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
func (l *Layer) towerSlot(r jlm.Role) *Tensor {
	switch r {
	case jlm.RoleVAttnQ:
		return &l.Wq
	case jlm.RoleVAttnK:
		return &l.Wk
	case jlm.RoleVAttnV:
		return &l.Wv
	case jlm.RoleVAttnOut:
		return &l.Wo
	case jlm.RoleVFFNGate:
		return &l.Gate
	case jlm.RoleVFC1:
		return &l.Up
	case jlm.RoleVFC2:
		return &l.Down
	}
	return nil
}

// towerFacts is the shapes of the tower's tensors in its block b (the
// tower's own index), jlm.DenseBlock the entry's and the head's.
func (m *Model) TowerFacts(b int32) graph.Facts {
	inv := map[jlm.Role]graph.Shape{}
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
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
func (m *Model) TowerFormats(g *graph.Graph) map[jlm.Role]jlm.Type {
	b := int32(jlm.DenseBlock)
	if g.Block >= 0 {
		b = int32(g.Block - m.TowerBase)
	}
	out := map[jlm.Role]jlm.Type{}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if e, ok := m.Container.Find(w.Role, b, -1); ok {
				out[w.Role] = e.Type
			}
		}
	}
	return out
}

// buildTowerGraphs lowers the container's stated tower, verifying each graph
// against the tensors it binds. A container whose tower is not stated has
// none to lower, and says which family it is so its converter can be told.
func (m *Model) BuildTowerGraphs() (*TowerGraphs, error) {
	v := m.Container.Vision()
	t := v.Tower
	if t == nil {
		return nil, fmt.Errorf("model: the container's vision tower (%v) is not stated: convert it again "+
			"with a converter that states this family (jlm.Tower)", v.Projector)
	}
	dense := m.TowerFacts(jlm.DenseBlock)
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
	tg := &TowerGraphs{Lead: int(ops[0].Offset), GridDiv: int(t.Span.GridDiv), Merge: merge, Window: window,
		Side: int(v.ImageSize / v.PatchSize), Patch: P, Fit: fit}
	sides := []int{0}
	if fit.Factor > 0 {
		sides = fit.classSides()
	}
	for _, sd := range sides {
		grid := tg.Side * tg.Side
		if sd > 0 {
			grid = min((sd/P)*(sd/P), fit.MaxPix/(P*P))
		}
		cl := towerClass{Entry: map[uint8]*graph.Graph{}, Rows: grid + tg.Lead}
		for px := graph.PixRGBA8; px < graph.PixFormats; px++ {
			g, err := towerEntry(t, px, grid, tg.Lead, sd, fit, dense)
			if err != nil {
				return nil, fmt.Errorf("model: the tower's entry: %w", err)
			}
			cl.Pic, cl.Entry[px] = g.Ops[0].Attr, g
		}
		tg.Classes = append(tg.Classes, cl)
		tg.Rows = max(tg.Rows, cl.Rows)
	}
	if r := t.Block.Rope; len(r.Runs) > 0 {
		tg.Table, tg.Planes = towerRotary(t)
	}
	for i := range int(v.NLayer) {
		tg.Full = append(tg.Full, window == 0 || slices.Contains(t.Block.Full, int32(i)))
		g, err := towerBlock(m.TowerBase+i, i, t, v, m.TowerFacts(int32(i)))
		if err != nil {
			return nil, fmt.Errorf("model: the tower's block %d: %w", i, err)
		}
		tg.Blocks = append(tg.Blocks, g)
	}
	h, out, err := towerProgram(t, &t.Head, v, tg.Rows, dense)
	if err != nil {
		return nil, fmt.Errorf("model: the tower's head: %w", err)
	}
	tg.Head, tg.Out = h, out
	D := int(v.ProjDim)
	for j := range t.Taps {
		tp := &t.Taps[j]
		if tp.Block >= int32(v.NLayer) {
			return nil, fmt.Errorf("model: the tower's tap %d reads block %d of %d", j, tp.Block, v.NLayer)
		}
		g, n, err := towerProgram(t, &tp.Prog, v, tg.Rows, m.towerTapFacts(tp.Block))
		if err != nil {
			return nil, fmt.Errorf("model: the tower's tap %d: %w", j, err)
		}
		if w := g.Values[g.Out[0]].Width; n != out || w != D {
			return nil, fmt.Errorf("model: the tower's tap %d writes %d rows of %d, the head %d of %d", j, n, w, out, D)
		}
		tg.Taps, tg.TapBlocks = append(tg.Taps, g), append(tg.TapBlocks, int(tp.Block))
	}
	return tg, nil
}

// pictureFit is how a picture becomes the tower's squares: resized as a
// whole to the picture op's size; its shorter side resized to short and its
// centre cropped to it; or cut into between minTiles and maxTiles tiles of
// it, a thumbnail of the whole after them where thumb says (jlm.PreTile).
type PictureFit struct {
	Short              int
	MinTiles, MaxTiles int
	Thumb              bool
	// factor, minPix and maxPix are a resize to the picture's own size
	// (jlm.PreSmartResize); factor 0 where there is none.
	Factor, MinPix, MaxPix int
	// lift scales a picture a side shorter than factor up first
	// (jlm.PreSmartResize's Ints[5]).
	lift bool
}

// smartSize is a w x h picture's own size (Qwen2-VL's smart_resize): each
// side rounded half to even to a multiple of factor, at least one; then,
// past maxPix pixels, both scaled down by sqrt(area/maxPix) and floored to
// one, or short of minPix, scaled up by sqrt(minPix/area) and ceiled -- in
// float64, as Python computes them.
func (f PictureFit) smartSize(w, h int) (ow, oh int) {
	fa := f.Factor
	if f.lift && (w < fa || h < fa) {
		s := math.Max(float64(fa)/float64(h), float64(fa)/float64(w))
		w, h = int(float64(w)*s), int(float64(h)*s)
	}
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
	case ow*oh > f.MaxPix:
		beta := math.Sqrt(area / float64(f.MaxPix))
		oh = max(fa, int(math.Floor(float64(h)/beta/float64(fa)))*fa)
		ow = max(fa, int(math.Floor(float64(w)/beta/float64(fa)))*fa)
	case ow*oh < f.MinPix:
		beta := math.Sqrt(float64(f.MinPix) / area)
		oh = int(math.Ceil(float64(h)*beta/float64(fa))) * fa
		ow = int(math.Ceil(float64(w)*beta/float64(fa))) * fa
	}
	return ow, oh
}

// classSides is the sides of the squares a picture of its own size is
// encoded in, smallest first: 16 factors, doubling, and last the most a
// side reaches, a picture's at most composer.PictureSide rounded up a
// factor.
func (f PictureFit) classSides() []int {
	most := (composer.PictureSide/f.Factor + 1) * f.Factor
	var out []int
	for s := 16 * f.Factor; s < most; s *= 2 {
		out = append(out, s)
	}
	return append(out, most)
}

// towerFit is how the record's steps make a picture the tower's squares,
// or why they cannot.
func towerFit(t *jlm.Tower) (PictureFit, error) {
	var fit PictureFit
	cropped, pil, sized := false, false, false
	for _, s := range t.Pre {
		switch s.Kind {
		case jlm.PreResize:
			sized, pil = true, jlm.Filter(s.Ints[2]) != jlm.FilterTwoTap
		case jlm.PreResizeShort:
			fit.Short, sized, pil = int(s.Ints[0]), true, jlm.Filter(s.Ints[1]) != jlm.FilterTwoTap
		case jlm.PreSmartResize:
			fit.Factor, fit.MinPix, fit.MaxPix = int(s.Ints[0]), int(s.Ints[1]), int(s.Ints[2])
			fit.lift = len(s.Ints) == 6 && s.Ints[5] == 1
			sized, pil = true, jlm.Filter(s.Ints[3]) != jlm.FilterTwoTap
		case jlm.PreCenterCrop:
			cropped = true
		case jlm.PreTile:
			fit.MinTiles, fit.MaxTiles, fit.Thumb = int(s.Ints[0]), int(s.Ints[1]), s.Ints[2] == 1
		}
	}
	ways := 0
	for _, w := range []bool{fit.Short > 0, fit.MaxTiles > 0, fit.Factor > 0} {
		if w {
			ways++
		}
	}
	switch {
	case !sized:
		return fit, fmt.Errorf("the picture is never resized to the tower's size")
	case fit.Short > 0 && !cropped:
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
func towerEntry(t *jlm.Tower, px uint8, grid, lead, side int, fit PictureFit, f graph.Facts) (*graph.Graph, error) {
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
		root := int(math.Sqrt(float64(fit.MaxPix)))
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
	out, err := LowerProgram(b, t, p, []graph.ValueID{pix}, grid+lead, f)
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
	if t == nil {
		return merge, 0
	}
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
	return composer.TableGraph(TowerRotaryOf(t), 0), append(r.TabPlanes(len(axes)), axes...)
}

// towerRotaryOf is the blocks' rotary as their graphs state it: every turned
// pair's element, each row's position the cell of the grid in the order the
// patch gather lays the rows out (TOpPatchify's Width and Offset).
func TowerRotaryOf(t *jlm.Tower) graph.Rotary {
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
func (f PictureFit) Tiles(w, h, s int) (cols, rows, n int) {
	if f.MaxTiles == 0 {
		return 1, 1, 1
	}
	cols, rows = TileGrid(w, h, s, f.MinTiles, f.MaxTiles)
	n = cols * rows
	if f.Thumb && n > 1 {
		n++
	}
	return cols, rows, n
}

// window is square t of a w x h picture as the picture op's taps take it:
// the size the picture is resized to and the square's corner in it.
func (f PictureFit) Window(w, h, pw, ph, t int) (ow, oh, x0, y0 int) {
	switch {
	case f.Short > 0:
		// transformers' center_crop: the offsets floored.
		ow, oh = shortSide(w, h, f.Short)
		return ow, oh, (ow - pw) / 2, (oh - ph) / 2
	case f.MaxTiles > 0:
		cols, rows, _ := f.Tiles(w, h, pw)
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
func TileGrid(w, h, s, minT, maxT int) (cols, rows int) {
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
type Square struct {
	Class                  int
	Ow, Oh, X0, Y0, Pw, Ph int
	Gw, Gh                 int
}

// square is square i of a w x h picture and how many the picture has: one
// at its own size in the smallest class that holds it, or a fixed square,
// a crop or a tile at the entry's size.
func (tg *TowerGraphs) Square(w, h, i int) (Square, int) {
	if tg.Fit.Factor > 0 {
		ow, oh := tg.Fit.smartSize(w, h)
		c := 0
		for c < len(tg.Classes)-1 && max(ow, oh) > tg.Classes[c].Pic.PicW {
			c++
		}
		return Square{Class: c, Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: ow / tg.Patch, Gh: oh / tg.Patch}, 1
	}
	a := tg.Classes[0].Pic
	_, _, n := tg.Fit.Tiles(w, h, a.PicW)
	ow, oh, x0, y0 := tg.Fit.Window(w, h, a.PicW, a.PicH, i)
	return Square{Ow: ow, Oh: oh, X0: x0, Y0: y0, Pw: a.PicW, Ph: a.PicH, Gw: tg.Side, Gh: tg.Side}, n
}

// squareOut is the rows the head writes for square sq: its patches over
// the head's ratio of rows out to patch rows in.
func (tg *TowerGraphs) SquareOut(sq Square) int {
	return sq.Gw * sq.Gh * tg.Out / (tg.Rows - tg.Lead)
}

// picRows is the rows a w x h picture becomes: its squares' every row.
func (tg *TowerGraphs) PicRows(w, h int) int {
	sq, n := tg.Square(w, h, 0)
	rows := tg.SquareOut(sq)
	for i := 1; i < n; i++ {
		sq, _ = tg.Square(w, h, i)
		rows += tg.SquareOut(sq)
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
func towerProgram(t *jlm.Tower, p *jlm.Program, v *jlm.Vision, rows int, f graph.Facts) (*graph.Graph, int, error) {
	b := graph.NewBlock(-1)
	x := b.In("rows", int(v.NEmbd))
	out, err := LowerProgram(b, t, p, []graph.ValueID{x}, rows, f)
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

// resampleOf is a record's resample (jlm.ResampleAligned, jlm.TapsLinear) as
// graph.OpPosResample states it.
func resampleOf(chain, filter uint8) (uint8, uint8, error) {
	var c, f uint8
	switch chain {
	case jlm.ResampleAligned:
		c = graph.ChainAligned
	case jlm.ResampleHalfPixel:
		c = graph.ChainHalfPixel
	}
	switch filter {
	case jlm.TapsBilinear:
		f = graph.FilterBilinear
	case jlm.TapsLinear:
		f = graph.FilterLinear
	case jlm.TapsCubic:
		f = graph.FilterCubic
	}
	if c == 0 || f == 0 {
		return 0, 0, fmt.Errorf("a resample by chain %d and taps %d", chain, filter)
	}
	return c, f, nil
}

// lowerProgram adds p's ops to b over its inputs in, each op the graph's of
// the same meaning, and is p's result. A value's width is its op's: a
// matrix's rows, a regroup's input times its group, and a patch gather's the
// width the matrix reading it takes.
func LowerProgram(b *graph.Builder, t *jlm.Tower, p *jlm.Program, in []graph.ValueID, rows int, f graph.Facts) (graph.ValueID, error) {
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
		case jlm.TOpPosResample:
			s, err := shape(op.Bind[0])
			if err != nil {
				return 0, err
			}
			if s.Rows != int(op.Span*op.Span) {
				return 0, fmt.Errorf("resamples a table %d a side from %d rows", op.Span, s.Rows)
			}
			chain, filter, err := resampleOf(op.Chain, op.Filter)
			if err != nil {
				return 0, err
			}
			out = b.Emit(graph.OpPosResample, nil, []graph.Result{{Name: "pos", Width: s.K}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				graph.Attrs{Span: int(op.Span), Offset: int(op.Offset), Width: int(op.Width), Window: window, Chain: chain, Filter: filter}, nil, nil)[0]
		case jlm.TOpAdd:
			out = b.Add(ins[0], ins[1])
		case jlm.TOpLayerNorm:
			out = b.LayerNorm(ins[0], op.Bind[0], float64(op.Eps), op.Bind[1:]...)
		case jlm.TOpRMSNorm:
			if len(op.Bind) == 0 {
				w := width(ins[0])
				out = b.Emit(graph.OpHeadNorm, ins, []graph.Result{{Name: name, Width: w}}, nil,
					graph.Attrs{Eps: float64(op.Eps), HeadDim: w}, nil, nil)[0]
				break
			}
			out = b.Norm(ins[0], op.Bind[0], float64(op.Eps))
		case jlm.TOpScale:
			s, err := shape(op.Bind[0])
			if err != nil {
				return 0, err
			}
			out = mapRow(b, name, graph.Attrs{Fn: graph.MapMul}, []graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				width(ins[0]), ins[0])
		case jlm.TOpSum, jlm.TOpMax:
			fn := graph.MapAdd
			if op.Code == jlm.TOpMax {
				fn = graph.MapMax
			}
			out = b.Emit(graph.OpReduce, ins, []graph.Result{{Name: name, Width: 1}}, nil, graph.Attrs{Fn: fn}, nil, nil)[0]
		case jlm.TOpConcat:
			w := 0
			for _, v := range ins {
				w += width(v)
			}
			out = b.Emit(graph.OpConcat, ins, []graph.Result{{Name: name, Width: w}}, nil, graph.Attrs{}, nil, nil)[0]
		case jlm.TOpSlice:
			out = b.Emit(graph.OpSlice, ins, []graph.Result{{Name: name, Width: int(op.Width)}}, nil,
				graph.Attrs{Offset: int(op.Offset), Width: int(op.Width)}, nil, nil)[0]
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
	rot := TowerRotaryOf(t)
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
func TowerConfig(base, rows int, v *jlm.Vision, t *jlm.Tower) *Config {
	b := &t.Block
	r := TowerRotaryOf(t)
	return &Config{Arch: "tower", NLayer: base + int(v.NLayer), NEmbd: int(v.NEmbd), NHead: int(b.NHead),
		NKVHead: int(b.NKVHead), HeadDim: int(b.HeadDim), NFFN: int(v.NFFN), NCtx: rows,
		RMSEps: float64(b.Eps), EmbdScale: 1, AttnFactor: 1, NonCausal: true, NoPosEnc: r.NRot == 0,
		NRot: r.NRot, RopeBase: r.Base, RopeNeox: b.Rope.Neox, LayerNorm: b.Norm == jlm.NormLayer}
}

// pictureLayout is how a decoded picture's samples sit in the step's picture
// table: their format, the picture's size and its rows' stride in bytes, and
// for a second and third plane (YCbCr's chroma, a palette) their stride, byte
// offsets and the chroma subsampling (composer.RequestPicSub).
type PictureLayout struct {
	Px                   uint8
	W, H, stride         int
	cstride, p2, p3, sub int
}

// request writes the layout's words into the step's request.
func (l PictureLayout) Request(req []uint32) {
	req[composer.RequestPicW], req[composer.RequestPicH], req[composer.RequestPicStride] = uint32(l.W), uint32(l.H), uint32(l.stride)
	req[composer.RequestPicCStride], req[composer.RequestPicP2], req[composer.RequestPicP3] = uint32(l.cstride), uint32(l.p2), uint32(l.p3)
	req[composer.RequestPicSub] = uint32(l.sub)
}

// picturePlanes is a decoded picture's layout and the planes the step's
// picture table holds for it, each from a word boundary after the last: its
// samples' bytes as the decoder left them, and a palette's entries -- four
// bytes each, the colour's straight red, green and blue and its alpha --
// written into ent.
func picturePlanes(img image.Image, ent *[4 * 256]byte) (PictureLayout, [3][]byte, error) {
	b := img.Bounds()
	l := PictureLayout{W: b.Dx(), H: b.Dy()}
	var planes [3][]byte
	if l.W <= 0 || l.H <= 0 {
		return l, planes, fmt.Errorf("model: an empty picture")
	}
	rows := func(pix []byte, stride, bpp int) []byte { return pix[:(l.H-1)*stride+l.W*bpp] }
	switch p := img.(type) {
	case *image.NRGBA:
		l.Px, l.stride, planes[0] = graph.PixRGBA8, p.Stride, rows(p.Pix, p.Stride, 4)
	case *image.RGBA:
		l.Px, l.stride, planes[0] = graph.PixRGBA8, p.Stride, rows(p.Pix, p.Stride, 4)
	case *image.Gray:
		l.Px, l.stride, planes[0] = graph.PixGray8, p.Stride, rows(p.Pix, p.Stride, 1)
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
		l.Px, l.stride = graph.PixPaletted8, p.Stride
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
func ycbcrPlanes(l PictureLayout, p *image.YCbCr) (PictureLayout, [3][]byte, error) {
	r := p.Rect
	ylen := p.YOffset(r.Max.X-1, r.Max.Y-1) + 1
	clen := p.COffset(r.Max.X-1, r.Max.Y-1) + 1
	planes := [3][]byte{p.Y[:ylen], p.Cb[:clen], p.Cr[:clen]}
	hs, mx, okx := chromaShift(r.Dx(), 1, func(k int) int { return p.COffset(r.Min.X+k, r.Min.Y) })
	vs, my, oky := chromaShift(r.Dy(), p.CStride, func(k int) int { return p.COffset(r.Min.X, r.Min.Y+k) })
	if !okx || !oky {
		return l, planes, fmt.Errorf("model: a %v YCbCr picture's chroma follows no shift", p.SubsampleRatio)
	}
	l.Px, l.stride, l.cstride, l.sub = graph.PixYCbCr8, p.YStride, p.CStride, hs|vs<<8|mx<<16|my<<24
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
func (l *PictureLayout) place(planes [3][]byte) {
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
func PictureWords(img image.Image, pad *[]uint32, ent *[4 * 256]byte) ([]uint32, PictureLayout, error) {
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

// SeesPictures reports whether the model carries a tower, so a prompt may hold
// a picture.
func (m *Model) SeesPictures() bool { return m.Tower != nil }

// PictureSpan is the span a picture stands in a prompt as, for ChatSpans or
// PrefillMixed: the picture, which the prefill encodes through the tower,
// the rows it takes, and the key naming them (pictureKey).
func (m *Model) PictureSpan(img image.Image) (Span, error) {
	if m.Tower == nil {
		return Span{}, fmt.Errorf("model: the model has no vision tower")
	}
	k, err := pictureKey(img)
	if err != nil {
		return Span{}, err
	}
	b := img.Bounds()
	return Span{Picture: img, PicRows: m.Tower.PicRows(b.Dx(), b.Dy()), Grid: m.Tower.SpanGrid(b.Dx(), b.Dy()), Key: k}, nil
}

// spanGrid is the grid a w x h picture's rows are for the language model's
// rotary (jlm.TowerSpan.GridDiv): its one square's patch grid with each side
// divided; none where the rows advance as text.
func (tg *TowerGraphs) SpanGrid(w, h int) ImageGrid {
	if tg.GridDiv == 0 {
		return ImageGrid{}
	}
	sq, _ := tg.Square(w, h, 0)
	return ImageGrid{T: 1, H: sq.Gh / tg.GridDiv, W: sq.Gw / tg.GridDiv}
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
	hdr[0] = l.Px
	for i, v := range [7]int{l.W, l.H, l.stride, l.cstride, l.p2, l.p3, l.sub} {
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

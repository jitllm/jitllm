package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"math"
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
	entry  map[uint8]*graph.Graph
	blocks []*graph.Graph
	head   *graph.Graph
	// rows is the patch rows the entry writes and every block reads, side
	// the patch grid's width; out the rows the head writes, one a token of
	// the picture's span.
	rows, side, out int
	// pic is the entry's picture op's attributes: its size, and for a
	// resize by Pillow's filters its taps.
	// short is the shorter side a picture is resized to before its centre
	// is cropped to the picture op's size, 0 where the whole picture is
	// resized to it.
	short int
	pic graph.Attrs
	// holders is what each chat template writes for a picture, by the
	// template's source (picturePlaceholder).
	holders sync.Map
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
		if e.Role.Expanded() || e.NDim < 2 {
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
		if e.NDim >= 2 {
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
	side := int(v.ImageSize / v.PatchSize)
	lead := 0
	if ops := t.Entry.Ops; len(ops) > 0 && ops[0].Code == jlm.TOpPatchify {
		lead = int(ops[0].Offset)
	}
	tg := &towerGraphs{entry: map[uint8]*graph.Graph{}, rows: side*side + lead, side: side}
	for px := graph.PixRGBA8; px < graph.PixFormats; px++ {
		g, short, err := towerEntry(t, px, side*side, lead, dense)
		if err != nil {
			return nil, fmt.Errorf("model: the tower's entry: %w", err)
		}
		tg.pic, tg.short = g.Ops[0].Attr, short
		tg.entry[px] = g
	}
	for i := range int(v.NLayer) {
		g, err := towerBlock(m.towerBase+i, t, v, m.towerFacts(int32(i)))
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

// towerEntry is the entry for pictures of sample format px: the record's
// preprocessing as one op (graph.OpPicture), then its entry program over
// the grid's patch rows after lead rows (a class row's), and the shorter
// side the picture is resized to before its centre is cropped (0 for a
// resize of the whole picture to the op's size).
func towerEntry(t *jlm.Tower, px uint8, grid, lead int, f graph.Facts) (*graph.Graph, int, error) {
	a := graph.Attrs{Pixels: px, Std: [3]float64{1, 1, 1}}
	short, cropped := 0, false
	for _, s := range t.Pre {
		switch s.Kind {
		case jlm.PreResize:
			a.PicW, a.PicH = int(s.Ints[0]), int(s.Ints[1])
			if err := resizeChain(&a, jlm.Filter(s.Ints[2])); err != nil {
				return nil, 0, err
			}
		case jlm.PreResizeShort:
			short = int(s.Ints[0])
			a.PicW, a.PicH = short, short
			if err := resizeChain(&a, jlm.Filter(s.Ints[1])); err != nil {
				return nil, 0, err
			}
			if a.Chain != graph.ChainPIL {
				return nil, 0, fmt.Errorf("a resize of the shorter side by filter %d, which crops no window", s.Ints[1])
			}
		case jlm.PreCenterCrop:
			a.PicW, a.PicH, cropped = int(s.Ints[0]), int(s.Ints[1]), true
		case jlm.PreNormalize:
			for c := range 3 {
				a.Mean[c], a.Std[c] = float64(s.Floats[c]), float64(s.Floats[3+c])
			}
		}
	}
	switch {
	case a.PicW == 0:
		return nil, 0, fmt.Errorf("the picture is never resized to the tower's size")
	case short > 0 && !cropped:
		return nil, 0, fmt.Errorf("a picture resized by its shorter side and never cropped has no one size")
	case short > 0 && (a.PicW > short || a.PicH > short):
		return nil, 0, fmt.Errorf("a %dx%d crop of a picture whose shorter side is %d", a.PicW, a.PicH, short)
	}
	if a.Chain == graph.ChainPIL {
		// A crop's taps are a window of the resize's, whose scale is at most
		// the crop's own.
		a.Taps = max(composer.PILTapCount(a.Filter, a.PicSide, a.PicW), composer.PILTapCount(a.Filter, a.PicSide, a.PicH))
	}
	p := &t.Entry
	if len(p.Ops) == 0 || p.Ops[0].Code != jlm.TOpPatchify || p.NIn != 1 {
		return nil, 0, fmt.Errorf("an entry that does not begin with the picture's patches")
	}
	b := graph.NewBlock(-1)
	P := int(p.Ops[0].Span)
	pix := b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: 3 * P * P}}, nil, a, nil, nil)[0]
	if a.PicW*a.PicH != grid*P*P {
		return nil, 0, fmt.Errorf("a %dx%d picture is not %d patches of %d", a.PicW, a.PicH, grid, P)
	}
	out, err := lowerProgram(b, p, []graph.ValueID{pix}, grid+lead, f)
	if err != nil {
		return nil, 0, err
	}
	b.Out(out)
	return b.G, short, graph.Verify(b.G, f)
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
	out, err := lowerProgram(b, &t.Head, []graph.ValueID{x}, rows, f)
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
func lowerProgram(b *graph.Builder, p *jlm.Program, in []graph.ValueID, rows int, f graph.Facts) (graph.ValueID, error) {
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
				graph.Attrs{Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
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
func towerBlock(li int, t *jlm.Tower, v *jlm.Vision, f graph.Facts) (*graph.Graph, error) {
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
	hist := graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, vv, 0)}
	scale := float64(bl.AttnScale)
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(hd))
	}
	att := b.AttendOf(q, hist, graph.Attrs{NHead: int(bl.NHead), NKVHead: int(bl.NKVHead), HeadDim: hd, Scale: scale, Bidir: true})
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
// attending both ways, with no rotary and no vocabulary. Its block indices
// are the model's (base on), so a block is asked by the index it runs at.
func towerConfig(base, rows int, v *jlm.Vision, t *jlm.Tower) *Config {
	b := &t.Block
	return &Config{Arch: "tower", NLayer: base + int(v.NLayer), NEmbd: int(v.NEmbd), NHead: int(b.NHead),
		NKVHead: int(b.NKVHead), HeadDim: int(b.HeadDim), NFFN: int(v.NFFN), NCtx: rows,
		RMSEps: float64(b.Eps), EmbdScale: 1, AttnFactor: 1, NonCausal: true, NoPosEnc: true,
		LayerNorm: b.Norm == jlm.NormLayer}
}

// towerRun is an instance's tower: an instance of its own over the tower's
// blocks (towerConfig), whose history holds one picture's rows at a time,
// and the entry's forms (one a sample format) and the head's on it, all
// composed when the instance is made.
type towerRun struct {
	s     *Instance
	entry map[uint8]*cpudev.Form
	head  *cpudev.Form
	// rows is the picture's patch rows at the tower's width, as the entry
	// writes them, and blk as the last block leaves them; out the head's at
	// the language model's; seq and pos the rows' sequence and positions,
	// posTab the step's live rows and positions as the entry reads them, mode
	// the blocks' run on a card (embedMode.rows).
	rows, blk, out []float32
	seq, pos       []int
	posTab         []uint32
	mode           embedMode
	req            []uint32
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
	tr := &towerRun{s: ts, entry: map[uint8]*cpudev.Form{}, rows: make([]float32, n*c.NEmbd),
		blk: make([]float32, n*c.NEmbd), out: make([]float32, tg.out*int(v.ProjDim)),
		seq: make([]int, n), pos: make([]int, n),
		posTab: make([]uint32, 1+n), req: make([]uint32, composer.RequestPicTaps+(tg.pic.PicW+tg.pic.PicH)*(tg.pic.Taps+1))}
	tr.posTab[0] = uint32(n)
	for i := range n {
		tr.pos[i], tr.posTab[1+i] = i, uint32(i)
	}
	for px, g := range tg.entry {
		f, err := m.formCPUGraph(ts, g, n, nil)
		if err != nil {
			ts.Close()
			return nil, fmt.Errorf("model: the tower's entry: %w", err)
		}
		tr.entry[px] = f
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
	return s.encodePicture(img, s.tower.out)
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
	words, pl, err := pictureWords(img, &tr.pad, &tr.ent)
	if err != nil {
		return nil, err
	}
	ts, m := tr.s, s.m
	tg := m.tower
	f := tr.entry[pl.px]
	if why := ts.cpuWeights(f, -1, &m.towerDense); why != composer.DeclineServes {
		return nil, fmt.Errorf("model: the tower's entry: %v", why)
	}
	tr.req[composer.RequestGrid] = uint32(tg.side)
	pl.request(tr.req)
	if a := tg.pic; a.Chain == graph.ChainPIL && (pl.w != tr.tapsW || pl.h != tr.tapsH) {
		if pl.w > a.PicSide || pl.h > a.PicSide {
			return nil, fmt.Errorf("model: a %dx%d picture is past the %d pixels a side the tower's resize reaches", pl.w, pl.h, a.PicSide)
		}
		// A crop is the centre window of the resize (transformers'
		// center_crop: the offsets floored).
		ow, oh := a.PicW, a.PicH
		if tg.short > 0 {
			ow, oh = shortSide(pl.w, pl.h, tg.short)
		}
		x0, y0 := (ow-a.PicW)/2, (oh-a.PicH)/2
		composer.PILTaps(a.Filter, pl.w, ow, x0, a.PicW, a.Taps, tr.req[composer.RequestPicTaps:])
		composer.PILTaps(a.Filter, pl.h, oh, y0, a.PicH, a.Taps, tr.req[composer.RequestPicTaps+a.PicW*(a.Taps+1):])
		tr.tapsW, tr.tapsH = pl.w, pl.h
	}
	b := &ts.cpu.bound
	*b = cpuBound{li: -1, l: &m.towerDense, own: ts.cpu.dev.Own(f), n: tg.rows, req: tr.req, pos: tr.posTab, pic: words}
	b.io[0] = unsafe.Pointer(&tr.rows[0])
	ts.cpuRunForm(b, f)
	if err := ts.towerBlocks(tr); err != nil {
		return nil, fmt.Errorf("model: the tower's blocks: %w", err)
	}
	f = tr.head
	if why := ts.cpuWeights(f, -1, &m.towerDense); why != composer.DeclineServes {
		return nil, fmt.Errorf("model: the tower's head: %v", why)
	}
	if len(dst) != tg.out*int(m.container.Vision().ProjDim) {
		return nil, fmt.Errorf("model: %d values for a picture's %d rows of %d", len(dst), tg.out,
			m.container.Vision().ProjDim)
	}
	*b = cpuBound{li: -1, l: &m.towerDense, own: ts.cpu.dev.Own(f), n: tg.rows, req: tr.req, pos: tr.posTab}
	b.io[0], b.io[1] = unsafe.Pointer(&tr.blk[0]), unsafe.Pointer(&dst[0])
	ts.cpuRunForm(b, f)
	return dst, nil
}

// towerBlocks runs the tower's blocks over the patch rows, one run in which
// every row's keys are every row's, leaving the last block's rows in blk: on
// the CPU device, the runner's CPU rows over the run, block by block; with
// blocks on a card, the tower instance's prefill (embedMode.rows), each block
// on the device that holds it.
func (ts *Instance) towerBlocks(tr *towerRun) error {
	n := len(tr.pos)
	ts.bidir = append(ts.bidir[:0], nn.KeyRun{Lo: 0, Hi: n})
	defer func() { ts.bidir = ts.bidir[:0] }()
	if ts.devCount() > 0 {
		ts.Reset()
		tr.mode = embedMode{bidir: true, rows: true, total: n, out: tr.blk}
		ts.emb = &tr.mode
		defer func() { ts.emb = nil }()
		_, err := ts.prefillSpans(0, Span{Embd: tr.rows})
		return err
	}
	copy(tr.blk, tr.rows)
	for li := ts.lo; li < ts.hi; li++ {
		if err := ts.m.pageIn(li); err != nil {
			return err
		}
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
	w := s.m.tower.out * int(s.m.container.Vision().ProjDim)
	tr.spans = append(tr.spans[:0], spans...)
	k, row := 0, 0
	for i := range tr.spans {
		sp := &tr.spans[i]
		r := sp.n(s.c.NEmbd)
		if sp.Picture != nil && sp.Embd == nil {
			if sp.PicRows != s.m.tower.out {
				return nil, fmt.Errorf("model: span %d is a picture of %d rows and the tower writes %d",
					i, sp.PicRows, s.m.tower.out)
			}
			if k == len(tr.pics) {
				tr.pics = append(tr.pics, make([]float32, w))
			}
			sp.Embd = tr.pics[k]
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
	return Span{Picture: img, PicRows: m.tower.out, Key: k}, nil
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

package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"image"
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
	// holders is what each chat template writes for a picture, by the
	// template's source (picturePlaceholder).
	holders sync.Map
}

// placeholder is what the template src writes for an image part, read off it
// once.
func (tg *towerGraphs) placeholder(tpl *jinja.Template, src, bos, eos string) (string, error) {
	if ph, ok := tg.holders.Load(src); ok {
		return ph.(string), nil
	}
	ph, err := picturePlaceholder(tpl, src, bos, eos)
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
		l.mats = append(l.mats, roleTensor{e.Role, &t})
	}
	return bindPacked(c, m.Cfg, &m.towerDense)
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
	tg := &towerGraphs{entry: map[uint8]*graph.Graph{}, rows: side * side, side: side}
	for _, px := range []uint8{graph.PixRGBA8, graph.PixGray8} {
		g, err := towerEntry(t, px, tg.rows, dense)
		if err != nil {
			return nil, fmt.Errorf("model: the tower's entry: %w", err)
		}
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
// preprocessing as one op (graph.OpPicture), then its entry program.
func towerEntry(t *jlm.Tower, px uint8, rows int, f graph.Facts) (*graph.Graph, error) {
	a := graph.Attrs{Pixels: px, Std: [3]float64{1, 1, 1}}
	for _, s := range t.Pre {
		switch s.Kind {
		case jlm.PreResize:
			if jlm.Filter(s.Ints[2]) != jlm.FilterTwoTap {
				return nil, fmt.Errorf("a resize by filter %d, which no picture op runs yet", s.Ints[2])
			}
			a.PicW, a.PicH, a.Chain, a.Filter = int(s.Ints[0]), int(s.Ints[1]), graph.ChainHalfPixel, graph.FilterLinear
		case jlm.PreNormalize:
			for c := range 3 {
				a.Mean[c], a.Std[c] = float64(s.Floats[c]), float64(s.Floats[3+c])
			}
		}
	}
	if a.PicW == 0 {
		return nil, fmt.Errorf("the picture is never resized to the tower's size")
	}
	p := &t.Entry
	if len(p.Ops) == 0 || p.Ops[0].Code != jlm.TOpPatchify || p.NIn != 1 {
		return nil, fmt.Errorf("an entry that does not begin with the picture's patches")
	}
	b := graph.NewBlock(-1)
	P := int(p.Ops[0].Span)
	pix := b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: 3 * P * P}}, nil, a, nil, nil)[0]
	if a.PicW*a.PicH != rows*P*P {
		return nil, fmt.Errorf("a %dx%d picture is not %d patches of %d", a.PicW, a.PicH, rows, P)
	}
	out, err := lowerProgram(b, p, []graph.ValueID{pix}, rows, f)
	if err != nil {
		return nil, err
	}
	b.Out(out)
	return b.G, graph.Verify(b.G, f)
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
	// rows is the picture's patch rows at the tower's width, out the head's
	// at the language model's; seq and pos the rows' sequence and positions,
	// posTab the step's live rows and positions as the entry reads them.
	rows, out []float32
	seq, pos  []int
	posTab    []uint32
	req       []uint32
	// pad holds a picture's samples where its bytes are not whole words.
	pad []uint32
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
		out: make([]float32, tg.out*int(v.ProjDim)), seq: make([]int, n), pos: make([]int, n),
		posTab: make([]uint32, 1+n), req: make([]uint32, composer.RequestPicStride+1)}
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

// pictureWords is a decoded picture's samples as the step's picture table
// reads them, the format they are in, and the picture's size and row stride
// in bytes; a picture whose bytes are not whole words is copied into pad,
// which grows to hold it.
func pictureWords(img image.Image, pad *[]uint32) (words []uint32, px uint8, w, h, stride int, err error) {
	var pix []byte
	switch p := img.(type) {
	case *image.NRGBA:
		pix, px, stride = p.Pix, graph.PixRGBA8, p.Stride
	case *image.RGBA:
		pix, px, stride = p.Pix, graph.PixRGBA8, p.Stride
	case *image.Gray:
		pix, px, stride = p.Pix, graph.PixGray8, p.Stride
	default:
		return nil, 0, 0, 0, 0, fmt.Errorf("model: a %T picture has no picture op reading its samples", img)
	}
	b := img.Bounds()
	w, h = b.Dx(), b.Dy()
	if w <= 0 || h <= 0 || len(pix) == 0 {
		return nil, 0, 0, 0, 0, fmt.Errorf("model: an empty picture")
	}
	if len(pix)%4 == 0 && uintptr(unsafe.Pointer(&pix[0]))%4 == 0 {
		return unsafe.Slice((*uint32)(unsafe.Pointer(&pix[0])), len(pix)/4), px, w, h, stride, nil
	}
	n := (len(pix) + 3) / 4
	if cap(*pad) < n {
		*pad = make([]uint32, n)
	}
	words = (*pad)[:n]
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), 4*n), pix)
	return words, px, w, h, stride, nil
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
	words, px, w, h, stride, err := pictureWords(img, &tr.pad)
	if err != nil {
		return nil, err
	}
	ts, m := tr.s, s.m
	tg := m.tower
	f := tr.entry[px]
	if why := ts.cpuWeights(f, -1, &m.towerDense); why != composer.DeclineServes {
		return nil, fmt.Errorf("model: the tower's entry: %v", why)
	}
	tr.req[composer.RequestGrid] = uint32(tg.side)
	tr.req[composer.RequestPicW], tr.req[composer.RequestPicH], tr.req[composer.RequestPicStride] = uint32(w), uint32(h), uint32(stride)
	b := &ts.cpu.bound
	*b = cpuBound{li: -1, l: &m.towerDense, own: ts.cpu.dev.Own(f), n: tg.rows, req: tr.req, pos: tr.posTab, pic: words}
	b.io[0] = unsafe.Pointer(&tr.rows[0])
	ts.cpuRunForm(b, f)
	// The blocks: one run over every row, each row's keys every row's.
	ts.bidir = append(ts.bidir[:0], nn.KeyRun{Lo: 0, Hi: tg.rows})
	defer func() { ts.bidir = ts.bidir[:0] }()
	for li := ts.lo; li < ts.hi; li++ {
		if err := m.pageIn(li); err != nil {
			return nil, err
		}
		why, err := ts.cpuRows(li, &m.layers[li], tr.rows, nil, nil, tr.seq, tr.pos, nil)
		if err != nil {
			return nil, err
		}
		if why != composer.DeclineServes {
			return nil, fmt.Errorf("model: the tower's block %d: %v", li-m.towerBase, why)
		}
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
	b.io[0], b.io[1] = unsafe.Pointer(&tr.rows[0]), unsafe.Pointer(&dst[0])
	ts.cpuRunForm(b, f)
	return dst, nil
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

// pictureKey names a picture by everything the tower reads of it: its sample
// format, its size and its samples row by row (a row's padding past its
// samples is not read, so it is not named).
func pictureKey(img image.Image) (*RowsKey, error) {
	var pad []uint32
	words, px, w, h, stride, err := pictureWords(img, &pad)
	if err != nil {
		return nil, err
	}
	bpp := 4
	if px == graph.PixGray8 {
		bpp = 1
	}
	pix := unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), 4*len(words))
	hs := sha256.New()
	var hdr [13]byte
	hdr[0] = px
	binary.LittleEndian.PutUint32(hdr[1:], uint32(w))
	binary.LittleEndian.PutUint32(hdr[5:], uint32(h))
	binary.LittleEndian.PutUint32(hdr[9:], uint32(bpp))
	hs.Write(hdr[:])
	for y := range h {
		hs.Write(pix[y*stride : y*stride+w*bpp])
	}
	var k RowsKey
	hs.Sum(k[:0])
	return &k, nil
}

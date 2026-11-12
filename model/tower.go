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
	// Chain says the blocks are a chained tower's (jlm.Tower.Chain), each a
	// graph over a grid of its own run after the one before it; HeadFrom
	// the blocks whose rows the head reads beside the last's.
	Chain    bool
	HeadFrom []int
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
	// Line and Ends are the rows the head adds to the merged grid's: Line
	// after each of its rows (TOpRowBreak), Ends before and after it
	// (TOpWrapRows); Frame the merge they count the grid over.
	Line, Ends, Frame int
	// Trim is the break rows the frame leaves off its last row: 1 where the
	// head breaks after each merged row but the last (TOpRowBreak's Offset).
	Trim int
	// Fixed is the rows the head writes for a square whatever its grid,
	// its marker rows aside (a resampler's queries, TOpRowsAttend); 0 for
	// a head whose rows follow the grid.
	Fixed int
	// Marks is the most marker rows the head writes around a square's rows
	// (TOpMarkRow's Span and Offset), each run where the square names one
	// (Square.Mark, Square.Trail).
	Marks int
	// Tail is the class rows after the patches (TOpTailRow).
	Tail int
	Full []bool
	// fit is how a picture becomes the tower's squares.
	Fit PictureFit
	// Video is how the tower reads a video (jlm.Video), nil for none, and
	// Videos its entry's classes over a pair of frames, one for each
	// picture class's size.
	Video  *jlm.Video
	Videos []towerClass
	// table and planes are the blocks' rotary table and its planes, nil for
	// blocks that turn nothing.
	Table  *graph.Graph
	Planes []float32
	// holders is what each chat template writes for a picture and for a
	// video, by the template's source (mediaPlaceholder).
	holders sync.Map
}

// towerClass is one size the entry is formed for: its graph for each sample
// format, its picture op's attributes and its rows.
type towerClass struct {
	Entry map[uint8]*graph.Graph
	// Band is the class's band graphs, where its resize is Pillow's
	// (graph.PicBand), by BandKey; BandRuns the band graphs each entry runs
	// first, in order. Its entries' picture ops then finish the bands' sums.
	Band     map[uint8]*graph.Graph
	BandRuns map[uint8][]uint8
	Pic      graph.Attrs
	Rows     int
}

// placeholder is what the template src writes for an image part, or a video
// part, read off it once.
func (tg *TowerGraphs) placeholder(tpl *jinja.Template, src, bos, eos, mark string, video bool) (string, error) {
	key := holderKey{src, video}
	if ph, ok := tg.holders.Load(key); ok {
		return ph.(string), nil
	}
	ph, err := mediaPlaceholder(tpl, src, bos, eos, mark, video)
	if err != nil {
		return "", err
	}
	tg.holders.Store(key, ph)
	return ph, nil
}

// holderKey names a template's placeholder: its source and the part's kind.
type holderKey struct {
	src   string
	video bool
}

// audioBlocks is how many audio blocks follow the vision blocks in v's
// page array (jlm.Audio).
func audioBlocks(v *jlm.Vision) int {
	if v.Audio == nil {
		return 0
	}
	return int(v.Audio.NLayer)
}

// bindTower makes the tower's blocks layers of the model from towerBase on,
// the page each is read from the container's vision block of that index, and
// its entry's and head's tensors towerDense: every vector read once, every
// matrix a descriptor recorded by its role (layer.mats), which an instance's
// copy binds (BindDense, BindBlock).
func (m *Model) bindTower(c *jlm.File) error {
	if m.TowerBase > int(c.H.NBlocks) {
		return fmt.Errorf("model: %d text blocks over a page array of %d: the tower's pages would be another block's",
			m.TowerBase, c.H.NBlocks)
	}
	// Blocks the container holds past the text model's (a decision model's
	// head blocks) keep their pages: the tower's start after them.
	m.Layers = m.Layers[:c.H.NBlocks]
	m.TowerBase = int(c.H.NBlocks)
	v := c.Vision()
	// The tower's blocks are bound where they live: a matrix is recorded by a
	// pointer into its layer (layer.mats), which a copy would leave behind.
	n := len(m.Layers)
	if cap(m.Layers)-n < int(v.NLayer) {
		return fmt.Errorf("model: room for %d tower blocks after the text's, %d made", v.NLayer, cap(m.Layers)-n)
	}
	m.Layers = m.Layers[:n+int(v.NLayer)+audioBlocks(v)]
	ls := m.Layers[n : n+int(v.NLayer)]
	audio := m.Layers[n+int(v.NLayer):]
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
		case e.Block >= 0 && int(e.Block) < len(ls)+len(audio):
			l = &audio[int(e.Block)-len(ls)]
		case e.Block != jlm.DenseBlock:
			return fmt.Errorf("model: the tower holds %v in block %d of %d", e.Role, e.Block, len(ls))
		}
		if e.Role.Expanded() || e.NDim < 2 || squeezed(e) {
			x, err := entryVector(c, e)
			if err != nil {
				return err
			}
			if e.Role == jlm.RoleVClamp {
				// The block's named vector (Layer.NamedVector), as every
				// device reads it.
				l.Clamp = x
				continue
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
			// Every trailing dimension's rows, as the tensor's load counts
			// them: a stack of tables is its tables' rows back to back.
			s.Rows = 1
			for d := 1; d < int(e.NDim); d++ {
				s.Rows *= int(e.Dims[d])
			}
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
			// Every trailing dimension's rows, as the tensor's load counts
			// them: a stack of tables is its tables' rows back to back.
			s.Rows = 1
			for d := 1; d < int(e.NDim); d++ {
				s.Rows *= int(e.Dims[d])
			}
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

// ClipRows is the frames a clip of n samples makes in the audio tower's
// entry and the rows of them the blocks read, both from the stated entry:
// the centred frames one more than its hops (TOpFrames), the clip's those
// the feature norm keeps (its Offset fewer), then each strided convolution's
// output side, (L + 2*Offset - Span)/Stride + 1.
func (m *Model) ClipRows(n int) (frames, rows int) {
	for _, op := range m.Container.Vision().Audio.Tower.Entry.Ops {
		switch {
		case op.Code == jlm.TOpFrames:
			frames = n/int(op.Stride) + 1
			rows = frames
		case op.Code == jlm.TOpFeatureNorm:
			rows -= int(op.Offset)
		case (op.Code == jlm.TOpUnfold || op.Code == jlm.TOpConv) && op.Stride > 1:
			rows = max(rows+2*int(op.Offset)-int(op.Span), 0)/int(op.Stride) + 1
		}
	}
	return frames, rows
}

// ClipLength is the samples the audio tower reads of a clip that makes
// outputs at its rate: cut to the longest clip the reference reads and padded
// with silence to half a second, as audio.waveform makes them.
func (m *Model) ClipLength(outputs int) int {
	a := m.Container.Vision().Audio
	return max(min(outputs, int(a.MaxSamples)), int(a.SampleRate)/2)
}

// ClipSamples is the samples a clip makes at the audio tower's rate as the
// tower reads them: ffmpeg's resample of its mono mix (composer.SWR,
// composer.MonoMix), cut and padded (ClipLength).
func (m *Model) ClipSamples(c Clip) (int, error) {
	if m.Audio == nil {
		return 0, fmt.Errorf("model: the model has no audio tower")
	}
	if composer.SoundBytes(c.Format) == 0 || c.Channels < 1 || c.Rate < 1 {
		return 0, fmt.Errorf("model: a clip of sample format %d, %d channel(s) at %d Hz", c.Format, c.Channels, c.Rate)
	}
	if c.Frames() == 0 {
		return 0, fmt.Errorf("model: an empty clip")
	}
	if _, err := composer.MonoMix(c.Channels, c.Mask); err != nil {
		return 0, err
	}
	s, err := composer.NewSWR(c.Rate, int(m.Container.Vision().Audio.SampleRate))
	if err != nil {
		return 0, err
	}
	if s.Taps*c.Channels > composer.SoundTapsMax {
		return 0, fmt.Errorf("model: a clip of %d channel(s) at %d Hz resamples through %d taps of each, past the sound op's %d",
			c.Channels, c.Rate, s.Taps, composer.SoundTapsMax)
	}
	_, n := s.Clip(c.Frames())
	return m.ClipLength(n), nil
}

// ClipSpan is a voice clip as a span of the prompt: the clip as its
// container stores it, the rows the audio tower writes for it (ClipSamples)
// and a key over what decides them.
func (m *Model) ClipSpan(clip Clip) (Span, error) {
	n, err := m.ClipSamples(clip)
	if err != nil {
		return Span{}, err
	}
	_, rows := m.ClipRows(n)
	h := sha256.New()
	h.Write([]byte("clip"))
	var w [8]byte
	for _, v := range [...]uint64{uint64(clip.Format), uint64(clip.Channels), uint64(clip.Rate), clip.Mask} {
		binary.LittleEndian.PutUint64(w[:], v)
		h.Write(w[:])
	}
	h.Write(clip.PCM)
	var k RowsKey
	h.Sum(k[:0])
	return Span{Clip: &clip, ClipRows: rows, Key: &k}, nil
}

// BuildAudioGraphs lowers the container's stated audio tower (jlm.Audio):
// its entry over the longest clip's frames, a chained tower whose blocks
// follow the vision tower's in the tower's block space, each block over the
// rows the subsampling leaves, and the head -- each graph verified against
// the tensors it binds.
func (m *Model) BuildAudioGraphs() (*TowerGraphs, error) {
	v := m.Container.Vision()
	a := v.Audio
	t := a.Tower
	dense := m.TowerFacts(jlm.DenseBlock)
	frames := int(a.MaxSamples/a.Hop) + 1
	// An entry a sample format: the clip's samples at the tower's rate, made
	// by the sound op from its PCM as it lies, are what the stated entry's
	// frames read.
	tg := &TowerGraphs{Classes: []towerClass{{Entry: map[uint8]*graph.Graph{}, Rows: frames}}, Rows: frames}
	var first *graph.Graph
	for f := graph.SoundU8; f < graph.SoundFormats; f++ {
		b := graph.NewBlock(-1)
		snd := b.Emit(graph.OpSound, nil, []graph.Result{{Name: "sound", Width: int(a.MaxSamples), Rows: 1}}, nil,
			graph.Attrs{Sound: f, Taps: composer.SoundTapsMax}, nil, nil)[0]
		at := len(b.G.Ops)
		out, err := LowerProgram(b, t, &t.Entry, nil, frames, dense)
		if err != nil {
			return nil, fmt.Errorf("model: the audio tower's entry: %w", err)
		}
		for i := at; i < len(b.G.Ops); i++ {
			if b.G.Ops[i].Kind == graph.OpFrames {
				b.G.Ops[i].In = []graph.ValueID{snd}
			}
		}
		b.Out(out)
		if err := graph.Verify(b.G, dense); err != nil {
			return nil, fmt.Errorf("model: the audio tower's entry: %w", err)
		}
		tg.Classes[0].Entry[f] = b.G
		if first == nil {
			first = b.G
		}
	}
	if err := m.chainGraphs(tg, t, int(a.NLayer), int(v.NLayer), first, dense); err != nil {
		return nil, fmt.Errorf("model: the audio %w", err)
	}
	return tg, nil
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
	if len(t.Chain) > 0 {
		// A chain's entry reads the picture's every patch of P: its grid is
		// the picture's, whatever the container's patch.
		tg.Side = int(v.ImageSize) / P
	}
	for _, op := range ops {
		if op.Code == jlm.TOpTailRow {
			tg.Tail = 1
		}
	}
	sides := []int{0}
	if fit.Own() {
		sides = fit.classSides()
	}
	for _, sd := range sides {
		grid := tg.Side * tg.Side
		if sd > 0 {
			grid = min((sd/P)*(sd/P), fit.maxPatches(P))
		}
		cl := towerClass{Entry: map[uint8]*graph.Graph{}, Rows: grid + tg.Lead + tg.Tail}
		for px := graph.PixRGBA8; px < graph.PixFormats; px++ {
			g, band, err := towerEntry(t, px, grid, tg.Lead+tg.Tail, sd, fit, dense)
			if err != nil {
				return nil, fmt.Errorf("model: the tower's entry: %w", err)
			}
			cl.Pic, cl.Entry[px] = g.Ops[0].Attr, g
			cl.Pic.PicPass = graph.PicWhole
			if band != nil {
				if cl.Band == nil {
					cl.Band, cl.BandRuns = map[uint8]*graph.Graph{}, map[uint8][]uint8{}
				}
				cl.Band[BandKey(0, px)], cl.BandRuns[px] = band, []uint8{BandKey(0, px)}
				if err := columnBands(cl.Band, BandKey(0, px), dense); err != nil {
					return nil, fmt.Errorf("model: the tower's entry columns: %w", err)
				}
			}
		}
		tg.Classes = append(tg.Classes, cl)
		tg.Rows = max(tg.Rows, cl.Rows)
	}
	if t.Video != nil {
		if len(t.Chain) > 0 || !t.Video.AsPictures() && !fit.Own() {
			return nil, fmt.Errorf("model: a video read by a tower whose pictures are not each of a size of its own")
		}
		tg.Video = t.Video
	}
	if t.Video != nil && !t.Video.AsPictures() {
		// A whole frame is a picture: its classes are the picture's.
		for i, sd := range sides {
			cl := towerClass{Entry: map[uint8]*graph.Graph{}, Rows: tg.Classes[i].Rows}
			for px0 := graph.PixRGBA8; px0 < graph.PixFormats; px0++ {
				for px1 := graph.PixRGBA8; px1 < graph.PixFormats; px1++ {
					g, err := videoEntry(t, px0, px1, cl.Rows-tg.Lead-tg.Tail, tg.Lead+tg.Tail, sd, dense)
					if err != nil {
						return nil, fmt.Errorf("model: the tower's video entry: %w", err)
					}
					cl.Pic, cl.Entry[PairKey(px0, px1)] = g.Ops[0].Attr, g
					cl.Pic.PicPass = graph.PicWhole
				}
			}
			if cl.Pic.Chain == graph.ChainPIL {
				// Each frame's bands into its own accumulator, the first
				// frame's then the second's.
				cl.Band, cl.BandRuns = map[uint8]*graph.Graph{}, map[uint8][]uint8{}
				for px := graph.PixRGBA8; px < graph.PixFormats; px++ {
					for frame := range 2 {
						a := cl.Pic
						a.Pixels, a.PicFrame, a.PicPass = px, frame == 1, graph.PicBand
						b := graph.NewBlock(-1)
						b.Out(b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "frame.band", Width: 3}}, nil, a, nil, nil)[0])
						if err := graph.Verify(b.G, dense); err != nil {
							return nil, fmt.Errorf("model: the tower's video band: %w", err)
						}
						cl.Band[BandKey(frame, px)] = b.G
						if err := columnBands(cl.Band, BandKey(frame, px), dense); err != nil {
							return nil, fmt.Errorf("model: the tower's video columns: %w", err)
						}
					}
				}
				for px0 := graph.PixRGBA8; px0 < graph.PixFormats; px0++ {
					for px1 := graph.PixRGBA8; px1 < graph.PixFormats; px1++ {
						cl.BandRuns[PairKey(px0, px1)] = []uint8{BandKey(0, px0), BandKey(1, px1)}
					}
				}
			}
			tg.Videos = append(tg.Videos, cl)
		}
	}
	if r := t.Block.Rope; len(r.Runs) > 0 {
		tg.Table, tg.Planes = towerRotary(t)
	}
	if len(t.Chain) > 0 {
		if err := m.chainGraphs(tg, t, int(v.NLayer), 0, tg.Classes[0].Entry[graph.PixRGBA8], dense); err != nil {
			return nil, err
		}
	} else {
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
	}
	for _, op := range t.Head.Ops {
		if op.Code == jlm.TOpMarkRow {
			tg.Marks = int(op.Span + op.Offset)
		}
		if op.Code == jlm.TOpRowsAttend {
			tg.Fixed = -1
		}
	}
	if tg.Fixed < 0 {
		tg.Fixed = tg.Out - tg.Marks
	}
	if tg.Line, tg.Frame, tg.Trim = programFrame(&t.Head); tg.Frame > 0 {
		for _, op := range t.Head.Ops {
			if op.Code == jlm.TOpWrapRows {
				tg.Ends = 1
			}
		}
	} else {
		for _, op := range t.Head.Ops {
			if op.Code == jlm.TOpWrapRows {
				tg.Ends, tg.Frame = 1, int(op.Span)
			}
		}
	}
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
		if w := g.Values[g.Out[0]].Width; n != tg.Out || w != D {
			return nil, fmt.Errorf("model: the tower's tap %d writes %d rows of %d, the head %d of %d", j, n, w, tg.Out, D)
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
	// Slice, SlicePatch and SliceMax are llava-uhd's cut (jlm.PreSliceUHD):
	// an overview of about Slice^2 pixels, each side whole SlicePatch-pixel
	// patches, and at most SliceMax slices; 0 where there is none.
	Slice, SlicePatch, SliceMax int
	// Canvas says the tiles are a best-fitting canvas's (jlm.PreBestFit):
	// the picture resized onto it without distortion, at its top left.
	Canvas bool
	// factor, minPix and maxPix are a resize to the picture's own size
	// (jlm.PreSmartResize); factor 0 where there is none.
	Factor, MinPix, MaxPix int
	// lift scales a picture a side shorter than factor up first
	// (jlm.PreSmartResize's Ints[5]); naflex sizes it by its patch count
	// instead, MinPix and MaxPix whole patches' pixels (jlm.PreNaFlex).
	lift, naflex bool
	// Limit, Pad and LimitPatch are a resize past a patch count and a pad
	// (jlm.PreLimitPad): a picture of more than Limit whole LimitPatch-pixel
	// patches scaled down to about that many, then padded to multiples of
	// Pad pixels; Limit 0 where there is none.
	Limit, Pad, LimitPatch int
	// Box and BoxMin are a resize by the longer side into a Box-pixel square
	// (jlm.PreLongBox): the shorter side at least BoxMin; Box 0 where there
	// is none.
	Box, BoxMin int
	// Cap and CapUnit are a resize that keeps the longer side at most Cap
	// pixels, each side then rounded up to a whole CapUnit (jlm.PreCapUnit);
	// Cap 0 where there is none.
	Cap, CapUnit int
	// SmartTile is the tile side of a picture tiled past its size bound or
	// sized its own way under it (jlm.PreTileSmart), 0 where there is none:
	// SmartMin to SmartMax tiles, its own size whole SmartUnits between
	// MinPix and MaxPix, cut where its sides, rounded so (at least a
	// SmartPatch each), hold more than Tolerance times its piece's bound.
	SmartTile, SmartMin, SmartMax, SmartPatch, SmartUnit, SmartMaxPix int
	Tolerance                                                         float64
	// Budget, BudgetPatch and BudgetUnit are a resize to the largest size
	// of the picture's aspect within Budget patches of BudgetPatch pixels,
	// each side whole BudgetUnits (jlm.PreAspectBudget); Budget 0 where
	// there is none.
	Budget, BudgetPatch, BudgetUnit int
}

// budgetSize is a w x h picture's size within the patch budget (Gemma 4's
// get_aspect_ratio_preserving_size): both sides scaled by
// sqrt(budget area / area) and floored to whole units; a side that floors
// to nothing one unit and the other the aspect's whole multiple of it, at
// most the budget's units a row -- in float64, as Python computes them.
func (f PictureFit) budgetSize(w, h int) (int, int) {
	p, u := f.BudgetPatch, f.BudgetUnit
	s := math.Sqrt(float64(f.Budget*p*p) / (float64(h) * float64(w)))
	th := int(math.Floor(s*float64(h)/float64(u))) * u
	tw := int(math.Floor(s*float64(w)/float64(u))) * u
	g := u / p
	most := (f.Budget / (g * g)) * u
	switch {
	case th == 0 && tw == 0:
		th, tw = u, u
	case th == 0:
		th, tw = u, min(int(math.Floor(float64(w)/float64(h)))*u, most)
	case tw == 0:
		tw, th = u, min(int(math.Floor(float64(h)/float64(w)))*u, most)
	}
	return tw, th
}

// capSize is a w x h picture's size under the cap (PixtralImageProcessor's
// get_resize_output_image_size): both sides divided by the longer over Cap
// and floored where it passes Cap, then each rounded up to a whole unit.
func (f PictureFit) capSize(w, h int) (int, int) {
	if r := math.Max(float64(w), float64(h)) / float64(f.Cap); r > 1 {
		w, h = int(math.Floor(float64(w)/r)), int(math.Floor(float64(h)/r))
	}
	u := f.CapUnit
	return ((max(w, 1)-1)/u + 1) * u, ((max(h, 1)-1)/u + 1) * u
}

// boxSize is a w x h picture's size inside its box (Janus's resize): the
// longer side the box's, the shorter rounded half to even to its share of
// it and at least BoxMin -- in float64, as Python computes them.
func (f PictureFit) boxSize(w, h int) (int, int) {
	d := float64(f.Box) / float64(max(w, h))
	return max(int(math.RoundToEven(float64(w)*d)), f.BoxMin), max(int(math.RoundToEven(float64(h)*d)), f.BoxMin)
}

// Own reports that a picture is encoded at a size of its own.
func (f PictureFit) Own() bool {
	return f.Factor > 0 || f.Limit > 0 || f.Cap > 0 || f.Budget > 0 || f.SmartTile > 0 || f.Slice > 0
}

// smartLayout is how a w x h picture is cut under jlm.PreTileSmart: cols x
// rows tiles where tiled, and its own size (the thumbnail, or the one piece
// where not).
func (f PictureFit) smartLayout(w, h int) (cols, rows int, tiled bool, ow, oh int) {
	ow, oh = PictureFit{Factor: f.SmartUnit, MinPix: f.MinPix, MaxPix: f.SmartMaxPix}.smartSize(w, h)
	u := f.SmartUnit
	round := func(v int) int { return max(f.SmartPatch, int(math.RoundToEven(float64(v)/float64(u)))*u) }
	if float64(round(h)*round(w)) <= float64(f.SmartMaxPix)*f.Tolerance {
		return 1, 1, false, ow, oh
	}
	cols, rows = TileGrid(w, h, f.SmartTile, f.SmartMin, f.SmartMax)
	return cols, rows, true, ow, oh
}

// unit is the multiple a picture of its own size's encoded sides are.
func (f PictureFit) unit() int {
	switch {
	case f.Limit > 0:
		return f.Pad
	case f.Cap > 0:
		return f.CapUnit
	case f.Budget > 0:
		return f.BudgetUnit
	case f.SmartTile > 0:
		return f.SmartUnit
	case f.Slice > 0:
		return f.SlicePatch
	}
	return f.Factor
}

// limitSize is a w x h picture's size under a patch limit (Kimi-VL's
// rescale): kept, unless its whole patches pass Limit, where both sides are
// scaled by sqrt(Limit / patches) and truncated -- in float64, as Python
// computes them.
func (f PictureFit) limitSize(w, h int) (int, int) {
	p := f.LimitPatch
	if n := (w / p) * (h / p); n > f.Limit {
		s := math.Sqrt(float64(f.Limit) / float64(n))
		return int(float64(w) * s), int(float64(h) * s)
	}
	return w, h
}

// maxPatches is the most patches a picture of its own size encodes in, of
// p pixels: under an area bound, the area's; under a patch limit, the most a
// picture pads to -- a side of a whole patches pads to at most a over the
// pad's patches plus one of them, and its other side has at most Limit/a
// (any number up to a picture's longest side where a is zero).
func (f PictureFit) maxPatches(p int) int {
	if f.Limit == 0 {
		return f.MaxPix / (p * p)
	}
	u := f.Pad / p
	pad := func(a int) int { return (a/u + 1) * u }
	most := composer.PictureSide / p
	best := 0
	for a := 0; a <= most; a++ {
		b := most
		if a > 0 {
			b = min(most, f.Limit/a)
		}
		best = max(best, pad(a)*pad(b))
	}
	return best
}

// smartSize is a w x h picture's own size (Qwen2-VL's smart_resize): each
// side rounded half to even to a multiple of factor, at least one; then,
// past maxPix pixels, both scaled down by sqrt(area/maxPix) and floored to
// one, or short of minPix, scaled up by sqrt(minPix/area) and ceiled -- in
// float64, as Python computes them.
func (f PictureFit) smartSize(w, h int) (ow, oh int) {
	fa := f.Factor
	if f.naflex {
		return naflexSize(w, h, fa, f.MinPix/(fa*fa), f.MaxPix/(fa*fa))
	}
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

// classSides is the sides of the squares whose areas a picture of its own
// size is encoded in (graph.Attrs.PicArea), smallest first: 16 units,
// doubling, and last composer.PictureSide rounded up a unit, or the side of
// the area the resize reaches where that is more.
func (f PictureFit) classSides() []int {
	u := f.unit()
	most := (composer.PictureSide/u + 1) * u
	if r := int(math.Ceil(math.Sqrt(float64(f.MaxPix)))); r > most {
		most = (r + u - 1) / u * u
	}
	var out []int
	for s := 16 * u; s < most; s *= 2 {
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
		case jlm.PreTileSmart:
			fit.SmartTile, fit.SmartMin, fit.SmartMax = int(s.Ints[0]), int(s.Ints[1]), int(s.Ints[2])
			fit.SmartPatch, fit.SmartUnit, fit.MinPix, fit.SmartMaxPix = int(s.Ints[3]), int(s.Ints[4]), int(s.Ints[5]), int(s.Ints[6])
			fit.Tolerance = float64(s.Floats[0])
			// The area the reduction reaches: a piece's bound or a tile's.
			fit.MaxPix = max(fit.SmartMaxPix, fit.SmartTile*fit.SmartTile)
			sized, pil = true, jlm.Filter(s.Ints[7]) != jlm.FilterTwoTap
		case jlm.PreSliceUHD:
			fit.Slice, fit.SlicePatch, fit.SliceMax = int(s.Ints[0]), int(s.Ints[1]), int(s.Ints[2])
			// The area the reduction reaches: a piece's, each side rounded up
			// a patch.
			fit.MaxPix = (fit.Slice + fit.SlicePatch) * (fit.Slice + fit.SlicePatch)
			sized, pil = true, jlm.Filter(s.Ints[3]) != jlm.FilterTwoTap
		case jlm.PreAspectBudget:
			fit.Budget, fit.BudgetPatch, fit.BudgetUnit = int(s.Ints[0]), int(s.Ints[1]), int(s.Ints[2])
			// The area the reduction reaches: the budget's.
			fit.MaxPix = fit.Budget * fit.BudgetPatch * fit.BudgetPatch
			sized, pil = true, jlm.Filter(s.Ints[3]) != jlm.FilterTwoTap
		case jlm.PreCapUnit:
			fit.Cap, fit.CapUnit = int(s.Ints[0]), int(s.Ints[1])
			// The area the reduction reaches: the cap's, rounded up a unit.
			side := (fit.Cap + fit.CapUnit - 1) / fit.CapUnit * fit.CapUnit
			fit.MaxPix = side * side
			sized, pil = true, jlm.Filter(s.Ints[2]) != jlm.FilterTwoTap
		case jlm.PreNaFlex:
			p := int(s.Ints[0])
			fit.Factor, fit.MinPix, fit.MaxPix, fit.naflex = p, int(s.Ints[1])*p*p, int(s.Ints[2])*p*p, true
			sized, pil = true, jlm.Filter(s.Ints[3]) != jlm.FilterTwoTap
		case jlm.PreLongBox:
			fit.Box, fit.BoxMin = int(s.Ints[0]), int(s.Ints[1])
			// The area the reduction reaches: the box's.
			fit.MaxPix = fit.Box * fit.Box
			sized, pil = true, jlm.Filter(s.Ints[5]) != jlm.FilterTwoTap
		case jlm.PreLimitPad:
			fit.LimitPatch, fit.Limit, fit.Pad = int(s.Ints[0]), int(s.Ints[1]), int(s.Ints[2])
			// The area the reduction reaches: the limit's patches.
			fit.MaxPix = fit.Limit * fit.LimitPatch * fit.LimitPatch
			sized, pil = true, jlm.Filter(s.Ints[3]) != jlm.FilterTwoTap
		case jlm.PreCenterCrop:
			cropped = true
		case jlm.PreTile:
			fit.MinTiles, fit.MaxTiles, fit.Thumb = int(s.Ints[0]), int(s.Ints[1]), s.Ints[2] == 1
		case jlm.PreBestFit:
			fit.MinTiles, fit.MaxTiles, fit.Thumb, fit.Canvas = 1, int(s.Ints[0]), true, true
		}
	}
	ways := 0
	for _, w := range []bool{fit.Short > 0, fit.MaxTiles > 0, fit.Factor > 0, fit.Limit > 0, fit.Box > 0, fit.Cap > 0, fit.Budget > 0, fit.SmartTile > 0, fit.Slice > 0} {
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
// the grid's patch rows and extra rows beside them (a class row's, before
// or after); for a picture of a size of its own, at most side pixels a side.
func towerEntry(t *jlm.Tower, px uint8, grid, extra, side int, fit PictureFit, f graph.Facts) (*graph.Graph, *graph.Graph, error) {
	a := graph.Attrs{Pixels: px, Std: [3]float64{1, 1, 1}}
	for _, s := range t.Pre {
		switch s.Kind {
		case jlm.PreResize:
			a.PicW, a.PicH = int(s.Ints[0]), int(s.Ints[1])
			if err := resizeChain(&a, jlm.Filter(s.Ints[2])); err != nil {
				return nil, nil, err
			}
			a.PicTorch = jlm.Precision(s.Ints[3]) == jlm.PrecTorch
		case jlm.PreResizeShort:
			a.PicW, a.PicH = int(s.Ints[0]), int(s.Ints[0])
			if err := resizeChain(&a, jlm.Filter(s.Ints[1])); err != nil {
				return nil, nil, err
			}
		case jlm.PreSmartResize:
			a.PicW, a.PicH, a.PicLive = side, side, true
			if err := resizeChain(&a, jlm.Filter(s.Ints[3])); err != nil {
				return nil, nil, err
			}
		case jlm.PreSliceUHD:
			a.PicW, a.PicH, a.PicLive = side, side, true
			if err := resizeChain(&a, jlm.Filter(s.Ints[3])); err != nil {
				return nil, nil, err
			}
			a.PicTorch = jlm.Precision(s.Ints[4]) == jlm.PrecTorch
		case jlm.PreTileSmart:
			a.PicW, a.PicH, a.PicLive = side, side, true
			if err := resizeChain(&a, jlm.Filter(s.Ints[7])); err != nil {
				return nil, nil, err
			}
			a.PicTorch = jlm.Precision(s.Ints[8]) == jlm.PrecTorch
		case jlm.PreAspectBudget:
			a.PicW, a.PicH, a.PicLive = side, side, true
			if err := resizeChain(&a, jlm.Filter(s.Ints[3])); err != nil {
				return nil, nil, err
			}
		case jlm.PreCapUnit:
			a.PicW, a.PicH, a.PicLive = side, side, true
			if err := resizeChain(&a, jlm.Filter(s.Ints[2])); err != nil {
				return nil, nil, err
			}
		case jlm.PreNaFlex:
			a.PicW, a.PicH, a.PicLive = side, side, true
			if err := resizeChain(&a, jlm.Filter(s.Ints[3])); err != nil {
				return nil, nil, err
			}
		case jlm.PreLongBox:
			a.PicW, a.PicH, a.PicLive, a.PicBox = int(s.Ints[0]), int(s.Ints[0]), true, true
			a.PicFill = [3]uint8{uint8(s.Ints[2]), uint8(s.Ints[3]), uint8(s.Ints[4])}
			if err := resizeChain(&a, jlm.Filter(s.Ints[5])); err != nil {
				return nil, nil, err
			}
		case jlm.PreLimitPad:
			a.PicW, a.PicH, a.PicLive, a.PicPad = side, side, true, int(s.Ints[2])
			if err := resizeChain(&a, jlm.Filter(s.Ints[3])); err != nil {
				return nil, nil, err
			}
		case jlm.PreCenterCrop:
			a.PicW, a.PicH = int(s.Ints[0]), int(s.Ints[1])
		case jlm.PreNormalize, jlm.PreNormalizeBF16:
			for c := range 3 {
				a.Mean[c], a.Std[c] = float64(s.Floats[c]), float64(s.Floats[3+c])
			}
			a.PicBF16 = s.Kind == jlm.PreNormalizeBF16
		}
	}
	if a.PicLive && !a.PicBox && side > 0 {
		// A picture of a size of its own holds its class's area, its sides
		// any a picture reaches.
		m := composer.PictureMaxSide
		if a.PicPad > 0 {
			m -= m % a.PicPad
		}
		a.PicW, a.PicH, a.PicArea = m, m, side*side
	}
	switch {
	case a.Chain == graph.ChainPIL && a.PicLive:
		// A picture's own size is its source's but for the rounding to the
		// factor and the area's bound: a side under twice smaller than the
		// source's times sqrt(area / maxPix), the source at most PicSide a
		// side (pictureFit.smartSize).
		root := int(math.Sqrt(float64(fit.MaxPix)))
		a.Taps = composer.LiveTaps(a.Filter, root)
	case a.Chain == graph.ChainPIL:
		// A crop's taps are a window of the resize's, whose scale is at most
		// the crop's own.
		a.Taps = max(composer.PILTapCount(a.Filter, composer.PictureMaxSide, a.PicW), composer.PILTapCount(a.Filter, composer.PictureMaxSide, a.PicH))
	}
	// A Pillow resize runs in bands of PicSide source rows (its band graph)
	// and the entry finishes their sums.
	var band *graph.Graph
	if a.Chain == graph.ChainPIL {
		ba := a
		ba.PicPass = graph.PicBand
		bb := graph.NewBlock(-1)
		bb.Out(bb.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels.band", Width: 3}}, nil, ba, nil, nil)[0])
		if err := graph.Verify(bb.G, f); err != nil {
			return nil, nil, err
		}
		band, a.PicPass = bb.G, graph.PicFinish
	}
	p := &t.Entry
	b := graph.NewBlock(-1)
	P := int(p.Ops[0].Span)
	pix := b.Emit(graph.OpPicture, nil, []graph.Result{{Name: "pixels", Width: 3 * P * P}}, nil, a, nil, nil)[0]
	if a.PicLive && grid*P*P > a.Area() || !a.PicLive && a.PicW*a.PicH != grid*P*P {
		return nil, nil, fmt.Errorf("a %dx%d picture is not %d patches of %d", a.PicW, a.PicH, grid, P)
	}
	out, err := LowerProgram(b, t, p, []graph.ValueID{pix}, grid+extra, f)
	if err != nil {
		return nil, nil, err
	}
	b.Out(out)
	return b.G, band, graph.Verify(b.G, f)
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
	g.Origin = int(t.Block.Rope.Origin)
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
	if f.Canvas {
		rows, cols = bestFit(w, h, s, f.MaxTiles)
	} else {
		cols, rows = TileGrid(w, h, s, f.MinTiles, f.MaxTiles)
	}
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
			if f.Canvas {
				ow, oh = canvasSize(w, h, pw, cols, rows)
				return ow, oh, (t % cols) * pw, (t / cols) * ph
			}
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
	// Mark and Trail are the first rows of the square's marker runs in the
	// head's table plus one, before its rows and after them, 0 for none, and
	// MarkN and TrailN their lengths (TOpMarkRow, composer.RequestMark and
	// RequestTrail).
	Mark, Trail, MarkN, TrailN int
}

// square is square i of a w x h picture and how many the picture has: one
// at its own size in the smallest class that holds it, or a fixed square,
// a crop or a tile at the entry's size.
func (tg *TowerGraphs) Square(w, h, i int) (Square, int) {
	if f := tg.Fit; f.Slice > 0 {
		cols, rows, cut := f.sliceGrid(w, h)
		n := 1
		if cut {
			n += cols * rows
		}
		if i == 0 {
			// The overview, at its own size, upscaled where nothing is cut.
			ow, oh := f.sliceSize(float64(w), float64(h), !cut)
			sq := Square{Class: tg.classOf(ow, oh), Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: ow / tg.Patch, Gh: oh / tg.Patch}
			if tg.Marks > 0 {
				sq.Mark, sq.MarkN, sq.Trail, sq.TrailN = 1, 1, 2, 1
			}
			return sq, n
		}
		// Slice i-1 of the picture resized to whole best-sized slices; a
		// slice row's first after the first takes the newline before it.
		rw, rh := f.refineSize(w, h, cols, rows)
		sw, sh, k := rw/cols, rh/rows, i-1
		sq := Square{Class: tg.classOf(sw, sh), Ow: rw, Oh: rh, X0: k % cols * sw, Y0: k / cols * sh, Pw: sw, Ph: sh,
			Gw: sw / tg.Patch, Gh: sh / tg.Patch}
		if tg.Marks > 0 {
			sq.Mark, sq.MarkN, sq.Trail, sq.TrailN = 3, 1, 4, 1
			if k > 0 && k%cols == 0 {
				sq.Mark, sq.MarkN = 5, 2
			}
		}
		return sq, n
	}
	if f := tg.Fit; f.SmartTile > 0 {
		cols, rows, tiled, ow, oh := f.smartLayout(w, h)
		n, span := 1, f.SmartMax
		if tiled {
			n = cols*rows + 1
		}
		class := tg.classOf
		if i < n-1 {
			// Tile i of the picture resized onto the grid, its marker the
			// tile's place.
			s := f.SmartTile
			return Square{Class: class(s, s), Ow: cols * s, Oh: rows * s, X0: i % cols * s, Y0: i / cols * s, Pw: s, Ph: s,
				Gw: s / tg.Patch, Gh: s / tg.Patch, Mark: i/cols*span + i%cols + 1, MarkN: 1}, n
		}
		sq := Square{Class: class(ow, oh), Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: ow / tg.Patch, Gh: oh / tg.Patch}
		if tiled {
			sq.Mark, sq.MarkN = span*span+1, 1
		}
		return sq, n
	}
	if tg.Fit.Budget > 0 {
		ow, oh := tg.Fit.budgetSize(w, h)
		c := 0
		for c < len(tg.Classes)-1 && !tg.Classes[c].Pic.Holds(ow, oh) {
			c++
		}
		return Square{Class: c, Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: ow / tg.Patch, Gh: oh / tg.Patch}, 1
	}
	if tg.Fit.Cap > 0 {
		ow, oh := tg.Fit.capSize(w, h)
		c := 0
		for c < len(tg.Classes)-1 && !tg.Classes[c].Pic.Holds(ow, oh) {
			c++
		}
		return Square{Class: c, Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: ow / tg.Patch, Gh: oh / tg.Patch}, 1
	}
	if tg.Fit.Box > 0 {
		// Resized inside the box, which the picture op lays it in.
		ow, oh := tg.Fit.boxSize(w, h)
		return Square{Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: tg.Side, Gh: tg.Side}, 1
	}
	if tg.Fit.Limit > 0 {
		// Resized to its own size and laid over its padded grid by the
		// picture op: the window the size itself, the grid the padded one.
		ow, oh := tg.Fit.limitSize(w, h)
		u := tg.Fit.Pad
		pw, ph := (ow+u-1)/u*u, (oh+u-1)/u*u
		c := 0
		for c < len(tg.Classes)-1 && !tg.Classes[c].Pic.Holds(pw, ph) {
			c++
		}
		return Square{Class: c, Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: pw / tg.Patch, Gh: ph / tg.Patch}, 1
	}
	if tg.Fit.Factor > 0 {
		ow, oh := tg.Fit.smartSize(w, h)
		c := 0
		for c < len(tg.Classes)-1 && !tg.Classes[c].Pic.Holds(ow, oh) {
			c++
		}
		return Square{Class: c, Ow: ow, Oh: oh, Pw: ow, Ph: oh, Gw: ow / tg.Patch, Gh: oh / tg.Patch}, 1
	}
	a := tg.Classes[0].Pic
	cols, _, n := tg.Fit.Tiles(w, h, a.PicW)
	ow, oh, x0, y0 := tg.Fit.Window(w, h, a.PicW, a.PicH, i)
	sq := Square{Ow: ow, Oh: oh, X0: x0, Y0: y0, Pw: a.PicW, Ph: a.PicH, Gw: tg.Side, Gh: tg.Side}
	if tg.Fit.Canvas && tg.Marks > 0 {
		// The canvas's markers (jlm.PreBestFit): before the last square the
		// third, after a tile its row's separator, after a row's last the
		// rows'.
		switch {
		case i == n-1:
			sq.Mark, sq.MarkN = 3, 1
		case i%cols < cols-1:
			sq.Trail, sq.TrailN = 1, 1
		default:
			sq.Trail, sq.TrailN = 2, 1
		}
	}
	return sq, n
}

// squareOut is the rows the head writes for square sq: its patches over
// the head's ratio of rows out to patch rows in.
func (tg *TowerGraphs) SquareOut(sq Square) int {
	if tg.Frame > 0 {
		m := tg.Frame
		return (sq.Gh/m)*(sq.Gw/m+tg.Line) - tg.Trim + 2*tg.Ends
	}
	grid := tg.Rows - tg.Lead - tg.Tail
	if tg.Fixed > 0 {
		n := tg.Fixed
		if sq.Mark > 0 {
			n += sq.MarkN
		}
		if sq.Trail > 0 {
			n += sq.TrailN
		}
		return n
	}
	if tg.Marks > 0 {
		// The head's rows less its marker rows over the patches, and the
		// markers the square names.
		n := sq.Gw * sq.Gh * (tg.Out - tg.Marks) / grid
		if sq.Mark > 0 {
			n += sq.MarkN
		}
		if sq.Trail > 0 {
			n += sq.TrailN
		}
		return n
	}
	return sq.Gw * sq.Gh * tg.Out / grid
}

// LastSquare is a w x h picture's last square: a tiled tower's thumbnail,
// the whole picture at a square's size, or its one square.
func (tg *TowerGraphs) LastSquare(w, h int) Square {
	sq, n := tg.Square(w, h, 0)
	if n > 1 {
		sq, _ = tg.Square(w, h, n-1)
	}
	return sq
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
	case jlm.TapsAntialias:
		f = graph.FilterAntialias
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
	vals, err := lowerValues(b, t, p, in, rows, f)
	if err != nil {
		return 0, err
	}
	return vals[p.Out], nil
}

// lowerValues is LowerProgram's every value: the inputs, then each op's.
func lowerValues(b *graph.Builder, t *jlm.Tower, p *jlm.Program, in []graph.ValueID, rows int, f graph.Facts) ([]graph.ValueID, error) {
	return lowerValuesWith(b, t, p, in, rows, f, nil)
}

// lowerValuesWith is lowerValues where blockOps lowers the ops only a block
// has (Model.blockOps).
func lowerValuesWith(b *graph.Builder, t *jlm.Tower, p *jlm.Program, in []graph.ValueID, rows int, f graph.Facts,
	blockOps func([]graph.ValueID, *jlm.TowerOp) (graph.ValueID, error)) ([]graph.ValueID, error) {
	merge, window := towerTiles(t)
	if len(in) != int(p.NIn) {
		return nil, fmt.Errorf("a program of %d inputs over %d", p.NIn, len(in))
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
	// An elementwise op's inputs are read at the width its result is: a
	// video's frames' difference is read by the second plane, padded as
	// the first is.
	for i := len(p.Ops) - 1; i >= 0; i-- {
		op := &p.Ops[i]
		if k := readAt[len(in)+i]; k > 0 && (op.Code == jlm.TOpSub || op.Code == jlm.TOpAdd) {
			for _, v := range op.In {
				readAt[int(v)] = max(readAt[int(v)], k)
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
				return nil, err
			}
			out = b.MatVec(ins[0], op.Bind[0], s.Rows, op.Bind[1:]...)
		case jlm.TOpPosTable:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			out = b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "pos", Width: s.K}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, graph.Attrs{Positional: true, Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpPosResample:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.Rows != int(op.Span*op.Span) {
				return nil, fmt.Errorf("resamples a table %d a side from %d rows", op.Span, s.Rows)
			}
			chain, filter, err := resampleOf(op.Chain, op.Filter)
			if err != nil {
				return nil, err
			}
			out = b.Emit(graph.OpPosResample, nil, []graph.Result{{Name: "pos", Width: s.K}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				graph.Attrs{Span: int(op.Span), Offset: int(op.Offset), Width: int(op.Width), Window: window, Chain: chain, Filter: filter}, nil, nil)[0]
		case jlm.TOpPosAxes:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.Rows != int(2*op.Span) {
				return nil, fmt.Errorf("reads two tables of %d rows from %d rows", op.Span, s.Rows)
			}
			out = b.Emit(graph.OpPosAxes, nil, []graph.Result{{Name: "pos", Width: s.K}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				graph.Attrs{Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpAdd:
			out = b.Add(ins[0], ins[1])
		case jlm.TOpSub:
			out = b.Emit(graph.OpMap, ins, []graph.Result{{Name: "diff", Width: width(ins[0])}}, nil,
				graph.Attrs{Fn: graph.MapSub}, nil, nil)[0]
		case jlm.TOpLayerNorm:
			out = b.LayerNorm(ins[0], op.Bind[0], float64(op.Eps), op.Bind[1:]...)
		case jlm.TOpRMSNorm:
			if len(op.Bind) == 0 {
				w, hd := width(ins[0]), width(ins[0])
				if op.Span > 0 {
					hd = int(op.Span)
				}
				out = b.Emit(graph.OpHeadNorm, ins, []graph.Result{{Name: name, Width: w}}, nil,
					graph.Attrs{Eps: float64(op.Eps), HeadDim: hd}, nil, nil)[0]
				break
			}
			if op.Span > 0 && int(op.Span) < width(ins[0]) {
				out = b.HeadNorm(ins[0], op.Bind[0], int(op.Span), float64(op.Eps))
				break
			}
			out = b.Norm(ins[0], op.Bind[0], float64(op.Eps))
		case jlm.TOpRope, jlm.TOpNGramRows, jlm.TOpTailConv:
			if blockOps == nil {
				return nil, fmt.Errorf("op %d is code %d, which only a block has", i, op.Code)
			}
			v, err := blockOps(ins, op)
			if err != nil {
				return nil, err
			}
			out = v
		case jlm.TOpScale, jlm.TOpShift:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			fn := graph.MapMul
			if op.Code == jlm.TOpShift {
				fn = graph.MapAdd
			}
			out = mapRow(b, name, graph.Attrs{Fn: fn}, []graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				width(ins[0]), ins[0])
		case jlm.TOpSum, jlm.TOpMax:
			fn := graph.MapAdd
			if op.Code == jlm.TOpMax {
				fn = graph.MapMax
			}
			w, a := 1, graph.Attrs{Fn: fn}
			if op.Span > 0 {
				w, a.Span = width(ins[0])/int(op.Span), int(op.Span)
			}
			out = b.Emit(graph.OpReduce, ins, []graph.Result{{Name: name, Width: w}}, nil, a, nil, nil)[0]
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
				Rows: rowsOf(ins[0]) / (S * S)}}, nil, graph.Attrs{Span: S, GridW: int(op.GridW)}, nil, nil)[0]
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
				return nil, fmt.Errorf("the pool's mean: %w", err)
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
		case jlm.TOpMarkRow:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.K != width(ins[0]) || s.Rows < 1 {
				return nil, fmt.Errorf("marks rows %d wide from a table of %d rows %d wide", width(ins[0]), s.Rows, s.K)
			}
			if op.Span < 1 || op.Offset < 0 {
				return nil, fmt.Errorf("marks rows with runs of %d before and %d after", op.Span, op.Offset)
			}
			out = b.Emit(graph.OpMarkRow, ins, []graph.Result{{Name: "marked", Width: width(ins[0]),
				Rows: rowsOf(ins[0]) + int(op.Span+op.Offset)}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				graph.Attrs{Width: int(op.Width), Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpPosBuckets:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.Rows != int(op.Span*op.Span) {
				return nil, fmt.Errorf("buckets onto a table %d a side from %d rows", op.Span, s.Rows)
			}
			out = b.Emit(graph.OpPosBuckets, nil, []graph.Result{{Name: "pos", Width: s.K}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				graph.Attrs{Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpTableRows:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			out = b.Emit(graph.OpTableRows, nil, []graph.Result{{Name: "table", Width: s.K, Rows: s.Rows}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, graph.Attrs{}, nil, nil)[0]
		case jlm.TOpRowsAttend:
			w, h, g := width(ins[0]), int(op.Span), int(op.Width)
			if g == 0 {
				g = h
			}
			if w%h != 0 || width(ins[1]) != g*(w/h) || width(ins[2]) != g*(w/h) || len(ins) == 4 && width(ins[3]) != w {
				return nil, fmt.Errorf("attends rows %d, %d and %d wide in %d heads over %d", w, width(ins[1]), width(ins[2]), h, g)
			}
			// The relative form reads its positions' table.
			var bind []graph.Binding
			if len(ins) == 4 {
				s, err := shape(op.Bind[0])
				if err != nil {
					return nil, err
				}
				bind = []graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}
			}
			out = b.Emit(graph.OpRowsAttend, ins, []graph.Result{{Name: "attended", Width: w, Rows: rowsOf(ins[0])}}, bind,
				graph.Attrs{NHead: h, NKVHead: g, HeadDim: w / h, Scale: 1 / math.Sqrt(float64(w/h))}, nil, nil)[0]
		case jlm.TOpFrames:
			out = b.Emit(graph.OpFrames, nil, []graph.Result{{Name: "frames", Width: int(op.Width), Rows: rows}}, nil,
				graph.Attrs{Width: int(op.Width), Stride: int(op.Stride), Const: float64(op.Eps)}, nil, nil)[0]
		case jlm.TOpFeatureNorm:
			out = b.Emit(graph.OpFeatureNorm, ins, []graph.Result{{Name: name, Width: width(ins[0])}}, nil,
				graph.Attrs{Offset: int(op.Offset), Eps: float64(op.Eps)}, nil, nil)[0]
		case jlm.TOpSplitRows:
			w := int(op.Width)
			if width(ins[0])%w != 0 {
				return nil, fmt.Errorf("splits rows of %d into rows of %d", width(ins[0]), w)
			}
			out = b.Emit(graph.OpSplitRows, ins, []graph.Result{{Name: name, Width: w, Rows: rowsOf(ins[0]) * (width(ins[0]) / w)}}, nil,
				graph.Attrs{Width: w}, nil, nil)[0]
		case jlm.TOpClipMask:
			out = b.Emit(graph.OpClipMask, ins, []graph.Result{{Name: name, Width: width(ins[0])}}, nil,
				graph.Attrs{Stride: int(op.Stride), Span: int(op.Span), GridW: int(op.GridW)}, nil, nil)[0]
		case jlm.TOpRowConv:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.K != int(op.Span) || s.Rows != width(ins[0]) {
				return nil, fmt.Errorf("convolves %d channels by %d taps from a %dx%d table", width(ins[0]), op.Span, s.Rows, s.K)
			}
			out = b.Emit(graph.OpRowConv, ins, []graph.Result{{Name: name, Width: width(ins[0])}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, graph.Attrs{Taps: int(op.Span)}, nil, nil)[0]
		case jlm.TOpUnfold, jlm.TOpConv:
			T, st, W := int(op.Span), int(op.Stride), int(op.GridW)
			R := rowsOf(ins[0])
			if R%W != 0 {
				return nil, fmt.Errorf("convolves %d rows as a grid %d wide", R, W)
			}
			res := graph.Result{Name: name, Width: width(ins[0])}
			if st > 1 {
				res.Rows = ((R/W + st - 1) / st) * ((W + st - 1) / st)
			}
			kind := graph.OpConv
			var bind []graph.Binding
			if op.Code == jlm.TOpUnfold {
				kind, res.Width = graph.OpUnfold, int(op.Width)
			} else {
				bind = []graph.Binding{{Role: op.Bind[0], K: width(ins[0]), Rows: T * T}}
			}
			out = b.Emit(kind, ins, []graph.Result{res}, bind, graph.Attrs{Taps: T, Stride: st, GridW: W, Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpUpsample:
			S := int(op.Span)
			out = b.Emit(graph.OpUpsample, ins, []graph.Result{{Name: name, Width: width(ins[0]), Rows: rowsOf(ins[0]) * S * S}}, nil,
				graph.Attrs{Span: S, GridW: int(op.GridW)}, nil, nil)[0]
		case jlm.TOpAppendRow:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.K != width(ins[0]) || int(op.Offset) >= max(s.Rows, 1) {
				return nil, fmt.Errorf("appends row %d of a %dx%d table to rows %d wide", op.Offset, s.Rows, s.K, width(ins[0]))
			}
			out = b.Emit(graph.OpAppendRow, ins, []graph.Result{{Name: name, Width: width(ins[0]), Rows: rowsOf(ins[0]) + 1}},
				[]graph.Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, graph.Attrs{Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpTailRow:
			out = b.Emit(graph.OpTailRow, ins, []graph.Result{{Name: "class", Width: width(ins[0])}},
				[]graph.Binding{{Role: op.Bind[0], K: width(ins[0])}}, graph.Attrs{}, nil, nil)[0]
		case jlm.TOpRowBreak:
			out = b.Emit(graph.OpRowBreak, ins, []graph.Result{{Name: "lines", Width: width(ins[0]), Rows: 2 * rowsOf(ins[0])}},
				[]graph.Binding{{Role: op.Bind[0], K: width(ins[0])}}, graph.Attrs{Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpWrapRows:
			line, _, _ := programFrame(p)
			out = b.Emit(graph.OpWrapRows, ins, []graph.Result{{Name: "framed", Width: width(ins[0]), Rows: rowsOf(ins[0]) + 2}},
				[]graph.Binding{{Role: op.Bind[0], K: width(ins[0])}, {Role: op.Bind[1], K: width(ins[0])}},
				graph.Attrs{Span: int(op.Span), Offset: int(t.Entry.Ops[0].Offset), Width: line}, nil, nil)[0]
		case jlm.TOpDropRows:
			out = b.Emit(graph.OpDropRows, ins, []graph.Result{{Name: "rows", Width: width(ins[0]),
				Rows: rowsOf(ins[0]) - int(op.Offset)}}, nil, graph.Attrs{Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpStated:
			w := int(op.Width)
			if w == 0 {
				w = width(ins[0])
			}
			v, err := b.Stated(name, op.Expr, ins, w, f)
			if err != nil {
				return nil, err
			}
			out = v
		default:
			return nil, fmt.Errorf("op %d is code %d, which this engine does not lower", i, op.Code)
		}
		vals = append(vals, out)
	}
	return vals, nil
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
	// clamp holds x, w wide, to the bounds at slot of the block's clipped
	// linears (RoleVClamp: four a matrix, input then output, min then max),
	// where it holds them.
	clamp := func(x graph.ValueID, w, slot int) graph.ValueID {
		if !has(jlm.RoleVClamp) {
			return x
		}
		return b.Emit(graph.OpClamp, []graph.ValueID{x}, []graph.Result{{Name: "clamp", Width: w}},
			[]graph.Binding{{Role: jlm.RoleVClamp, K: 28}}, graph.Attrs{Offset: slot}, nil, nil)[0]
	}
	// mv is the matrix r over x, in wide to out wide, both clamped by
	// matrix m's bounds.
	mv := func(x graph.ValueID, r jlm.Role, in, out, m int, bias ...jlm.Role) graph.ValueID {
		return clamp(b.MatVec(clamp(x, in, 4*m), r, out, bias...), out, 4*m+2)
	}
	x := b.In("x", d)
	n := norm(x, jlm.RoleVAttnNorm, jlm.RoleVAttnNormBias)
	q := mv(n, jlm.RoleVAttnQ, d, qdim, 0, stored(jlm.RoleVAttnQBias)...)
	k := mv(n, jlm.RoleVAttnK, d, kvDim, 1, stored(jlm.RoleVAttnKBias)...)
	vv := mv(n, jlm.RoleVAttnV, d, kvDim, 2, stored(jlm.RoleVAttnVBias)...)
	if has(jlm.RoleVAttnQNorm) {
		q = b.HeadNorm(q, jlm.RoleVAttnQNorm, hd, eps)
	}
	if has(jlm.RoleVAttnKNorm) {
		k = b.HeadNorm(k, jlm.RoleVAttnKNorm, hd, eps)
	}
	if bl.VNorm {
		vv = b.Emit(graph.OpHeadNorm, []graph.ValueID{vv}, []graph.Result{{Name: "v.norm", Width: kvDim}}, nil,
			graph.Attrs{Eps: eps, HeadDim: hd}, nil, nil)[0]
	}
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
	ao := mv(att, jlm.RoleVAttnOut, qdim, d, 3, stored(jlm.RoleVAttnOutBias)...)
	if has(jlm.RoleVPostAttnNorm) {
		ao = b.Norm(ao, jlm.RoleVPostAttnNorm, eps)
	}
	x2 := b.Add(x, ao)
	n2 := norm(x2, jlm.RoleVFFNNorm, jlm.RoleVFFNNormBias)
	ffn := f.Inventory[jlm.RoleVFC1].Rows
	var acts []graph.ValueID
	if has(jlm.RoleVFFNGate) {
		acts = append(acts, mv(n2, jlm.RoleVFFNGate, d, ffn, 4, stored(jlm.RoleVFFNGateBias)...))
	}
	acts = append(acts, mv(n2, jlm.RoleVFC1, d, ffn, 5, stored(jlm.RoleVFC1Bias)...))
	h, err := b.Stated("act", &bl.Act, acts, ffn, f)
	if err != nil {
		return nil, fmt.Errorf("the stated activation: %w", err)
	}
	fo := mv(h, jlm.RoleVFC2, ffn, d, 6, stored(jlm.RoleVFC2Bias)...)
	if has(jlm.RoleVPostFFNNorm) {
		fo = b.Norm(fo, jlm.RoleVPostFFNNorm, eps)
	}
	b.Out(b.Add(x2, fo))
	return b.G, graph.Verify(b.G, f)
}

// towerConfig is the tower's segment as a Config: its blocks' geometry,
// attending both ways, with its own rotary where it turns q and k and no
// vocabulary. Its block indices are the model's (base on), so a block is
// asked by the index it runs at.
func TowerConfig(base, rows int, v *jlm.Vision, t *jlm.Tower) *Config {
	b := &t.Block
	r := TowerRotaryOf(t)
	return &Config{NLayer: base + int(v.NLayer), NEmbd: int(v.NEmbd), NHead: int(b.NHead),
		NKVHead: int(b.NKVHead), HeadDim: int(b.HeadDim), NFFN: int(v.NFFN), NCtx: rows,
		RMSEps: float64(b.Eps), EmbdScale: 1, AttnFactor: 1, NonCausal: true, NoPosEnc: r.NRot == 0,
		NRot: r.NRot, RopeBase: r.Base, RopeNeox: b.Rope.Neox, LayerNorm: b.Norm == jlm.NormLayer}
}

// pictureLayout is how a decoded picture's samples sit in the step's picture
// table: their format, the picture's size and its rows' stride in bytes, and
// for a second and third plane (YCbCr's chroma, a palette) their stride, byte
// offsets and the chroma subsampling (composer.RequestPicSub).
type PictureLayout struct {
	Px                              uint8
	W, H, stride                    int
	cstride, p2, p3, sub, wide, ink int
}

// request writes the layout's words into the step's request.
func (l PictureLayout) Request(req []uint32) {
	req[composer.RequestPicW], req[composer.RequestPicH], req[composer.RequestPicStride] = uint32(l.W), uint32(l.H), uint32(l.stride)
	req[composer.RequestPicCStride], req[composer.RequestPicP2], req[composer.RequestPicP3] = uint32(l.cstride), uint32(l.p2), uint32(l.p3)
	req[composer.RequestPicSub], req[composer.RequestPicWide] = uint32(l.sub), uint32(l.wide)
	req[composer.RequestPicInk] = uint32(l.ink)
}

// RequestFrame writes the layout's words as a video pair's second frame's.
func (l PictureLayout) RequestFrame(req []uint32) {
	req[composer.RequestFrameStride], req[composer.RequestFrameCStride] = uint32(l.stride), uint32(l.cstride)
	req[composer.RequestFrameP2], req[composer.RequestFrameP3], req[composer.RequestFrameSub] = uint32(l.p2), uint32(l.p3), uint32(l.sub)
	req[composer.RequestFrameWide], req[composer.RequestFrameInk] = uint32(l.wide), uint32(l.ink)
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
	case *image.NRGBA64:
		l.Px, l.stride, l.wide, planes[0] = graph.PixRGBA8, p.Stride, 1, rows(p.Pix, p.Stride, 8)
	case *image.RGBA64:
		l.Px, l.stride, l.wide, planes[0] = graph.PixRGBA8, p.Stride, 1, rows(p.Pix, p.Stride, 8)
	case *image.Gray16:
		l.Px, l.stride, l.wide, planes[0] = graph.PixGray8, p.Stride, 1, rows(p.Pix, p.Stride, 2)
	case *image.CMYK:
		l.Px, l.stride, l.ink, planes[0] = graph.PixRGBA8, p.Stride, 1, rows(p.Pix, p.Stride, 4)
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
	end := len(p)
	if planes[2] != nil {
		end = l.p3 + len(planes[2])
	} else if planes[1] != nil {
		end = l.p2 + len(planes[1])
	}
	n := (end + 3) / 4
	if n > composer.PictureWords {
		return nil, l, fmt.Errorf("model: a picture of %d bytes is past the %d its samples' U32 offsets reach", end, 4*composer.PictureWords)
	}
	if planes[1] == nil && len(p)%4 == 0 && uintptr(unsafe.Pointer(&p[0]))%4 == 0 {
		return unsafe.Slice((*uint32)(unsafe.Pointer(&p[0])), len(p)/4), l, nil
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
	return ImageGrid{T: 1, H: sq.Gh / tg.GridDiv, W: sq.Gw / tg.GridDiv, Line: tg.Line, Ends: tg.Ends}
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
	var hdr [37]byte
	hdr[0] = l.Px
	for i, v := range [9]int{l.W, l.H, l.stride, l.cstride, l.p2, l.p3, l.sub, l.wide, l.ink} {
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

// programFrame is the rows a program's TOpRowBreak adds after each row of
// the merged grid -- 1 where it has one, 0 where not -- the merge it counts
// that grid over, and the breaks it leaves off the last row.
func programFrame(p *jlm.Program) (line, merge, trim int) {
	for _, op := range p.Ops {
		if op.Code == jlm.TOpRowBreak {
			return 1, int(op.Span), int(op.Offset)
		}
	}
	return 0, 0, 0
}

// naflexSize is the size SigLIP2's NaFlex processor resamples a w x h
// picture to: its own whole patches' count clamped to [lo, hi] patches, then
// transformers' get_image_size_for_max_num_patches -- a bisection for the
// largest scale whose sides, ceiled to whole patches, hold that many.
func naflexSize(w, h, patch, lo, hi int) (int, int) {
	n := max((w/patch)*(h/patch), 1)
	target := min(max(n, lo), hi)
	scaled := func(scale float64, size int) int {
		v := math.Ceil(float64(size)*scale/float64(patch)) * float64(patch)
		return int(math.Max(float64(patch), v))
	}
	const eps = 1e-5
	smin, smax := eps/10, 100.0
	for smax-smin >= eps {
		sc := (smin + smax) / 2
		th, tw := scaled(sc, h), scaled(sc, w)
		if float64(th)/float64(patch)*(float64(tw)/float64(patch)) <= float64(target) {
			smin = sc
		} else {
			smax = sc
		}
	}
	return scaled(smin, w), scaled(smin, h)
}

// canvases is every rows x columns canvas of at most most tiles in the
// order jlm.PreBestFit walks them: tile counts from most down, each count's
// factor pairs by rows ascending, grouped by aspect in the order each
// first appears.
func canvases(most int) [][2]int {
	var ratios []float64
	byRatio := map[float64][][2]int{}
	for n := most; n >= 1; n-- {
		for r := 1; r <= n; r++ {
			if n%r != 0 {
				continue
			}
			c := n / r
			q := float64(r) / float64(c)
			if _, ok := byRatio[q]; !ok {
				ratios = append(ratios, q)
			}
			byRatio[q] = append(byRatio[q], [2]int{r, c})
		}
	}
	var out [][2]int
	for _, q := range ratios {
		out = append(out, byRatio[q]...)
	}
	return out
}

// bestFit is a w x h picture's canvas of s-pixel tiles (jlm.PreBestFit):
// the one whose limiting scale is the smallest upscale, or failing any the
// largest downscale, the smallest area breaking a tie -- the scales
// float32, as the reference divides two integer tensors.
func bestFit(w, h, s, most int) (rows, cols int) {
	cs := canvases(most)
	scale := make([]float32, len(cs))
	for i, c := range cs {
		sw := float32(c[1]*s) / float32(w)
		sh := float32(c[0]*s) / float32(h)
		scale[i] = min(sh, sw)
	}
	sel, up := float32(0), false
	for _, v := range scale {
		if v >= 1 && (!up || v < sel) {
			sel, up = v, true
		}
	}
	if !up {
		for _, v := range scale {
			sel = max(sel, v)
		}
	}
	best := -1
	for i, v := range scale {
		if v == sel && (best < 0 || cs[i][0]*cs[i][1] < cs[best][0]*cs[best][1]) {
			best = i
		}
	}
	return cs[best][0], cs[best][1]
}

// canvasSize is the size a w x h picture is resized to on a cols x rows
// canvas of s-pixel tiles: the largest of its aspect within the canvas and
// within its own size raised to one tile, the shorter scale's side floored.
func canvasSize(w, h, s, cols, rows int) (ow, oh int) {
	th, tw := min(max(h, s), rows*s), min(max(w, s), cols*s)
	sw, sh := float64(tw)/float64(w), float64(th)/float64(h)
	if sw < sh {
		ow, oh = tw, min(int(math.Floor(float64(h)*sw)), th)
	} else {
		oh, ow = th, min(int(math.Floor(float64(w)*sh)), tw)
	}
	return max(ow, 1), max(oh, 1)
}

// classOf is the class a picture of side pixels is encoded in: the smallest
// whose picture holds it, else the largest.
func (tg *TowerGraphs) classOf(w, h int) int {
	c := 0
	for c < len(tg.Classes)-1 && !tg.Classes[c].Pic.Holds(w, h) {
		c++
	}
	return c
}

// sliceGrid is jlm.PreSliceUHD's grid of a w x h picture: cols x rows
// slices, and whether it is cut at all -- the counts one either side of the
// picture's area over Slice^2 rounded up, each count's factor pairs in
// order, the first whose aspect is closest to the picture's
// (get_sliced_grid, in float64 as Python computes it).
func (f PictureFit) sliceGrid(w, h int) (cols, rows int, cut bool) {
	logRatio := math.Log(float64(w) / float64(h))
	ratio := float64(w) * float64(h) / float64(f.Slice*f.Slice)
	multiple := min(int(math.Ceil(ratio)), f.SliceMax)
	if multiple <= 1 {
		return 1, 1, false
	}
	best, least := [2]int{1, 1}, math.Inf(1)
	for _, n := range []int{multiple - 1, multiple, multiple + 1} {
		if n == 1 || n > f.SliceMax {
			continue
		}
		for m := 1; m <= n; m++ {
			if n%m != 0 {
				continue
			}
			if e := math.Abs(logRatio - math.Log(float64(m)/float64(n/m))); e < least {
				best, least = [2]int{m, n / m}, e
			}
		}
	}
	return best[0], best[1], true
}

// sliceSize is a w x h piece's size under jlm.PreSliceUHD (find_best_resize):
// past Slice^2 pixels, or where up, Slice^2 at its aspect -- the height
// truncated, the width the truncated height times the aspect, truncated --
// then each side whole patches, rounded half to even, at least one.
func (f PictureFit) sliceSize(w, h float64, up bool) (int, int) {
	if w*h > float64(f.Slice*f.Slice) || up {
		r := w / h
		h = math.Trunc(float64(f.Slice) / math.Sqrt(r))
		w = math.Trunc(h * r)
	}
	return wholeUnits(w, f.SlicePatch), wholeUnits(h, f.SlicePatch)
}

// refineSize is the size a w x h picture is resized to before it is cut
// into cols x rows slices (get_refine_size): whole slices, each the best
// size of the picture's own slice, upscaled where it must be.
func (f PictureFit) refineSize(w, h, cols, rows int) (int, int) {
	rw, rh := wholeUnits(float64(w), cols), wholeUnits(float64(h), rows)
	sw, sh := f.sliceSize(float64(rw)/float64(cols), float64(rh)/float64(rows), true)
	return sw * cols, sh * rows
}

// wholeUnits is length rounded half to even to whole units, at least one
// (the processor's ensure_divide).
func wholeUnits(length float64, unit int) int {
	return max(int(math.RoundToEven(length/float64(unit)))*unit, unit)
}

// chainGraphs lowers a chained tower's blocks, each over the grid the block
// before it left (the entry's, for the first) and bound to its own tensors,
// and its head over the last block's rows and each HeadFrom block's: the
// head writes as many rows whatever the picture.
func (m *Model) chainGraphs(tg *TowerGraphs, t *jlm.Tower, n, base int, e *graph.Graph, dense graph.Facts) error {
	if len(t.Chain) != n {
		return fmt.Errorf("model: the tower chains %d blocks of its %d", len(t.Chain), n)
	}
	ev := e.Values[e.Out[0]]
	rows, width := ev.Rows, ev.Width
	if rows == 0 {
		rows = tg.Classes[0].Rows
	}
	outs := make([][2]int, len(t.Chain))
	for i := range t.Chain {
		cb := &t.Chain[i]
		if int(cb.Rows) != rows || int(cb.Width) != width {
			return fmt.Errorf("model: the tower's block %d reads %d rows of %d after %d rows of %d", i, cb.Rows, cb.Width, rows, width)
		}
		f := m.TowerFacts(int32(base + i))
		b := graph.NewBlock(m.TowerBase + base + i)
		x := b.In("x", width)
		b.G.Values[x].Rows = rows
		out, err := LowerProgram(b, t, &cb.Prog, []graph.ValueID{x}, rows, f)
		if err != nil {
			return fmt.Errorf("model: the tower's block %d: %w", i, err)
		}
		b.Out(out)
		if err := graph.Verify(b.G, f); err != nil {
			return fmt.Errorf("model: the tower's block %d: %w", i, err)
		}
		tg.Blocks, tg.Full = append(tg.Blocks, b.G), append(tg.Full, true)
		o := b.G.Values[out]
		rows, width = o.Rows, o.Width
		outs[i] = [2]int{rows, width}
	}
	b := graph.NewBlock(-1)
	ins := []graph.ValueID{b.In("x", width)}
	b.G.Values[ins[0]].Rows = rows
	for _, j := range t.HeadFrom {
		x := b.In("from", outs[j][1])
		b.G.Values[x].Rows = outs[j][0]
		ins = append(ins, x)
		tg.HeadFrom = append(tg.HeadFrom, int(j))
	}
	out, err := LowerProgram(b, t, &t.Head, ins, rows, dense)
	if err != nil {
		return fmt.Errorf("model: the tower's head: %w", err)
	}
	b.Out(out)
	if err := graph.Verify(b.G, dense); err != nil {
		return fmt.Errorf("model: the tower's head: %w", err)
	}
	hn := b.G.Values[out].Rows
	tg.Head, tg.Out, tg.Fixed, tg.Chain = b.G, hn, hn, true
	return nil
}

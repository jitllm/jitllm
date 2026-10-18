package model

import (
	"fmt"
	"image"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// Llama 4's vision: a CLIP-style tower over ImageSz tiles, and the adapter.
//
// What is the family's own, each read off the reference:
//
//	the tiles           Llama4ImageProcessor: the best-fitting canvas of up
//	                    to MaxTiles tiles (find_supported_resolutions,
//	                    get_best_fit), the picture resized onto it without
//	                    distortion (torchvision's antialiased bilinear) and
//	                    padded at the bottom and right with black, cut into
//	                    tiles row-major, then -- for more than one tile -- the
//	                    whole picture squashed to one more; rescaled and
//	                    normalised in bfloat16, as the original
//	                    implementation does
//	the class token     LAST, with the position table's last row
//	                    (Llama4VisionModel: cat([patches, class]))
//	the rotary          Llama4VisionRotaryEmbedding: adjacent pairs, the first
//	                    HeadDim/4 turned by the patch's column plus one and the
//	                    next HeadDim/4 by its row plus one, each half from the
//	                    first frequency of base^(-4i/HeadDim); the class token
//	                    does not turn
//	the adapter         the 2x2 shuffle (pixel_shuffle's reshapes come to the
//	                    shuffle every tower here runs: dy outer, dx inner),
//	                    two matrices each followed by a GELU, the projector
//	the prompt          <|image_start|>, each tile's rows followed by
//	                    <|tile_x_separator|> or, at a row's end,
//	                    <|tile_y_separator|>, then <|image|>, the global
//	                    tile's rows and <|image_end|> (Llama4Processor)
//
// The block -- LayerNorms with biases, biased q/k/v/o, a GELU MLP with biases
// -- is the kit every CLIP tower runs.

// llama4Rope is the tower's 2-D rotary as a multi-axis table: NRot is half
// a head, so frequency index i is base^(-4i/HeadDim), and each run of
// HeadDim/4 adjacent pairs starts from index 0 -- the column's first, then
// the row's.
func (c TowerConfig) llama4Rope() nn.Rope {
	q := c.HeadDim / 4
	return nn.Rope{NRot: c.HeadDim / 2, Base: 10000,
		Runs: []nn.RopeRun{{Pairs: q, Axis: 0}, {Pairs: q, Axis: 1}}}
}

// llama4Coords is the rotary position of residual row i of a tile of np
// patches, gw a side: (column+1, row+1) for a patch, (0, 0) -- no turn -- for
// the class token, which is the last row.
func llama4Coords(f towerFault, i, np, gw int, dst []int) {
	if i >= np {
		dst[0], dst[1] = 0, 0
		return
	}
	switch f {
	case faultLlama4FromZero:
		dst[0], dst[1] = i%gw, i/gw
	case faultLlama4RowFirst:
		dst[0], dst[1] = i/gw+1, i%gw+1
	default:
		dst[0], dst[1] = i%gw+1, i/gw+1
	}
}

// llama4Canvases is find_supported_resolutions: every (rows, columns) of up
// to max tiles, in the reference's order -- tile counts from max down, each
// count's factor pairs ascending, grouped by aspect ratio in the order each
// ratio first appears.
func llama4Canvases(max int) [][2]int {
	var ratios []float64
	byRatio := map[float64][][2]int{}
	for n := max; n >= 1; n-- {
		for f := 1; f <= n; f++ {
			if n%f != 0 {
				continue
			}
			h, w := f, n/f
			r := float64(h) / float64(w)
			if _, ok := byRatio[r]; !ok {
				ratios = append(ratios, r)
			}
			byRatio[r] = append(byRatio[r], [2]int{h, w})
		}
	}
	var out [][2]int
	for _, r := range ratios {
		out = append(out, byRatio[r]...)
	}
	return out
}

// llama4BestFit is get_best_fit on an h x w picture: the canvas (in tiles)
// whose limiting scale is the smallest upscale, or failing any the largest
// downscale, the smallest area breaking a tie. The scales are float32, as
// the reference divides two integer tensors.
func llama4BestFit(h, w, tile, max int) (rows, cols int) {
	cs := llama4Canvases(max)
	scale := make([]float32, len(cs))
	for i, c := range cs {
		sw := float32(c[1]*tile) / float32(w)
		sh := float32(c[0]*tile) / float32(h)
		scale[i] = sh
		if sh > sw {
			scale[i] = sw
		}
	}
	sel, up := float32(0), false
	for _, s := range scale {
		if s >= 1 && (!up || s < sel) {
			sel, up = s, true
		}
	}
	if !up {
		for _, s := range scale {
			if s > sel {
				sel = s
			}
		}
	}
	best := -1
	for i, s := range scale {
		if s == sel && (best < 0 || cs[i][0]*cs[i][1] < cs[best][0]*cs[best][1]) {
			best = i
		}
	}
	return cs[best][0], cs[best][1]
}

// llama4Layout is a w x h picture's canvas in tiles and the size it is
// resized to on that canvas (get_max_res_without_distortion, after the
// processor's cap on upscaling at one tile).
func (c TowerConfig) llama4Layout(w, h int) (rows, cols, nw, nh int) {
	n := c.ImageSz
	rows, cols = llama4BestFit(h, w, n, c.MaxTiles)
	th, tw := min(max(h, n), rows*n), min(max(w, n), cols*n)
	sw, sh := float64(tw)/float64(w), float64(th)/float64(h)
	if sw < sh {
		nw, nh = tw, min(int(math.Floor(float64(h)*sw)), th)
	} else {
		nh, nw = th, min(int(math.Floor(float64(w)*sh)), tw)
	}
	return rows, cols, max(nw, 1), max(nh, 1)
}

// llama4Tiles is how many tiles a w x h picture becomes: its canvas, and a
// global tile when the canvas is more than one.
func (c TowerConfig) llama4Tiles(w, h int) int {
	rows, cols, _, _ := c.llama4Layout(w, h)
	if rows*cols == 1 {
		return 1
	}
	return rows*cols + 1
}

// llama4Preprocess is Llama4ImageProcessor's pixels, every tile ImageSz
// square back to back, [tile][y][x][channel], and the canvas in tiles.
func (c TowerConfig) llama4Preprocess(p rgb8) (px []float32, rows, cols int) {
	n := c.ImageSz
	one := n * n * 3
	rows, cols, nw, nh := c.llama4Layout(p.w, p.h)
	canvas := rgb8{w: cols * n, h: rows * n, px: make([]uint8, 3*cols*n*rows*n)}
	res := p.resize(nw, nh, aaBilinear)
	for y := 0; y < nh; y++ {
		copy(canvas.px[3*y*canvas.w:], res.px[3*y*nw:3*(y+1)*nw])
	}
	tiles := rows * cols
	if tiles > 1 {
		tiles++
	}
	px = make([]float32, tiles*one)
	lut := bf16NormLUT(c.Mean, c.Std)
	for i := 0; i < rows*cols; i++ {
		canvas.crop((i%cols)*n, (i/cols)*n, n, n).lookupInto(px[i*one:(i+1)*one], &lut)
	}
	if tiles > rows*cols {
		p.resize(n, n, aaBilinear).lookupInto(px[(tiles-1)*one:], &lut)
	}
	return px, rows, cols
}

// lookupInto writes each sample's value from a per-channel table, on
// generated code.
func (p rgb8) lookupInto(dst []float32, lut *[3][256]float32) { nn.PixLUTJIT(dst, p.px, lut) }

// bf16 rounds to bfloat16, to nearest even, as torch does.
func bf16(x float32) float32 {
	b := math.Float32bits(x)
	b += 0x7FFF + (b>>16)&1
	return math.Float32frombits(b &^ 0xFFFF)
}

// bf16NormLUT is the original implementation's rescale and normalisation for
// every 8-bit sample, in bfloat16: the sample times 1/255 rounded to
// bfloat16, minus the mean rounded again, divided by the std rounded again
// (torch's arithmetic on a bfloat16 tensor: each op in float32, each result
// rounded).
func bf16NormLUT(mean, std [3]float64) (lut [3][256]float32) {
	const rescale = float32(1.0 / 255)
	for ch := 0; ch < 3; ch++ {
		m, sd := bf16(float32(mean[ch])), bf16(float32(std[ch]))
		for v := range lut[ch] {
			x := bf16(float32(v) * rescale)
			lut[ch][v] = bf16(bf16(x-m) / sd)
		}
	}
	return lut
}

// projectLlama4 is the adapter and the projector over a tile's patch rows
// (the class row, last, left off): the 2x2 shuffle, a matrix and a GELU,
// another and another GELU, then the projector's matrix.
func (s *State) projectLlama4(patches []float32) ([]float32, error) {
	v := s.vis
	t, c := v.t, v.t.Cfg
	gh, gw := s.patchGrid()
	sc := c.Scale
	mh, mw := gh/sc, gw/sc
	nt := mh * mw
	w := c.NEmbd * sc * sc
	for gy := 0; gy < mh; gy++ {
		for gx := 0; gx < mw; gx++ {
			dst := v.grp[(gy*mw+gx)*w : (gy*mw+gx+1)*w]
			for dy := 0; dy < sc; dy++ {
				for dx := 0; dx < sc; dx++ {
					p := (gy*sc+dy)*gw + gx*sc + dx
					slot := dy*sc + dx
					if c.fault == towerFaultShuffleSwap {
						slot = dx*sc + dy
					}
					copy(dst[slot*c.NEmbd:], patches[p*c.NEmbd:(p+1)*c.NEmbd])
				}
			}
		}
	}
	v.stage("proj_in", v.grp[:nt*w])
	s.jit.NewInput()
	if err := s.mm(v.hid, t.projW, v.grp, nt); err != nil {
		return nil, err
	}
	s.actAll(v.hid[:nt*t.projW.rows], nn.ActGELU)
	s.jit.NewInput()
	if err := s.mm(v.grp, t.projW2, v.hid, nt); err != nil {
		return nil, err
	}
	s.actAll(v.grp[:nt*t.projW2.rows], nn.ActGELU)
	s.jit.NewInput()
	if err := s.mm(v.out, t.projW3, v.grp, nt); err != nil {
		return nil, err
	}
	v.stage("out", v.out[:nt*c.ProjDim])
	return v.out[:nt*c.ProjDim], nil
}

// loadLlama4 reads the adapter's two matrices and the projector's.
func (t *Tower) loadLlama4(get func(jlm.Role, int32) (tensor, error)) error {
	c := &t.Cfg
	var err error
	for _, w := range []struct {
		r   jlm.Role
		dst *tensor
	}{{jlm.RoleVProj, &t.projW}, {jlm.RoleVProj2, &t.projW2}, {jlm.RoleVProj3, &t.projW3}} {
		if *w.dst, err = get(w.r, jlm.DenseBlock); err != nil {
			return err
		}
	}
	if t.projW.k != c.projK() || t.projW2.k != t.projW.rows || t.projW3.k != t.projW2.rows ||
		t.projW3.rows != c.ProjDim {
		return fmt.Errorf("model: OpenTower: llama4's adapter and projector are %dx%d, %dx%d, %dx%d; "+
			"want %d in and %d out", t.projW.k, t.projW.rows, t.projW2.k, t.projW2.rows,
			t.projW3.k, t.projW3.rows, c.projK(), c.ProjDim)
	}
	return nil
}

// Llama4Pieces cuts img into Llama 4's tiles -- the canvas's, row-major, then
// the global tile -- each a picture of its own, named, and the canvas's rows
// and columns, which Llama4Spans lays out. The image kernels are the
// process's rather than a State's, so it needs no State.
func (t *Tower) Llama4Pieces(img image.Image) (rows, cols int, pieces []*Picture, err error) {
	c := t.Cfg
	if c.Kind != jlm.ProjLlama4 {
		return 0, 0, nil, fmt.Errorf("model: projector %s does not tile as Llama 4 does", c.Projector)
	}
	p := toRGB8(img)
	if p.w == 0 || p.h == 0 {
		return 0, 0, nil, fmt.Errorf("model: an empty picture")
	}
	px, rows, cols := c.llama4Preprocess(p)
	one := c.ImageSz * c.ImageSz * 3
	side := c.ImageSz / c.PatchSz
	for i := 0; i < len(px)/one; i++ {
		pc, err := c.picture(px[i*one:(i+1)*one], side, side)
		if err != nil {
			return 0, 0, nil, err
		}
		pieces = append(pieces, pc)
	}
	return rows, cols, pieces, nil
}

// Llama4Spans lays out one picture's run of the prompt as Llama4Processor
// does: <|image_start|>, the canvas's tiles row by row with
// <|tile_x_separator|> between two of a row and <|tile_y_separator|> after
// its last, then <|image|>, the global tile and <|image_end|>. A picture of
// one tile has no separators and no global tile.
func (m *Model) Llama4Spans(rows, cols int, pieces []*Picture) ([]Span, error) {
	want := rows * cols
	if want > 1 {
		want++
	}
	if len(pieces) != want {
		return nil, fmt.Errorf("model: %d pieces for a %dx%d canvas", len(pieces), rows, cols)
	}
	var out []Span
	var err error
	tok := func(s string) {
		if err != nil {
			return
		}
		id, ok := m.Vocab.ID(s)
		if !ok {
			err = fmt.Errorf("model: the vocabulary has no %q, so the image markers cannot be built", s)
			return
		}
		out = append(out, Span{Tokens: []int32{id}})
	}
	tok("<|image_start|>")
	if rows*cols > 1 {
		for r := 0; r < rows; r++ {
			for x := 0; x < cols; x++ {
				out = append(out, Span{Picture: pieces[r*cols+x]})
				if x < cols-1 {
					tok("<|tile_x_separator|>")
				}
			}
			tok("<|tile_y_separator|>")
		}
	}
	tok("<|image|>")
	out = append(out, Span{Picture: pieces[len(pieces)-1]})
	tok("<|image_end|>")
	if err != nil {
		return nil, err
	}
	return mergeTokenSpans(out), nil
}

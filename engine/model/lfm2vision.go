package model

import (
	"fmt"
	"image"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// LFM2-VL's vision (LiquidAI's LFM2-VL and LFM2.5-VL, and d1): SigLIP2's
// NaFlex tower over each piece of a picture at the piece's own grid, then the
// unshuffle and an MLP.
//
// What is the family's own, each read off the reference (transformers'
// Lfm2VlImageProcessor, Lfm2VlProcessor and Lfm2VlMultiModalProjector):
//
//	the pieces    a picture whose size, rounded to whole merge units, holds
//	              more than Tolerance times MaxPixels is cut: the grid of
//	              MinTiles to MaxTiles TileSize squares closest to its aspect
//	              (InternVL's choice, gridOf), the picture resized onto it and
//	              cut row-major, then a thumbnail at the smart-resized size.
//	              Any other picture is one piece at that size. The smart
//	              resize rounds each side to a merge unit (Python's round,
//	              half to even) and scales into [MinPixels, MaxPixels].
//	the resize    torchvision's on the 8-bit picture (rgb8.resize), bicubic
//	              where the file says so (FlagVisionBicubic) and bilinear,
//	              the class default, where it does not
//	the tower     Phi-4-reasoning-vision's (the square table resized to the
//	              grid, phi4Table) with the post-LayerNorm last_hidden_state
//	              carries
//	the projector the Scale x Scale unshuffle, dy outer and dx inner (the
//	              shuffle every tower here runs), the optional LayerNorm,
//	              linear, GELU, linear
//	the prompt    <|image_start|>, each tile as <|img_row_R_col_C|> and its
//	              rows, then <|img_thumbnail|> and the thumbnail's rows,
//	              <|image_end|>; a one-piece picture is its rows between the
//	              first and last marker alone

// Gates' violations are towerFault values (faultLFM2TanhGELU, faultLFM2NoThumb,
// faultLFM2Bilinear, faultLFM2NoProjNorm), set on the tower's configuration.

// lfm2Pixels gives the tower the processor's bounds the container states.
// MaxPixels is also the most patches a piece holds: a tile is TileSize
// square and a thumbnail at most MaxPixels.
func lfm2Pixels(c *TowerConfig, v *jlm.Vision) error {
	c.MinPixels, c.MaxPixels = int(v.MinPixels), int(v.MaxPixels)
	c.TileSize, c.Tolerance = int(v.TileSize), float64(v.Tolerance)
	c.Bicubic = v.Flags.Has(jlm.FlagVisionBicubic)
	unit := c.PatchSz * max(c.Scale, 1)
	switch {
	case unit == 0 || c.MinPixels <= 0 || c.MaxPixels < c.MinPixels:
		return fmt.Errorf("model: OpenTower: lfm2 reads %d to %d pixels", c.MinPixels, c.MaxPixels)
	case c.TileSize <= 0 || c.TileSize%unit != 0 || c.TileSize*c.TileSize > c.MaxPixels:
		return fmt.Errorf("model: OpenTower: lfm2's %d-pixel tile is not whole %d-pixel units within %d pixels",
			c.TileSize, unit, c.MaxPixels)
	case c.Tolerance <= 0 || c.MinTiles < 1 || c.MaxTiles < c.MinTiles:
		return fmt.Errorf("model: OpenTower: lfm2 tiles %d to %d past %g times its budget",
			c.MinTiles, c.MaxTiles, c.Tolerance)
	}
	return nil
}

// lfm2Layout is how a w x h picture is cut: cols x rows TileSize tiles and a
// thumbnail of tw x th pixels, or (cols, rows) = (1, 1) and the one piece at
// tw x th.
func (c TowerConfig) lfm2Layout(w, h int) (cols, rows, tw, th int, err error) {
	if w <= 0 || h <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("model: a %dx%d picture", w, h)
	}
	unit := c.PatchSz * c.Scale
	round := func(v int) int { return int(math.RoundToEven(float64(v)/float64(unit))) * unit }
	tw, th = c.lfm2Smart(w, h)
	// _is_image_too_large: the floor is one patch, not one unit.
	hb, wb := max(c.PatchSz, round(h)), max(c.PatchSz, round(w))
	if float64(hb*wb) <= float64(c.MaxPixels)*c.Tolerance {
		return 1, 1, tw, th, nil
	}
	cols, rows = gridOf(w, h, c.TileSize, c.MinTiles, c.MaxTiles)
	return cols, rows, tw, th, nil
}

// lfm2Tokens is how many rows a w x h picture's pieces emit, markers aside:
// a tile's Tokens each and the thumbnail's (or the one piece's) own.
func (c TowerConfig) lfm2Tokens(w, h int) int {
	cols, rows, tw, th, err := c.lfm2Layout(w, h)
	if err != nil {
		return 0
	}
	unit := c.PatchSz * c.Scale
	n := (tw / unit) * (th / unit)
	if cols*rows > 1 {
		n += cols * rows * (c.TileSize / unit) * (c.TileSize / unit)
	}
	return n
}

// lfm2Smart is Lfm2VlImageProcessor.smart_resize: each side rounded to a
// whole merge unit, then scaled down past MaxPixels or up under MinPixels.
func (c TowerConfig) lfm2Smart(w, h int) (int, int) {
	unit := c.PatchSz * c.Scale
	f := float64(unit)
	hb := max(unit, int(math.RoundToEven(float64(h)/f))*unit)
	wb := max(unit, int(math.RoundToEven(float64(w)/f))*unit)
	switch {
	case hb*wb > c.MaxPixels:
		beta := math.Sqrt(float64(h) * float64(w) / float64(c.MaxPixels))
		hb = max(unit, int(math.Floor(float64(h)/beta/f))*unit)
		wb = max(unit, int(math.Floor(float64(w)/beta/f))*unit)
	case hb*wb < c.MinPixels:
		beta := math.Sqrt(float64(c.MinPixels) / (float64(h) * float64(w)))
		hb = int(math.Ceil(float64(h)*beta/f)) * unit
		wb = int(math.Ceil(float64(w)*beta/f)) * unit
	}
	return wb, hb
}

// lfm2Filter is the resampling filter the container states.
func (c TowerConfig) lfm2Filter() aaFilter {
	if c.Bicubic && c.fault != faultLFM2Bilinear {
		return aaBicubic
	}
	return aaBilinear
}

// lfm2Preprocess is every piece of the picture, normalised, [y][x][channel]
// each, in the order the prompt takes them: the tiles row-major, then the
// thumbnail; or the one piece. sizes holds each piece's (w, h).
func (c TowerConfig) lfm2Preprocess(p rgb8) (px [][]float32, sizes [][2]int, cols, rows int, err error) {
	cols, rows, tw, th, err := c.lfm2Layout(p.w, p.h)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	f := c.lfm2Filter()
	norm := func(q rgb8) []float32 {
		out := make([]float32, q.w*q.h*3)
		q.normalizeInto(out, c.Mean, c.Std)
		return out
	}
	if cols*rows == 1 {
		return [][]float32{norm(p.resize(tw, th, f))}, [][2]int{{tw, th}}, 1, 1, nil
	}
	n := c.TileSize
	big := p.resize(cols*n, rows*n, f)
	for i := 0; i < cols*rows; i++ {
		px = append(px, norm(big.crop((i%cols)*n, (i/cols)*n, n, n)))
		sizes = append(sizes, [2]int{n, n})
	}
	if c.fault == faultLFM2NoThumb {
		return px, sizes, cols, rows, nil
	}
	px = append(px, norm(p.resize(tw, th, f)))
	sizes = append(sizes, [2]int{tw, th})
	return px, sizes, cols, rows, nil
}

// LFM2Pieces is the picture's pieces, preprocessed and named, and its grid
// of tiles ((1, 1) for a picture that is one piece).
func (t *Tower) LFM2Pieces(img image.Image) (cols, rows int, pieces []*Picture, err error) {
	c := t.Cfg
	if c.Kind != jlm.ProjLFM2 {
		return 0, 0, nil, fmt.Errorf("model: projector %s does not tile as LFM2-VL does", c.Projector)
	}
	p := toRGB8(img)
	if p.w == 0 || p.h == 0 {
		return 0, 0, nil, fmt.Errorf("model: an empty picture")
	}
	px, sizes, cols, rows, err := c.lfm2Preprocess(p)
	if err != nil {
		return 0, 0, nil, err
	}
	for i := range px {
		pc, err := c.picture(px[i], sizes[i][1]/c.PatchSz, sizes[i][0]/c.PatchSz)
		if err != nil {
			return 0, 0, nil, err
		}
		pieces = append(pieces, pc)
	}
	return cols, rows, pieces, nil
}

// LFM2Spans lays out one picture's run of the prompt as Lfm2VlProcessor does
// with its image special tokens (use_image_special_tokens, true for every
// published checkpoint).
func (m *Model) LFM2Spans(cols, rows int, pieces []*Picture) ([]Span, error) {
	want := 1
	if cols*rows > 1 {
		want = cols*rows + 1
	}
	if len(pieces) != want {
		return nil, fmt.Errorf("model: %d pieces for a %dx%d grid", len(pieces), cols, rows)
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
	if cols*rows > 1 {
		for r := 0; r < rows; r++ {
			for x := 0; x < cols; x++ {
				tok(fmt.Sprintf("<|img_row_%d_col_%d|>", r+1, x+1))
				out = append(out, Span{Picture: pieces[r*cols+x]})
			}
		}
		tok("<|img_thumbnail|>")
	}
	out = append(out, Span{Picture: pieces[len(pieces)-1]})
	tok("<|image_end|>")
	if err != nil {
		return nil, err
	}
	return mergeTokenSpans(out), nil
}

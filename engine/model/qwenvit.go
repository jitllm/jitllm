package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// The Qwen-VL towers' geometry: a patch grid that follows the picture
// (dynamic resolution), and Qwen2.5-VL's window attention.
//
// Both are about which ROWS a tower runs and in what order. The arithmetic is
// the tower's own (vision.go); what is here decides the grid an image becomes,
// the order the patches go through the blocks in, and the windows a windowed
// block attends inside. Every one of them is transformers' and llama.cpp's
// own layout, read from their source rather than re-derived:
//
//	smart_resize        transformers' image_processing_qwen2_vl.smart_resize
//	window order        transformers' vision_utils.get_vision_window_index,
//	                    llama.cpp's clip.cpp idx/inv_idx for QWEN25VL
//	merge order         each 2x2 merge group in four consecutive rows, dy
//	                    outer and dx inner (get_vision_position_ids' block-major
//	                    layout)

// The pixel bounds a dynamic-resolution tower defaults to. The floor is
// transformers' min_pixels (56*56). The ceiling is the square the mmproj
// names (clip.vision.image_size, 560 for every Qwen-VL mmproj here), which
// is the size the tower ran at before its grid followed the picture: the
// reference's own ceilings (transformers' 12845056 pixels, llama.cpp's 4096
// tokens) make a tower's attention scratch quadratic in a number nothing here
// bounds, so a larger one is the caller's choice (WithImageMaxPixels). Every
// picture under the ceiling becomes the reference's grid exactly.
const qwenMinPixels = 56 * 56

// dynamicPixels gives a Qwen-VL tower its pixel bounds.
func dynamicPixels(c *TowerConfig) {
	c.MinPixels = qwenMinPixels
	if c.MaxPixels == 0 {
		c.MaxPixels = c.ImageSz * c.ImageSz
	}
}

// Dynamic reports whether the tower's patch grid follows the picture.
func (c TowerConfig) Dynamic() bool { return c.MaxPixels > 0 }

// ResizeTo is the pixel size a w x h picture is resized to before it is
// patched: transformers' smart_resize, whose rounding is Python's round()
// (half to even), floor and ceil, each where the reference uses it. A
// fixed-size tower resizes every picture to its square.
func (c TowerConfig) ResizeTo(w, h int) (rw, rh int, err error) {
	if !c.Dynamic() {
		return c.ImageSz, c.ImageSz, nil
	}
	switch c.Kind {
	case jlm.ProjGLM4V:
		return c.glmResize(w, h)
	case jlm.ProjKimiVL:
		return c.kimiPadded(w, h)
	case jlm.ProjPixtral:
		return c.pixtralResize(w, h)
	case jlm.ProjGemma4V:
		return c.gemma4Resize(w, h)
	case jlm.ProjPhi4:
		return c.phi4Resize(w, h)
	}
	return c.qwenResize(w, h)
}

// qwenResize is ResizeTo for a Qwen-VL tower: smart_resize.
func (c TowerConfig) qwenResize(w, h int) (rw, rh int, err error) {
	if w <= 0 || h <= 0 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture", w, h)
	}
	if float64(max(w, h))/float64(min(w, h)) > 200 {
		return 0, 0, fmt.Errorf("model: a %dx%d picture's aspect is over 200, which the reference refuses", w, h)
	}
	f := float64(c.PatchSz * c.Scale)
	hb := math.RoundToEven(float64(h)/f) * f
	wb := math.RoundToEven(float64(w)/f) * f
	hw := float64(h) * float64(w)
	switch {
	case hb*wb > float64(c.MaxPixels):
		beta := math.Sqrt(hw / float64(c.MaxPixels))
		hb = math.Max(f, math.Floor(float64(h)/beta/f)*f)
		wb = math.Max(f, math.Floor(float64(w)/beta/f)*f)
	case hb*wb < float64(c.MinPixels):
		beta := math.Sqrt(float64(c.MinPixels) / hw)
		hb = math.Ceil(float64(h)*beta/f) * f
		wb = math.Ceil(float64(w)*beta/f) * f
	}
	return int(wb), int(hb), nil
}

// GridFor is the embeddings' grid a w x h picture becomes (ResizeTo, patched
// and merged): what a prompt is sized and its rows are laid out by before the
// picture is encoded.
func (c TowerConfig) GridFor(w, h int) (ImageGrid, error) {
	rw, rh, err := c.ResizeTo(w, h)
	if err != nil {
		return ImageGrid{}, err
	}
	s := c.PatchSz * max(c.Scale, 1)
	return c.gridOf(rh/s, rw/s), nil
}

// SetImageSize says the picture the next Encode is handed is w x h pixels,
// which Preprocess says by itself: it is for a caller that preprocessed on
// another state. A dynamic tower takes any size of whole merge units within
// its bounds; a fixed one takes only its square.
func (s *State) SetImageSize(w, h int) error {
	c := s.vis.t.Cfg
	u := c.PatchSz * max(c.Scale, 1)
	switch {
	case !c.Dynamic() && (w != c.ImageSz || h != c.ImageSz):
		return fmt.Errorf("model: a %dx%d picture for a tower that takes %dx%d", w, h, c.ImageSz, c.ImageSz)
	case w <= 0 || h <= 0 || w%u != 0 || h%u != 0:
		return fmt.Errorf("model: a %dx%d picture is not whole %d-pixel merge units", w, h, u)
	case (w/c.PatchSz)*(h/c.PatchSz) > c.Patches():
		return fmt.Errorf("model: a %dx%d picture is %d patches, over the %d this tower holds",
			w, h, (w/c.PatchSz)*(h/c.PatchSz), c.Patches())
	}
	s.vis.gh, s.vis.gw = h/c.PatchSz, w/c.PatchSz
	return nil
}

// patchGrid is the current image's patch grid.
func (s *State) patchGrid() (gh, gw int) {
	if s.vis.gh > 0 {
		return s.vis.gh, s.vis.gw
	}
	side := s.vis.t.Cfg.ImageSz / s.vis.t.Cfg.PatchSz
	return side, side
}

// Grid is the embeddings' grid of the image this state last preprocessed (or
// will encode): where its rows turn on an M-RoPE text model.
func (s *State) Grid() ImageGrid {
	gh, gw := s.patchGrid()
	sc := max(s.vis.t.Cfg.Scale, 1)
	return s.vis.t.Cfg.gridOf(gh/sc, gw/sc)
}

// patchAt is the raster patch row r holds.
func (s *State) patchAt(r int) int {
	if s.vis.ord == nil {
		return r
	}
	return s.vis.ord[r]
}

// windowed reports whether block li attends inside windows: every block but
// each WinPattern'th (llama.cpp: full when (il+1) % n_wa_pattern == 0).
func (s *State) windowed(li int) bool {
	p := s.vis.t.Cfg.WinPattern
	return p > 0 && (li+1)%p != 0
}

// windowOrder lays a gh x gw patch grid out in window order: windows of
// WinSize pixels (WinSize/PatchSz/Scale merge units on a side) row-major over
// the merged grid, the merge units inside a window row-major, each unit's
// Scale x Scale patches in consecutive rows, dy outer. segs is each window's
// row range. A tower with no windows keeps raster order.
func (s *State) windowOrder(gh, gw int) {
	c := s.vis.t.Cfg
	if c.WinPattern == 0 {
		s.vis.ord, s.vis.segs = nil, nil
		return
	}
	sc := c.Scale
	mh, mw := gh/sc, gw/sc
	win := c.WinSize / c.PatchSz / sc
	s.vis.ord, s.vis.segs = s.vis.ord[:0], s.vis.segs[:0]
	for wy := 0; wy < mh; wy += win {
		for wx := 0; wx < mw; wx += win {
			lo := len(s.vis.ord)
			for y := wy; y < min(wy+win, mh); y++ {
				for x := wx; x < min(wx+win, mw); x++ {
					for dy := 0; dy < sc; dy++ {
						for dx := 0; dx < sc; dx++ {
							s.vis.ord = append(s.vis.ord, (y*sc+dy)*gw+x*sc+dx)
						}
					}
				}
			}
			s.vis.segs = append(s.vis.segs, lo, len(s.vis.ord))
		}
	}
}

// unwindow puts the merged rows, emitted in window order, back into raster
// order of the merged grid: transformers' reverse_indices, llama.cpp's
// get_rows(window_idx). Merged row i of the window order is merge unit
// ord[i*Scale^2]'s, whose raster unit is its patch's (y/Scale, x/Scale).
func (s *State) unwindow(dst, src []float32, nt int) {
	c := s.vis.t.Cfg
	if s.tfault() == faultQwenNoUnwindow {
		copy(dst[:nt*c.ProjDim], src[:nt*c.ProjDim])
		return
	}
	_, gw := s.patchGrid()
	sc := c.Scale
	mw := gw / sc
	for i := 0; i < nt; i++ {
		p := s.vis.ord[i*sc*sc]
		u := (p/gw/sc)*mw + (p%gw)/sc
		copy(dst[u*c.ProjDim:(u+1)*c.ProjDim], src[i*c.ProjDim:(i+1)*c.ProjDim])
	}
}

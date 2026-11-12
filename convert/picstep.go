package convert

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// A processor's own picture step, as a family's stater writes it. The
// container holds none of these: the converter states the picture's
// geometry from them (GeometryOf) and lowers them to the container's
// generic steps (lowerPicture) -- a resize to a fixed window
// (jlm.PreResize) or to each square's own size (jlm.PreResizeLive), a
// canvas (jlm.PreCanvas), the normalisation. Each is one processor's
// resize.
const (
	// PreResizeShort resamples the picture, aspect kept, so its shorter side
	// is Ints[0] pixels and its longer int(Ints[0] * long / short), with
	// filter Ints[1] in arithmetic Ints[2].
	PreResizeShort jlm.PreKind = 100 + iota
	// PreCenterCrop is the Ints[0] x Ints[1] window of the picture as the
	// step before left it, at ((w - Ints[0]) / 2, (h - Ints[1]) / 2)
	// truncated.
	PreCenterCrop
	// PreTile cuts the picture into tiles, each a picture the tower sees:
	// between Ints[0] and Ints[1] of them, the columns x rows grid whose
	// aspect is closest to the picture's (a tie to the larger grid while the
	// picture's area is more than half of it, the grids every columns x rows
	// pair, columns outer, stably sorted by tile count), the picture resized
	// by the resize step to the grid's size in squares and cut row-major;
	// then, where Ints[2] is 1 and the grid is more than one tile, the whole
	// picture resized to one square. A picture's rows are every tile's in
	// that order.
	PreTile
	// PreSmartResize resamples the picture to a size of its own, each side a
	// multiple of Ints[0] pixels: each rounded half to even to one; then,
	// where the area passes Ints[2] pixels, both scaled down by
	// sqrt(area / Ints[2]) and floored to one, or where it falls short of
	// Ints[1], scaled up by sqrt(Ints[1] / area) and ceiled to one
	// (Qwen2-VL's smart_resize); with filter Ints[3] in arithmetic Ints[4].
	// Ints[5], where stated, is 1 for a picture a side shorter than Ints[0]
	// scaled up first so that side is Ints[0], each side truncated
	// (GLM-4V's smart_resize), 0 for none.
	PreSmartResize
	// PreLimitPad keeps the picture at its size unless its whole Ints[0]
	// pixel patches pass Ints[1], where both sides are scaled by
	// sqrt(Ints[1] / patches) and truncated, with filter Ints[3] in
	// arithmetic Ints[4]; then pads it black, before it is normalised, on
	// the right and at the bottom to multiples of Ints[2] pixels (Kimi-VL's
	// rescale with pad_input).
	PreLimitPad
	// PreLongBox resizes the picture so its longer side is Ints[0] pixels
	// and its shorter round-half-even(side * Ints[0] / longer), at least
	// Ints[1], with filter Ints[5] in arithmetic Ints[6]; then centres it
	// on an Ints[0]-pixel square of the colour Ints[2..4] (bytes, before
	// normalising), each margin the floor of half the slack (Janus's
	// resize and pad_to_square).
	PreLongBox
	// PreNaFlex resamples the picture to a size of its own in whole
	// Ints[0]-pixel patches: its own whole patches' count clamped to
	// [Ints[1], Ints[2]], then the largest scale whose sides, each ceiled
	// to a patch and at least one, hold that many, found by bisection
	// (SigLIP2's get_image_size_for_max_num_patches); with filter Ints[3] in
	// arithmetic Ints[4].
	PreNaFlex
	// PreCapUnit resamples the picture so neither side passes Ints[0]
	// pixels -- both divided by longer / Ints[0] and floored, where the
	// longer passes it -- then each side rounded up to a whole Ints[1]
	// pixels, with filter Ints[2] in arithmetic Ints[3] (Pixtral's resize).
	PreCapUnit
	// PreAspectBudget resamples the picture to the largest size of its
	// aspect within Ints[0] patches of Ints[1] pixels, each side a whole
	// Ints[2] pixels: both scaled by sqrt(budget area / area) and floored
	// to the unit; a side that floors to nothing one unit, the other the
	// aspect's whole multiple of it, at most the budget's units a row
	// (Gemma 4's get_aspect_ratio_preserving_size); with filter Ints[3] in
	// arithmetic Ints[4].
	PreAspectBudget
	// PreTileSmart is a picture one piece at its own size or cut: its own
	// size each side a whole Ints[4] pixels, rounded half to even, then
	// scaled into [Ints[5], Ints[6]] pixels as PreSmartResize scales it; a
	// picture whose sides rounded so (at least an Ints[3]-pixel patch each)
	// hold more than Floats[0] times Ints[6] pixels is cut into the grid of
	// Ints[1] to Ints[2] Ints[0]-pixel tiles closest to its aspect (as
	// PreTile chooses one), resized onto the grid and cut row-major, then
	// the whole at its own size as a thumbnail (LFM2-VL's); with filter
	// Ints[7] in arithmetic Ints[8].
	PreTileSmart
	// PreBestFit cuts the picture into tiles as PreTile does, on the canvas
	// of at most Ints[0] tiles whose limiting scale is the smallest upscale,
	// or failing any the largest downscale, the smallest area breaking a tie
	// (the canvases every rows x columns pair by tile count from Ints[0]
	// down, grouped by aspect in the order each first appears): the picture
	// resized by the resize step onto it without distortion, at most its
	// canvas and upscaled to at most one tile, laid at its top left with
	// black past it and cut row-major; then, for more than one tile, the
	// whole picture resized to one tile more (Llama 4's). A head's marker
	// rows (TOpMarkRow) are a table of three: after a tile not its row's
	// last the first, after a row's last the second, before the picture's
	// last tile the third.
	PreBestFit
	// PreSliceUHD is llava-uhd's cut (MiniCPM-V's): an overview at its own
	// size -- Ints[0]^2 pixels at the picture's aspect, each side a whole
	// Ints[1]-pixel patch, rounded half to even -- then, where the picture's
	// area over Ints[0]^2 rounds up past one, the slices of the grid of at
	// most Ints[2] whose aspect is closest to the picture's (of the counts
	// one either side of that multiple): the picture resized to a whole
	// number of best-sized slices and cut row-major; the overview upscaled
	// only where nothing is cut. With filter Ints[3] in arithmetic Ints[4].
	// A head's marker runs (TOpMarkRow) are a table of six: <image> and
	// </image> about the overview, <slice> and </slice> about a slice, a
	// slice row's first after the first taking the last two (a newline and
	// <slice>) before it.
	PreSliceUHD
)

// liveFilter is where a processor's own resize of each square's own size
// states its filter, and its arithmetic where the picture op reads it (-1
// where it reads the processor's filter's own).
func liveFilter(k jlm.PreKind) (filter, prec int, ok bool) {
	switch k {
	case PreSmartResize, PreAspectBudget, PreNaFlex:
		return 3, -1, true
	case PreLimitPad:
		return 3, -1, true
	case PreSliceUHD:
		return 3, 4, true
	case PreTileSmart:
		return 7, 8, true
	case PreCapUnit:
		return 2, -1, true
	case PreLongBox:
		return 5, -1, true
	}
	return 0, 0, false
}

// lowerPicture lowers steps, a stater's picture steps, to the container's
// generic ones: a processor's resize to each square's own size a live
// resize, with a canvas where it pads or boxes the picture; a shorter
// side's resize and its crop the crop's fixed window; a tiling step
// nothing, the geometry stating the tiles. The resize, the normalisation
// and every other generic step is kept as it is.
func lowerPicture(steps []jlm.PreStep) ([]jlm.PreStep, error) {
	var out []jlm.PreStep
	short := false
	for _, s := range steps {
		in := func(i int) (int32, error) {
			if i >= len(s.Ints) {
				return 0, fmt.Errorf("convert: a picture step of kind %d with %d ints", s.Kind, len(s.Ints))
			}
			return s.Ints[i], nil
		}
		if f, p, ok := liveFilter(s.Kind); ok {
			filter, err := in(f)
			if err != nil {
				return nil, err
			}
			prec := int32(jlm.PrecPIL)
			if p >= 0 {
				if prec, err = in(p); err != nil {
					return nil, err
				}
			}
			out = append(out, jlm.PreStep{Kind: jlm.PreResizeLive, Ints: []int32{filter, prec}})
			switch s.Kind {
			case PreLimitPad:
				out = append(out, jlm.PreStep{Kind: jlm.PreCanvas, Ints: []int32{0, 0, 0, 0, 0}})
			case PreLongBox:
				box := s.Ints[0]
				out = append(out, jlm.PreStep{Kind: jlm.PreCanvas, Ints: []int32{s.Ints[2], s.Ints[3], s.Ints[4], box, box}})
			}
			continue
		}
		switch s.Kind {
		case PreResizeShort:
			// The shorter side's resize lands on its crop's fixed window.
			short = true
			out = append(out, jlm.PreStep{Kind: jlm.PreResize, Ints: []int32{0, 0, s.Ints[1], s.Ints[2]}})
		case PreCenterCrop:
			if !short || len(out) == 0 || out[len(out)-1].Kind != jlm.PreResize {
				return nil, fmt.Errorf("convert: a centre crop after no shorter side's resize")
			}
			out[len(out)-1].Ints[0], out[len(out)-1].Ints[1] = s.Ints[0], s.Ints[1]
		case PreTile, PreBestFit:
		default:
			out = append(out, jlm.PreStep{Kind: s.Kind, Ints: slices.Clone(s.Ints), Floats: slices.Clone(s.Floats)})
		}
	}
	return out, nil
}

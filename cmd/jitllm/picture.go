package main

import (
	"fmt"
	"image"
	"os"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
)

// picture is the image of a prompt as its spans carry it: one model.Picture,
// or for a tower that cuts a picture into an overview and slices (MiniCPM-V's
// resampler) one per piece of lay. The prompt's prefill encodes them.
type picture struct {
	path   string
	lay    *model.Layout
	pieces []*model.Picture
	// tiles is a Llama 4 picture's canvas, rows and columns of tiles: its
	// pieces are the tiles and the global one (Tower.Llama4Pieces).
	tiles *[2]int
	// lfm2 is an LFM2-VL picture's grid of tiles, columns and rows: its pieces
	// are the tiles and the thumbnail, or the one piece (Tower.LFM2Pieces).
	lfm2 *[2]int
}

// openPicture lays out the picture at path before the State exists, which
// the prompt's size needs. A slicing tower's pieces are resampled here --
// Pillow's resampler needs no generated code -- and every other tower's
// picture is sized from the header alone and preprocessed by the State
// (preprocess), whose JIT runs the bilinear resize.
func openPicture(tw *model.Tower, path string) (*picture, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	p := &picture{path: path}
	if tw.Cfg.Kind == jlm.ProjLlama4 {
		img, _, err := image.Decode(f)
		if err != nil {
			return nil, err
		}
		rows, cols, pieces, err := tw.Llama4Pieces(img)
		if err != nil {
			return nil, err
		}
		p.tiles, p.pieces = &[2]int{rows, cols}, pieces
		return p, nil
	}
	if tw.Cfg.Kind == jlm.ProjLFM2 {
		img, _, err := image.Decode(f)
		if err != nil {
			return nil, err
		}
		cols, rows, pieces, err := tw.LFM2Pieces(img)
		if err != nil {
			return nil, err
		}
		p.lfm2, p.pieces = &[2]int{cols, rows}, pieces
		return p, nil
	}
	if tw.Cfg.PosBuckets > 0 {
		img, _, err := image.Decode(f)
		if err != nil {
			return nil, err
		}
		lay, err := tw.Layout(img.Bounds().Dx(), img.Bounds().Dy())
		if err != nil {
			return nil, err
		}
		p.lay = &lay
		p.pieces, err = tw.PicturePieces(img, lay)
		return p, err
	}
	// Its row count follows the picture -- a Qwen-VL tower's grid its
	// aspect, a tiling tower's its tile count -- which the header gives.
	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return nil, err
	}
	pc := &model.Picture{Grid: tw.Cfg.Grid(), Rows: tw.Cfg.TokensFor(cfg.Width, cfg.Height)}
	if tw.Cfg.Dynamic() {
		if pc.Grid, err = tw.Cfg.GridFor(cfg.Width, cfg.Height); err != nil {
			return nil, err
		}
		pc.Rows = tw.Cfg.GridRows(pc.Grid)
	}
	p.pieces = []*model.Picture{pc}
	return p, nil
}

// preprocess fills a picture sized from its header with its pixels and its
// name, on st's vision segment.
func (p *picture) preprocess(st *model.State) error {
	if p.lay != nil || p.tiles != nil || p.lfm2 != nil {
		return nil
	}
	f, err := os.Open(p.path)
	if err != nil {
		return err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return err
	}
	pc, err := st.Picture(img)
	if err != nil {
		return err
	}
	if pc.Rows != p.pieces[0].Rows {
		return fmt.Errorf("the picture preprocessed to %d rows and the prompt was sized for %d",
			pc.Rows, p.pieces[0].Rows)
	}
	*p.pieces[0] = *pc
	return nil
}

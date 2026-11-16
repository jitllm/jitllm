package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok"
)

// statedTower is the tower s carries written down as its reference runs it
// (jlm.Tower), as the projector library states it (libraryTower), so the
// engine knows no family. A projector the library states no tower for gets
// none, and the engine refuses its container by name.
func statedTower(s *jlm.Source, o options) (*jlm.Tower, error) {
	if s.Vision == nil {
		return nil, nil
	}
	lt, by, err := libraryTower(s.Vision.Projector)
	if err != nil || lt == nil {
		return nil, err
	}
	t, err := lt.state(by, s)
	if err != nil {
		return nil, err
	}
	if o.longestEdge > 0 {
		steps := [][]jlm.PreStep{t.Pre}
		if t.Video != nil {
			steps = append(steps, t.Video.Pre)
		}
		for _, pre := range steps {
			for i := range pre {
				if pre[i].Kind == PreLongTiles {
					pre[i].Ints[0] = o.longestEdge
				}
			}
		}
	}
	if o.pictureStep != nil {
		edit := func(steps []jlm.PreStep) {
			for i := range steps {
				o.pictureStep(&steps[i])
			}
		}
		edit(t.Pre)
		if t.Video != nil {
			edit(t.Video.Pre)
		}
	}
	// The picture's geometry is stated from the stater's steps, which are
	// then lowered to the container's generic ones; a video's frames read
	// as pictures under steps of their own take a geometry of their own and
	// the picture's steps.
	if t.Geometry, err = GeometryOf(t, s.Vision); err != nil {
		return nil, err
	}
	switch v := t.Video; {
	case v != nil && v.AsPictures() && len(v.Pre) > 0:
		frames := *t
		frames.Pre = v.Pre
		if v.Frames, err = GeometryOf(&frames, s.Vision); err != nil {
			return nil, fmt.Errorf("convert: the video's frames: %w", err)
		}
		v.Pre = nil
	case v != nil && !v.AsPictures():
		if v.Frames, err = FramesOf(v); err != nil {
			return nil, err
		}
		if v.Pre, err = lowerPicture(v.Pre); err != nil {
			return nil, err
		}
	}
	if t.Pre, err = lowerPicture(t.Pre); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: the stated tower (%v): %w", s.Vision.Projector, err)
	}
	return t, nil
}

// StatesTower reports whether the converter states projector p's tower, so a
// container of p written without one is a stale conversion.
func StatesTower(p jlm.Projector) bool {
	t, _, err := libraryTower(p)
	return err == nil && t != nil
}

// StatesVideo reports whether the converter states how a projector's tower
// reads a video (jlm.Video).
func StatesVideo(p jlm.Projector) bool {
	if t, _, err := libraryTower(p); err == nil && t != nil {
		return t.Video != nil
	}
	return false
}

// cardinal is n in English words as num2words writes a cardinal below a
// thousand: "zero", "twenty-one", "one hundred and five".
func cardinal(n int) string {
	ones := [...]string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
		"eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
	tens := [...]string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
	switch {
	case n < 20:
		return ones[n]
	case n < 100 && n%10 == 0:
		return tens[n/10]
	case n < 100:
		return tens[n/10] + "-" + ones[n%10]
	case n%100 == 0:
		return ones[n/100] + " hundred"
	}
	return ones[n/100] + " hundred and " + cardinal(n%100)
}

// erfGELU is the activation of a tower whose class's hidden_act is "gelu"
// (ACT2FN's GELUActivation, the erf form): the file says only that the tower
// runs GELU, which llama.cpp's graph runs in the tanh form.
func erfGELU(v *jlm.Vision, family string) (jlm.Expr, error) {
	if !v.Flags.Has(jlm.FlagGELU) || v.Flags.Has(jlm.FlagQuickGELU) {
		return jlm.Expr{}, fmt.Errorf("convert: %s: the tower's activation is not GELU", family)
	}
	var e jlm.ExprBuilder
	e.GELUErf(jlm.Operand(0))
	x := jlm.Expr{Terms: e.T}
	return x, x.Valid()
}

// towerFFNByShape names each tower block's two MLP matrices by what they do,
// not what the file calls them: the one that expands NEmbd to the MLP's
// width is RoleVFC1, the one that contracts it back RoleVFC2, their biases
// with them. SmolVLM's mmproj names them the other way round (llama.cpp's
// loader warns "ffn up/down are swapped"). A block whose MLP is as wide as
// the tower cannot be told by shape and keeps the file's names; one whose
// matrices fit neither arrangement is refused.
func towerFFNByShape(s *jlm.Source) error {
	d := uint64(s.Vision.NEmbd)
	type pair struct{ up, down, upB, downB *jlm.Tensor }
	blocks := map[int32]*pair{}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Block == jlm.DenseBlock || !t.Role.Vision() {
			continue
		}
		p := blocks[t.Block]
		if p == nil {
			p = &pair{}
			blocks[t.Block] = p
		}
		switch t.Role {
		case jlm.RoleVFC1:
			p.up = t
		case jlm.RoleVFC2:
			p.down = t
		case jlm.RoleVFC1Bias:
			p.upB = t
		case jlm.RoleVFC2Bias:
			p.downB = t
		}
	}
	for b, p := range blocks {
		if p.up == nil || p.down == nil {
			continue
		}
		upK, upRows := p.up.Dims[0], p.up.Dims[1]
		downK, downRows := p.down.Dims[0], p.down.Dims[1]
		switch {
		case upK == d && downRows == d && upRows == downK:
			// As named, or as wide as the tower and so as named.
		case upRows == d && downK == d && upK == downRows:
			p.up.Role, p.down.Role = jlm.RoleVFC2, jlm.RoleVFC1
			if p.upB != nil {
				p.upB.Role = jlm.RoleVFC2Bias
			}
			if p.downB != nil {
				p.downB.Role = jlm.RoleVFC1Bias
			}
		default:
			return fmt.Errorf("convert: the tower's block %d has MLP matrices of %dx%d and %dx%d over a width of %d: "+
				"neither expands it", b, upK, upRows, downK, downRows, d)
		}
	}
	return nil
}

// towerRoles is the roles of s's tower tensors outside its blocks.
func towerRoles(s *jlm.Source) map[jlm.Role]bool {
	has := map[jlm.Role]bool{}
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && t.Role.Vision() {
			has[t.Role] = true
		}
	}
	return has
}

// markerIDs is each marker's id in s's vocabulary, found by its decoded piece.
func markerIDs(s *jlm.Source, pieces ...string) ([]int32, error) {
	v, err := tok.New(s.Vocab)
	if err != nil {
		return nil, fmt.Errorf("convert: the tower's markers: %w", err)
	}
	out := make([]int32, len(pieces))
	for i, p := range pieces {
		id, ok := v.ID(p)
		if !ok {
			return nil, fmt.Errorf("convert: the tower's marker %q is no token of this vocabulary", p)
		}
		out[i] = id
	}
	return out, nil
}

// widestPicture is the widest side of a picture a tower's picture op reads
// (composer.PictureSide), the bound on a piece's grid.
const widestPicture = 4096

// sincosAxes writes the resampler's key positions as two tables of length
// rows each (jlm.RoleVRsPosAxes): row x of the first sin(x*w_i) then
// cos(x*w_i) over the first half, row y of the second the same of y over
// the second, w_i = 10000^(-i/(d/4)) for i < d/4 -- resampler.py's
// get_2d_sincos_pos_embed, whose first half is the column's (it meshgrids
// (w, h)), in float32 as numpy computes it.
func sincosAxes(s *jlm.Source, d, length int) error {
	q := d / 4
	omega := make([]float32, q)
	for i := range omega {
		omega[i] = 1 / float32(math.Pow(10000, float64(float32(i)/float32(q))))
	}
	table := make([]float32, 2*length*d)
	for p := range length {
		x, y := table[p*d:(p+1)*d], table[(length+p)*d:(length+p+1)*d]
		for i, w := range omega {
			a := float64(float32(p) * w)
			sn, cs := float32(math.Sin(a)), float32(math.Cos(a))
			x[i], x[q+i] = sn, cs
			y[2*q+i], y[3*q+i] = sn, cs
		}
	}
	e := jlm.Tensor{Role: jlm.RoleVRsPosAxes, Block: jlm.DenseBlock, Index: -1, Name: "resampler.pos_axes"}
	setF32Dims(&e, table, uint64(d), uint64(2*length))
	s.Tensors = append(s.Tensors, e)
	return nil
}

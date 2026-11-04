package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok"
)

// statedTower is the tower s carries written down as its reference runs it
// (jlm.Tower): the converter knows the family, so the engine need not. A
// family whose statement is not written yet states none, and the engine
// refuses its container by name until it is.
func statedTower(s *jlm.Source) (*jlm.Tower, error) {
	if s.Vision == nil {
		return nil, nil
	}
	switch s.Vision.Projector {
	case jlm.ProjIdefics3:
		return idefics3Tower(s)
	}
	return nil, nil
}

// idefics3Tower is SmolVLM's tower as llama.cpp's mtmd runs it, which this
// container matches: the picture squashed to the tower's square by two-tap
// bilinear and normalised, patches through one matrix plus the learned
// position table, the SigLIP blocks, a post LayerNorm, the pixel shuffle and
// the one projector matrix, and the overview image's markers around its
// rows. transformers' Idefics3ImageProcessor also cuts a large picture into
// tiles with a Lanczos resize; this states the square alone (docs/divergences.md).
func idefics3Tower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	need := []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPosEmbd, jlm.RoleVPostNorm, jlm.RoleVProj}
	for _, r := range need {
		if !has[r] {
			return nil, fmt.Errorf("convert: idefics3: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || v.Scale == 0 || v.PatchSize == 0 || v.ImageSize%v.PatchSize != 0 {
		return nil, fmt.Errorf("convert: idefics3: %d heads over %d, a shuffle of %d, patches of %d in %d",
			v.NHead, v.NEmbd, v.Scale, v.PatchSize, v.ImageSize)
	}
	if side := v.ImageSize / v.PatchSize; side%v.Scale != 0 {
		return nil, fmt.Errorf("convert: idefics3: a grid of %d patches does not shuffle by %d", side, v.Scale)
	}
	act, err := v.Activation(false)
	if err != nil {
		return nil, err
	}
	bias := func(w, b jlm.Role) []jlm.Role {
		if has[b] {
			return []jlm.Role{w, b}
		}
		return []jlm.Role{w}
	}
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreResize, Ints: []int32{int32(v.ImageSize), int32(v.ImageSize), int32(jlm.FilterTwoTap), int32(jlm.PrecFloat)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 4, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize)},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpPosTable, Bind: []jlm.Role{jlm.RoleVPosEmbd}},
			{Code: jlm.TOpAdd, In: []uint16{2, 3}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act},
		// The last block's rows are value 0.
		Head: jlm.Program{NIn: 1, Out: 3, Ops: []jlm.TowerOp{
			{Code: jlm.TOpLayerNorm, In: []uint16{0}, Bind: bias(jlm.RoleVPostNorm, jlm.RoleVPostNormBias), Eps: v.Eps},
			{Code: jlm.TOpRegroup, In: []uint16{1}, Span: int32(v.Scale)},
			{Code: jlm.TOpLinear, In: []uint16{2}, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)},
		}},
	}
	// mtmd's overview image: "\n\n" <fake_token_around_image> <global-img>,
	// the rows, <fake_token_around_image>, each looked up by its decoded
	// piece (tok.Vocab.ID), as mtmd's lookup_token does.
	if t.Span.Before, err = markerIDs(s, "\n\n", "<fake_token_around_image>", "<global-img>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "<fake_token_around_image>"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: idefics3: %w", err)
	}
	return t, nil
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

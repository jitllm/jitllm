package convert

import (
	"fmt"
	"maps"
	"math"
	"slices"

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
	f := towerStater(s.Vision.Projector)
	if f == nil {
		return nil, nil
	}
	return f(s)
}

// towerStater is what states projector p's tower, nil for a projector whose
// tower is not stated yet.
func towerStater(p jlm.Projector) func(*jlm.Source) (*jlm.Tower, error) {
	switch p {
	case jlm.ProjIdefics3:
		return idefics3Tower
	case jlm.ProjGemma3:
		return gemma3Tower
	case jlm.ProjMLP:
		return llavaTower
	case jlm.ProjInternVL:
		return internvlTower
	case jlm.ProjQwen2VL:
		return qwen2vlTower
	case jlm.ProjQwen25VL:
		return qwen25vlTower
	case jlm.ProjQwen3VL:
		return qwen3vlTower
	case jlm.ProjGLM4V:
		return glm4vTower
	case jlm.ProjKimiVL:
		return kimivlTower
	case jlm.ProjHunyuanVL:
		return hunyuanvlTower
	case jlm.ProjJanus:
		return janusTower
	case jlm.ProjPhi4:
		return phi4Tower
	case jlm.ProjPixtral:
		return pixtralTower
	case jlm.ProjGemma4V:
		return gemma4vTower
	case jlm.ProjLFM2:
		return lfm2Tower
	}
	return nil
}

// StatesTower reports whether the converter states projector p's tower, so a
// container of p written without one is a stale conversion.
func StatesTower(p jlm.Projector) bool { return towerStater(p) != nil }

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

// gemma3Tower is Gemma 3's tower as transformers' Gemma3ForConditionalGeneration
// runs it: Gemma3ImageProcessor resizes the picture to the tower's square by
// Pillow's BILINEAR filter and normalises it; SigLIP's patches, their
// embedding and position table, its blocks; then the projector -- the
// post-norm, the patch grid average-pooled Scale x Scale, the soft-embedding
// RMS norm (its weight, as llama.cpp's converter stores it, already 1 + w)
// and the input projection. Gemma3Processor writes each picture as
// "\n\n<start_of_image>", its rows, "<end_of_image>\n\n", tokenized with the
// text beside it, and the rows attend to each other both ways.
func gemma3Tower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPosEmbd, jlm.RoleVPostNorm, jlm.RoleVProjNorm, jlm.RoleVProj} {
		if !has[r] {
			return nil, fmt.Errorf("convert: gemma3: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || v.Scale == 0 || v.PatchSize == 0 || v.ImageSize%v.PatchSize != 0 {
		return nil, fmt.Errorf("convert: gemma3: %d heads over %d, a pool of %d, patches of %d in %d",
			v.NHead, v.NEmbd, v.Scale, v.PatchSize, v.ImageSize)
	}
	if side := v.ImageSize / v.PatchSize; side%v.Scale != 0 {
		return nil, fmt.Errorf("convert: gemma3: a grid of %d patches does not pool by %d", side, v.Scale)
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
			{Kind: jlm.PreResize, Ints: []int32{int32(v.ImageSize), int32(v.ImageSize), int32(jlm.FilterPILBilinear), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		Entry: jlm.Program{NIn: 1, Out: 4, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize)},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpPosTable, Bind: []jlm.Role{jlm.RoleVPosEmbd}},
			{Code: jlm.TOpAdd, In: []uint16{2, 3}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act},
		Head: jlm.Program{NIn: 1, Out: 4, Ops: []jlm.TowerOp{
			{Code: jlm.TOpLayerNorm, In: []uint16{0}, Bind: bias(jlm.RoleVPostNorm, jlm.RoleVPostNormBias), Eps: v.Eps},
			{Code: jlm.TOpPool, In: []uint16{1}, Span: int32(v.Scale)},
			{Code: jlm.TOpRMSNorm, In: []uint16{2}, Bind: []jlm.Role{jlm.RoleVProjNorm}, Eps: v.Eps},
			{Code: jlm.TOpLinear, In: []uint16{3}, Bind: []jlm.Role{jlm.RoleVProj}},
		}},
		Span: jlm.TowerSpan{BeforeText: "\n\n<start_of_image>", AfterText: "<end_of_image>\n\n", Bidir: true},
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: gemma3: %w", err)
	}
	return t, nil
}

// llavaTower is llava's tower as transformers' LlavaForConditionalGeneration
// runs it over CLIPImageProcessor's pixels: the shorter side resized to the
// tower's square by Pillow's BICUBIC filter and the centre cropped to it,
// then normalised; CLIP's patches after a class row, the position table over
// both, the pre-norm; the blocks the file holds (llama.cpp's converter stops
// it at the feature layer, the second from last, so a post-norm is applied
// only where the file has one); then the class row dropped and the two-layer
// projector with transformers' GELU between. The processor's template
// writes "<image>\n" before a message's text, and "<image>" becomes the
// picture's rows: a template that reads a message as a string carries the
// mark, and the newline is the text after the rows.
func llavaTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVClassEmbd, jlm.RoleVPosEmbd, jlm.RoleVProj, jlm.RoleVProj2} {
		if !has[r] {
			return nil, fmt.Errorf("convert: llava: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || v.PatchSize == 0 || v.ImageSize%v.PatchSize != 0 {
		return nil, fmt.Errorf("convert: llava: %d heads over %d, patches of %d in %d", v.NHead, v.NEmbd, v.PatchSize, v.ImageSize)
	}
	var hidden int32
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProj {
			hidden = int32(t.Dims[1])
		}
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
	// The pixels are value 0.
	entry := jlm.Program{NIn: 1, Out: 5, Ops: []jlm.TowerOp{
		{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize), Offset: 1},
		{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
		{Code: jlm.TOpLeadRow, In: []uint16{2}, Bind: []jlm.Role{jlm.RoleVClassEmbd}, Offset: 1},
		{Code: jlm.TOpPosTable, Bind: []jlm.Role{jlm.RoleVPosEmbd}},
		{Code: jlm.TOpAdd, In: []uint16{3, 4}},
	}}
	if has[jlm.RoleVPreNorm] {
		entry.Ops = append(entry.Ops, jlm.TowerOp{Code: jlm.TOpLayerNorm, In: []uint16{5},
			Bind: bias(jlm.RoleVPreNorm, jlm.RoleVPreNormBias), Eps: v.Eps})
		entry.Out = 6
	}
	// The last block's rows are value 0.
	var head jlm.Program
	head.NIn = 1
	at := uint16(0)
	op := func(o jlm.TowerOp) {
		o.In = []uint16{at}
		head.Ops = append(head.Ops, o)
		at = uint16(len(head.Ops))
	}
	if has[jlm.RoleVPostNorm] {
		op(jlm.TowerOp{Code: jlm.TOpLayerNorm, Bind: bias(jlm.RoleVPostNorm, jlm.RoleVPostNormBias), Eps: v.Eps})
	}
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	op(jlm.TowerOp{Code: jlm.TOpDropRows, Offset: 1})
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)})
	op(jlm.TowerOp{Code: jlm.TOpStated, Expr: &jlm.Expr{Terms: gelu.T}, Width: hidden})
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)})
	head.Out = at
	side := int32(v.ImageSize)
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreResizeShort, Ints: []int32{side, int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreCenterCrop, Ints: []int32{side, side}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		Entry: entry,
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act},
		Head: head,
		Span: jlm.TowerSpan{Mark: "<image>\n", AfterText: "\n"},
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: llava: %w", err)
	}
	return t, nil
}

// internvlTower is InternVL's tower as its reference runs it (the model
// card's load_image: dynamic_preprocess, then build_transform): the picture
// cut into between 1 and 12 tiles of the tower's square -- resized by
// Pillow's BICUBIC filter to the grid closest to its aspect and cut
// row-major, a thumbnail of the whole after them -- each normalised;
// InternViT's patch embedding after a class row and the position table over
// both; its blocks, their layer scales folded into the matrices they follow
// at conversion (foldLayerScales), the MLP's activation transformers' exact
// GELU; then the class row dropped, the pixel shuffle (each 2x2 of the grid
// one row, the raster order its view and permutes leave), mlp1's LayerNorm
// (nn.LayerNorm's epsilon) and its two matrices with GELU between. A
// picture is "<img>", its tiles' rows, "</img>" in place of the "<image>"
// the chat writes before the question, whose newline is the text after.
func internvlTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVClassEmbd, jlm.RoleVPosEmbd, jlm.RoleVProjNorm, jlm.RoleVProj, jlm.RoleVProj2} {
		if !has[r] {
			return nil, fmt.Errorf("convert: internvl: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || v.Scale == 0 || v.PatchSize == 0 || v.ImageSize%v.PatchSize != 0 {
		return nil, fmt.Errorf("convert: internvl: %d heads over %d, a shuffle of %d, patches of %d in %d",
			v.NHead, v.NEmbd, v.Scale, v.PatchSize, v.ImageSize)
	}
	if side := v.ImageSize / v.PatchSize; side%v.Scale != 0 {
		return nil, fmt.Errorf("convert: internvl: a grid of %d patches does not shuffle by %d", side, v.Scale)
	}
	var hidden int32
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProj {
			hidden = int32(t.Dims[1])
		}
	}
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	act := jlm.Expr{Terms: gelu.T}
	bias := func(w, b jlm.Role) []jlm.Role {
		if has[b] {
			return []jlm.Role{w, b}
		}
		return []jlm.Role{w}
	}
	side := int32(v.ImageSize)
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreTile, Ints: []int32{1, 12, 1}},
			{Kind: jlm.PreResize, Ints: []int32{side, side, int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 5, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize), Offset: 1},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpLeadRow, In: []uint16{2}, Bind: []jlm.Role{jlm.RoleVClassEmbd}, Offset: 1},
			{Code: jlm.TOpPosTable, Bind: []jlm.Role{jlm.RoleVPosEmbd}},
			{Code: jlm.TOpAdd, In: []uint16{3, 4}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act},
		// The last block's rows are value 0.
		Head: jlm.Program{NIn: 1, Out: 6, Ops: []jlm.TowerOp{
			{Code: jlm.TOpDropRows, In: []uint16{0}, Offset: 1},
			{Code: jlm.TOpRegroup, In: []uint16{1}, Span: int32(v.Scale)},
			{Code: jlm.TOpLayerNorm, In: []uint16{2}, Bind: bias(jlm.RoleVProjNorm, jlm.RoleVProjNormBias), Eps: 1e-5},
			{Code: jlm.TOpLinear, In: []uint16{3}, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)},
			{Code: jlm.TOpStated, In: []uint16{4}, Expr: &jlm.Expr{Terms: gelu.T}, Width: hidden},
			{Code: jlm.TOpLinear, In: []uint16{5}, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)},
		}},
		Span: jlm.TowerSpan{Mark: "<image>\n", AfterText: "\n"},
	}
	var err error
	if t.Span.Before, err = markerIDs(s, "<img>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "</img>"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: internvl: %w", err)
	}
	return t, nil
}

// Qwen2VLImageProcessor's own bounds on a picture's area, which an mmproj
// does not state: its class's defaults (docs/divergences.md).
const (
	qwen2vlMinPixels = 56 * 56
	qwen2vlMaxPixels = 28 * 28 * 1280
)

// Qwen3-VL's bounds on a picture's area as its published
// preprocessor_config.json states them (size.shortest_edge and
// size.longest_edge), which its mmproj does not (docs/divergences.md).
const (
	qwen3vlMinPixels = 65536
	qwen3vlMaxPixels = 16777216
)

// qwen2vlTower is Qwen2-VL's tower as transformers' Qwen2VLImageProcessorPil
// and Qwen2VisionTransformerPretrainedModel run it: the picture resized by
// Pillow's BICUBIC filter to a size of its own (smart_resize: each side a
// multiple of a merge window's pixels, the area within the processor
// class's bounds) and normalised; its patches -- the conv3d's two temporal
// planes folded at conversion, the picture repeated in time -- in the
// raster order of the merge windows, through the one matrix; the blocks, q
// and k turned NEOX-paired by the rotary over the patch grid, the first half
// of the turned pairs by a patch's row and the second by its column, each
// at theta 10000 over half a head (VisionRotaryEmbedding(head_dim/2)); then
// the merger: its LayerNorm, each window's patches one row, two matrices
// with the exact GELU between. A picture is "<|vision_start|>", its rows,
// "<|vision_end|>" in place of the chat template's three, its grid the text
// model's M-RoPE grid with each side over the merge.
func qwen2vlTower(s *jlm.Source) (*jlm.Tower, error) { return qwenVLTower(s, qwen2VL) }

// qwen25vlTower is Qwen2.5-VL's tower as transformers'
// Qwen2_5_VisionTransformerPretrainedModel runs it: Qwen2-VL's picture,
// patches and rotary, its blocks RMSNorm with a gated SiLU MLP, every one
// but each n_wa_pattern'th attending inside windows of window_size pixels
// a side (get_window_index: the merge windows reordered window by window,
// the windows at the edges cut), the merger's norm an RMSNorm, and its rows
// put back in the merge windows' raster order (reverse_indices).
func qwen25vlTower(s *jlm.Source) (*jlm.Tower, error) { return qwenVLTower(s, qwen25VL) }

// qwen3vlTower is Qwen3-VL's tower, and Qwen3.5-VL's, as transformers'
// Qwen3VLVisionModel runs it: Qwen2-VL's picture under the published
// processor's bounds, its patches (a biased matrix) and rotary; a learned
// table resampled to the patch grid (fast_pos_embed_interpolate: each axis
// over linspace(0, side-1, n), the four rows around a cell weighted by its
// fractions) added to them; LayerNorm blocks with an ungated GELU-tanh MLP;
// then Qwen2-VL's merger.
func qwen3vlTower(s *jlm.Source) (*jlm.Tower, error) { return qwenVLTower(s, qwen3VL) }

// qwenVLKind is which of the Qwen VL towers qwenVLTower states.
type qwenVLKind uint8

const (
	qwen2VL qwenVLKind = iota
	qwen25VL
	qwen3VL
)

// qwenVLTower is qwen2vlTower, qwen25vlTower and qwen3vlTower.
func qwenVLTower(s *jlm.Source, kind qwenVLKind) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	windowed := kind == qwen25VL
	need := []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPostNorm, jlm.RoleVProj, jlm.RoleVProj2}
	if kind == qwen3VL {
		need = append(need, jlm.RoleVPosEmbd)
	}
	for _, r := range need {
		if !has[r] {
			return nil, fmt.Errorf("convert: qwen2vl: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || (v.NEmbd/v.NHead)%4 != 0 || v.PatchSize == 0 || v.Scale < 2 {
		return nil, fmt.Errorf("convert: qwen2vl: %d heads over %d, patches of %d merged by %d",
			v.NHead, v.NEmbd, v.PatchSize, v.Scale)
	}
	var hidden, table int32
	// A deepstack tap's block, and its merger's first matrix's rows.
	taps := map[int32]int32{}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch {
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProj:
			hidden = int32(t.Dims[1])
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVPosEmbd:
			table = int32(t.Dims[1])
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVDsFC1:
			taps[t.Index] = int32(t.Dims[1])
		}
	}
	if len(taps) > 0 && kind != qwen3VL {
		return nil, fmt.Errorf("convert: qwen2vl: deepstack taps on a tower of no deepstack")
	}
	side := int32(math.Sqrt(float64(table)))
	if kind == qwen3VL && (side < 2 || side*side != table) {
		return nil, fmt.Errorf("convert: qwen3vl: a position table of %d rows is not a square grid", table)
	}
	act, err := v.Activation(windowed)
	if err != nil {
		return nil, err
	}
	bias := func(w, b jlm.Role) []jlm.Role {
		if has[b] {
			return []jlm.Role{w, b}
		}
		return []jlm.Role{w}
	}
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	hd := int32(v.NEmbd / v.NHead)
	merge := int32(v.Scale)
	norm, mergerNorm := jlm.NormLayer, jlm.TowerOp{Code: jlm.TOpLayerNorm, In: []uint16{0},
		Bind: bias(jlm.RoleVPostNorm, jlm.RoleVPostNormBias), Eps: v.Eps}
	var window int32
	var full []int32
	if windowed {
		if v.WinPattern == 0 || v.WinSize == 0 || v.WinSize%(v.PatchSize*v.Scale) != 0 {
			return nil, fmt.Errorf("convert: qwen2.5vl: windows of %d pixels every %d blocks over patches of %d merged by %d",
				v.WinSize, v.WinPattern, v.PatchSize, v.Scale)
		}
		norm = jlm.NormRMS
		mergerNorm = jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{0}, Bind: []jlm.Role{jlm.RoleVPostNorm}, Eps: v.Eps}
		window = int32(v.WinSize / v.PatchSize)
		for i := range int32(v.NLayer) {
			if (i+1)%int32(v.WinPattern) == 0 {
				full = append(full, i)
			}
		}
	}
	minPix, maxPix := int32(qwen2vlMinPixels), int32(qwen2vlMaxPixels)
	if kind == qwen3VL {
		minPix, maxPix = qwen3vlMinPixels, qwen3vlMaxPixels
	}
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreSmartResize, Ints: []int32{int32(v.PatchSize) * merge, minPix, maxPix,
				int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 2, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize), Width: merge},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: norm, Eps: v.Eps, Act: act, Window: window, Full: full,
			Rope: jlm.TowerRope{Base: 10000, Dim: hd / 2, Neox: true,
				Runs: []jlm.TowerRopeRun{{Pairs: hd / 4, Axis: 0}, {Pairs: hd / 4, Axis: 1}}}},
		// The last block's rows are value 0.
		Head: jlm.Program{NIn: 1, Out: 5, Ops: []jlm.TowerOp{
			mergerNorm,
			{Code: jlm.TOpFold, In: []uint16{1}, Span: merge * merge},
			{Code: jlm.TOpLinear, In: []uint16{2}, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)},
			{Code: jlm.TOpStated, In: []uint16{3}, Expr: &jlm.Expr{Terms: gelu.T}, Width: hidden},
			{Code: jlm.TOpLinear, In: []uint16{4}, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)},
		}},
		Span: jlm.TowerSpan{Mark: "<|vision_start|><|image_pad|><|vision_end|>", GridDiv: merge},
	}
	if t.Span.Before, err = markerIDs(s, "<|vision_start|>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "<|vision_end|>"); err != nil {
		return nil, err
	}
	if windowed {
		t.Head.Ops = append(t.Head.Ops, jlm.TowerOp{Code: jlm.TOpUntile, In: []uint16{5}})
		t.Head.Out = 6
	}
	for _, b := range slices.Sorted(maps.Keys(taps)) {
		// The tapped block's rows, each merge window's patches one row,
		// normed over that row (use_postshuffle_norm).
		t.Taps = append(t.Taps, jlm.TowerTap{Block: b, Prog: jlm.Program{NIn: 1, Out: 5, Ops: []jlm.TowerOp{
			{Code: jlm.TOpFold, In: []uint16{0}, Span: merge * merge},
			{Code: jlm.TOpLayerNorm, In: []uint16{1}, Bind: []jlm.Role{jlm.RoleVDsNorm, jlm.RoleVDsNormBias}, Eps: v.Eps},
			{Code: jlm.TOpLinear, In: []uint16{2}, Bind: []jlm.Role{jlm.RoleVDsFC1, jlm.RoleVDsFC1Bias}},
			{Code: jlm.TOpStated, In: []uint16{3}, Expr: &jlm.Expr{Terms: gelu.T}, Width: taps[b]},
			{Code: jlm.TOpLinear, In: []uint16{4}, Bind: []jlm.Role{jlm.RoleVDsFC2, jlm.RoleVDsFC2Bias}},
		}}})
	}
	if kind == qwen3VL {
		// The table's rows at each patch's cell, in the patches' order.
		t.Entry.Ops = append(t.Entry.Ops,
			jlm.TowerOp{Code: jlm.TOpPosResample, Bind: []jlm.Role{jlm.RoleVPosEmbd}, Span: side, Width: merge,
				Chain: jlm.ResampleAligned, Filter: jlm.TapsLinear},
			jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{2, 3}})
		t.Entry.Out = 4
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: qwen2vl: %w", err)
	}
	return t, nil
}

// GLM-4V's bounds on a picture's area as its published preprocessor_config
// states them (size.shortest_edge and longest_edge, 12544 and 9633792), which
// its processor holds a still picture's two temporal frames to: per frame,
// as the tower takes them, half each (docs/divergences.md).
const (
	glm4vMinPixels = 12544 / 2
	glm4vMaxPixels = 9633792 / 2
)

// glm4vTower is GLM-4V's tower as transformers' Glm4vImageProcessorPil and
// Glm4vVisionModel run it: the picture resized by Pillow's BICUBIC filter to
// a size of its own (its smart_resize: a side shorter than a merge window
// scaled up first) and normalised; Qwen2-VL's patches (the conv3d's planes
// folded) through a biased matrix and RMSNormed (post_conv_layernorm), a
// learned table resampled bicubic to the patch grid (grid_sample,
// align_corners false, border) added in the merge windows' order; RMSNorm
// blocks with a gated SiLU MLP, turned by Qwen2-VL's rotary; the rows
// RMSNormed (post_layernorm), each merge window's patches one row through
// the 2x2 convolution as a matrix (downsample), then the merger: a matrix,
// LayerNorm, the exact GELU and a gated SiLU MLP. A picture is
// "<|begin_of_image|>", its rows, "<|end_of_image|>", its grid the text
// model's M-RoPE grid over the merge.
func glm4vTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVEmbdNorm, jlm.RoleVPosEmbd, jlm.RoleVPostNorm,
		jlm.RoleVMergeConv, jlm.RoleVProj, jlm.RoleVProjNorm, jlm.RoleVProjGate, jlm.RoleVProjUp, jlm.RoleVProjDown} {
		if !has[r] {
			return nil, fmt.Errorf("convert: glm4v: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || (v.NEmbd/v.NHead)%4 != 0 || v.PatchSize == 0 || v.Scale < 2 {
		return nil, fmt.Errorf("convert: glm4v: %d heads over %d, patches of %d merged by %d",
			v.NHead, v.NEmbd, v.PatchSize, v.Scale)
	}
	var table, ctx int32
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch {
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVPosEmbd:
			table = int32(t.Dims[1])
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProjGate:
			ctx = int32(t.Dims[1])
		}
	}
	side := int32(math.Sqrt(float64(table)))
	if side < 2 || side*side != table {
		return nil, fmt.Errorf("convert: glm4v: a position table of %d rows is not a square grid", table)
	}
	act, err := v.Activation(true)
	if err != nil {
		return nil, err
	}
	bias := func(w, b jlm.Role) []jlm.Role {
		if has[b] {
			return []jlm.Role{w, b}
		}
		return []jlm.Role{w}
	}
	var gelu, swiglu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	swiglu.Fn(jlm.FnMul, swiglu.SiLU(jlm.Operand(0)), jlm.Operand(1))
	hd := int32(v.NEmbd / v.NHead)
	merge := int32(v.Scale)
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreSmartResize, Ints: []int32{int32(v.PatchSize) * merge, glm4vMinPixels, glm4vMaxPixels,
				int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL), 1}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 5, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize), Width: merge},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpRMSNorm, In: []uint16{2}, Bind: []jlm.Role{jlm.RoleVEmbdNorm}, Eps: v.Eps},
			{Code: jlm.TOpPosResample, Bind: []jlm.Role{jlm.RoleVPosEmbd}, Span: side, Width: merge,
				Chain: jlm.ResampleHalfPixel, Filter: jlm.TapsCubic},
			{Code: jlm.TOpAdd, In: []uint16{3, 4}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormRMS, Eps: v.Eps, Act: act,
			Rope: jlm.TowerRope{Base: 10000, Dim: hd / 2, Neox: true,
				Runs: []jlm.TowerRopeRun{{Pairs: hd / 4, Axis: 0}, {Pairs: hd / 4, Axis: 1}}}},
		// The last block's rows are value 0.
		Head: jlm.Program{NIn: 1, Out: 10, Ops: []jlm.TowerOp{
			{Code: jlm.TOpRMSNorm, In: []uint16{0}, Bind: []jlm.Role{jlm.RoleVPostNorm}, Eps: v.Eps},
			{Code: jlm.TOpFold, In: []uint16{1}, Span: merge * merge},
			{Code: jlm.TOpLinear, In: []uint16{2}, Bind: bias(jlm.RoleVMergeConv, jlm.RoleVMergeConvBias)},
			{Code: jlm.TOpLinear, In: []uint16{3}, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)},
			// torch's LayerNorm at its default epsilon.
			{Code: jlm.TOpLayerNorm, In: []uint16{4}, Bind: bias(jlm.RoleVProjNorm, jlm.RoleVProjNormBias), Eps: 1e-5},
			{Code: jlm.TOpStated, In: []uint16{5}, Expr: &jlm.Expr{Terms: gelu.T}, Width: int32(v.ProjDim)},
			{Code: jlm.TOpLinear, In: []uint16{6}, Bind: bias(jlm.RoleVProjGate, jlm.RoleVProjGateBias)},
			{Code: jlm.TOpLinear, In: []uint16{6}, Bind: bias(jlm.RoleVProjUp, jlm.RoleVProjUpBias)},
			{Code: jlm.TOpStated, In: []uint16{7, 8}, Expr: &jlm.Expr{Terms: swiglu.T}, Width: ctx},
			{Code: jlm.TOpLinear, In: []uint16{9}, Bind: bias(jlm.RoleVProjDown, jlm.RoleVProjDownBias)},
		}},
		Span: jlm.TowerSpan{Mark: "<|begin_of_image|><|image|><|end_of_image|>", GridDiv: merge},
	}
	if t.Span.Before, err = markerIDs(s, "<|begin_of_image|>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "<|end_of_image|>"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: glm4v: %w", err)
	}
	return t, nil
}

// Kimi-VL's processor's in_token_limit, which its mmproj does not state
// (preprocessor_config.json).
const kimivlTokenLimit = 4096

// kimivlTower is Kimi-VL's tower as Moonshot's KimiVLImageProcessor and
// MoonViT run it: a picture of more whole patches than the processor's
// limit scaled down to about that many by Pillow's BICUBIC filter, then
// padded black on the right and at the bottom to whole merge windows
// (pad_input) and normalised; its patches in raster order through a biased
// matrix, plus a learned square table resampled bicubic to the patch grid
// (interpolate, align_corners false); LayerNorm blocks with a GELU-tanh MLP,
// q and k turned in complex pairs by the 2-D rotary over the whole head --
// pair 2j by the patch's column and 2j+1 by its row, both at theta 10000 to
// the -4j/head (Rope2DPosEmb); the final LayerNorm; then the projector: a
// LayerNorm on each patch, each 2x2 window's patches one row, a matrix, the
// exact GELU and a matrix. A picture is "<|media_start|>image<|media_content|>",
// its rows, "<|media_end|>", the rows advancing as text.
func kimivlTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPosEmbd, jlm.RoleVPostNorm, jlm.RoleVProjInNorm,
		jlm.RoleVProj, jlm.RoleVProj2} {
		if !has[r] {
			return nil, fmt.Errorf("convert: kimivl: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || (v.NEmbd/v.NHead)%4 != 0 || v.PatchSize == 0 || v.Scale < 2 {
		return nil, fmt.Errorf("convert: kimivl: %d heads over %d, patches of %d merged by %d",
			v.NHead, v.NEmbd, v.PatchSize, v.Scale)
	}
	var hidden, table int32
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch {
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProj:
			hidden = int32(t.Dims[1])
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVPosEmbd:
			table = int32(t.Dims[1])
		}
	}
	side := int32(math.Sqrt(float64(table)))
	if side < 2 || side*side != table {
		return nil, fmt.Errorf("convert: kimivl: a position table of %d rows is not a square grid", table)
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
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	hd := int32(v.NEmbd / v.NHead)
	merge := int32(v.Scale)
	var runs []jlm.TowerRopeRun
	for j := range hd / 4 {
		runs = append(runs, jlm.TowerRopeRun{Pairs: 1, Axis: 1, Freq: 2 * j}, jlm.TowerRopeRun{Pairs: 1, Axis: 0, Freq: 2 * j})
	}
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreLimitPad, Ints: []int32{int32(v.PatchSize), kimivlTokenLimit, int32(v.PatchSize) * merge,
				int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 4, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize)},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpPosResample, Bind: []jlm.Role{jlm.RoleVPosEmbd}, Span: side,
				Chain: jlm.ResampleHalfPixel, Filter: jlm.TapsCubic},
			{Code: jlm.TOpAdd, In: []uint16{2, 3}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act,
			Rope: jlm.TowerRope{Base: 10000, Dim: hd, Runs: runs}},
		// The last block's rows are value 0.
		Head: jlm.Program{NIn: 1, Out: 6, Ops: []jlm.TowerOp{
			{Code: jlm.TOpLayerNorm, In: []uint16{0}, Bind: bias(jlm.RoleVPostNorm, jlm.RoleVPostNormBias), Eps: v.Eps},
			// torch's LayerNorm at the projector's epsilon.
			{Code: jlm.TOpLayerNorm, In: []uint16{1}, Bind: bias(jlm.RoleVProjInNorm, jlm.RoleVProjInNormBias), Eps: 1e-5},
			{Code: jlm.TOpRegroup, In: []uint16{2}, Span: merge},
			{Code: jlm.TOpLinear, In: []uint16{3}, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)},
			{Code: jlm.TOpStated, In: []uint16{4}, Expr: &jlm.Expr{Terms: gelu.T}, Width: hidden},
			{Code: jlm.TOpLinear, In: []uint16{5}, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)},
		}},
		Span: jlm.TowerSpan{Mark: "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"},
	}
	if t.Span.Before, err = markerIDs(s, "<|media_start|>", "image", "<|media_content|>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "<|media_end|>"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: kimivl: %w", err)
	}
	return t, nil
}

// hunyuanvlTower is HunyuanVL's tower as transformers' HunYuanVL classes run
// it: the picture resized by Pillow's BICUBIC filter to a size of its own
// (Qwen2-VL's smart_resize, a merge window's pixels a side, the file's
// bounds on the area) and normalised; its patches in raster order -- the
// processor's, unlike Qwen-VL's merge windows -- through a biased matrix,
// plus a learned square table resampled bilinear (interpolate,
// align_corners false); LayerNorm blocks with no rotary; then the
// merger: an RMSNorm, each 2x2 raster window one row through the 2x2
// convolution as a matrix, the exact GELU, the 1x1 convolution, a newline
// row after each merged row, a linear, the begin and end rows around the
// picture and an RMSNorm -- H*(W+1)+2 rows. A picture is its start marker,
// its rows and its end marker, its grid the text model's XD-RoPE grid over
// the merge with the newline column and the two end rows.
func hunyuanvlTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPosEmbd, jlm.RoleVProjInNorm, jlm.RoleVMergeConv,
		jlm.RoleVProj, jlm.RoleVImgNewline, jlm.RoleVProj2, jlm.RoleVImgBegin, jlm.RoleVImgEnd, jlm.RoleVProjNorm} {
		if !has[r] {
			return nil, fmt.Errorf("convert: hunyuanvl: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || v.PatchSize == 0 || v.Scale < 2 || v.MinPixels == 0 || v.MaxPixels < v.MinPixels {
		return nil, fmt.Errorf("convert: hunyuanvl: %d heads over %d, patches of %d merged by %d, %d to %d pixels",
			v.NHead, v.NEmbd, v.PatchSize, v.Scale, v.MinPixels, v.MaxPixels)
	}
	var merged, table int32
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch {
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVMergeConv:
			merged = int32(t.Dims[1])
		case t.Block == jlm.DenseBlock && t.Role == jlm.RoleVPosEmbd:
			table = int32(t.Dims[1])
		}
	}
	side := int32(math.Sqrt(float64(table)))
	if side < 2 || side*side != table {
		return nil, fmt.Errorf("convert: hunyuanvl: a position table of %d rows is not a square grid", table)
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
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	merge := int32(v.Scale)
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreSmartResize, Ints: []int32{int32(v.PatchSize) * merge, int32(v.MinPixels), int32(v.MaxPixels),
				int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 4, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize)},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpPosResample, Bind: []jlm.Role{jlm.RoleVPosEmbd}, Span: side,
				Chain: jlm.ResampleHalfPixel, Filter: jlm.TapsLinear},
			{Code: jlm.TOpAdd, In: []uint16{2, 3}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act},
		// The last block's rows are value 0.
		Head: jlm.Program{NIn: 1, Out: 9, Ops: []jlm.TowerOp{
			{Code: jlm.TOpRMSNorm, In: []uint16{0}, Bind: []jlm.Role{jlm.RoleVProjInNorm}, Eps: v.Eps},
			{Code: jlm.TOpRegroup, In: []uint16{1}, Span: merge},
			{Code: jlm.TOpLinear, In: []uint16{2}, Bind: bias(jlm.RoleVMergeConv, jlm.RoleVMergeConvBias)},
			{Code: jlm.TOpStated, In: []uint16{3}, Expr: &jlm.Expr{Terms: gelu.T}, Width: merged},
			{Code: jlm.TOpLinear, In: []uint16{4}, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)},
			{Code: jlm.TOpRowBreak, In: []uint16{5}, Bind: []jlm.Role{jlm.RoleVImgNewline}, Span: merge},
			{Code: jlm.TOpLinear, In: []uint16{6}, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)},
			// after_rms before the frame: the begin and end rows were
			// normalised by it at conversion (normFrameRows).
			{Code: jlm.TOpRMSNorm, In: []uint16{7}, Bind: []jlm.Role{jlm.RoleVProjNorm}, Eps: v.Eps},
			{Code: jlm.TOpWrapRows, In: []uint16{8}, Bind: []jlm.Role{jlm.RoleVImgBegin, jlm.RoleVImgEnd}, Span: merge},
		}},
		Span: jlm.TowerSpan{Mark: "<\uff5chy_place\u2581holder\u2581no\u2581100\uff5c><\uff5chy_place\u2581holder\u2581no\u2581102\uff5c>" +
			"<\uff5chy_place\u2581holder\u2581no\u2581101\uff5c>", GridDiv: merge},
	}
	if t.Span.Before, err = markerIDs(s, "<\uff5chy_place\u2581holder\u2581no\u2581100\uff5c>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "<\uff5chy_place\u2581holder\u2581no\u2581101\uff5c>"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: hunyuanvl: %w", err)
	}
	return t, nil
}

// janusMinSide is the shorter side Janus's processor keeps at least
// (JanusImageProcessor's min_size).
const janusMinSide = 14

// janusTower is Janus-Pro's tower as transformers' JanusImageProcessorPil,
// JanusVisionModel and JanusVisionAlignerMLP run it: the picture resized by
// Pillow's BICUBIC filter so its longer side is the tower's and centred on
// the tower's square of the processor's background colour (the mean as
// bytes), then normalised; SigLIP's patches through a biased matrix plus
// the position table, its blocks and post-LayerNorm; then the aligner --
// llava's two matrices with the exact GELU between. A picture is
// "<begin_of_image>", its rows, "<end_of_image>", the template's
// "<image_placeholder>" its mark.
func janusTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPosEmbd, jlm.RoleVProj, jlm.RoleVProj2} {
		if !has[r] {
			return nil, fmt.Errorf("convert: janus: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || v.PatchSize == 0 || v.ImageSize%v.PatchSize != 0 {
		return nil, fmt.Errorf("convert: janus: %d heads over %d, patches of %d in %d", v.NHead, v.NEmbd, v.PatchSize, v.ImageSize)
	}
	var hidden int32
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProj {
			hidden = int32(t.Dims[1])
		}
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
	var fill [3]int32
	for c := range 3 {
		// transformers' tuple(int(x * 255) for x in image_mean).
		fill[c] = int32(v.Mean[c] * 255)
	}
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	var head jlm.Program
	head.NIn = 1
	at := uint16(0)
	op := func(o jlm.TowerOp) {
		o.In = []uint16{at}
		head.Ops = append(head.Ops, o)
		at = uint16(len(head.Ops))
	}
	if has[jlm.RoleVPostNorm] {
		op(jlm.TowerOp{Code: jlm.TOpLayerNorm, Bind: bias(jlm.RoleVPostNorm, jlm.RoleVPostNormBias), Eps: v.Eps})
	}
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)})
	op(jlm.TowerOp{Code: jlm.TOpStated, Expr: &jlm.Expr{Terms: gelu.T}, Width: hidden})
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)})
	head.Out = at
	side := int32(v.ImageSize)
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreLongBox, Ints: []int32{side, janusMinSide, fill[0], fill[1], fill[2],
				int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
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
		Head: head,
		Span: jlm.TowerSpan{Mark: "<image_placeholder>"},
	}
	if t.Span.Before, err = markerIDs(s, "<begin_of_image>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "<end_of_image>"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: janus: %w", err)
	}
	return t, nil
}

// phi4Tower is Phi-4-reasoning-vision's tower as its processor and
// Siglip2VisionModel run it: the picture resized by Pillow's BILINEAR filter
// to whole patches of its own count, clamped to the file's bounds
// (Siglip2ImageProcessorNoUpscale's NaFlex size), and normalised; the
// patches through a biased matrix plus the square position table resized to
// the picture's grid by torch's antialiased bilinear (resize_positional_
// embeddings); the blocks llama.cpp's converter kept, which end at
// hidden_states[-2] with no post-LayerNorm; then mlp2x_gelu, two matrices
// with the exact GELU between. The template's "<image>" is the picture's
// rows with no markers.
func phi4Tower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPosEmbd, jlm.RoleVProj, jlm.RoleVProj2} {
		if !has[r] {
			return nil, fmt.Errorf("convert: phi4: the tower has no %v", r)
		}
	}
	P := int32(v.PatchSize)
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || P == 0 || v.ImageSize%v.PatchSize != 0 ||
		v.MinPixels < uint32(P*P) || v.MaxPixels < v.MinPixels {
		return nil, fmt.Errorf("convert: phi4: %d heads over %d, patches of %d in %d, %d to %d pixels",
			v.NHead, v.NEmbd, P, v.ImageSize, v.MinPixels, v.MaxPixels)
	}
	var hidden int32
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProj {
			hidden = int32(t.Dims[1])
		}
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
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	side := int32(v.ImageSize) / P
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreNaFlex, Ints: []int32{P, int32(v.MinPixels) / (P * P), int32(v.MaxPixels) / (P * P),
				int32(jlm.FilterPILBilinear), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 4, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: P},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpPosResample, Bind: []jlm.Role{jlm.RoleVPosEmbd}, Span: side,
				Chain: jlm.ResampleHalfPixel, Filter: jlm.TapsAntialias},
			{Code: jlm.TOpAdd, In: []uint16{2, 3}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act},
		// The last block's rows are value 0.
		Head: jlm.Program{NIn: 1, Out: 3, Ops: []jlm.TowerOp{
			{Code: jlm.TOpLinear, In: []uint16{0}, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)},
			{Code: jlm.TOpStated, In: []uint16{1}, Expr: &jlm.Expr{Terms: gelu.T}, Width: hidden},
			{Code: jlm.TOpLinear, In: []uint16{2}, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)},
		}},
		Span: jlm.TowerSpan{Mark: "<image>"},
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: phi4: %w", err)
	}
	return t, nil
}

// pixtralTower is Pixtral's and Mistral 3's tower as transformers'
// PixtralImageProcessor, PixtralVisionModel and Mistral3MultiModalProjector
// run it: the picture resized by Pillow's BICUBIC filter so its longer side
// is at most the tower's, each side then rounded up to a whole merge unit,
// and normalised; the patches through the matrix and ln_pre; RMSNorm blocks
// with the gated SiLU MLP, turned by the 2-D rotary -- the first quarter of
// a head's pairs by the patch's grid row at the even frequencies of
// base^(-2i/head_dim), the next by its column at the odd; then the
// projector's RMSNorm, the merger (the converter put its columns in the
// regroup's order) where the file merges, the two matrices with the exact
// GELU between, and the [IMG_BREAK] row after each merged row but the last.
// A picture is its rows then [IMG_END], the template's "[IMG]" its mark.
func pixtralTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPostNorm, jlm.RoleVProj, jlm.RoleVProj2, jlm.RoleVImgBreak} {
		if !has[r] {
			return nil, fmt.Errorf("convert: pixtral: the tower has no %v", r)
		}
	}
	merge := int32(max(v.Scale, 1))
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || v.PatchSize == 0 || v.ImageSize == 0 || (v.NEmbd/v.NHead)%4 != 0 ||
		merge > 1 && !has[jlm.RoleVMerge] {
		return nil, fmt.Errorf("convert: pixtral: %d heads over %d, patches of %d up to %d, a merge of %d",
			v.NHead, v.NEmbd, v.PatchSize, v.ImageSize, merge)
	}
	var hidden int32
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProj {
			hidden = int32(t.Dims[1])
		}
	}
	act, err := v.Activation(true)
	if err != nil {
		return nil, err
	}
	bias := func(w, b jlm.Role) []jlm.Role {
		if has[b] {
			return []jlm.Role{w, b}
		}
		return []jlm.Role{w}
	}
	hd := int32(v.NEmbd / v.NHead)
	var runs []jlm.TowerRopeRun
	for j := range hd / 4 {
		runs = append(runs, jlm.TowerRopeRun{Pairs: 1, Axis: 0, Freq: 2 * j})
	}
	for j := range hd / 4 {
		runs = append(runs, jlm.TowerRopeRun{Pairs: 1, Axis: 1, Freq: 2*j + 1})
	}
	entry := jlm.Program{NIn: 1, Out: 2, Ops: []jlm.TowerOp{
		{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize)},
		{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
	}}
	if has[jlm.RoleVPreNorm] {
		entry.Ops = append(entry.Ops, jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{2}, Bind: []jlm.Role{jlm.RoleVPreNorm}, Eps: v.Eps})
		entry.Out = 3
	}
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	var head jlm.Program
	head.NIn = 1
	at := uint16(0)
	op := func(o jlm.TowerOp) {
		o.In = []uint16{at}
		head.Ops = append(head.Ops, o)
		at = uint16(len(head.Ops))
	}
	op(jlm.TowerOp{Code: jlm.TOpRMSNorm, Bind: []jlm.Role{jlm.RoleVPostNorm}, Eps: v.Eps})
	if merge > 1 {
		op(jlm.TowerOp{Code: jlm.TOpRegroup, Span: merge})
		op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: []jlm.Role{jlm.RoleVMerge}})
	}
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)})
	op(jlm.TowerOp{Code: jlm.TOpStated, Expr: &jlm.Expr{Terms: gelu.T}, Width: hidden})
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)})
	op(jlm.TowerOp{Code: jlm.TOpRowBreak, Bind: []jlm.Role{jlm.RoleVImgBreak}, Span: merge, Offset: 1})
	head.Out = at
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreCapUnit, Ints: []int32{int32(v.ImageSize), int32(v.PatchSize) * merge,
				int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		Entry: entry,
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormRMS, Eps: v.Eps, Act: act,
			Rope: jlm.TowerRope{Base: 10000, Dim: hd, Runs: runs}},
		Head: head,
		Span: jlm.TowerSpan{Mark: "[IMG]"},
	}
	if t.Span.After, err = markerIDs(s, "[IMG_END]"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: pixtral: %w", err)
	}
	return t, nil
}

// gemma4SoftTokens is Gemma 4's soft-token budget (max_soft_tokens): the
// mmproj does not state it, every published processor_config does.
const gemma4SoftTokens = 280

// gemma4vTower is Gemma 4's tower as transformers' Gemma4ImageProcessor and
// Gemma4VisionModel run it: the picture resized by Pillow's BICUBIC filter to
// the largest size of its aspect within the soft-token budget's patches,
// each side whole pool groups, and normalised (2x - 1); the patches through
// the matrix plus two learned tables, the column's row of the first and the
// row's of the second; RMSNorm blocks with sandwich norms, per-head q and k
// norms and a weightless v norm, scores at scale one, the gated GELU MLP,
// every clipped linear's input and output held to its bounds, turned by the
// 2-D rotary -- NEOX pairs over the head the converter laid out, the
// column's quarter first then the row's, each from base^0 over half the
// head at base 100; then the Scale x Scale average pool, times
// sqrt(NEmbd), the standardisation where the checkpoint has it (the
// converter stores its bias negated, the shift it adds), an RMSNorm with no
// weight and the projection. A picture is <|image>, its rows, <image|>, the
// template's <|image|> its mark.
func gemma4vTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPosEmbd, jlm.RoleVProj} {
		if !has[r] {
			return nil, fmt.Errorf("convert: gemma4v: the tower has no %v", r)
		}
	}
	if has[jlm.RoleVStdBias] != has[jlm.RoleVStdScale] {
		return nil, fmt.Errorf("convert: gemma4v: a standardisation of a bias %v and a scale %v", has[jlm.RoleVStdBias], has[jlm.RoleVStdScale])
	}
	merge := int32(v.Scale)
	hd := int32(v.NEmbd / max(v.NHead, 1))
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || hd%4 != 0 || v.PatchSize == 0 || merge < 1 {
		return nil, fmt.Errorf("convert: gemma4v: %d heads over %d, patches of %d, a pool of %d", v.NHead, v.NEmbd, v.PatchSize, merge)
	}
	// The tables are one [NEmbd, length, 2] tensor: the column's, then the
	// row's.
	var table int32
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && t.Role == jlm.RoleVPosEmbd {
			if t.NDim != 3 || t.Dims[2] != 2 {
				return nil, fmt.Errorf("convert: gemma4v: position tables of shape %v, not two of one length", t.Dims[:t.NDim])
			}
			table = int32(t.Dims[1])
		}
	}
	if table < 1 {
		return nil, fmt.Errorf("convert: gemma4v: the tower has no position tables")
	}
	act, err := v.Activation(true)
	if err != nil {
		return nil, err
	}
	bias := func(w, b jlm.Role) []jlm.Role {
		if has[b] {
			return []jlm.Role{w, b}
		}
		return []jlm.Role{w}
	}
	var scale jlm.ExprBuilder
	scale.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(float32(math.Sqrt(float64(v.NEmbd)))))
	var head jlm.Program
	head.NIn = 1
	at := uint16(0)
	op := func(o jlm.TowerOp) {
		o.In = []uint16{at}
		head.Ops = append(head.Ops, o)
		at = uint16(len(head.Ops))
	}
	op(jlm.TowerOp{Code: jlm.TOpPool, Span: merge})
	op(jlm.TowerOp{Code: jlm.TOpStated, Expr: &jlm.Expr{Terms: scale.T}})
	if has[jlm.RoleVStdBias] {
		op(jlm.TowerOp{Code: jlm.TOpShift, Bind: []jlm.Role{jlm.RoleVStdBias}})
		op(jlm.TowerOp{Code: jlm.TOpScale, Bind: []jlm.Role{jlm.RoleVStdScale}})
	}
	op(jlm.TowerOp{Code: jlm.TOpRMSNorm, Eps: v.Eps})
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)})
	head.Out = at
	P := int32(v.PatchSize)
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreAspectBudget, Ints: []int32{gemma4SoftTokens * merge * merge, P, P * merge,
				int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 4, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: P},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpPosAxes, Bind: []jlm.Role{jlm.RoleVPosEmbd}, Span: table},
			{Code: jlm.TOpAdd, In: []uint16{2, 3}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: uint32(hd), Norm: jlm.NormRMS, Eps: v.Eps,
			AttnScale: 1, Act: act, VNorm: true,
			Rope: jlm.TowerRope{Base: 100, Dim: hd / 2, Neox: true,
				Runs: []jlm.TowerRopeRun{{Pairs: hd / 4, Axis: 1}, {Pairs: hd / 4, Axis: 0}}}},
		Head: head,
		Span: jlm.TowerSpan{Mark: "<|image|>"},
	}
	if t.Span.Before, err = markerIDs(s, "<|image>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "<image|>"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: gemma4v: %w", err)
	}
	return t, nil
}

// lfm2Tower is LFM2-VL's tower (and d1's) as transformers'
// Lfm2VlImageProcessor, Siglip2VisionModel and Lfm2VlMultiModalProjector run
// it: a picture whose sides, rounded to whole merge units, hold more than
// the file's tolerance times its piece's pixel bound cut into the grid of
// tiles closest to its aspect, resized onto the grid and cut row-major, then
// the whole at its own size as a thumbnail; any other picture one piece at
// that size (each side rounded half to even to a merge unit and scaled into
// the file's bounds); resized by the file's filter as torchvision resizes an
// 8-bit picture, and normalised. Each
// piece runs SigLIP2's NaFlex tower -- the patches through a biased matrix
// plus the square position table resized to the piece's grid by torch's
// antialiased bilinear, LayerNorm blocks, the post-LayerNorm -- then the
// projector: the Scale x Scale unshuffle, the LayerNorm where the file has
// one, two matrices with the exact GELU between. A tiled picture's pieces
// each follow their marker, <|img_row_R_col_C|> for a tile and
// <|img_thumbnail|> for the thumbnail, rows the converter copies from the
// text table (tileMarks); the picture is <|image_start|>, its pieces,
// <|image_end|>, the template's <image> its mark.
func lfm2Tower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPosEmbd, jlm.RoleVPostNorm, jlm.RoleVProj, jlm.RoleVProj2} {
		if !has[r] {
			return nil, fmt.Errorf("convert: lfm2: the tower has no %v", r)
		}
	}
	P, merge := int32(v.PatchSize), int32(v.Scale)
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || P == 0 || merge < 1 || v.ImageSize%v.PatchSize != 0 ||
		v.TileSize == 0 || v.MinTiles < 1 || v.MaxTiles < v.MinTiles || !(v.Tolerance > 0) {
		return nil, fmt.Errorf("convert: lfm2: %d heads over %d, patches of %d merged by %d, tiles of %d (%d to %d past %g)",
			v.NHead, v.NEmbd, P, merge, v.TileSize, v.MinTiles, v.MaxTiles, v.Tolerance)
	}
	var hidden int32
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Block == jlm.DenseBlock && t.Role == jlm.RoleVProj {
			hidden = int32(t.Dims[1])
		}
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
	span := int32(v.MaxTiles)
	if err := tileMarks(s, int(span)); err != nil {
		return nil, err
	}
	filter := jlm.FilterPILBilinear
	if v.Flags.Has(jlm.FlagVisionBicubic) {
		filter = jlm.FilterPILBicubic
	}
	var gelu jlm.ExprBuilder
	gelu.GELUErf(jlm.Operand(0))
	var head jlm.Program
	head.NIn = 1
	at := uint16(0)
	op := func(o jlm.TowerOp) {
		o.In = []uint16{at}
		head.Ops = append(head.Ops, o)
		at = uint16(len(head.Ops))
	}
	op(jlm.TowerOp{Code: jlm.TOpLayerNorm, Bind: bias(jlm.RoleVPostNorm, jlm.RoleVPostNormBias), Eps: v.Eps})
	op(jlm.TowerOp{Code: jlm.TOpRegroup, Span: merge})
	if has[jlm.RoleVProjNorm] {
		// torch's LayerNorm at its default epsilon.
		op(jlm.TowerOp{Code: jlm.TOpLayerNorm, Bind: bias(jlm.RoleVProjNorm, jlm.RoleVProjNormBias), Eps: 1e-5})
	}
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj, jlm.RoleVProjBias)})
	op(jlm.TowerOp{Code: jlm.TOpStated, Expr: &jlm.Expr{Terms: gelu.T}, Width: hidden})
	op(jlm.TowerOp{Code: jlm.TOpLinear, Bind: bias(jlm.RoleVProj2, jlm.RoleVProj2Bias)})
	op(jlm.TowerOp{Code: jlm.TOpMarkRow, Bind: []jlm.Role{jlm.RoleVTileMarks}, Span: span, Width: merge})
	head.Out = at
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreTileSmart, Ints: []int32{int32(v.TileSize), int32(v.MinTiles), int32(v.MaxTiles), P, P * merge,
				int32(v.MinPixels), int32(v.MaxPixels), int32(filter), int32(jlm.PrecTorch)}, Floats: []float32{v.Tolerance}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 4, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: P},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
			{Code: jlm.TOpPosResample, Bind: []jlm.Role{jlm.RoleVPosEmbd}, Span: int32(v.ImageSize) / P,
				Chain: jlm.ResampleHalfPixel, Filter: jlm.TapsAntialias},
			{Code: jlm.TOpAdd, In: []uint16{2, 3}},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act},
		Head: head,
		Span: jlm.TowerSpan{Mark: "<image>"},
	}
	if t.Span.Before, err = markerIDs(s, "<|image_start|>"); err != nil {
		return nil, err
	}
	if t.Span.After, err = markerIDs(s, "<|image_end|>"); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: lfm2: %w", err)
	}
	return t, nil
}

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
func qwen2vlTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	has := towerRoles(s)
	for _, r := range []jlm.Role{jlm.RoleVPatchEmbd, jlm.RoleVPostNorm, jlm.RoleVProj, jlm.RoleVProj2} {
		if !has[r] {
			return nil, fmt.Errorf("convert: qwen2vl: the tower has no %v", r)
		}
	}
	if v.NHead == 0 || v.NEmbd%v.NHead != 0 || (v.NEmbd/v.NHead)%4 != 0 || v.PatchSize == 0 || v.Scale < 2 {
		return nil, fmt.Errorf("convert: qwen2vl: %d heads over %d, patches of %d merged by %d",
			v.NHead, v.NEmbd, v.PatchSize, v.Scale)
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
	hd := int32(v.NEmbd / v.NHead)
	merge := int32(v.Scale)
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreSmartResize, Ints: []int32{int32(v.PatchSize) * merge, qwen2vlMinPixels, qwen2vlMaxPixels,
				int32(jlm.FilterPILBicubic), int32(jlm.PrecPIL)}},
			{Kind: jlm.PreNormalize, Floats: []float32{v.Mean[0], v.Mean[1], v.Mean[2], v.Std[0], v.Std[1], v.Std[2]}},
		},
		// The pixels are value 0.
		Entry: jlm.Program{NIn: 1, Out: 2, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: int32(v.PatchSize), Width: merge},
			{Code: jlm.TOpLinear, In: []uint16{1}, Bind: bias(jlm.RoleVPatchEmbd, jlm.RoleVPatchBias)},
		}},
		Block: jlm.TowerBlock{NHead: v.NHead, NKVHead: v.NHead, HeadDim: v.NEmbd / v.NHead,
			Norm: jlm.NormLayer, Eps: v.Eps, Act: act,
			Rope: jlm.TowerRope{Base: 10000, Dim: hd / 2, Neox: true,
				Runs: []jlm.TowerRopeRun{{Pairs: hd / 4, Axis: 0}, {Pairs: hd / 4, Axis: 1}}}},
		// The last block's rows are value 0.
		Head: jlm.Program{NIn: 1, Out: 5, Ops: []jlm.TowerOp{
			{Code: jlm.TOpLayerNorm, In: []uint16{0}, Bind: bias(jlm.RoleVPostNorm, jlm.RoleVPostNormBias), Eps: v.Eps},
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
	if err := t.Valid(); err != nil {
		return nil, fmt.Errorf("convert: qwen2vl: %w", err)
	}
	return t, nil
}

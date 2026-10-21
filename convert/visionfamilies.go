package convert

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The per-projector half of the tower converter: what gemma3 and InternVL ship
// that the shared name table and the shared transforms do not cover. Every
// transform here is exact, or states the one rounding it adds.

// projectorRole is a projector's own tensor vocabulary, consulted before the
// shared table because one position means different things in different
// projectors: InternVL's mm.model.mlp.0 is a LayerNorm over the shuffled row,
// where llava's mm.0 is a matrix.
func projectorRole(p jlm.Projector, name string) (jlm.Role, bool) {
	if p == jlm.ProjPixtral {
		// Pixtral's MLP is mm.1 and mm.2, the positions llava's are mm.0 and
		// mm.2. Mistral 3's input norm is an RMSNorm over each patch at the
		// tower's width before the merge, which is what a tower's post-norm is.
		r, ok := map[string]jlm.Role{
			"mm.1.weight":            jlm.RoleVProj,
			"mm.1.bias":              jlm.RoleVProjBias,
			"mm.input_norm.weight":   jlm.RoleVPostNorm,
			"mm.patch_merger.weight": jlm.RoleVMerge,
			"v.token_embd.img_break": jlm.RoleVImgBreak,
		}[name]
		return r, ok
	}
	if p == jlm.ProjLlama4 {
		// The adapter's two matrices and the projector's: mlp.1 and mlp.2 are
		// the positions llava's mm.0 and mm.2 hold, and fc is the third.
		r, ok := map[string]jlm.Role{
			"mm.model.mlp.1.weight": jlm.RoleVProj,
			"mm.model.mlp.2.weight": jlm.RoleVProj2,
			"mm.model.fc.weight":    jlm.RoleVProj3,
		}[name]
		return r, ok
	}
	if p == jlm.ProjLFM2 {
		// LFM2-VL's optional LayerNorm over the unshuffled row: InternVL's
		// norm, in the position InternVL's has.
		r, ok := map[string]jlm.Role{
			"mm.input_norm.weight": jlm.RoleVProjNorm,
			"mm.input_norm.bias":   jlm.RoleVProjNormBias,
		}[name]
		return r, ok
	}
	if p != jlm.ProjInternVL {
		return 0, false
	}
	r, ok := map[string]jlm.Role{
		"mm.model.mlp.0.weight": jlm.RoleVProjNorm,
		"mm.model.mlp.0.bias":   jlm.RoleVProjNormBias,
		"mm.model.mlp.1.weight": jlm.RoleVProj,
		"mm.model.mlp.1.bias":   jlm.RoleVProjBias,
		"mm.model.mlp.3.weight": jlm.RoleVProj2,
		"mm.model.mlp.3.bias":   jlm.RoleVProj2Bias,
	}[name]
	return r, ok
}

// layerScaleOf parses InternViT's per-block layer scale, "v.blk.N.ls1.weight"
// (after attention) or "...ls2.weight" (after the MLP). It has no role: the
// converter folds it into the matrix whose output it scales (foldLayerScales).
func layerScaleOf(name string) (block int32, which int, ok bool) {
	rest, found := strings.CutPrefix(name, "v.blk.")
	if !found {
		return 0, 0, false
	}
	i := strings.IndexByte(rest, '.')
	if i <= 0 {
		return 0, 0, false
	}
	n, err := strconv.Atoi(rest[:i])
	if err != nil || n < 0 {
		return 0, 0, false
	}
	switch rest[i+1:] {
	case "ls1.weight":
		return int32(n), 1, true
	case "ls2.weight":
		return int32(n), 2, true
	}
	return 0, 0, false
}

// floatRows decodes a 2-D tensor to f32, k values per row.
func floatRows(t *jlm.Tensor) (vals []float32, k, rows int, err error) {
	if t.NDim != 2 {
		return nil, 0, 0, fmt.Errorf("convert: %s is %d-dimensional, want a matrix", t.Name, t.NDim)
	}
	src, ok := sourceType(t.Type)
	if !ok {
		return nil, 0, 0, fmt.Errorf("convert: %s is %v, which has no source type", t.Name, t.Type)
	}
	k, rows = int(t.Dims[0]), int(t.Dims[1])
	b, err := t.Bytes()
	if err != nil {
		return nil, 0, 0, err
	}
	vals = make([]float32, k*rows)
	if err := quant.Dequant32(src, b, vals); err != nil {
		return nil, 0, 0, fmt.Errorf("convert: %s: %w", t.Name, err)
	}
	return vals, k, rows, nil
}

// setF32Dims is setF32 with a new shape. The tower's matrices are quantized
// afterwards by quantizeTower, whose one lossy step is then the only one.
func setF32Dims(t *jlm.Tensor, vals []float32, dims ...uint64) {
	setF32(t, vals)
	t.NDim = uint8(len(dims))
	t.Dims = [4]uint64{}
	copy(t.Dims[:], dims)
}

// vecOf decodes a vector to f32.
func vecOf(t *jlm.Tensor) ([]float32, error) { return normF32(t) }

// foldLayerScales folds InternViT's layer scales into the matrices they
// follow. ls1 multiplies the attention output and ls2 the MLP output, each per
// output channel, so diag(ls)(Wx + b) = (diag(ls)W)x + ls*b: row r of the
// matrix and element r of its bias are scaled by ls[r]. The block the engine
// and every device tier run is then CLIP's, unchanged.
//
// The scale is applied in f32 before quantizeTower rounds the matrix; on a
// Q8_0 source that only rescales each block's f16 scale, since the codes are
// a row's shape and the row is scaled as a whole.
//
// The MLP's contraction is found by SHAPE (rows == NEmbd, k == NFFN), as
// buildTower finds it: InternVL's mmproj names it ffn_down, SmolVLM's names
// the same position ffn_up.
func foldLayerScales(s *jlm.Source, scales map[int32][2][]float32) error {
	if len(scales) == 0 {
		return nil
	}
	v := s.Vision
	type key struct {
		role  jlm.Role
		block int32
	}
	idx := map[key]int{}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Block >= 0 {
			idx[key{t.Role, t.Block}] = i
		}
	}
	scale := func(b int32, mat, bias jlm.Role, ls []float32) error {
		mi, ok := idx[key{mat, b}]
		if !ok {
			return fmt.Errorf("convert: block %d has a layer scale and no %v to fold it into", b, mat)
		}
		m := &s.Tensors[mi]
		vals, k, rows, err := floatRows(m)
		if err != nil {
			return err
		}
		if rows != len(ls) {
			return fmt.Errorf("convert: %s has %d rows and its layer scale %d", m.Name, rows, len(ls))
		}
		for r := 0; r < rows; r++ {
			for j := 0; j < k; j++ {
				vals[r*k+j] *= ls[r]
			}
		}
		setF32Dims(m, vals, uint64(k), uint64(rows))
		bi, ok := idx[key{bias, b}]
		if !ok {
			return nil // an unbiased matrix: nothing more to scale
		}
		bt := &s.Tensors[bi]
		bv, err := vecOf(bt)
		if err != nil {
			return err
		}
		if len(bv) != len(ls) {
			return fmt.Errorf("convert: %s is %d floats and its layer scale %d", bt.Name, len(bv), len(ls))
		}
		for r := range bv {
			bv[r] *= ls[r]
		}
		setF32Dims(bt, bv, uint64(len(bv)))
		return nil
	}
	for b, pair := range scales {
		if b < 0 || uint32(b) >= v.NLayer {
			return fmt.Errorf("convert: a layer scale for block %d of %d", b, v.NLayer)
		}
		if pair[0] == nil || pair[1] == nil {
			return fmt.Errorf("convert: block %d carries one layer scale of two", b)
		}
		if err := scale(b, jlm.RoleVAttnOut, jlm.RoleVAttnOutBias, pair[0]); err != nil {
			return err
		}
		// The contraction, by shape.
		mat, bias := jlm.RoleVFC2, jlm.RoleVFC2Bias
		if i, ok := idx[key{jlm.RoleVFC1, b}]; ok && s.Tensors[i].NDim == 2 &&
			uint32(s.Tensors[i].Dims[1]) == v.NEmbd && uint32(s.Tensors[i].Dims[0]) == v.NFFN {
			mat, bias = jlm.RoleVFC1, jlm.RoleVFC1Bias
		}
		if err := scale(b, mat, bias, pair[1]); err != nil {
			return err
		}
	}
	return nil
}

// transposeGemma3Projection lays gemma3's input projection out as every other
// matrix in the container is: k first. The source stores it as HF does,
// [n_embd_text, n_embd_vision] in ggml order, because the reference computes
// x @ W; llama.cpp transposes it inside its graph on every image.
func transposeGemma3Projection(s *jlm.Source) error {
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role != jlm.RoleVProj {
			continue
		}
		vals, k, rows, err := floatRows(t)
		if err != nil {
			return err
		}
		// k is the text width here and rows the tower's; the transpose swaps them.
		if uint32(k) != s.Vision.ProjDim || uint32(rows) != s.Vision.NEmbd {
			return fmt.Errorf("convert: gemma3's input projection is %dx%d, want %d (text) x %d (tower)",
				k, rows, s.Vision.ProjDim, s.Vision.NEmbd)
		}
		out := make([]float32, len(vals))
		for r := 0; r < rows; r++ {
			for j := 0; j < k; j++ {
				out[j*rows+r] = vals[r*k+j]
			}
		}
		setF32Dims(t, out, uint64(rows), uint64(k))
		return nil
	}
	return fmt.Errorf("convert: gemma3 projector has no mm.input_projection.weight")
}

// permuteMerger lays Mistral 3's patch merger out in the order the engine's
// shuffle hands it a group. The reference unfolds each Scale x Scale block of
// the patch grid with torch's unfold (llama.cpp: im2col), which is
// CHANNEL-major -- column c*S^2 + dy*S + dx reads channel c of patch (dy, dx)
// -- where the shuffle concatenates whole patch rows, dy outer and dx inner:
// column (dy*S + dx)*NEmbd + c. A column permutation of the matrix is the
// same function on the shuffle's rows, exactly.
func permuteMerger(s *jlm.Source) error {
	v := s.Vision
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role != jlm.RoleVMerge {
			continue
		}
		vals, k, rows, err := floatRows(t)
		if err != nil {
			return err
		}
		sq := int(v.Scale * v.Scale)
		d := int(v.NEmbd)
		if k != d*sq || rows != d {
			return fmt.Errorf("convert: %s is %dx%d, want %d rows of %d (%d patches of %d)",
				t.Name, k, rows, d, d*sq, sq, d)
		}
		out := make([]float32, len(vals))
		for r := 0; r < rows; r++ {
			src, dst := vals[r*k:(r+1)*k], out[r*k:(r+1)*k]
			for c := 0; c < d; c++ {
				for p := 0; p < sq; p++ {
					dst[p*d+c] = src[c*sq+p]
				}
			}
		}
		setF32Dims(t, out, uint64(k), uint64(rows))
	}
	return nil
}

// imgBreakRow gives a Pixtral tower its [IMG_BREAK] row when the mmproj does
// not carry one. llama.cpp's converter copies the token's embedding into the
// mmproj only from a file whose table it can see: from a transformers
// checkpoint it skips every language_model tensor first, so the row is
// absent. The container holds the text model too, so the row is the text
// table's own, read once here: the same row the reference looks up for the
// token, at the precision the text file keeps it.
func imgBreakRow(s *jlm.Source) error {
	for i := range s.Tensors {
		if s.Tensors[i].Role == jlm.RoleVImgBreak {
			return nil
		}
	}
	id := -1
	if s.Vocab != nil {
		for i := range s.Vocab.Tokens.Len() {
			if s.Vocab.Tokens.At(i) == "[IMG_BREAK]" {
				id = i
				break
			}
		}
	}
	if id < 0 {
		return fmt.Errorf("convert: a pixtral tower and no [IMG_BREAK] row: the mmproj carries no " +
			"v.token_embd.img_break and the text vocabulary has no [IMG_BREAK]")
	}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role != jlm.RoleTokenEmbd {
			continue
		}
		src, ok := sourceType(t.Type)
		if !ok || t.NDim != 2 || uint64(id) >= t.Dims[1] {
			return fmt.Errorf("convert: %s cannot give token %d's row", t.Name, id)
		}
		k := int(t.Dims[0])
		be, bb := int(src.BlockElems()), int(src.BlockBytes())
		if k%be != 0 {
			return fmt.Errorf("convert: %s rows are %d wide, not whole %v blocks", t.Name, k, src)
		}
		rb := k / be * bb
		b, err := t.Bytes()
		if err != nil {
			return err
		}
		row := make([]float32, k)
		if err := quant.Dequant32(src, b[id*rb:(id+1)*rb], row); err != nil {
			return fmt.Errorf("convert: %s row %d: %w", t.Name, id, err)
		}
		e := jlm.Tensor{Role: jlm.RoleVImgBreak, Block: jlm.DenseBlock, Index: -1,
			Name: "v.token_embd.img_break"}
		setF32Dims(&e, row, uint64(k))
		s.Tensors = append(s.Tensors, e)
		return nil
	}
	return fmt.Errorf("convert: a pixtral tower and no text token table to take [IMG_BREAK]'s row from")
}

// padFFN widens a tower's MLP to a whole quantization block when its width is
// not one: SigLIP-so400m's is 4304, 16 short of a block, and Qwen2.5-VL's
// 3420 (the published mmproj keeps that contraction F16 for the reason). The
// expansions -- the up projection, and a gated MLP's gate -- gain
// zero rows and a zero bias, so their extra outputs are exactly zero, every
// activation the tower uses maps zero to zero, and the contraction's zero
// columns then add exactly nothing. The result is the same function with a
// width every packed kernel, host and device, can read.
func padFFN(s *jlm.Source) error {
	v := s.Vision
	if v.NFFN%q8Elems == 0 {
		return nil
	}
	wide := (v.NFFN + q8Elems - 1) / q8Elems * q8Elems
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch t.Role {
		case jlm.RoleVFC1, jlm.RoleVFC2, jlm.RoleVFFNGate:
			vals, k, rows, err := floatRows(t)
			if err != nil {
				return err
			}
			switch {
			case uint32(rows) == v.NFFN && uint32(k) == v.NEmbd:
				// The expansion: zero rows appended.
				out := make([]float32, int(wide)*k)
				copy(out, vals)
				setF32Dims(t, out, uint64(k), uint64(wide))
			case uint32(k) == v.NFFN && uint32(rows) == v.NEmbd:
				// The contraction: zero columns appended to every row.
				out := make([]float32, int(wide)*rows)
				for r := 0; r < rows; r++ {
					copy(out[r*int(wide):], vals[r*k:(r+1)*k])
				}
				setF32Dims(t, out, uint64(wide), uint64(rows))
			default:
				return fmt.Errorf("convert: %s is %dx%d, neither NEmbd %d -> NFFN %d nor back",
					t.Name, k, rows, v.NEmbd, v.NFFN)
			}
		case jlm.RoleVFC1Bias, jlm.RoleVFC2Bias, jlm.RoleVFFNGateBias:
			bv, err := vecOf(t)
			if err != nil {
				return err
			}
			if uint32(len(bv)) == v.NFFN {
				out := make([]float32, wide)
				copy(out, bv)
				setF32Dims(t, out, uint64(wide))
			}
		}
	}
	v.NFFN = wide
	return nil
}

// towerRoleOf is the role a tower file's tensor takes: a projector's own
// names first, then the name table after towerName's renames. sourceOf and
// the role gate both read it, so the gate sees the names conversion sees.
func towerRoleOf(v *jlm.Vision, name string) (jlm.Role, int32, int32, error) {
	if r, ok := projectorRole(v.Projector, name); ok {
		return r, jlm.DenseBlock, -1, nil
	}
	return identify(towerName(v, name))
}

// towerUnread reports a tower file's tensor that no graph here reads: the
// unreadTower list, and Gemma 4's E-models' audio encoder, which ships in the
// same mmproj (a.*, and its projection mm.a.*) -- a second modality nothing
// of the vision or text path reads.
func towerUnread(v *jlm.Vision, name string) bool {
	if _, ok := unreadTower[name]; ok {
		return true
	}
	switch v.Projector {
	case jlm.ProjGemma4V:
		return strings.HasPrefix(name, "a.") || strings.HasPrefix(name, "mm.a.")
	case jlm.ProjGemma3nV:
		// Gemma 3n's mmproj carries its audio encoder too.
		return strings.HasPrefix(name, "a.") || strings.HasPrefix(name, "mm.a.")
	}
	return false
}

// towerGathered reports a tower file's tensor that has no role of its own
// because conversion gathers it into another: Gemma 4's clipped-linear
// bounds (RoleVClamp) and InternViT's layer scales (folded into the matrices
// they follow).
func towerGathered(name string) bool {
	if _, _, ok := clampOf(name); ok {
		return true
	}
	_, _, ok := layerScaleOf(name)
	return ok
}

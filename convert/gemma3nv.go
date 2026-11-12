package convert

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// Gemma 3n's MobileNet-V5 tower (jlm.ProjGemma3nV), converter half.
//
// llama.cpp's mmproj names a block by its stage and its place in the stage,
// "v.blk.S.I.*"; the container's blocks are one index, stage by stage
// (towerNames), each carrying its place in a RoleVMNGeom vector. Every
// convolution weight is laid out tap-major (ky, kx, c), which is the order
// the engine's im2col and depthwise rows read: a full or pointwise one as a
// matrix of k = K*K*Cin, a depthwise filter as K*K rows of channels. A block's
// layer scale is folded into the norm or the matrix it follows.

func gemma3nvRoleOf(name string) (jlm.Role, bool) {
	switch name {
	case "v.conv_stem.conv.weight":
		return jlm.RoleVMNStem, true
	case "v.conv_stem.conv.bias":
		return jlm.RoleVMNStemBias, true
	case "v.conv_stem.bn.weight":
		return jlm.RoleVMNStemNorm, true
	case "v.conv_exp.weight":
		return jlm.RoleVMNConvExp, true
	case "v.bn1.weight":
		return jlm.RoleVMNNorm1, true
	case "v.conv_pwl.weight":
		return jlm.RoleVMNConvPwl, true
	case "v.bn2.weight":
		return jlm.RoleVMNNorm2, true
	case "v.dw_start.conv.weight":
		return jlm.RoleVMNDwStart, true
	case "v.dw_start.bn.weight":
		return jlm.RoleVMNDwStartNorm, true
	case "v.pw_exp.conv.weight":
		return jlm.RoleVMNPwExp, true
	case "v.pw_exp.bn.weight":
		return jlm.RoleVMNPwExpNorm, true
	case "v.dw_mid.conv.weight":
		return jlm.RoleVMNDwMid, true
	case "v.dw_mid.bn.weight":
		return jlm.RoleVMNDwMidNorm, true
	case "v.pw_proj.conv.weight":
		return jlm.RoleVMNPwProj, true
	case "v.pw_proj.bn.weight":
		return jlm.RoleVMNPwProjNorm, true
	case "v.layer_scale.gamma":
		return jlm.RoleVMNLayerScale, true
	case "v.norm.weight":
		return jlm.RoleVAttnNorm, true
	case "v.attn.query.proj.weight":
		return jlm.RoleVAttnQ, true
	case "v.attn.key.proj.weight":
		return jlm.RoleVAttnK, true
	case "v.attn.value.proj.weight":
		return jlm.RoleVAttnV, true
	case "v.attn.output.proj.weight":
		return jlm.RoleVAttnOut, true
	case "v.attn.key.down_conv.weight":
		return jlm.RoleVMNKDown, true
	case "v.attn.key.norm.weight":
		return jlm.RoleVMNKDownNorm, true
	case "v.attn.value.down_conv.weight":
		return jlm.RoleVMNVDown, true
	case "v.attn.value.norm.weight":
		return jlm.RoleVMNVDownNorm, true
	case "v.msfa.ffn.pw_exp.conv.weight":
		return jlm.RoleVMNFusionExp, true
	case "v.msfa.ffn.pw_exp.bn.weight":
		return jlm.RoleVMNFusionExpNorm, true
	case "v.msfa.ffn.pw_proj.conv.weight":
		return jlm.RoleVMNFusionProj, true
	case "v.msfa.ffn.pw_proj.bn.weight":
		return jlm.RoleVMNFusionProjNorm, true
	case "v.msfa.norm.weight":
		return jlm.RoleVMNFusionNorm, true
	case "mm.embedding.weight":
		return jlm.RoleVMNHardTable, true
	case "mm.hard_emb_norm.weight":
		return jlm.RoleVMNHardNorm, true
	}
	return 0, false
}

// mobilenetPlace parses "v.blk.S.I.rest", a MobileNet-V5 block tensor.
func mobilenetPlace(name string) (stage, idx int, rest string, ok bool) {
	r, found := strings.CutPrefix(name, "v.blk.")
	if !found {
		return 0, 0, "", false
	}
	f := strings.SplitN(r, ".", 3)
	if len(f) != 3 {
		return 0, 0, "", false
	}
	s, e1 := strconv.Atoi(f[0])
	i, e2 := strconv.Atoi(f[1])
	if e1 != nil || e2 != nil || s < 0 || i < 0 {
		return 0, 0, "", false
	}
	return s, i, f[2], true
}

// mobilenetPlaces is every block a Gemma 3n mmproj carries, in order: by
// stage, then by place in the stage.
func mobilenetPlaces(f *meta.File) [][2]int {
	seen := map[[2]int]bool{}
	var out [][2]int
	for i := range f.Tensors {
		s, b, _, ok := mobilenetPlace(f.Tensors[i].Name)
		if !ok || seen[[2]int{s, b}] {
			continue
		}
		seen[[2]int{s, b}] = true
		out = append(out, [2]int{s, b})
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a][0] != out[b][0] {
			return out[a][0] < out[b][0]
		}
		return out[a][1] < out[b][1]
	})
	return out
}

// towerNames is the renames a tower file's tensors take before the name
// table: Gemma 3n's double-indexed blocks become one index. nil for every
// other tower. sourceOf and the role gate both read it.
func towerNames(f *meta.File, v *jlm.Vision) map[string]string {
	if v == nil || v.Projector != jlm.ProjGemma3nV {
		return nil
	}
	flat := map[[2]int]int{}
	for i, p := range mobilenetPlaces(f) {
		flat[p] = i
	}
	out := map[string]string{}
	for i := range f.Tensors {
		n := f.Tensors[i].Name
		if s, b, rest, ok := mobilenetPlace(n); ok {
			out[n] = "v.blk." + strconv.Itoa(flat[[2]int{s, b}]) + "." + rest
		}
	}
	return out
}

// mobilenetGeom appends each block's place, {stage, index in the stage}.
func mobilenetGeom(s *jlm.Source, places [][2]int) {
	for i, p := range places {
		e := jlm.Tensor{Role: jlm.RoleVMNGeom, Block: int32(i), Index: -1,
			Name: "v.blk." + strconv.Itoa(i) + ".mn_geom"}
		setF32Dims(&e, []float32{float32(p[0]), float32(p[1])}, 2)
		s.Tensors = append(s.Tensors, e)
	}
}

// gemma3nVocabOffset is the first of Gemma 3n's hard vision token ids
// (Gemma3nVisionConfig.vocab_offset): its end-of-image marker and image
// placeholder, which transformers embeds through the vision embedder rather
// than the text table. The mmproj does not state it; the architecture does.
const gemma3nVocabOffset = 262144

// prepareGemma3nV lays every convolution out tap-major, folds the layer
// scales and computes the hard tokens' embeddings, before quantizeTower rounds
// the matrices.
func prepareGemma3nV(s *jlm.Source) error {
	if err := gemma3nHardEmbd(s); err != nil {
		return err
	}
	scales := map[int32][]float32{}
	kept := s.Tensors[:0]
	for i := range s.Tensors {
		t := s.Tensors[i]
		if t.Role == jlm.RoleVMNLayerScale {
			x, err := vecOf(&t)
			if err != nil {
				return err
			}
			scales[t.Block] = x
			continue
		}
		kept = append(kept, t)
	}
	s.Tensors = kept
	for i := range s.Tensors {
		t := &s.Tensors[i]
		var err error
		switch t.Role {
		case jlm.RoleVMNStem, jlm.RoleVMNConvExp, jlm.RoleVMNConvPwl, jlm.RoleVMNPwExp, jlm.RoleVMNPwProj,
			jlm.RoleVMNFusionExp, jlm.RoleVMNFusionProj, jlm.RoleVAttnQ, jlm.RoleVAttnK, jlm.RoleVAttnV,
			jlm.RoleVAttnOut:
			err = convMatrix(t)
		case jlm.RoleVMNDwStart, jlm.RoleVMNDwMid, jlm.RoleVMNKDown, jlm.RoleVMNVDown:
			err = depthwiseTaps(t)
		case jlm.RoleVMNStemBias:
			var x []float32
			if x, err = vecOf(t); err == nil {
				setF32Dims(t, x, uint64(len(x)))
			}
		}
		if err != nil {
			return err
		}
	}
	// A layer scale multiplies its block's output channels: the projection's
	// norm in an inverted residual, the output matrix's rows in an attention
	// block. Both are per output channel, so the fold is exact.
	for b, g := range scales {
		folded := false
		for i := range s.Tensors {
			t := &s.Tensors[i]
			if t.Block != b {
				continue
			}
			switch t.Role {
			case jlm.RoleVMNPwProjNorm:
				x, err := vecOf(t)
				if err != nil {
					return err
				}
				if len(x) != len(g) {
					return fmt.Errorf("convert: block %d's layer scale is %d and its norm %d", b, len(g), len(x))
				}
				for j := range x {
					x[j] *= g[j]
				}
				setF32Dims(t, x, uint64(len(x)))
				folded = true
			case jlm.RoleVAttnOut:
				vals, k, rows, err := floatRows(t)
				if err != nil {
					return err
				}
				if rows != len(g) {
					return fmt.Errorf("convert: block %d's layer scale is %d and its output %d rows", b, len(g), rows)
				}
				for r := 0; r < rows; r++ {
					for j := 0; j < k; j++ {
						vals[r*k+j] *= g[r]
					}
				}
				setF32Dims(t, vals, uint64(k), uint64(rows))
				folded = true
			}
		}
		if !folded {
			return fmt.Errorf("convert: block %d has a layer scale and nothing to fold it into", b)
		}
	}
	return nil
}

// convMatrix turns a [kw kh Cin Cout] convolution (torch's [Cout][Cin][kh][kw])
// into the matrix the engine multiplies by: Cout rows of k = kh*kw*Cin in
// (ky, kx, c) order, padded to a whole quantization block. A 2-D tensor is
// already one (a linear's).
func convMatrix(t *jlm.Tensor) error {
	if t.NDim == 2 {
		return padK(t)
	}
	if t.NDim != 4 {
		return fmt.Errorf("convert: %s is %d-dimensional, want a convolution", t.Name, t.NDim)
	}
	kw, kh, cin, cout := int(t.Dims[0]), int(t.Dims[1]), int(t.Dims[2]), int(t.Dims[3])
	src, err := vecOf(t)
	if err != nil {
		return err
	}
	k := kh * kw * cin
	out := make([]float32, k*cout)
	for o := 0; o < cout; o++ {
		for c := 0; c < cin; c++ {
			for y := 0; y < kh; y++ {
				for x := 0; x < kw; x++ {
					out[o*k+(y*kw+x)*cin+c] = src[((o*cin+c)*kh+y)*kw+x]
				}
			}
		}
	}
	setF32Dims(t, out, uint64(k), uint64(cout))
	return padK(t)
}

// padK pads a matrix's k to a whole quantization block with zero columns.
func padK(t *jlm.Tensor) error {
	k, rows := int(t.Dims[0]), int(t.Dims[1])
	if k%q8Elems == 0 {
		return nil
	}
	vals, _, _, err := floatRows(t)
	if err != nil {
		return err
	}
	padded := (k + q8Elems - 1) / q8Elems * q8Elems
	out := make([]float32, padded*rows)
	for r := 0; r < rows; r++ {
		copy(out[r*padded:], vals[r*k:(r+1)*k])
	}
	setF32Dims(t, out, uint64(padded), uint64(rows))
	return nil
}

// depthwiseTaps turns a [kw kh 1 C] depthwise filter (torch's [C][1][kh][kw])
// into K*K rows of C channels, tap-major.
func depthwiseTaps(t *jlm.Tensor) error {
	if t.NDim != 4 || t.Dims[2] != 1 || t.Dims[0] != t.Dims[1] {
		return fmt.Errorf("convert: %s is %v, want a square depthwise filter [K K 1 C]", t.Name, t.Dims[:t.NDim])
	}
	kk, c := int(t.Dims[0]*t.Dims[1]), int(t.Dims[3])
	src, err := vecOf(t)
	if err != nil {
		return err
	}
	out := make([]float32, kk*c)
	for ch := 0; ch < c; ch++ {
		for tap := 0; tap < kk; tap++ {
			out[tap*c+ch] = src[ch*kk+tap]
		}
	}
	setF32Dims(t, out, uint64(c), uint64(kk))
	return nil
}

// gemma3nHardEmbd folds the embedder's hard-token path into one table:
// transformers' Gemma3nMultimodalEmbedder for an id, post_norm(W · norm(
// table[id - offset])), with W the projection the picture's rows go through
// and post_norm unweighted. Offline arithmetic in float64, so the reader does
// none of it.
func gemma3nHardEmbd(s *jlm.Source) error {
	var tab, nrm, proj *jlm.Tensor
	kept := s.Tensors[:0]
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch t.Role {
		case jlm.RoleVMNHardTable:
			x := *t
			tab = &x
			continue
		case jlm.RoleVMNHardNorm:
			x := *t
			nrm = &x
			continue
		case jlm.RoleVProj:
			x := *t
			proj = &x
		}
		kept = append(kept, *t)
	}
	s.Tensors = kept
	if tab == nil && nrm == nil {
		return nil
	}
	if tab == nil || nrm == nil || proj == nil {
		return fmt.Errorf("convert: gemma3nv's hard-token embedder is missing its table, its norm or its projection")
	}
	tv, err := vecOf(tab)
	if err != nil {
		return err
	}
	nv, err := vecOf(nrm)
	if err != nil {
		return err
	}
	pv, k, rows, err := floatRows(proj)
	if err != nil {
		return err
	}
	if len(nv) != k || len(tv)%k != 0 {
		return fmt.Errorf("convert: gemma3nv's hard table is %d floats and its norm %d, for a %d-wide projection",
			len(tv), len(nv), k)
	}
	eps := float64(s.Vision.Eps)
	n := len(tv) / k
	out := make([]float32, n*rows)
	x := make([]float64, k)
	y := make([]float64, rows)
	for i := 0; i < n; i++ {
		row := tv[i*k : (i+1)*k]
		ss := 0.0
		for j, v := range row {
			ss += float64(v) * float64(v)
			x[j] = float64(v)
		}
		inv := 1 / math.Sqrt(ss/float64(k)+eps)
		for j := range x {
			x[j] *= inv * float64(nv[j])
		}
		ss = 0
		for r := 0; r < rows; r++ {
			acc := 0.0
			w := pv[r*k : (r+1)*k]
			for j := range x {
				acc += float64(w[j]) * x[j]
			}
			y[r] = acc
			ss += acc * acc
		}
		inv = 1 / math.Sqrt(ss/float64(rows)+eps)
		for r := range y {
			out[i*rows+r] = float32(y[r] * inv)
		}
	}
	e := jlm.Tensor{Role: jlm.RoleVMNHardEmbd, Block: jlm.DenseBlock, Index: -1, Name: "mm.hard_embd"}
	setF32Dims(&e, out, uint64(rows), uint64(n))
	o := jlm.Tensor{Role: jlm.RoleVMNHardOffset, Block: jlm.DenseBlock, Index: -1, Name: "mm.hard_offset"}
	setF32Dims(&o, []float32{gemma3nVocabOffset}, 1)
	s.Tensors = append(s.Tensors, e, o)
	return nil
}

// gemma3nvTower is Gemma 3n's MobileNet-V5 encoder as transformers'
// Gemma3nForConditionalGeneration runs it (timm's MobileNetV5Encoder and
// Gemma3nMultimodalEmbedder) over SiglipImageProcessorFast's pixels: the
// picture resized to the tower's square by torch's bilinear and read as
// [0, 1]; the stem's 3x3 stride-2 convolution, RmsNorm2d and GELU; each
// block a program of its own over its own grid (jlm.Tower.Chain) -- an edge
// residual, a universal inverted residual or multi-query attention -- the
// first of a stage downsampling by two, every window padded as TensorFlow
// pads; then the fusion adapter over the last blocks of the last two stages
// -- the coarser upsampled to the finer by nearest, the two concatenated, an
// inverted residual's FFN with no skip, an average pool to the tower's
// output grid and an RMSNorm -- and the embedder: the rows times the root of
// their width (get_image_features), its RMSNorm, the matrix and an
// unweighted RMSNorm. After them the end-of-image marker's row, which
// transformers embeds through the embedder's hard path, folded at conversion
// (gemma3nHardEmbd). timm's GELU is the tanh form. The blocks' places
// (RoleVMNGeom) say which downsample and which the adapter reads; stated,
// they leave the container.
func gemma3nvTower(s *jlm.Source) (*jlm.Tower, error) {
	v := s.Vision
	type key struct {
		r jlm.Role
		b int32
	}
	at := map[key]*jlm.Tensor{}
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Role.Vision() {
			at[key{t.Role, t.Block}] = t
		}
	}
	has := func(r jlm.Role, b int32) bool { return at[key{r, b}] != nil }
	// dims is a tensor's k and rows as the reader counts them.
	dims := func(r jlm.Role, b int32) (int, int) {
		t := at[key{r, b}]
		if t == nil {
			return 0, 0
		}
		rows := 1
		for d := 1; d < int(t.NDim); d++ {
			rows *= int(t.Dims[d])
		}
		return int(t.Dims[0]), rows
	}
	need := func(b int32, rs ...jlm.Role) error {
		for _, r := range rs {
			if !has(r, b) {
				return fmt.Errorf("convert: gemma3nv: block %d has no %v", b, r)
			}
		}
		return nil
	}
	if err := need(jlm.DenseBlock, jlm.RoleVMNStem, jlm.RoleVMNStemNorm, jlm.RoleVMNFusionExp, jlm.RoleVMNFusionExpNorm,
		jlm.RoleVMNFusionProj, jlm.RoleVMNFusionProjNorm, jlm.RoleVMNFusionNorm, jlm.RoleVProjNorm, jlm.RoleVProj,
		jlm.RoleVMNHardEmbd); err != nil {
		return nil, err
	}
	// Gemma3nConfig's eoi_token_id is its first hard token; a GGUF whose
	// vocabulary stops at the text table's end does not spell it.
	eoi := gemma3nVocabOffset
	if s.Vocab != nil {
		if i := slices.Index(s.Vocab.Tokens, "<end_of_image>"); i >= 0 {
			eoi = i
		}
	}
	_, hard := dims(jlm.RoleVMNHardEmbd, jlm.DenseBlock)
	if eoi < gemma3nVocabOffset || eoi-gemma3nVocabOffset >= hard {
		return nil, fmt.Errorf("convert: gemma3nv: <end_of_image> is id %d, not one of the %d hard tokens from %d", eoi, hard, gemma3nVocabOffset)
	}
	eps := v.Eps
	var gelu jlm.ExprBuilder
	gelu.GELUTanh(jlm.Operand(0))
	act := jlm.Expr{Terms: gelu.T}
	// same is a side's output at stride s under TensorFlow's SAME padding.
	same := func(in, s int) int { return (in + s - 1) / s }
	side := int(v.ImageSize)
	stemK, stemC := dims(jlm.RoleVMNStem, jlm.DenseBlock)
	if side < 1 || stemK < 27 || stemC < 1 {
		return nil, fmt.Errorf("convert: gemma3nv: a %d-pixel picture and a stem of %dx%d", side, stemC, stemK)
	}
	stem := []jlm.Role{jlm.RoleVMNStem}
	if has(jlm.RoleVMNStemBias, jlm.DenseBlock) {
		stem = append(stem, jlm.RoleVMNStemBias)
	}
	t := &jlm.Tower{
		Version: jlm.TowerVersion,
		Pre: []jlm.PreStep{
			{Kind: jlm.PreResize, Ints: []int32{int32(side), int32(side), int32(jlm.FilterPILBilinear), int32(jlm.PrecTorch)}},
		},
		Entry: jlm.Program{NIn: 1, Out: 5, Ops: []jlm.TowerOp{
			{Code: jlm.TOpPatchify, In: []uint16{0}, Span: 1},
			{Code: jlm.TOpUnfold, In: []uint16{1}, Span: 3, Stride: 2, GridW: int32(side), Width: int32(stemK)},
			{Code: jlm.TOpLinear, In: []uint16{2}, Bind: stem},
			{Code: jlm.TOpRMSNorm, In: []uint16{3}, Bind: []jlm.Role{jlm.RoleVMNStemNorm}, Eps: eps},
			{Code: jlm.TOpStated, In: []uint16{4}, Expr: &act},
		}},
		Span: jlm.TowerSpan{BeforeText: "\n\n<start_of_image>", AfterText: "\n\n"},
	}
	h, w, c := same(side, 2), same(side, 2), stemC
	type shape struct{ h, w, c, stage int }
	var outs []shape
	for i := range int(v.NLayer) {
		b := int32(i)
		g, ok := at[key{jlm.RoleVMNGeom, b}]
		if !ok {
			return nil, fmt.Errorf("convert: gemma3nv: block %d states no place", i)
		}
		place, err := vecOf(g)
		if err != nil || len(place) != 2 {
			return nil, fmt.Errorf("convert: gemma3nv: block %d's place is %d numbers: %v", i, len(place), err)
		}
		stride := 1
		if place[1] == 0 {
			stride = 2
		}
		p := jlm.Program{NIn: 1}
		op := func(o jlm.TowerOp) uint16 {
			p.Ops = append(p.Ops, o)
			return p.NIn + uint16(len(p.Ops)) - 1
		}
		norm := func(x uint16, r jlm.Role) uint16 {
			return op(jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{x}, Bind: []jlm.Role{r}, Eps: eps})
		}
		lin := func(x uint16, r jlm.Role) uint16 {
			return op(jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{x}, Bind: []jlm.Role{r}})
		}
		gel := func(x uint16) uint16 { return op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{x}, Expr: &act}) }
		// dw is a depthwise convolution by filter r over the grid gw wide
		// at stride st, its side read off the filter's rows.
		dw := func(x uint16, r jlm.Role, ch, gw, st int) (uint16, int, error) {
			k, taps := dims(r, b)
			side := int(math.Round(math.Sqrt(float64(taps))))
			if k != ch || side*side != taps || side%2 == 0 {
				return 0, 0, fmt.Errorf("convert: gemma3nv: block %d's %v is %d taps over %d channels, want a square over %d", i, r, taps, k, ch)
			}
			return op(jlm.TowerOp{Code: jlm.TOpConv, In: []uint16{x}, Bind: []jlm.Role{r}, Span: int32(side), Stride: int32(st),
				GridW: int32(gw)}), side, nil
		}
		ho, wo, co := h, w, c
		var out uint16
		switch {
		case has(jlm.RoleVMNConvExp, b):
			if err := need(b, jlm.RoleVMNNorm1, jlm.RoleVMNConvPwl, jlm.RoleVMNNorm2); err != nil {
				return nil, err
			}
			k, _ := dims(jlm.RoleVMNConvExp, b)
			_, co = dims(jlm.RoleVMNConvPwl, b)
			if k < 9*c {
				return nil, fmt.Errorf("convert: gemma3nv: block %d's full convolution reads %d of a 3x3 window of %d", i, k, c)
			}
			x := op(jlm.TowerOp{Code: jlm.TOpUnfold, In: []uint16{0}, Span: 3, Stride: int32(stride), GridW: int32(w), Width: int32(k)})
			x = gel(norm(lin(x, jlm.RoleVMNConvExp), jlm.RoleVMNNorm1))
			out = norm(lin(x, jlm.RoleVMNConvPwl), jlm.RoleVMNNorm2)
			ho, wo = same(h, stride), same(w, stride)
			if stride == 1 && co == c {
				out = op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{0, out}})
			}
		case has(jlm.RoleVAttnQ, b):
			if err := need(b, jlm.RoleVAttnNorm, jlm.RoleVAttnK, jlm.RoleVAttnV, jlm.RoleVAttnOut); err != nil {
				return nil, err
			}
			_, qd := dims(jlm.RoleVAttnQ, b)
			_, kd := dims(jlm.RoleVAttnK, b)
			if kd == 0 || qd%kd != 0 {
				return nil, fmt.Errorf("convert: gemma3nv: block %d's attention is %d query and %d key wide", i, qd, kd)
			}
			n := norm(0, jlm.RoleVAttnNorm)
			q := lin(n, jlm.RoleVAttnQ)
			kin, vin := n, n
			if has(jlm.RoleVMNKDown, b) {
				if err := need(b, jlm.RoleVMNKDownNorm, jlm.RoleVMNVDown, jlm.RoleVMNVDownNorm); err != nil {
					return nil, err
				}
				x, _, err := dw(n, jlm.RoleVMNKDown, c, w, 2)
				if err != nil {
					return nil, err
				}
				kin = norm(x, jlm.RoleVMNKDownNorm)
				if x, _, err = dw(n, jlm.RoleVMNVDown, c, w, 2); err != nil {
					return nil, err
				}
				vin = norm(x, jlm.RoleVMNVDownNorm)
			}
			k, vv := lin(kin, jlm.RoleVAttnK), lin(vin, jlm.RoleVAttnV)
			a := op(jlm.TowerOp{Code: jlm.TOpRowsAttend, In: []uint16{q, k, vv}, Span: int32(qd / kd), Width: 1})
			out = op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{0, lin(a, jlm.RoleVAttnOut)}})
		default:
			if err := need(b, jlm.RoleVMNPwExp, jlm.RoleVMNPwExpNorm, jlm.RoleVMNPwProj, jlm.RoleVMNPwProjNorm); err != nil {
				return nil, err
			}
			_, cexp := dims(jlm.RoleVMNPwExp, b)
			_, co = dims(jlm.RoleVMNPwProj, b)
			// The stride goes to the depthwise middle, or to the start where
			// there is no middle (timm's UniversalInvertedResidual).
			sStart, sMid := 1, stride
			if !has(jlm.RoleVMNDwMid, b) {
				sStart, sMid = stride, 1
			}
			x := uint16(0)
			hm, wm := h, w
			if has(jlm.RoleVMNDwStart, b) {
				y, _, err := dw(x, jlm.RoleVMNDwStart, c, w, sStart)
				if err != nil {
					return nil, err
				}
				x = norm(y, jlm.RoleVMNDwStartNorm)
				hm, wm = same(h, sStart), same(w, sStart)
			} else if sStart != 1 {
				return nil, fmt.Errorf("convert: gemma3nv: block %d downsamples and has no depthwise filter to do it", i)
			}
			x = gel(norm(lin(x, jlm.RoleVMNPwExp), jlm.RoleVMNPwExpNorm))
			ho, wo = hm, wm
			if has(jlm.RoleVMNDwMid, b) {
				y, _, err := dw(x, jlm.RoleVMNDwMid, cexp, wm, sMid)
				if err != nil {
					return nil, err
				}
				x = gel(norm(y, jlm.RoleVMNDwMidNorm))
				ho, wo = same(hm, sMid), same(wm, sMid)
			}
			out = norm(lin(x, jlm.RoleVMNPwProj), jlm.RoleVMNPwProjNorm)
			if ho == h && wo == w && co == c {
				out = op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{0, out}})
			}
		}
		p.Out = out
		t.Chain = append(t.Chain, jlm.ChainBlock{Rows: int32(h * w), Width: int32(c), Prog: p})
		h, w, c = ho, wo, co
		outs = append(outs, shape{h, w, c, int(place[0])})
	}
	if len(outs) == 0 {
		return nil, fmt.Errorf("convert: gemma3nv: the tower has no blocks")
	}
	// The fusion adapter reads the last block of the last two stages.
	last := len(outs) - 1
	from := -1
	for i, o := range outs {
		if o.stage == outs[last].stage-1 {
			from = i
		}
	}
	grid := int(v.ImageSize / max(v.PatchSize, 1))
	if from < 0 {
		return nil, fmt.Errorf("convert: gemma3nv: the tower has one stage; the fusion adapter reads two")
	}
	a, z := outs[from], outs[last]
	if a.h%z.h != 0 || a.w%z.w != 0 || a.h/z.h != a.w/z.w || a.h != a.w || grid < 1 || a.h%grid != 0 {
		return nil, fmt.Errorf("convert: gemma3nv: the adapter fuses a %dx%d and a %dx%d grid into %d a side", a.h, a.w, z.h, z.w, grid)
	}
	fk, _ := dims(jlm.RoleVMNFusionExp, jlm.DenseBlock)
	if fk != a.c+z.c {
		return nil, fmt.Errorf("convert: gemma3nv: the adapter reads %d channels of %d and %d", fk, a.c, z.c)
	}
	E, _ := dims(jlm.RoleVProjNorm, jlm.DenseBlock)
	var scale jlm.ExprBuilder
	scale.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(float32(math.Sqrt(float64(E)))))
	sq := jlm.Expr{Terms: scale.T}
	t.HeadFrom = []int32{int32(from)}
	t.Head = jlm.Program{NIn: 2, Out: 15, Ops: []jlm.TowerOp{
		{Code: jlm.TOpUpsample, In: []uint16{0}, Span: int32(a.h / z.h), GridW: int32(z.w)},
		{Code: jlm.TOpConcat, In: []uint16{1, 2}},
		{Code: jlm.TOpLinear, In: []uint16{3}, Bind: []jlm.Role{jlm.RoleVMNFusionExp}},
		{Code: jlm.TOpRMSNorm, In: []uint16{4}, Bind: []jlm.Role{jlm.RoleVMNFusionExpNorm}, Eps: eps},
		{Code: jlm.TOpStated, In: []uint16{5}, Expr: &act},
		{Code: jlm.TOpLinear, In: []uint16{6}, Bind: []jlm.Role{jlm.RoleVMNFusionProj}},
		{Code: jlm.TOpRMSNorm, In: []uint16{7}, Bind: []jlm.Role{jlm.RoleVMNFusionProjNorm}, Eps: eps},
		{Code: jlm.TOpPool, In: []uint16{8}, Span: int32(a.h / grid), GridW: int32(a.w)},
		{Code: jlm.TOpRMSNorm, In: []uint16{9}, Bind: []jlm.Role{jlm.RoleVMNFusionNorm}, Eps: eps},
		{Code: jlm.TOpStated, In: []uint16{10}, Expr: &sq},
		{Code: jlm.TOpRMSNorm, In: []uint16{11}, Bind: []jlm.Role{jlm.RoleVProjNorm}, Eps: eps},
		{Code: jlm.TOpLinear, In: []uint16{12}, Bind: []jlm.Role{jlm.RoleVProj}},
		{Code: jlm.TOpRMSNorm, In: []uint16{13}, Eps: eps},
		{Code: jlm.TOpAppendRow, In: []uint16{14}, Bind: []jlm.Role{jlm.RoleVMNHardEmbd}, Offset: int32(eoi - gemma3nVocabOffset)},
	}}
	kept := s.Tensors[:0]
	for _, x := range s.Tensors {
		if x.Role != jlm.RoleVMNGeom && x.Role != jlm.RoleVMNHardOffset {
			kept = append(kept, x)
		}
	}
	s.Tensors = kept
	return t, nil
}

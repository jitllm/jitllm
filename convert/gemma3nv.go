package convert

import (
	"fmt"
	"math"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// Gemma 3n's MobileNet-V5 tower (jlm.ProjGemma3nV), converter half.
//
// llama.cpp's mmproj names a block by its stage and its place in the stage
// (convert/stagedblocks.go); the tensor transforms are convert/towerlayouts.go's.

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
// (hardTokenEmbeddings). timm's GELU is the tanh form. The blocks' places
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
	// The first hard token is the end-of-image marker (Gemma3nConfig's
	// eoi_token_id); a GGUF whose vocabulary stops at the text table's end does
	// not spell it.
	first := -1
	for i := range s.Tensors {
		if t := &s.Tensors[i]; t.Role == jlm.RoleVMNHardOffset {
			x, err := vecOf(t)
			if err != nil || len(x) != 1 {
				return nil, fmt.Errorf("convert: gemma3nv: the hard tokens' offset is not one number")
			}
			first = int(x[0])
		}
	}
	if first < 0 {
		return nil, fmt.Errorf("convert: gemma3nv: no hard tokens' offset")
	}
	eoi := first
	if s.Vocab != nil {
		if i := slices.Index(s.Vocab.Tokens, "<end_of_image>"); i >= 0 {
			eoi = i
		}
	}
	_, hard := dims(jlm.RoleVMNHardEmbd, jlm.DenseBlock)
	if eoi < first || eoi-first >= hard {
		return nil, fmt.Errorf("convert: gemma3nv: <end_of_image> is id %d, not one of the %d hard tokens from %d", eoi, hard, first)
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
		{Code: jlm.TOpAppendRow, In: []uint16{14}, Bind: []jlm.Role{jlm.RoleVMNHardEmbd}, Offset: int32(eoi - first)},
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

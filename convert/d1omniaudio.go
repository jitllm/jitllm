package convert

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// d1-omni's audio half (clip.audio.projector_type "d1omni_a"), carried as the
// tower's audio blocks after its vision blocks (jlm.Audio). What the file
// states is read from it; what audio.py hardcodes and the file does not
// state -- the front end's framing -- is the class's, written into the
// container, with the tower itself as a stated program (d1OmniAudioTower)
// and the tables its front end reads. Layouts change on the way:
// pointwise_conv1 is cut into the GLU's value and gate halves, the
// subsampling's depthwise convolutions become tap-major (a tap's channels in
// a row), its output matrix reads its 4096 inputs channel-last
// (frequency-major), the layout the grid's rows take where the reference
// flattens them channel-major, and each block's linear_pos becomes the
// product with the sinusoids its attention reads.

// AudioProjectors is the list of audio towers (clip.audio.projector_type)
// that convert, as Projectors is the vision towers'.
func AudioProjectors() []string { return []string{"d1omni_a"} }

// d1OmniAudioNames is every per-block audio tensor's role, by its name after
// "a.blk.N.".
func d1OmniAudioNames(name string) (jlm.Role, bool) {
	switch name {
	case "ffn_norm.weight":
		return jlm.RoleAFF1Norm, true
	case "ffn_norm.bias":
		return jlm.RoleAFF1NormB, true
	case "ffn_up.weight":
		return jlm.RoleAFF1Up, true
	case "ffn_up.bias":
		return jlm.RoleAFF1UpB, true
	case "ffn_down.weight":
		return jlm.RoleAFF1Down, true
	case "ffn_down.bias":
		return jlm.RoleAFF1DownB, true
	case "ln1.weight":
		return jlm.RoleAAttnNorm, true
	case "ln1.bias":
		return jlm.RoleAAttnNormB, true
	case "attn_q.weight":
		return jlm.RoleAQ, true
	case "attn_q.bias":
		return jlm.RoleAQB, true
	case "attn_k.weight":
		return jlm.RoleAK, true
	case "attn_k.bias":
		return jlm.RoleAKB, true
	case "attn_v.weight":
		return jlm.RoleAV, true
	case "attn_v.bias":
		return jlm.RoleAVB, true
	case "attn_out.weight":
		return jlm.RoleAOut, true
	case "attn_out.bias":
		return jlm.RoleAOutB, true
	case "linear_pos.weight":
		return jlm.RoleAPos, true
	case "pos_bias_u":
		return jlm.RoleAPosU, true
	case "pos_bias_v":
		return jlm.RoleAPosV, true
	case "norm_conv.weight":
		return jlm.RoleAConvNorm, true
	case "norm_conv.bias":
		return jlm.RoleAConvNormB, true
	case "conv_dw.weight":
		return jlm.RoleADW, true
	case "conv_dw.bias":
		return jlm.RoleADWB, true
	case "conv_norm.weight":
		return jlm.RoleABNScale, true
	case "conv_norm.bias":
		return jlm.RoleABNShift, true
	case "conv_pw2.weight":
		return jlm.RoleAPW2, true
	case "conv_pw2.bias":
		return jlm.RoleAPW2B, true
	case "ffn_norm_1.weight":
		return jlm.RoleAFF2Norm, true
	case "ffn_norm_1.bias":
		return jlm.RoleAFF2NormB, true
	case "ffn_up_1.weight":
		return jlm.RoleAFF2Up, true
	case "ffn_up_1.bias":
		return jlm.RoleAFF2UpB, true
	case "ffn_down_1.weight":
		return jlm.RoleAFF2Down, true
	case "ffn_down_1.bias":
		return jlm.RoleAFF2DownB, true
	case "ln2.weight":
		return jlm.RoleAOutNorm, true
	case "ln2.bias":
		return jlm.RoleAOutNormB, true
	case "conv_pw1.weight":
		return jlm.RoleAGLUVal, true
	case "conv_pw1.bias":
		return jlm.RoleAGLUValB, true
	}
	return 0, false
}

// d1OmniAudioDense is the dense audio tensors' roles.
func d1OmniAudioDense(name string) (jlm.Role, bool) {
	switch name {
	case "a.pre_encode.out.weight":
		return jlm.RoleASubOut, true
	case "a.pre_encode.out.bias":
		return jlm.RoleASubOutB, true
	case "mm.a.mlp.0.weight":
		return jlm.RoleAAdNorm, true
	case "mm.a.mlp.0.bias":
		return jlm.RoleAAdNormB, true
	case "mm.a.mlp.1.weight":
		return jlm.RoleAAd1, true
	case "mm.a.mlp.1.bias":
		return jlm.RoleAAd1B, true
	case "mm.a.mlp.3.weight":
		return jlm.RoleAAd2, true
	case "mm.a.mlp.3.bias":
		return jlm.RoleAAd2B, true
	case "mm.a.mlp.4.weight":
		return jlm.RoleAResNorm, true
	case "mm.a.mlp.4.bias":
		return jlm.RoleAResNrmB, true
	case "mm.a.mlp.5.weight":
		return jlm.RoleAResDown, true
	case "mm.a.mlp.5.bias":
		return jlm.RoleAResDnB, true
	case "mm.a.mlp.6.weight":
		return jlm.RoleAResUp, true
	case "mm.a.mlp.6.bias":
		return jlm.RoleAResUpB, true
	}
	return 0, false
}

// audioOf reads f's audio tower, its blocks after nVis vision blocks, or nil
// for an mmproj with none. An audio projector other than d1-omni's is
// refused by name.
func audioOf(f *meta.File, nVis uint32) (*jlm.Audio, []jlm.Tensor, error) {
	if v, ok := f.KV["clip.has_audio_encoder"]; !ok || v.Type != meta.Bool || len(v.Raw) != 1 || v.Raw[0] == 0 {
		return nil, nil, nil
	}
	// Another family's audio half (Gemma 3n's) is not built: set aside
	// (withoutAudio), as it was before this one.
	if proj, _ := f.KV["clip.audio.projector_type"].String(); !slices.Contains(AudioProjectors(), proj) {
		return nil, nil, nil
	}
	key := func(k string) uint32 {
		v, ok := f.KV["clip.audio."+k]
		if !ok {
			return 0
		}
		n, _ := v.Uint()
		return uint32(n)
	}
	a := &jlm.Audio{
		NLayer: key("block_count"), NEmbd: key("embedding_length"), NHead: key("attention.head_count"),
		NFFN: key("feed_forward_length"), MelBins: key("num_mel_bins"), ProjDim: key("projection_dim"),
		Projector: jlm.ProjD1OmniA,
		// NeMo's FilterbankFeatures as audio.py builds it (MelFrontend): 25 ms
		// windows every 10 ms at 16 kHz, a 512-point transform.
		SampleRate: 16000, NFFT: 512, Window: 400, Hop: 160,
		// audio.waveform cuts a clip to 30 seconds.
		MaxSamples: 30 * 16000,
	}
	if v, ok := f.KV["clip.audio.attention.layer_norm_epsilon"]; ok {
		e, _ := v.Float()
		a.Eps = float32(e)
	}
	if a.NLayer == 0 || a.NEmbd == 0 || a.NHead == 0 || a.NEmbd%a.NHead != 0 || a.NFFN == 0 || a.MelBins == 0 ||
		a.ProjDim == 0 || a.Eps <= 0 {
		return nil, nil, fmt.Errorf("convert: clip: the audio encoder states %d blocks of %d (%d heads, FFN %d), "+
			"%d mel bins, projection %d, eps %g", a.NLayer, a.NEmbd, a.NHead, a.NFFN, a.MelBins, a.ProjDim, a.Eps)
	}
	var out []jlm.Tensor
	tensor := func(t *meta.Tensor, role jlm.Role, block int32, index int32) (jlm.Tensor, error) {
		e := jlm.Tensor{Role: role, Block: block, Index: index, NDim: uint8(len(t.Dims)), Data: f.Bytes(t), Name: t.Name}
		copy(e.Dims[:], t.Dims)
		var err error
		e.Type, err = typeOf(t.Type)
		return e, err
	}
	for i := range f.Tensors {
		t := &f.Tensors[i]
		name := t.Name
		switch {
		case strings.HasPrefix(name, "a.blk."):
			rest := name[len("a.blk."):]
			dot := strings.IndexByte(rest, '.')
			// A block index the file states, parsed to 31 bits so every
			// conversion below holds it (RULE 9).
			n, err := strconv.ParseInt(rest[:max(dot, 0)], 10, 31)
			if dot <= 0 || err != nil || n < 0 || uint64(n) >= uint64(a.NLayer) {
				return nil, nil, fmt.Errorf("convert: %q: an audio block index outside %d", name, a.NLayer)
			}
			role, ok := d1OmniAudioNames(rest[dot+1:])
			if !ok {
				return nil, nil, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary; "+
					"a tensor dropped at conversion is a model that loads and is wrong, so this is a refusal", name)
			}
			e, err := tensor(t, role, int32(nVis)+int32(n), -1)
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			es, err := d1OmniAudioLayout(e, a)
			if err != nil {
				return nil, nil, err
			}
			out = append(out, es...)
		case strings.HasPrefix(name, "a.conv1d."):
			rest := strings.TrimPrefix(name, "a.conv1d.")
			dot := strings.IndexByte(rest, '.')
			n64, err := strconv.ParseInt(rest[:max(dot, 0)], 10, 31)
			if err == nil && n64 < 0 {
				err = fmt.Errorf("a negative index")
			}
			if err == nil && (dot <= 0 || rest[dot+1:] != "weight" && rest[dot+1:] != "bias") {
				err = fmt.Errorf("not a weight or a bias")
			}
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %q: %v", name, err)
			}
			role, ok := d1OmniSubRole(int(n64), rest[dot+1:] == "bias")
			if !ok {
				return nil, nil, fmt.Errorf("convert: %s: the subsampling has no layer %d", name, n64)
			}
			e, err := tensor(t, role, jlm.DenseBlock, -1)
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			if err := d1OmniSubConv(&e, a); err != nil {
				return nil, nil, err
			}
			out = append(out, e)
		case strings.HasPrefix(name, "a.") || strings.HasPrefix(name, "mm.a."):
			role, ok := d1OmniAudioDense(name)
			if !ok {
				return nil, nil, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary; "+
					"a tensor dropped at conversion is a model that loads and is wrong, so this is a refusal", name)
			}
			e, err := tensor(t, role, jlm.DenseBlock, -1)
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			if role == jlm.RoleAResDown {
				a.ResWidth = uint32(e.Dims[1])
			}
			out = append(out, e)
		}
	}
	// The subsampling's output matrix, once its channel count is known.
	for i := range out {
		if out[i].Role == jlm.RoleASubOut {
			if err := d1OmniSubOut(&out[i], a); err != nil {
				return nil, nil, err
			}
		}
	}
	if a.ConvKernel == 0 || a.SubChannels == 0 || a.ResWidth == 0 {
		return nil, nil, fmt.Errorf("convert: clip: the audio encoder carries no depthwise convolution, "+
			"subsampling or residual correction (%d taps, %d channels, width %d)", a.ConvKernel, a.SubChannels, a.ResWidth)
	}
	out, err := d1OmniFoldDWBias(out)
	if err != nil {
		return nil, nil, err
	}
	out = append(out, d1OmniFrontTables(a)...)
	if err := quantizeTower(out); err != nil {
		return nil, nil, err
	}
	a.Tower = d1OmniAudioTower(a)
	if err := a.Tower.Valid(); err != nil {
		return nil, nil, fmt.Errorf("convert: the audio tower: %w", err)
	}
	return a, out, nil
}

// audioRows is the rows an audio block is formed for: the centred frames of
// the longest clip (one more than its hops), halved three times as the
// subsampling's stride-two convolutions padded by one halve them.
func audioRows(a *jlm.Audio) int {
	t := int(a.MaxSamples/a.Hop) + 1
	for range 3 {
		t = (t + 1) / 2
	}
	return t
}

// d1OmniAudioTower is d1-omni's audio tower as a stated program, audio.py's
// graph op for op. The entry: the clip's centred frames under the
// preemphasis, |DFT|^2 through the windowed bases, the filterbank, log(x +
// 2^-24), each feature normalised over the clip's frames (all but the last
// centred one), then the features as a grid of one channel through the
// subsampling -- each stride-two convolution padded by one, each ReLU, the
// grid rows past the clip's length zeroed after each as the reference masks
// them -- and the linear from a time row's frequencies and channels. Each
// block: half a SiLU MLP, the relative-position attention (q plus u over
// the keys, q plus v over the positions), the convolution module (a GLU, the
// centred depthwise convolution, the folded BatchNorm, SiLU, pointwise), the
// other half MLP and a LayerNorm. The head: the adapter and the residual
// correction.
func d1OmniAudioTower(a *jlm.Audio) *jlm.Tower {
	eps, d := a.Eps, int32(a.NEmbd)
	f1 := (int32(a.MelBins) + 1) / 2
	f2, f3 := (f1+1)/2, ((f1+1)/2+1)/2
	hop := int32(a.Hop)
	relu := func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMax, jlm.Operand(0), jlm.Num(0)) }
	silu := func(e *jlm.ExprBuilder) { e.SiLU(jlm.Operand(0)) }
	gelu := func(e *jlm.ExprBuilder) { e.GELUErf(jlm.Operand(0)) }
	half := func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnAdd, jlm.Operand(0), e.Fn(jlm.FnMul, jlm.Operand(1), jlm.Num(0.5)))
	}
	op := func(p *program, o jlm.TowerOp) uint16 { return p.op(o) }
	lin := func(p *program, x uint16, w, b jlm.Role) uint16 {
		return op(p, jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{x}, Bind: []jlm.Role{w, b}})
	}
	ln := func(p *program, x uint16, w, b jlm.Role) uint16 {
		return op(p, jlm.TowerOp{Code: jlm.TOpLayerNorm, In: []uint16{x}, Bind: []jlm.Role{w, b}, Eps: eps})
	}
	mask := func(p *program, x uint16, gw, halvings int32) uint16 {
		return op(p, jlm.TowerOp{Code: jlm.TOpClipMask, In: []uint16{x}, GridW: gw, Stride: hop, Span: halvings})
	}
	conv := func(p *program, x uint16, w, b jlm.Role, gw int32) uint16 {
		c := op(p, jlm.TowerOp{Code: jlm.TOpConv, In: []uint16{x}, Bind: []jlm.Role{w}, Span: 3, Stride: 2, GridW: gw, Offset: 1})
		return op(p, jlm.TowerOp{Code: jlm.TOpShift, In: []uint16{c}, Bind: []jlm.Role{b}})
	}

	e := newProgram(0)
	x := op(e, jlm.TowerOp{Code: jlm.TOpFrames, Width: int32(a.NFFT), Stride: hop, Eps: 0.97})
	re := op(e, jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{x}, Bind: []jlm.Role{jlm.RoleADFTCos}})
	im := op(e, jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{x}, Bind: []jlm.Role{jlm.RoleADFTSin}})
	x = e.expr(0, func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnAdd, e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(0)), e.Fn(jlm.FnMul, jlm.Operand(1), jlm.Operand(1)))
	}, re, im)
	x = op(e, jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{x}, Bind: []jlm.Role{jlm.RoleAMelBank}})
	x = e.expr(0, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnLog, e.Fn(jlm.FnAdd, jlm.Operand(0), jlm.Num(1.0/(1<<24)))) }, x)
	x = op(e, jlm.TowerOp{Code: jlm.TOpFeatureNorm, In: []uint16{x}, Offset: 1, Eps: 1e-5})
	x = op(e, jlm.TowerOp{Code: jlm.TOpSplitRows, In: []uint16{x}, Width: 1})
	x = op(e, jlm.TowerOp{Code: jlm.TOpUnfold, In: []uint16{x}, Span: 3, Stride: 2, GridW: int32(a.MelBins), Width: 9, Offset: 1})
	x = mask(e, e.expr(0, relu, lin(e, x, jlm.RoleASub0, jlm.RoleASub0B)), f1, 1)
	x = mask(e, conv(e, x, jlm.RoleASubDW1, jlm.RoleASubDW1B, f1), f2, 2)
	x = mask(e, e.expr(0, relu, lin(e, x, jlm.RoleASubPW1, jlm.RoleASubPW1B)), f2, 2)
	x = conv(e, x, jlm.RoleASubDW2, jlm.RoleASubDW2B, f2)
	x = e.expr(0, relu, lin(e, x, jlm.RoleASubPW2, jlm.RoleASubPW2B))
	x = op(e, jlm.TowerOp{Code: jlm.TOpFold, In: []uint16{x}, Span: f3})
	lin(e, x, jlm.RoleASubOut, jlm.RoleASubOutB)
	e.p.Out = e.p.NIn + uint16(len(e.p.Ops)-1)

	ff := func(p *program, x uint16, nw, nb, up, upB, down, downB jlm.Role) uint16 {
		h := p.expr(0, silu, lin(p, ln(p, x, nw, nb), up, upB))
		return p.expr(0, half, x, lin(p, h, down, downB))
	}
	t := &jlm.Tower{Version: jlm.TowerVersion, Entry: *e.p}
	for range a.NLayer {
		b := newProgram(1)
		x := ff(b, 0, jlm.RoleAFF1Norm, jlm.RoleAFF1NormB, jlm.RoleAFF1Up, jlm.RoleAFF1UpB, jlm.RoleAFF1Down, jlm.RoleAFF1DownB)
		h := ln(b, x, jlm.RoleAAttnNorm, jlm.RoleAAttnNormB)
		q, k, v := lin(b, h, jlm.RoleAQ, jlm.RoleAQB), lin(b, h, jlm.RoleAK, jlm.RoleAKB), lin(b, h, jlm.RoleAV, jlm.RoleAVB)
		qu := op(b, jlm.TowerOp{Code: jlm.TOpShift, In: []uint16{q}, Bind: []jlm.Role{jlm.RoleAPosU}})
		qv := op(b, jlm.TowerOp{Code: jlm.TOpShift, In: []uint16{q}, Bind: []jlm.Role{jlm.RoleAPosV}})
		h = op(b, jlm.TowerOp{Code: jlm.TOpRowsAttend, In: []uint16{qu, k, v, qv}, Bind: []jlm.Role{jlm.RoleAPos}, Span: int32(a.NHead)})
		x = b.add(x, lin(b, h, jlm.RoleAOut, jlm.RoleAOutB))
		h = ln(b, x, jlm.RoleAConvNorm, jlm.RoleAConvNormB)
		h = b.expr(0, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, jlm.Operand(0), e.Fn(jlm.FnSigmoid, jlm.Operand(1))) },
			lin(b, h, jlm.RoleAGLUVal, jlm.RoleAGLUValB), lin(b, h, jlm.RoleAGLUGate, jlm.RoleAGLUGateB))
		h = op(b, jlm.TowerOp{Code: jlm.TOpRowConv, In: []uint16{h}, Bind: []jlm.Role{jlm.RoleADW}, Span: int32(a.ConvKernel)})
		h = op(b, jlm.TowerOp{Code: jlm.TOpScale, In: []uint16{h}, Bind: []jlm.Role{jlm.RoleABNScale}})
		h = op(b, jlm.TowerOp{Code: jlm.TOpShift, In: []uint16{h}, Bind: []jlm.Role{jlm.RoleABNShift}})
		x = b.add(x, lin(b, b.expr(0, silu, h), jlm.RoleAPW2, jlm.RoleAPW2B))
		x = ff(b, x, jlm.RoleAFF2Norm, jlm.RoleAFF2NormB, jlm.RoleAFF2Up, jlm.RoleAFF2UpB, jlm.RoleAFF2Down, jlm.RoleAFF2DownB)
		b.p.Out = ln(b, x, jlm.RoleAOutNorm, jlm.RoleAOutNormB)
		t.Chain = append(t.Chain, jlm.ChainBlock{Rows: int32(audioRows(a)), Width: d, Prog: *b.p})
	}
	h := newProgram(1)
	x5 := lin(h, h.expr(0, gelu, lin(h, ln(h, 0, jlm.RoleAAdNorm, jlm.RoleAAdNormB), jlm.RoleAAd1, jlm.RoleAAd1B)), jlm.RoleAAd2, jlm.RoleAAd2B)
	r := h.expr(0, gelu, lin(h, ln(h, x5, jlm.RoleAResNorm, jlm.RoleAResNrmB), jlm.RoleAResDown, jlm.RoleAResDnB))
	h.p.Out = h.add(x5, lin(h, r, jlm.RoleAResUp, jlm.RoleAResUpB))
	t.Head = *h.p
	return t
}

// d1OmniFrontTables is the front end's tables (audio.py's MelFrontend),
// computed in float64 and stored F32: the DFT bases with
// torch.hann_window(Window, periodic=False) folded in, centred in the NFFT
// frame as torch.stft pads it, and librosa's Slaney filterbank.
func d1OmniFrontTables(a *jlm.Audio) []jlm.Tensor {
	n, win, nm := int(a.NFFT), int(a.Window), int(a.MelBins)
	bins := n/2 + 1
	w := make([]float64, n)
	left := (n - win) / 2
	for i := 0; i < win; i++ {
		w[left+i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(win-1))
	}
	cs, sn := make([]float32, bins*n), make([]float32, bins*n)
	for k := 0; k < bins; k++ {
		for t := 0; t < n; t++ {
			ang := 2 * math.Pi * float64((k*t)%n) / float64(n)
			cs[k*n+t] = float32(w[t] * math.Cos(ang))
			sn[k*n+t] = float32(w[t] * math.Sin(ang))
		}
	}
	var out []jlm.Tensor
	for _, tb := range []struct {
		role    jlm.Role
		name    string
		vals    []float32
		k, rows int
	}{
		{jlm.RoleADFTCos, "a.dft.cos", cs, n, bins},
		{jlm.RoleADFTSin, "a.dft.sin", sn, n, bins},
		{jlm.RoleAMelBank, "a.mel", slaneyFilterbank(int(a.SampleRate), n, nm), bins, nm},
	} {
		e := jlm.Tensor{Role: tb.role, Block: jlm.DenseBlock, Index: -1, Name: tb.name}
		setF32Dims(&e, tb.vals, uint64(tb.k), uint64(tb.rows))
		out = append(out, e)
	}
	return out
}

// slaneyFilterbank is librosa.filters.mel(sr, n_fft, n_mels, norm="slaney")
// as audio.py computes it: n_mels rows of n_fft/2+1.
func slaneyFilterbank(sr, nfft, nmels int) []float32 {
	const fsp, minLogHz = 200.0 / 3, 1000.0
	logstep := math.Log(6.4) / 27
	minLogMel := minLogHz / fsp
	hzToMel := func(f float64) float64 {
		if f >= minLogHz {
			return minLogMel + math.Log(f/minLogHz)/logstep
		}
		return f / fsp
	}
	melToHz := func(m float64) float64 {
		if m >= minLogMel {
			return minLogHz * math.Exp(logstep*(m-minLogMel))
		}
		return fsp * m
	}
	bins := nfft/2 + 1
	lo, hi := hzToMel(0), hzToMel(float64(sr)/2)
	melF := make([]float64, nmels+2)
	for i := range melF {
		// np.linspace's formula: lo + i*step, the last point exact.
		melF[i] = melToHz(lo + float64(i)*(hi-lo)/float64(nmels+1))
	}
	melF[nmels+1] = melToHz(hi)
	out := make([]float32, nmels*bins)
	for i := 0; i < nmels; i++ {
		fd0, fd1 := melF[i+1]-melF[i], melF[i+2]-melF[i+1]
		enorm := 2 / (melF[i+2] - melF[i])
		for j := 0; j < bins; j++ {
			f := float64(j) * float64(sr) / float64(nfft)
			v := math.Max(0, math.Min(-(melF[i]-f)/fd0, (melF[i+2]-f)/fd1))
			out[i*bins+j] = float32(v) * float32(enorm)
		}
	}
	return out
}

// d1OmniSubRole is the subsampling's layer's role, its weight's or its
// bias's, by the layer's index in the module.
func d1OmniSubRole(layer int, bias bool) (jlm.Role, bool) {
	var w, b jlm.Role
	switch layer {
	case 0:
		w, b = jlm.RoleASub0, jlm.RoleASub0B
	case 2:
		w, b = jlm.RoleASubDW1, jlm.RoleASubDW1B
	case 3:
		w, b = jlm.RoleASubPW1, jlm.RoleASubPW1B
	case 5:
		w, b = jlm.RoleASubDW2, jlm.RoleASubDW2B
	case 6:
		w, b = jlm.RoleASubPW2, jlm.RoleASubPW2B
	default:
		return 0, false
	}
	if bias {
		return b, true
	}
	return w, true
}

// d1OmniFoldDWBias folds each block's depthwise bias into its BatchNorm
// shift, dwB*scale + shift (jlm.RoleABNShift), and drops the bias: the
// depthwise convolution feeds the BatchNorm alone, so the module is one
// affine and no load or token folds it again.
func d1OmniFoldDWBias(out []jlm.Tensor) ([]jlm.Tensor, error) {
	type trio struct{ dwB, scale, shift int }
	blocks := map[int32]*trio{}
	at := func(b int32) *trio {
		if blocks[b] == nil {
			blocks[b] = &trio{-1, -1, -1}
		}
		return blocks[b]
	}
	for i := range out {
		switch out[i].Role {
		case jlm.RoleADWB:
			at(out[i].Block).dwB = i
		case jlm.RoleABNScale:
			at(out[i].Block).scale = i
		case jlm.RoleABNShift:
			at(out[i].Block).shift = i
		}
	}
	for b, t := range blocks {
		if t.dwB < 0 || t.scale < 0 || t.shift < 0 {
			return nil, fmt.Errorf("convert: audio block %d carries the depthwise bias, the BatchNorm scale "+
				"and its shift only in part", b)
		}
		bias, err := normF32(&out[t.dwB])
		if err != nil {
			return nil, err
		}
		scale, err := normF32(&out[t.scale])
		if err != nil {
			return nil, err
		}
		shift, err := normF32(&out[t.shift])
		if err != nil {
			return nil, err
		}
		if len(bias) != len(scale) || len(shift) != len(scale) {
			return nil, fmt.Errorf("convert: audio block %d: depthwise bias %d, BatchNorm %d and %d wide",
				b, len(bias), len(scale), len(shift))
		}
		folded := make([]float32, len(shift))
		for c := range folded {
			folded[c] = bias[c]*scale[c] + shift[c]
		}
		setF32Dims(&out[t.shift], folded, uint64(len(folded)))
	}
	kept := out[:0]
	for _, e := range out {
		if e.Role != jlm.RoleADWB {
			kept = append(kept, e)
		}
	}
	return kept, nil
}

// d1OmniAudioLayout is one block tensor in the engine's layout: the GLU's
// halves cut apart, the depthwise convolution F32 as the file lays it out
// (channel-major), the per-head biases one vector, and linear_pos the table
// of the projected positions.
func d1OmniAudioLayout(e jlm.Tensor, a *jlm.Audio) ([]jlm.Tensor, error) {
	d := int(a.NEmbd)
	switch e.Role {
	case jlm.RoleAGLUVal, jlm.RoleAGLUValB:
		vals, err := normF32(&e)
		if err != nil {
			return nil, err
		}
		if len(vals)%2 != 0 {
			return nil, fmt.Errorf("convert: %s has %d values, not two halves", e.Name, len(vals))
		}
		half := len(vals) / 2
		g := e
		g.Name = e.Name + "/gate"
		if e.Role == jlm.RoleAGLUVal {
			g.Role = jlm.RoleAGLUGate
			setF32Dims(&e, vals[:half], uint64(d), uint64(half/d))
			setF32Dims(&g, vals[half:], uint64(d), uint64(half/d))
		} else {
			g.Role = jlm.RoleAGLUGateB
			setF32Dims(&e, vals[:half], uint64(half))
			setF32Dims(&g, vals[half:], uint64(half))
		}
		return []jlm.Tensor{e, g}, nil
	case jlm.RoleADW:
		// [chan][tap] in the file (conv_dw.weight, {k, d}), as graph.OpRowConv
		// reads it.
		vals, err := normF32(&e)
		if err != nil {
			return nil, err
		}
		k := int(e.Dims[0])
		if e.NDim != 2 || int(e.Dims[1]) != d || k < 1 || k%2 != 1 {
			return nil, fmt.Errorf("convert: %s is %v, want an odd number of taps over %d channels",
				e.Name, e.Dims[:e.NDim], d)
		}
		setF32Dims(&e, vals, uint64(k), uint64(d))
		if a.ConvKernel != 0 && a.ConvKernel != uint32(k) {
			return nil, fmt.Errorf("convert: %s has %d taps where another block has %d", e.Name, k, a.ConvKernel)
		}
		a.ConvKernel = uint32(k)
	case jlm.RoleAPosU, jlm.RoleAPosV:
		vals, err := normF32(&e)
		if err != nil {
			return nil, err
		}
		if len(vals) != d {
			return nil, fmt.Errorf("convert: %s has %d values over %d channels", e.Name, len(vals), d)
		}
		setF32Dims(&e, vals, uint64(d))
	case jlm.RoleAPos:
		// linear_pos times the sinusoids of every position the block's rows
		// reach, positions rows-1 down to 1-rows (sin at the even elements,
		// cos at the odd, NeMo's RelPositionalEncoding): the product the
		// relative attention reads, computed here once rather than every clip.
		w, err := normF32(&e)
		if err != nil {
			return nil, err
		}
		if e.NDim != 2 || int(e.Dims[0]) != d || int(e.Dims[1]) != d || d%2 != 0 {
			return nil, fmt.Errorf("convert: %s is %v, want %dx%d", e.Name, e.Dims[:e.NDim], d, d)
		}
		rows := audioRows(a)
		n := 2*rows - 1
		pe := make([]float64, d)
		p := make([]float32, n*d)
		for k := 0; k < n; k++ {
			pos := float64(rows - 1 - k)
			for c := 0; c < d/2; c++ {
				div := math.Exp(float64(2*c) * -(math.Log(10000.0) / float64(d)))
				pe[2*c], pe[2*c+1] = math.Sin(pos*div), math.Cos(pos*div)
			}
			for r := 0; r < d; r++ {
				s := 0.0
				for c := 0; c < d; c++ {
					s += float64(w[r*d+c]) * pe[c]
				}
				p[k*d+r] = float32(s)
			}
		}
		setF32Dims(&e, p, uint64(d), uint64(n))
	}
	return []jlm.Tensor{e}, nil
}

// d1OmniSubConv is a subsampling layer's weight in the engine's layout: the
// first convolution (one input channel) as a matrix of 9 taps a channel,
// the depthwise ones tap-major (ky, kx, c), the pointwise ones a matrix of
// out x in. Biases stay vectors.
func d1OmniSubConv(e *jlm.Tensor, a *jlm.Audio) error {
	vals, err := normF32(e)
	if err != nil {
		return err
	}
	switch e.Role {
	case jlm.RoleASub0B, jlm.RoleASubDW1B, jlm.RoleASubPW1B, jlm.RoleASubDW2B, jlm.RoleASubPW2B:
		setF32Dims(e, vals, uint64(len(vals)))
		return nil
	}
	if e.NDim != 4 {
		return fmt.Errorf("convert: %s is %d-dimensional, want a 4-D convolution", e.Name, e.NDim)
	}
	kx, ky, cin, cout := int(e.Dims[0]), int(e.Dims[1]), int(e.Dims[2]), int(e.Dims[3])
	if a.SubChannels == 0 {
		a.SubChannels = uint32(cout)
	}
	switch e.Role {
	case jlm.RoleASub0, jlm.RoleASubPW1, jlm.RoleASubPW2:
		// A full convolution: out x (in*ky*kx), the file's own order.
		if e.Role == jlm.RoleASub0 && (cin != 1 || kx != 3 || ky != 3) || e.Role != jlm.RoleASub0 && (kx != 1 || ky != 1) {
			return fmt.Errorf("convert: %s is %dx%dx%dx%d, not the subsampling's %v", e.Name, kx, ky, cin, cout, e.Role)
		}
		setF32Dims(e, vals, uint64(cin*ky*kx), uint64(cout))
	case jlm.RoleASubDW1, jlm.RoleASubDW2:
		if cin != 1 || kx != 3 || ky != 3 {
			return fmt.Errorf("convert: %s is %dx%dx%dx%d, not a 3x3 depthwise convolution", e.Name, kx, ky, cin, cout)
		}
		tm := make([]float32, len(vals))
		for c := 0; c < cout; c++ {
			for u := 0; u < 9; u++ {
				tm[u*cout+c] = vals[c*9+u]
			}
		}
		setF32Dims(e, tm, uint64(cout), 9)
	default:
		return fmt.Errorf("convert: %s: the subsampling has no %v", e.Name, e.Role)
	}
	return nil
}

// d1OmniSubOut reorders the subsampling's output matrix's inputs from the
// reference's channel-major flattening (c*F + f) to channel-last (f*C + c).
func d1OmniSubOut(e *jlm.Tensor, a *jlm.Audio) error {
	vals, err := normF32(e)
	if err != nil {
		return err
	}
	k, rows := int(e.Dims[0]), int(e.Dims[1])
	c := int(a.SubChannels)
	if c == 0 || k%c != 0 {
		return fmt.Errorf("convert: %s reads %d inputs, not a whole number of %d channels", e.Name, k, c)
	}
	fr := k / c
	out := make([]float32, len(vals))
	for r := 0; r < rows; r++ {
		for ch := 0; ch < c; ch++ {
			for f := 0; f < fr; f++ {
				out[r*k+f*c+ch] = vals[r*k+ch*fr+f]
			}
		}
	}
	setF32Dims(e, out, uint64(k), uint64(rows))
	return nil
}

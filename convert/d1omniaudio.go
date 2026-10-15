package convert

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// d1-omni's audio half (clip.audio.projector_type "d1omni_a"), carried as the
// tower's audio blocks after its vision blocks (jlm.Audio). What the file
// states is read from it; what audio.py hardcodes and the file does not
// state -- the front end's framing -- is the class's, written into the
// container. Three layouts change on the way: pointwise_conv1 is cut into the
// GLU's value and gate halves, the depthwise convolutions become tap-major
// (a tap's channels in a row), and the subsampling's output matrix reads
// its 4096 inputs channel-last (frequency-major), the layout the engine's
// activations take, where the reference flattens them channel-major.

// d1OmniAudioNames is every per-block audio tensor's role, by its name after
// "a.blk.N.".
var d1OmniAudioNames = map[string]jlm.Role{
	"ffn_norm.weight": jlm.RoleAFF1Norm, "ffn_norm.bias": jlm.RoleAFF1NormB,
	"ffn_up.weight": jlm.RoleAFF1Up, "ffn_up.bias": jlm.RoleAFF1UpB,
	"ffn_down.weight": jlm.RoleAFF1Down, "ffn_down.bias": jlm.RoleAFF1DownB,
	"ln1.weight": jlm.RoleAAttnNorm, "ln1.bias": jlm.RoleAAttnNormB,
	"attn_q.weight": jlm.RoleAQ, "attn_q.bias": jlm.RoleAQB,
	"attn_k.weight": jlm.RoleAK, "attn_k.bias": jlm.RoleAKB,
	"attn_v.weight": jlm.RoleAV, "attn_v.bias": jlm.RoleAVB,
	"attn_out.weight": jlm.RoleAOut, "attn_out.bias": jlm.RoleAOutB,
	"linear_pos.weight": jlm.RoleAPos, "pos_bias_u": jlm.RoleAPosU, "pos_bias_v": jlm.RoleAPosV,
	"norm_conv.weight": jlm.RoleAConvNorm, "norm_conv.bias": jlm.RoleAConvNormB,
	"conv_dw.weight": jlm.RoleADW, "conv_dw.bias": jlm.RoleADWB, // folded below
	"conv_norm.weight": jlm.RoleABNScale, "conv_norm.bias": jlm.RoleABNShift,
	"conv_pw2.weight": jlm.RoleAPW2, "conv_pw2.bias": jlm.RoleAPW2B,
	"ffn_norm_1.weight": jlm.RoleAFF2Norm, "ffn_norm_1.bias": jlm.RoleAFF2NormB,
	"ffn_up_1.weight": jlm.RoleAFF2Up, "ffn_up_1.bias": jlm.RoleAFF2UpB,
	"ffn_down_1.weight": jlm.RoleAFF2Down, "ffn_down_1.bias": jlm.RoleAFF2DownB,
	"ln2.weight": jlm.RoleAOutNorm, "ln2.bias": jlm.RoleAOutNormB,
	// pointwise_conv1 is cut in two below.
	"conv_pw1.weight": jlm.RoleAGLUVal, "conv_pw1.bias": jlm.RoleAGLUValB,
}

// d1OmniAudioDense is the dense audio tensors' roles.
var d1OmniAudioDense = map[string]jlm.Role{
	"a.pre_encode.out.weight": jlm.RoleASubOut, "a.pre_encode.out.bias": jlm.RoleASubOutB,
	"mm.a.mlp.0.weight": jlm.RoleAAdNorm, "mm.a.mlp.0.bias": jlm.RoleAAdNormB,
	"mm.a.mlp.1.weight": jlm.RoleAAd1, "mm.a.mlp.1.bias": jlm.RoleAAd1B,
	"mm.a.mlp.3.weight": jlm.RoleAAd2, "mm.a.mlp.3.bias": jlm.RoleAAd2B,
	"mm.a.mlp.4.weight": jlm.RoleAResNorm, "mm.a.mlp.4.bias": jlm.RoleAResNrmB,
	"mm.a.mlp.5.weight": jlm.RoleAResDown, "mm.a.mlp.5.bias": jlm.RoleAResDnB,
	"mm.a.mlp.6.weight": jlm.RoleAResUp, "mm.a.mlp.6.bias": jlm.RoleAResUpB,
}

// d1OmniSubLayers is the subsampling's convolution layers (audio.py's
// nn.Sequential indices): a full 3x3, then depthwise 3x3 and pointwise twice.
var d1OmniSubLayers = []int{0, 2, 3, 5, 6}

// audioOf reads f's audio tower, its blocks after nVis vision blocks, or nil
// for an mmproj with none. An audio projector other than d1-omni's is
// refused by name.
func audioOf(f *meta.File, nVis uint32) (*jlm.Audio, []jlm.Tensor, error) {
	if v, ok := f.KV["clip.has_audio_encoder"]; !ok || v.Type != meta.Bool || len(v.Raw) != 1 || v.Raw[0] == 0 {
		return nil, nil, nil
	}
	// Another family's audio half (Gemma 3n's) is not built: set aside
	// (withoutAudio), as it was before this one.
	if proj, _ := f.KV["clip.audio.projector_type"].String(); proj != "d1omni_a" {
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
			role, ok := d1OmniAudioNames[rest[dot+1:]]
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
			n := int(n64)
			role := jlm.RoleAConv
			if dot > 0 && rest[dot+1:] == "bias" {
				role = jlm.RoleAConvBias
			} else if dot <= 0 || rest[dot+1:] != "weight" {
				err = fmt.Errorf("not a weight or a bias")
			}
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %q: %v", name, err)
			}
			e, err := tensor(t, role, jlm.DenseBlock, int32(n))
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			if err := d1OmniSubConv(&e, n, a); err != nil {
				return nil, nil, err
			}
			out = append(out, e)
		case strings.HasPrefix(name, "a.") || strings.HasPrefix(name, "mm.a."):
			role, ok := d1OmniAudioDense[name]
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
	if err := quantizeTower(out); err != nil {
		return nil, nil, err
	}
	return a, out, nil
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
// halves cut apart, the depthwise convolution tap-major.
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
		// [chan][tap] in the file (conv_dw.weight, {k, d}); [tap][chan] here.
		vals, err := normF32(&e)
		if err != nil {
			return nil, err
		}
		k := int(e.Dims[0])
		if e.NDim != 2 || int(e.Dims[1]) != d || k < 1 || k%2 != 1 {
			return nil, fmt.Errorf("convert: %s is %v, want an odd number of taps over %d channels",
				e.Name, e.Dims[:e.NDim], d)
		}
		tm := make([]float32, len(vals))
		for c := 0; c < d; c++ {
			for u := 0; u < k; u++ {
				tm[u*d+c] = vals[c*k+u]
			}
		}
		setF32Dims(&e, tm, uint64(d), uint64(k))
		if a.ConvKernel != 0 && a.ConvKernel != uint32(k) {
			return nil, fmt.Errorf("convert: %s has %d taps where another block has %d", e.Name, k, a.ConvKernel)
		}
		a.ConvKernel = uint32(k)
	}
	return []jlm.Tensor{e}, nil
}

// d1OmniSubConv is a subsampling layer's weight in the engine's layout: the
// first convolution (one input channel) as a matrix of 9 taps a channel,
// the depthwise ones tap-major (ky, kx, c), the pointwise ones a matrix of
// out x in. Biases stay vectors.
func d1OmniSubConv(e *jlm.Tensor, layer int, a *jlm.Audio) error {
	vals, err := normF32(e)
	if err != nil {
		return err
	}
	if e.Role == jlm.RoleAConvBias {
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
	switch layer {
	case 0, 3, 6:
		// A full convolution: out x (in*ky*kx), the file's own order.
		if layer == 0 && (cin != 1 || kx != 3 || ky != 3) || layer != 0 && (kx != 1 || ky != 1) {
			return fmt.Errorf("convert: %s is %dx%dx%dx%d, not the subsampling's layer %d", e.Name, kx, ky, cin, cout, layer)
		}
		setF32Dims(e, vals, uint64(cin*ky*kx), uint64(cout))
	case 2, 5:
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
		return fmt.Errorf("convert: %s: the subsampling has no layer %d", e.Name, layer)
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

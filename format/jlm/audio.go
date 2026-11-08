package jlm

// An audio tower: d1-omni's (its audio.py; llama.cpp's "d1omni_a"), the
// container's first. It rides the tower's page array after the vision
// blocks -- audio block j is tower block Vision.NLayer+j -- and its
// description is an optional tail of the Vision record (Vision.Audio), so a
// container with no audio is the bytes it was.
//
// The graph: 16 kHz mono samples; NeMo's log-mel front end (preemphasis,
// |STFT|^2 under a Hann window, a Slaney mel filterbank, log, each feature
// normalised over the clip); an 8x time subsampling (a 3x3 stride-2
// convolution of one channel into SubChannels, two depthwise-separable
// stride-2 ones, each followed by a ReLU, and a linear from the channels and
// the remaining frequencies into NEmbd); NLayer FastConformer blocks (half
// a SiLU MLP, Transformer-XL relative-position attention with per-head u and
// v biases, a convolution module -- pointwise into a GLU, a depthwise
// ConvKernel-tap convolution, BatchNorm folded into a scale and a shift at
// conversion, SiLU, pointwise -- the other half MLP and a LayerNorm); then
// the adapter (LayerNorm, linear, erf GELU, linear into ProjDim) and the
// residual correction (x + up(GELU(down(LayerNorm(x))))). One embedding row
// per 80 ms.
type Audio struct {
	NLayer, NEmbd, NHead, NFFN uint32
	ConvKernel                 uint32 // the convolution module's depthwise taps
	MelBins                    uint32 // log-mel features per frame
	SubChannels                uint32 // the subsampling's channels
	ProjDim                    uint32 // the adapter's output: the text model's width
	ResWidth                   uint32 // the residual correction's bottleneck
	Eps                        float32
	Projector                  Projector
	// The front end's framing: a window of Window samples every Hop, padded
	// to NFFT, at SampleRate.
	SampleRate, NFFT, Window, Hop uint32
	// MaxSamples is the longest clip the reference reads (audio.waveform
	// cuts one to it); the blocks are formed for its rows.
	MaxSamples uint32
	// Tower is the tower as the converter states it: the front end and the
	// subsampling its entry, each conformer block a chained block, the
	// adapter and the residual correction its head. A container that does
	// not state it is not heard.
	Tower *Tower
}

// ProjD1OmniA is d1-omni's audio half (llama.cpp's "d1omni_a"); see Audio.
const ProjD1OmniA Projector = 65529

// The audio tower's roles. Every one is a tower tensor (Role.Vision) and
// pages with the tower's blocks; the subsampling, adapter and residual sit
// in the dense region. Block j of an audio tensor is tower block
// Vision.NLayer+j.
const (
	RoleASubOut  Role = 1002 // the linear after the subsampling
	RoleASubOutB Role = 1003

	// The front end's tables, F32: the DFT bases with the window folded in
	// (NFFT/2+1 rows of NFFT) and the Slaney filterbank (MelBins rows of
	// NFFT/2+1).
	RoleADFTCos  Role = 1004
	RoleADFTSin  Role = 1005
	RoleAMelBank Role = 1006

	// A conformer block.
	RoleAFF1Norm   Role = 1010
	RoleAFF1NormB  Role = 1011
	RoleAFF1Up     Role = 1012
	RoleAFF1UpB    Role = 1013
	RoleAFF1Down   Role = 1014
	RoleAFF1DownB  Role = 1015
	RoleAAttnNorm  Role = 1016
	RoleAAttnNormB Role = 1017
	RoleAQ         Role = 1018
	RoleAQB        Role = 1019
	RoleAK         Role = 1020
	RoleAKB        Role = 1021
	RoleAV         Role = 1022
	RoleAVB        Role = 1023
	RoleAOut       Role = 1024
	RoleAOutB      Role = 1025
	// RoleAPos is linear_pos times the sinusoids of every position the
	// block's rows reach, F32: graph.OpRowsAttend's table, row k the
	// position (rows+1)/2-1-k.
	RoleAPos       Role = 1026
	RoleAPosU      Role = 1027 // pos_bias_u, NHead x head width
	RoleAPosV      Role = 1028
	RoleAConvNorm  Role = 1029 // the LayerNorm before the convolution module
	RoleAConvNormB Role = 1030
	RoleAGLUVal    Role = 1031 // pointwise_conv1's first half, the GLU's value
	RoleAGLUValB   Role = 1032
	RoleAGLUGate   Role = 1033 // its second half, the GLU's gate
	RoleAGLUGateB  Role = 1034
	RoleADW        Role = 1035 // the depthwise convolution, channel-major [chan][tap] (graph.OpRowConv's)
	// RoleADWB is the depthwise convolution's bias as the converter reads
	// it: folded into RoleABNShift since version 30 and written by no
	// converter since; a version 29 container carries it, which migrate29
	// folds.
	RoleADWB     Role = 1036
	RoleABNScale Role = 1037 // BatchNorm folded: x*scale + shift
	// RoleABNShift is the BatchNorm's shift with the depthwise bias folded
	// in at conversion, dwB*scale + shift, so the module is one affine.
	RoleABNShift  Role = 1038
	RoleAPW2      Role = 1039
	RoleAPW2B     Role = 1040
	RoleAFF2Norm  Role = 1041
	RoleAFF2NormB Role = 1042
	RoleAFF2Up    Role = 1043
	RoleAFF2UpB   Role = 1044
	RoleAFF2Down  Role = 1045
	RoleAFF2DownB Role = 1046
	RoleAOutNorm  Role = 1047
	RoleAOutNormB Role = 1048

	// The subsampling, F32: the first convolution a matrix of its 9 taps (one
	// input channel), the depthwise ones tap-major (ky, kx) rows of the
	// channels, the pointwise ones out x in; a bias each.
	RoleASub0    Role = 1049
	RoleASub0B   Role = 1050
	RoleASubDW1  Role = 1051
	RoleASubDW1B Role = 1052
	RoleASubPW1  Role = 1053
	RoleASubPW1B Role = 1054
	RoleASubDW2  Role = 1055
	RoleASubDW2B Role = 1056
	RoleASubPW2  Role = 1057
	RoleASubPW2B Role = 1058

	// The adapter and the residual correction (dense).
	RoleAAdNorm  Role = 1060
	RoleAAdNormB Role = 1061
	RoleAAd1     Role = 1062
	RoleAAd1B    Role = 1063
	RoleAAd2     Role = 1064
	RoleAAd2B    Role = 1065
	RoleAResNorm Role = 1066
	RoleAResNrmB Role = 1067
	RoleAResDown Role = 1068
	RoleAResDnB  Role = 1069
	RoleAResUp   Role = 1070
	RoleAResUpB  Role = 1071
)

func audioRoleName(r Role) (string, bool) {
	switch r {
	case RoleADFTCos:
		return "a.dft.cos", true
	case RoleADFTSin:
		return "a.dft.sin", true
	case RoleAMelBank:
		return "a.mel", true
	case RoleASub0:
		return "a.conv1d.0", true
	case RoleASub0B:
		return "a.conv1d.0.bias", true
	case RoleASubDW1:
		return "a.conv1d.2", true
	case RoleASubDW1B:
		return "a.conv1d.2.bias", true
	case RoleASubPW1:
		return "a.conv1d.3", true
	case RoleASubPW1B:
		return "a.conv1d.3.bias", true
	case RoleASubDW2:
		return "a.conv1d.5", true
	case RoleASubDW2B:
		return "a.conv1d.5.bias", true
	case RoleASubPW2:
		return "a.conv1d.6", true
	case RoleASubPW2B:
		return "a.conv1d.6.bias", true
	case RoleASubOut:
		return "a.pre_encode.out", true
	case RoleASubOutB:
		return "a.pre_encode.out.bias", true
	case RoleAFF1Norm:
		return "a.ffn_norm", true
	case RoleAFF1NormB:
		return "a.ffn_norm.bias", true
	case RoleAFF1Up:
		return "a.ffn_up", true
	case RoleAFF1UpB:
		return "a.ffn_up.bias", true
	case RoleAFF1Down:
		return "a.ffn_down", true
	case RoleAFF1DownB:
		return "a.ffn_down.bias", true
	case RoleAAttnNorm:
		return "a.ln1", true
	case RoleAAttnNormB:
		return "a.ln1.bias", true
	case RoleAQ:
		return "a.attn_q", true
	case RoleAQB:
		return "a.attn_q.bias", true
	case RoleAK:
		return "a.attn_k", true
	case RoleAKB:
		return "a.attn_k.bias", true
	case RoleAV:
		return "a.attn_v", true
	case RoleAVB:
		return "a.attn_v.bias", true
	case RoleAOut:
		return "a.attn_out", true
	case RoleAOutB:
		return "a.attn_out.bias", true
	case RoleAPos:
		return "a.linear_pos.table", true
	case RoleAPosU:
		return "a.pos_bias_u", true
	case RoleAPosV:
		return "a.pos_bias_v", true
	case RoleAConvNorm:
		return "a.norm_conv", true
	case RoleAConvNormB:
		return "a.norm_conv.bias", true
	case RoleAGLUVal:
		return "a.conv_pw1.value", true
	case RoleAGLUValB:
		return "a.conv_pw1.value.bias", true
	case RoleAGLUGate:
		return "a.conv_pw1.gate", true
	case RoleAGLUGateB:
		return "a.conv_pw1.gate.bias", true
	case RoleADW:
		return "a.conv_dw", true
	case RoleADWB:
		return "a.conv_dw.bias", true
	case RoleABNScale:
		return "a.conv_norm", true
	case RoleABNShift:
		return "a.conv_norm.bias+conv_dw.bias", true
	case RoleAPW2:
		return "a.conv_pw2", true
	case RoleAPW2B:
		return "a.conv_pw2.bias", true
	case RoleAFF2Norm:
		return "a.ffn_norm_1", true
	case RoleAFF2NormB:
		return "a.ffn_norm_1.bias", true
	case RoleAFF2Up:
		return "a.ffn_up_1", true
	case RoleAFF2UpB:
		return "a.ffn_up_1.bias", true
	case RoleAFF2Down:
		return "a.ffn_down_1", true
	case RoleAFF2DownB:
		return "a.ffn_down_1.bias", true
	case RoleAOutNorm:
		return "a.ln2", true
	case RoleAOutNormB:
		return "a.ln2.bias", true
	case RoleAAdNorm:
		return "mm.a.mlp.0", true
	case RoleAAdNormB:
		return "mm.a.mlp.0.bias", true
	case RoleAAd1:
		return "mm.a.mlp.1", true
	case RoleAAd1B:
		return "mm.a.mlp.1.bias", true
	case RoleAAd2:
		return "mm.a.mlp.3", true
	case RoleAAd2B:
		return "mm.a.mlp.3.bias", true
	case RoleAResNorm:
		return "mm.a.mlp.4", true
	case RoleAResNrmB:
		return "mm.a.mlp.4.bias", true
	case RoleAResDown:
		return "mm.a.mlp.5", true
	case RoleAResDnB:
		return "mm.a.mlp.5.bias", true
	case RoleAResUp:
		return "mm.a.mlp.6", true
	case RoleAResUpB:
		return "mm.a.mlp.6.bias", true
	}
	return "", false
}

// audioMatrix is the audio roles a tower multiplies by; every other is a
// vector read once at load (Role.Expanded) or a convolution's taps.
func audioMatrix(r Role) bool {
	switch r {
	case RoleASubOut, RoleAFF1Up, RoleAFF1Down, RoleAQ, RoleAK, RoleAV, RoleAOut, RoleAGLUVal, RoleAGLUGate, RoleAPW2, RoleAFF2Up, RoleAFF2Down, RoleAAd1, RoleAAd2, RoleAResDown, RoleAResUp:
		return true
	}
	return false
}

// AudioMatrix reports whether r is an audio tower matrix (the converter's
// quantized set, a reader's bound one).
func AudioMatrix(r Role) bool { return audioMatrix(r) }

// audioTable is the audio roles a tower multiplies by at F32, never
// quantized: the front end's tables, the subsampling's convolutions and the
// blocks' position tables, whose error every later block carries.
func audioTable(r Role) bool {
	switch r {
	case RoleADFTCos, RoleADFTSin, RoleAMelBank, RoleAPos, RoleASub0, RoleASubPW1, RoleASubPW2:
		return true
	}
	return false
}

// audioRole reports whether r is one of the audio tower's roles.
func audioRole(r Role) bool {
	_, ok := audioRoleName(r)
	return ok
}

// audioExpanded is Role.Expanded for the audio roles: every vector, and the
// convolutions' taps, which a reader copies out at load.
func audioExpanded(r Role) bool { return audioRole(r) && !audioMatrix(r) && !audioTable(r) }

// encodeAudio is the Vision record's audio tail.
func encodeAudio(w *wbuf, a *Audio) {
	for _, x := range []uint32{a.NLayer, a.NEmbd, a.NHead, a.NFFN, a.ConvKernel, a.MelBins, a.SubChannels,
		a.ProjDim, a.ResWidth} {
		w.u32(x)
	}
	w.f32(a.Eps)
	w.u16(uint16(a.Projector))
	for _, x := range []uint32{a.SampleRate, a.NFFT, a.Window, a.Hop} {
		w.u32(x)
	}
}

func decodeAudio(r *rbuf) *Audio {
	a := &Audio{}
	for _, p := range []*uint32{&a.NLayer, &a.NEmbd, &a.NHead, &a.NFFN, &a.ConvKernel, &a.MelBins,
		&a.SubChannels, &a.ProjDim, &a.ResWidth} {
		*p = r.u32()
	}
	a.Eps = r.f32()
	a.Projector = Projector(r.u16())
	for _, p := range []*uint32{&a.SampleRate, &a.NFFT, &a.Window, &a.Hop} {
		*p = r.u32()
	}
	return a
}

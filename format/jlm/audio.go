package jlm

import "fmt"

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
}

// ProjD1OmniA is d1-omni's audio half (llama.cpp's "d1omni_a"); see Audio.
const ProjD1OmniA Projector = 65529

// The audio tower's roles. Every one is a tower tensor (Role.Vision) and
// pages with the tower's blocks; the subsampling, adapter and residual sit
// in the dense region. Block j of an audio tensor is tower block
// Vision.NLayer+j.
const (
	// The subsampling: three convolutions over (time, frequency) and their
	// pointwise partners, tap-major (ky, kx, c) and channel-last, with a
	// bias each; Index is the module's layer (0, 2, 3, 5, 6).
	RoleAConv     Role = 1000
	RoleAConvBias Role = 1001
	RoleASubOut   Role = 1002 // the linear after the subsampling
	RoleASubOutB  Role = 1003

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
	RoleAPos       Role = 1026 // linear_pos, no bias
	RoleAPosU      Role = 1027 // pos_bias_u, NHead x head width
	RoleAPosV      Role = 1028
	RoleAConvNorm  Role = 1029 // the LayerNorm before the convolution module
	RoleAConvNormB Role = 1030
	RoleAGLUVal    Role = 1031 // pointwise_conv1's first half, the GLU's value
	RoleAGLUValB   Role = 1032
	RoleAGLUGate   Role = 1033 // its second half, the GLU's gate
	RoleAGLUGateB  Role = 1034
	RoleADW        Role = 1035 // the depthwise convolution, tap-major [tap][chan]
	// RoleADWB is the depthwise convolution's bias as the converter reads
	// it; it is folded into RoleABNShift and never written to a container.
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

var audioRoleNames = map[Role]string{
	RoleAConv: "a.conv1d", RoleAConvBias: "a.conv1d.bias",
	RoleASubOut: "a.pre_encode.out", RoleASubOutB: "a.pre_encode.out.bias",
	RoleAFF1Norm: "a.ffn_norm", RoleAFF1NormB: "a.ffn_norm.bias",
	RoleAFF1Up: "a.ffn_up", RoleAFF1UpB: "a.ffn_up.bias",
	RoleAFF1Down: "a.ffn_down", RoleAFF1DownB: "a.ffn_down.bias",
	RoleAAttnNorm: "a.ln1", RoleAAttnNormB: "a.ln1.bias",
	RoleAQ: "a.attn_q", RoleAQB: "a.attn_q.bias", RoleAK: "a.attn_k", RoleAKB: "a.attn_k.bias",
	RoleAV: "a.attn_v", RoleAVB: "a.attn_v.bias", RoleAOut: "a.attn_out", RoleAOutB: "a.attn_out.bias",
	RoleAPos: "a.linear_pos", RoleAPosU: "a.pos_bias_u", RoleAPosV: "a.pos_bias_v",
	RoleAConvNorm: "a.norm_conv", RoleAConvNormB: "a.norm_conv.bias",
	RoleAGLUVal: "a.conv_pw1.value", RoleAGLUValB: "a.conv_pw1.value.bias",
	RoleAGLUGate: "a.conv_pw1.gate", RoleAGLUGateB: "a.conv_pw1.gate.bias",
	RoleADW:      "a.conv_dw",
	RoleABNScale: "a.conv_norm", RoleABNShift: "a.conv_norm.bias+conv_dw.bias",
	RoleAPW2: "a.conv_pw2", RoleAPW2B: "a.conv_pw2.bias",
	RoleAFF2Norm: "a.ffn_norm_1", RoleAFF2NormB: "a.ffn_norm_1.bias",
	RoleAFF2Up: "a.ffn_up_1", RoleAFF2UpB: "a.ffn_up_1.bias",
	RoleAFF2Down: "a.ffn_down_1", RoleAFF2DownB: "a.ffn_down_1.bias",
	RoleAOutNorm: "a.ln2", RoleAOutNormB: "a.ln2.bias",
	RoleAAdNorm: "mm.a.mlp.0", RoleAAdNormB: "mm.a.mlp.0.bias",
	RoleAAd1: "mm.a.mlp.1", RoleAAd1B: "mm.a.mlp.1.bias",
	RoleAAd2: "mm.a.mlp.3", RoleAAd2B: "mm.a.mlp.3.bias",
	RoleAResNorm: "mm.a.mlp.4", RoleAResNrmB: "mm.a.mlp.4.bias",
	RoleAResDown: "mm.a.mlp.5", RoleAResDnB: "mm.a.mlp.5.bias",
	RoleAResUp: "mm.a.mlp.6", RoleAResUpB: "mm.a.mlp.6.bias",
}

// audioMatrix is the audio roles a tower multiplies by; every other is a
// vector read once at load (Role.Expanded) or a convolution's taps.
var audioMatrix = map[Role]bool{
	RoleASubOut: true, RoleAFF1Up: true, RoleAFF1Down: true, RoleAQ: true, RoleAK: true, RoleAV: true,
	RoleAOut: true, RoleAPos: true, RoleAGLUVal: true, RoleAGLUGate: true, RoleAPW2: true,
	RoleAFF2Up: true, RoleAFF2Down: true, RoleAAd1: true, RoleAAd2: true, RoleAResDown: true, RoleAResUp: true,
}

// AudioMatrix reports whether r is an audio tower matrix (the converter's
// quantized set, a reader's bound one).
func AudioMatrix(r Role) bool { return audioMatrix[r] }

func init() {
	for r, n := range audioRoleNames {
		if was, ok := roleNames[r]; ok {
			panic(fmt.Sprintf("jlm: role %d is both %q and %q", r, was, n))
		}
		roleNames[r] = n
	}
	visionFamProjNames[ProjD1OmniA] = "d1omni_a"
}

// audioRole reports whether r is one of the audio tower's roles.
func audioRole(r Role) bool {
	_, ok := audioRoleNames[r]
	return ok
}

// audioExpanded is Role.Expanded for the audio roles: every vector, and the
// convolutions' taps, which a reader copies out at load.
func audioExpanded(r Role) bool { return audioRole(r) && !audioMatrix[r] }

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

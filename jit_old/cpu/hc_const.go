package cpu

// DeepSeek V4's two small kernels, the hyper-connection mixer and the
// compressor's pool. This file holds what every tier shares: the constant
// block and the ABI; hc.go (AVX2), sse_hc.go and hc_a64.go are the bodies.
//
// EmitHCMix turns one row's (2+H)*H mixes, H = HCStreams, into pre (sigma +
// eps), post (2*sigma) and the 4 x 4 comb: a row softmax plus eps, one column
// normalisation, then iters-1 rounds of row and column (iters 0 stops after
// the softmax). Out the 24 outputs (8 for a head, which computes pre alone),
// Q32 the mixes, AScale their scales, Q2 their bases, Scr HCMixConsts(eps).
// Every tier keeps the mixer one row per 128-bit lane.
//
// EmitColPool is the compressor's per-channel softmax over positions,
// out[c] = sum_s softmax_s(gate[s,c]) * kv[s,c]: Out W wide, Q32 kv and Q2
// gate (S rows of W), Scr ActConsts(), K and Rows W's whole vectors and tail
// (ElemLanes), Cols S (at least one), RowStr 4*W.
//
// The formulas and the ABI in full: docs/engineering-history/cpu-kernels.md,
// "jit_old/cpu/hc_const.go".

// HCStreams is the stream count the mixer is built for: DeepSeek V4's.
const HCStreams = 4

// HCMixConsts is the mixer's constant block: exp's block, padded to 64 bytes,
// then pre/post's multiplier (1 x4, 2 x4) at 64, its offset (eps x4, 0 x4) at
// 96 and eps at 128.
func HCMixConsts(eps float32) []float32 {
	c := make([]float32, 16, 33)
	copy(c, expConsts())
	c = append(c, 1, 1, 1, 1, 2, 2, 2, 2)
	c = append(c, eps, eps, eps, eps, 0, 0, 0, 0)
	return append(c, eps)
}

const (
	hcMulOff = 64
	hcAddOff = 96
	hcEpsOff = 128
)

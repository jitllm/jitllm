package jlm

// The state-space hybrids and the short convolutions. Their codes sit in the
// middle of the 6-bit range (43..49), clear of the dense group counting down
// from 63 and of additions counting up from ArchHunyuan; Arch.Valid asks the
// name tables, so the gaps are not codes.
//
// Each is a code of its own so a reader that predates it refuses the
// container: every one of their layers would otherwise run as a gated delta
// rule, fluently.
const (
	// ArchMamba2 is Mamba-2 (state-spaces/mamba2, Mamba-Codestral): every
	// block a Mamba-2 mixer (LayerSSD) under one RMSNorm, and no FFN.
	ArchMamba2 Arch = 100

	// ArchGraniteHybrid is IBM Granite 4.0-H: Granite's four scales
	// (ArchGranite's) over Mamba-2 blocks and attention blocks with no rotary,
	// each followed by a SwiGLU FFN or a softmax mixture beside a shared MLP.
	ArchGraniteHybrid Arch = 101

	// ArchNemotronH is NVIDIA's Nemotron-H and Nemotron Nano 2, and its
	// mixture Nemotron 3 Nano (nemotron_h_moe): every source layer is ONE
	// mixer under its own RMSNorm -- Mamba-2, attention with no rotary, or a
	// squared-ReLU MLP (a mixture of them in the MoE) -- merged at conversion
	// into blocks of a mixer and an optional FFN.
	ArchNemotronH Arch = 102

	// ArchFalconH1 is TII's Falcon-H1: every block runs attention (NEOX
	// rotary) and a Mamba-2 mixer in parallel on one normed input
	// (jlm.LayerSSDAttn), then a SwiGLU FFN. Its muP multipliers are folded
	// into the weights by the converter that writes the GGUF.
	ArchFalconH1 Arch = 103

	// ArchLFM2 is Liquid's LFM2: gated short-convolution blocks
	// (jlm.LayerShortConv) with attention blocks among them (NEOX rotary, a
	// q/k RMSNorm per head), each followed by a SwiGLU FFN.
	ArchLFM2 Arch = 104

	// ArchMamba is Mamba-1 (state-spaces/mamba, FalconMamba): every block a
	// Mamba-1 mixer (LayerMamba1) under one RMSNorm, and no FFN.
	ArchMamba Arch = 105

	// ArchJamba is AI21's Jamba: Mamba-1 mixers with weighted dt/B/C norms
	// and attention blocks with no rotary among them, each followed by a
	// SwiGLU FFN or a softmax top-k mixture (on the blocks with a router).
	ArchJamba Arch = 106

	// ArchLFM2Enc is LFM2 made an encoder (LiquidAI's LFM2.5-Encoder, d1-omni's
	// trunk): LFM2's blocks with no causal mask, a centred 3-tap convolution
	// where LFM2's is causal, and the final norm over every row. A code of its
	// own so a reader that knows only the decoder refuses the container rather
	// than running it with a causal mask. A media prefix's rows attend only to
	// each other and the convolution never lets them read the text.
	ArchLFM2Enc Arch = 107
)

var ssmArchNames = map[Arch]string{
	ArchMamba2: "mamba2", ArchGraniteHybrid: "granitehybrid", ArchNemotronH: "nemotron_h",
	ArchFalconH1: "falcon-h1", ArchLFM2: "lfm2", ArchMamba: "mamba", ArchJamba: "jamba",
	ArchLFM2Enc: "lfm2enc",
}

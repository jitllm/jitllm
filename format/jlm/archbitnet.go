package jlm

// ArchBitNet is Microsoft's BitNet b1.58 (BitNetForCausalLM; llama.cpp's
// "bitnet", bitnet.cpp's "bitnet-b1.58"): llama's block with ternary
// projections (TypeTQ2), every projection input quantized to int8 per token
// by its absolute maximum (the engine's activation window is the whole row),
// an RMSNorm over the attention output before wo (RoleAttnSubNorm) and over
// act(gate)*up before down (RoleFFNSubNorm), and a squared-ReLU gate. The code
// carries the per-token quantization and the sub-norms' places, which no file
// states. A dense-group code, from the top down (archdense.go).
const ArchBitNet Arch = 56

// RoleAttnSubNorm and RoleFFNSubNorm are BitNet's inner norms: RMSNorm
// weights over the attention output (NEmbd wide, before wo) and over the
// gated FFN's product (NFFN wide, before down). Per-block vectors, like every
// norm. Their codes are apart from every other group's so a concurrent
// addition does not collide.
const (
	RoleAttnSubNorm Role = 501
	RoleFFNSubNorm  Role = 502
)

func bitnetRoleName(r Role) (string, bool) {
	switch r {
	case RoleAttnSubNorm:
		return "attn_sub_norm", true
	case RoleFFNSubNorm:
		return "ffn_sub_norm", true
	}
	return "", false
}

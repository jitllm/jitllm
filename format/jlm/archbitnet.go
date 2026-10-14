package jlm

import "fmt"

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

var bitnetRoleNames = map[Role]string{RoleAttnSubNorm: "attn_sub_norm", RoleFFNSubNorm: "ffn_sub_norm"}

func init() {
	for r, n := range bitnetRoleNames {
		if was, ok := roleNames[r]; ok {
			panic(fmt.Sprintf("jlm: role %d is both %q and %q", r, was, n))
		}
		roleNames[r] = n
	}
	if was, ok := denseArchNames[ArchBitNet]; ok {
		panic(fmt.Sprintf("jlm: arch %d is both %q and bitnet", ArchBitNet, was))
	}
	denseArchNames[ArchBitNet] = "bitnet"
}

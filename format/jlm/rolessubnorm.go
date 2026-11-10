package jlm

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

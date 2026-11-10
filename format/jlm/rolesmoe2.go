package jlm

// RoleXIELU is an Apertus block's xIELU, four f32 in their effective form:
// alpha_p = softplus(a_p), alpha_n = beta + softplus(a_n), beta and eps. The
// converter folds the softplus once (the source carries a_p and a_n raw), so
// the kernels read the numbers as they are. A per-block vector, like a norm.
// Its code is apart from every other group's so a concurrent addition does
// not collide.
const RoleXIELU Role = 500

func moe2RoleName(r Role) (string, bool) {
	switch r {
	case RoleXIELU:
		return "ffn_xielu", true
	}
	return "", false
}

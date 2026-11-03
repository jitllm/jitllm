package jlm

// The modern text families added as one group: Ling 2.0, dots.llm1,
// Phi-3.5-MoE and Apertus. Their codes are 39..42 of the 6-bit range, between
// the families counting up from ArchDBRX and the dense group counting down
// from 63; Arch.Valid asks the name tables, so the gaps are not codes.
//
// Each is a code of its own so a reader that predates it refuses the container
// rather than running it as a neighbouring graph with a feature missing.
const (
	// ArchBailingMoE2 is inclusionAI's Ling and Ring 2.0 (Ling-mini-2.0,
	// Ling-flash-2.0): DeepSeek-V3's router -- sigmoid gating, a selection-only
	// bias (expert_bias), the group-limited top-k, renormalised, a routed scale
	// -- beside shared experts after a dense lead, on GQA attention with a
	// per-head q/k RMSNorm and NEOX rotary on part of each head. Its fused
	// query_key_value is unfused at conversion.
	ArchBailingMoE2 Arch = 42

	// ArchDots1 is rednote-hilab's dots.llm1: DeepSeek-V3's router (sigmoid,
	// a selection-only bias, renormalised, a routed scale) beside shared
	// experts after a dense lead, on attention (MHA in every published
	// checkpoint) with qwen3's per-head q/k RMSNorm and NEOX rotary over the
	// whole head.
	ArchDots1 Arch = 41

	// ArchPhiMoE is Microsoft's Phi-3.5-MoE: the classic sequential block --
	// LayerNorm with biases, biased q/k/v/o, a biased head behind a biased
	// LayerNorm -- with phi3's NEOX LongRoPE, and a mixture routed by
	// SPARSEMIXER's inference path: the top two logits, each weighted by a
	// softmax over the logits within a relative 2*PhiMoEJitter of it, not
	// renormalised. The code carries the router: a GGUF states no gating key
	// for it, and llama.cpp runs a renormalised softmax top-2 there.
	ArchPhiMoE Arch = 40

	// ArchApertus is the Swiss AI Initiative's Apertus: llama's block with
	// qwen3's per-head q/k RMSNorm, NEOX rotary with llama 3's per-pair
	// factors (rope_freqs), and an UNGATED MLP -- up, xIELU, down -- whose
	// activation carries four numbers per block (RoleXIELU). The code carries
	// the activation, since a GGUF states it only through those numbers.
	ArchApertus Arch = 39
)

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

// PhiMoEJitter is sparsemixer's jitter_eps: router_jitter_noise, which every
// published Phi-3.5-MoE (and PhimoeConfig's default) sets to 0.01. A GGUF does
// not carry it, so the architecture does; the safetensors converter refuses a
// config that states another value.
const PhiMoEJitter = 0.01

func moe2ArchName(a Arch) (string, bool) {
	switch a {
	case ArchBailingMoE2:
		return "bailingmoe2", true
	case ArchDots1:
		return "dots1", true
	case ArchPhiMoE:
		return "phimoe", true
	case ArchApertus:
		return "apertus", true
	}
	return "", false
}

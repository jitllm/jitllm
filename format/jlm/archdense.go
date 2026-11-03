package jlm

// The dense llama-family architectures added as one group. Their codes are
// taken from the top of the 6-bit range down, so a concurrent addition counting
// up from ArchDBRX does not collide with them; Arch.Valid asks the name tables
// rather than a maximum, so the gap between the two ends is not a code.
//
// Each is a code of its own so a reader that predates it refuses the container
// rather than running it as llama's block with a feature missing -- every one
// of those omissions is fluent.
const (
	// ArchSmolLM3 is HuggingFace's SmolLM3: llama's block with no rotary on
	// every SWAPeriod-th layer, (il+1) % SWAPeriod == 0 (FlagNoPEGlobal with
	// SWAWindow 0: the period alone names the NoPE layers, and no layer slides).
	// llama.cpp hardcodes the period at 4 (n_no_rope_layer_step).
	ArchSmolLM3 Arch = 63

	// ArchArcee is Arcee's AFM-4.5B: llama's block with an ungated squared-ReLU
	// FFN (up, relu^2, down; FlagReLU2 and no gate tensor).
	ArchArcee Arch = 62

	// ArchSeedOSS is ByteDance's Seed-OSS: llama's block with biased q, k and v
	// (tensors) and NEOX rotary. Its post_attention_norm is the FFN's pre-norm,
	// which the converter writes as RoleFFNNorm.
	ArchSeedOSS Arch = 61

	// ArchOLMo2 is AI2's OLMo 2 and OLMo 3: no pre-norms at all. q and k are
	// RMSNormed over the whole projection (FlagQKNormWide) before NEOX rotary,
	// and the attention and FFN outputs are RMSNormed (RolePostAttnNorm,
	// RolePostFFNNorm) before their residual adds. OLMo 3 adds a sliding window
	// on three layers in four.
	ArchOLMo2 Arch = 60

	// ArchExaone4 is LG's EXAONE 4.0: OLMo 2's post-norm block with qwen3's
	// per-head q/k RMSNorm, NEOX rotary, and on the 32B a sliding window on
	// three layers in four with no rotary on the global one (FlagNoPEGlobal).
	ArchExaone4 Arch = 59

	// ArchMistral3 is Mistral Small 3.1/3.2, Magistral, Devstral and Ministral
	// 3, text only: llama's block with an attention temperature on every layer
	// (AttnTemp*, offset 0, floor the YaRN original context) and YaRN.
	ArchMistral3 Arch = 58

	// ArchOLMo3 is AI2's OLMo 3: OLMo 2's block with a sliding window on
	// three layers in four, whose rotary carries no scaling -- neither YaRN's
	// frequencies nor its magnitude (AttnFactor), which reach the global
	// layers only. A code of its own so a reader that would scale the local
	// layers refuses the container.
	ArchOLMo3 Arch = 57
)

func denseArchName(a Arch) (string, bool) {
	switch a {
	case ArchBitNet:
		return "bitnet", true
	case ArchSmolLM3:
		return "smollm3", true
	case ArchArcee:
		return "arcee", true
	case ArchSeedOSS:
		return "seed_oss", true
	case ArchOLMo2:
		return "olmo2", true
	case ArchExaone4:
		return "exaone4", true
	case ArchMistral3:
		return "mistral3", true
	case ArchOLMo3:
		return "olmo3", true
	}
	return "", false
}

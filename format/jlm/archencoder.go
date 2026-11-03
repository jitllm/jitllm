package jlm

// The encoders after BERT, counting up from 150: clear of the state-space
// group (100..) and of the vision-language text models (200..).
const (
	// ArchModernBERT is ModernBERT (answerdotai/ModernBERT, mmBERT): a
	// pre-norm encoder with no biases, LayerNorms without bias, a fused q|k|v
	// unfused at conversion, NEOX rotary on every layer, and GeGLU. Every
	// SWAPeriod-th layer from the first attends globally at RopeBase; the
	// rest attend within a symmetric window (SWAWindow/2 positions either
	// side) at RopeBaseSWA. Layer 0 has no attention norm; a final norm ends
	// the stack. Laya's decision head (Config.DecisionBlocks pre-norm blocks
	// with biases, a token-type row per question type, a scorer) follows it.
	ArchModernBERT Arch = 150
)

func encoderArchName(a Arch) (string, bool) {
	switch a {
	case ArchModernBERT:
		return "modern-bert", true
	}
	return "", false
}

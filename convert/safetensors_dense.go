package convert

// The dense llama family from HuggingFace safetensors: the same containers
// convert/dense.go writes from a GGUF, read off transformers' own classes
// (models/<arch>/modeling_*.py and configuration_*.py).
//
// The trap these share is the class default. A key config.json does not carry
// takes the default of ITS class, and the classes disagree: num_key_value_heads
// is 4 for SmolLM3, 8 for Ministral3, 32 for Exaone4 and the attention head
// count for the rest; head_dim is 128 for Ministral3 and hidden/heads
// elsewhere; rope_theta is 2e6 for SmolLM3 and 5e5 for OLMo 3.
// llamaHFConfig's defaults are llama's, so each entry states its class's first.

// hfClassDefaults is what a class's config dataclass declares for the keys
// llamaHFConfig reads. A zero kvHeads or headDim means the generic derivation
// (MHA, hidden/heads), which is what those classes' __post_init__ does too.
type hfClassDefaults struct {
	kvHeads, headDim uint32
	ropeTheta        float64
	rmsEps           float64
	tied             bool
	nCtx             uint32
}

// apply writes d into the keys config.json does not carry. A key that is
// present, null included, is the file's answer.
func (d hfClassDefaults) apply(c *hfConfig) {
	if c.absent("num_key_value_heads") && d.kvHeads != 0 {
		v := d.kvHeads
		c.NumKeyValueHeads = &v
	}
	if c.absent("head_dim") && d.headDim != 0 {
		v := d.headDim
		c.HeadDim = &v
	}
	// RopeTheta is zero only when neither rope_theta nor rope_parameters
	// states a base (readHFConfig reads both).
	if c.RopeTheta == 0 {
		c.RopeTheta = d.ropeTheta
	}
	if c.absent("rms_norm_eps") {
		c.RMSNormEps = d.rmsEps
	}
	if c.absent("tie_word_embeddings") {
		t := d.tied
		c.TieWordEmbeddings = &t
	}
	if c.absent("max_position_embeddings") {
		c.MaxPositionEmbeddings = d.nCtx
	}
}

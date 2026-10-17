package model

// The latent-attention fixtures, whose MLA blocks are lowered through the
// block graph, join the five principles' gates by name: the dense MLA block
// (synth-deepseek-dense) as a principle fixture, and the MLA mixtures --
// DeepSeek-V3's grouped sigmoid router with its selection bias, routed scale
// and ungated shared expert after a dense lead (synth-deepseek), Glm4MoeLite's
// (synth-glm4moelite) and Kimi-Linear's NoPE MLA beside its delta
// layers (synth-kimilinear) -- as principle mixtures, with
// TestGraphMixturesPageWithTheSameAnswer their paging gate.
var graphMLAMixtures = []string{"synth-deepseek.jlm", "synth-glm4moelite.jlm", "synth-kimilinear.jlm"}

func init() {
	principleFixtures = append(principleFixtures, "synth-deepseek-dense.jlm")
	principleMixtures = append(principleMixtures, graphMLAMixtures...)
}

package convert

import "github.com/jitllm/jitllm/format/jlm"

// StateRevision states what this revision of the converter states beyond
// the earlier tails: the sliding layers' rotary magnitude where the
// architecture sets it apart (OLMo 3's local layers rotate at their base
// unscaled: transformers builds them a default rotary, llama.cpp passes
// attn_factor 1 there), and the layers a picture's rows see each other on
// (Gemma 4's 26B and 31B, use_bidirectional_attention "vision": the sliding
// ones; the E-models and Gemma 3 every layer, as the engine runs them).
func StateRevision(c *jlm.Config) {
	c.Revision = jlm.StatementRevision
	if c.Arch == jlm.ArchOLMo3 {
		c.RopeScaleSWA = 1
	}
	if c.Arch == jlm.ArchGemma4 && c.PLEDim == 0 && c.SWAWindow > 0 {
		c.BidirLayers = make([]bool, c.NLayer)
		for li := range c.BidirLayers {
			c.BidirLayers[li] = c.Sliding(li)
		}
	}
}

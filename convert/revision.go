package convert

import "github.com/jitllm/jitllm/format/jlm"

// StateRevision states what this revision of the converter states beyond
// the earlier tails, as the library's revision statement for the source's
// architecture says (mjStatesRevision), once the blocks' geometry is read:
// a model's activation quantization (BitNet b1.58's BitLinear: int8 per
// token by its absolute maximum), the sliding layers' rotary magnitude
// where they rotate apart from the model's (OLMo 3's local layers rotate
// at their base unscaled: transformers builds them a default rotary,
// llama.cpp passes attn_factor 1 there), a windowed model's period that
// starts at its global layer, and the layers a picture's rows see each
// other on (Gemma 4's 26B and 31B, use_bidirectional_attention "vision":
// the sliding ones, which its tower's span then states bidirectional; none
// stated is every layer, Gemma 3's, and the E-models' pictures are causal).
func StateRevision(c *Config) error {
	r := c.Stated
	if r == nil {
		return unstated("the revision")
	}
	c.Revision = jlm.StatementRevision
	stateRopeTable(c)
	if r.actQuant != jlm.ActQuantNone {
		c.ActQuant = r.actQuant
	}
	if r.ropeScaleSWA != 0 {
		c.RopeScaleSWA = r.ropeScaleSWA
	}
	if r.swaGlobalFirst && c.SWAWindow != 0 {
		c.Flags2 |= jlm.Flag2SWAGlobalFirst
	}
	if r.pictureBidirSliding && c.SWAWindow > 0 {
		c.BidirLayers = make([]bool, c.NLayer)
		for li := range c.BidirLayers {
			c.BidirLayers[li] = c.Sliding(li)
		}
	}
	return nil
}

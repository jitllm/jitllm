package model

// The gated delta net: the other half of a hybrid's block loop.
//
// The graph is transcribed tensor by tensor from the GGUF oracle's running
// graph (JITLLM_LCPP; layer 0 of Qwen3-Next-80B-A3B), not re-derived from
// transformers' Qwen3NextGatedDeltaNet, and six of its steps are ones a
// from-paper reading gets wrong:
//
//	(1) SiLU on the convolution's output, before it is split
//	(2) q and k L2-normalised: no weight, divided by the vector's own length
//	(3) 16 key heads and 32 value heads, interleaved: value head h reads h/2
//	(4) the decay softplus(a + dt_bias) * ssm_a, ssm_a already -exp(A_log)
//	(5) beta a plain sigmoid, no bias
//	(6) the output gate SiLU(z), where the full-attention layers use sigmoid
//
// The state is not a KV cache: it is the same size at every position, so
// residency, paging and placement cannot treat it as KV.
//
// The node-by-node transcript: .notes/engineering-history/model-correctness.md,
// "model/delta.go".

// recTailLen is what each linear block's recurrent state carries per
// sequence after the delta rule's: the room the stated programs' tail ops
// keep (jlm.Program.Tail), a token window's predecessors or a convolution's
// history, which the block stating them fills and every other leaves
// unread. Every linear block has the same row, [delta rule | tail], so a
// device keeps one slot width for all of them, the slot each graph's delta
// rule states (graph.OpDeltaRule's Span).
func (c *Config) RecTailLen() int {
	t := 0
	if c.stated != nil {
		for i := range c.stated.Blocks {
			t = max(t, c.stated.Blocks[i].Pre.Read.Tail())
		}
	}
	return t
}

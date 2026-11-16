package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// promptBuilder writes a jlm.MarkerPrompt node by node.
type promptBuilder struct{ p jlm.MarkerPrompt }

func (b *promptBuilder) node(n jlm.PromptNode) uint32 {
	b.p.Nodes = append(b.p.Nodes, n)
	return uint32(len(b.p.Nodes) - 1)
}

func (b *promptBuilder) op(op jlm.PromptOp, args ...uint32) uint32 {
	return b.node(jlm.PromptNode{Op: op, Args: args})
}

func (b *promptBuilder) opInt(op jlm.PromptOp, i int64, args ...uint32) uint32 {
	return b.node(jlm.PromptNode{Op: op, Int: i, Args: args})
}

func (b *promptBuilder) num(i int64) uint32 { return b.opInt(jlm.PromptInt, i) }

func (b *promptBuilder) text(s string) uint32 {
	return b.node(jlm.PromptNode{Op: jlm.PromptText, Text: s})
}

// StateMarkerPrompt states a marker head's prompt in s's config, as its
// readout's statement writes it (readouts.json), its sequence cut to maxLen
// tokens (decision.max_tokens; 0 where the file is silent, the statement's).
// A model with no marker head states none.
func StateMarkerPrompt(s *jlm.Source, maxLen int64) error {
	c, v := s.Config, s.Vocab
	if c == nil || c.Flags2&(jlm.Flag2HeadBlocks|jlm.Flag2MarkerPointer) == 0 {
		return nil
	}
	if v == nil {
		return fmt.Errorf("convert: a marker head's prompt needs its vocabulary")
	}
	r := libraryReadout(c.Decision)
	if r == nil || r.Prompt == nil {
		return fmt.Errorf("convert: %v states head blocks and no prompt of its kind: %w", c.Decision, ErrNotImplemented)
	}
	p, err := r.Prompt.program(c.Decision.String(), v, maxLen, int64(c.DecisionHeadTokens))
	if err != nil {
		return err
	}
	c.MarkerPrompt = p
	return nil
}

// tokenOf is the id of the token spelled s, -1 for none.
func tokenOf(v *jlm.Vocab, s string) int32 {
	for i, t := range v.Tokens {
		if t == s {
			return int32(i)
		}
	}
	return -1
}

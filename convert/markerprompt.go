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

func (b *promptBuilder) text(s string) uint32 { return b.node(jlm.PromptNode{Op: jlm.PromptText, Text: s}) }

func (b *promptBuilder) ids(ids ...int32) uint32 { return b.node(jlm.PromptNode{Op: jlm.PromptIDs, IDs: ids}) }

// StateMarkerPrompt states a marker head's prompt in s's config: Laya's, from
// rl_common.build_sequence, its sequence cut to maxLen tokens. A model with
// no marker head states none.
func StateMarkerPrompt(s *jlm.Source, maxLen int64) error {
	c, v := s.Config, s.Vocab
	if c == nil || c.Flags2&jlm.Flag2HeadBlocks == 0 || c.Decision != jlm.DecisionLaya {
		return nil
	}
	if v == nil || v.BOS < 0 || v.Sep < 0 || v.Mask < 0 || int(v.Mask) >= len(v.Tokens) {
		return fmt.Errorf("convert: a laya readout needs its vocabulary's [CLS], [SEP] and [MASK]")
	}
	if maxLen <= 0 {
		return fmt.Errorf("convert: a laya readout's sequence of %d tokens", maxLen)
	}
	p, err := layaPrompt(v.BOS, v.Sep, v.Mask, v.Tokens[v.Mask], int64(c.DecisionHeadTokens), maxLen)
	if err != nil {
		return err
	}
	c.MarkerPrompt = p
	return nil
}

// layaPrompt is rl_common.build_sequence as a program:
//
//	[CLS] "<type> question: <ins>" [SEP] ([MASK] " <option>")* [SEP] <state> [SEP]
//
// each option's text cut to 48 tokens; where the options leave the question
// fewer than 16 of the head's headMax, every option sequence cut to
// max(4, (headMax-16)/k); the question cut to max(8, what the options leave);
// the state to the room maxLen leaves (or the instance, the smaller); the
// whole to it. The question's instructions are a string or their JSON
// (ASCII-escaped, _to_internal's json.dumps), the state its JSON
// (serialize_state's, unescaped), an option's description Python's str() of
// it, and the mask's text is a space in every piece. Its confidence is one
// minus the normalised entropy (confidence_from_probs).
func layaPrompt(cls, sep, mask int32, maskText string, headMax, maxLen int64) (*jlm.MarkerPrompt, error) {
	var b promptBuilder
	space, colon, noMask := b.text(" "), b.text(": "), b.text(maskText)
	unmask := func(t uint32) uint32 { return b.op(jlm.PromptReplace, t, noMask, space) }
	key := b.op(jlm.PromptOptionKey)
	desc := b.opInt(jlm.PromptOptionText, int64(jlm.PromptPyStr))
	empty := b.opInt(jlm.PromptOptionEmpty, int64(jlm.PromptFalsy))
	idx := b.op(jlm.PromptOptionIndex)
	choice := b.op(jlm.PromptSelect, empty, key, b.op(jlm.PromptConcat, key, colon, desc))
	score := b.op(jlm.PromptConcat, b.text("level "), b.op(jlm.PromptFormat, idx), colon, desc)
	noulDefault := b.op(jlm.PromptSelect, b.op(jlm.PromptLess, idx, b.num(1)), b.text("no, the statement does not hold"),
		b.text("yes, the statement holds"))
	noul := b.op(jlm.PromptConcat, key, colon, b.op(jlm.PromptSelect, empty, noulDefault, desc))
	opt := b.op(jlm.PromptPick, b.op(jlm.PromptTypeIndex), choice, score, noul)
	optIDs := b.op(jlm.PromptCat, b.node(jlm.PromptNode{Op: jlm.PromptMarked, IDs: []int32{mask}}),
		b.op(jlm.PromptCut, b.op(jlm.PromptEncode, b.op(jlm.PromptConcat, space, unmask(opt))), b.num(48)))
	head := b.num(headMax)
	budget := b.op(jlm.PromptSub, head, b.op(jlm.PromptSum, b.op(jlm.PromptLen, optIDs)))
	per := b.op(jlm.PromptMax, b.num(4), b.op(jlm.PromptDiv, b.op(jlm.PromptSub, head, b.num(16)), b.op(jlm.PromptOptions)))
	opts := b.op(jlm.PromptSelect, b.op(jlm.PromptLess, budget, b.num(16)), b.op(jlm.PromptCut, optIDs, per), optIDs)
	budget = b.op(jlm.PromptSub, head, b.op(jlm.PromptSum, b.op(jlm.PromptLen, opts)))
	question := b.op(jlm.PromptConcat, b.op(jlm.PromptType), b.text(" question: "),
		unmask(b.opInt(jlm.PromptInstructions, int64(jlm.PromptJSONASCII))))
	qIDs := b.op(jlm.PromptCut, b.op(jlm.PromptEncode, question), b.op(jlm.PromptMax, b.num(8), budget))
	sepIDs := b.ids(sep)
	pre := b.op(jlm.PromptCat, b.ids(cls), qIDs, sepIDs, b.op(jlm.PromptFlatten, opts), sepIDs)
	ctx := b.op(jlm.PromptMin, b.op(jlm.PromptContext), b.num(maxLen))
	room := b.op(jlm.PromptMax, b.num(0), b.op(jlm.PromptSub, b.op(jlm.PromptSub, ctx, b.op(jlm.PromptLen, pre)), b.num(1)))
	state := b.op(jlm.PromptCut, b.op(jlm.PromptEncode, unmask(b.opInt(jlm.PromptState, int64(jlm.PromptJSON)))), room)
	b.p.Out = b.op(jlm.PromptCut, b.op(jlm.PromptCat, pre, state, sepIDs), ctx)
	b.p.EntropyConfidence = true
	return &b.p, b.p.Valid()
}

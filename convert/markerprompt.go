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

// StateMarkerPrompt states a marker head's prompt in s's config, its
// sequence cut to maxLen tokens (decision.max_tokens; 0 where the file is
// silent, the class's): Laya's, from rl_common.build_sequence (512,
// rl_agent_config's max_len), or d1-omni's, from prompt.encode (16384,
// config.max_length). A model with no marker head states none.
func StateMarkerPrompt(s *jlm.Source, maxLen int64) error {
	c, v := s.Config, s.Vocab
	if c == nil || c.Flags2&(jlm.Flag2HeadBlocks|jlm.Flag2MarkerPointer) == 0 {
		return nil
	}
	if v == nil {
		return fmt.Errorf("convert: a marker head's prompt needs its vocabulary")
	}
	if v.BOS < 0 && c.Decision != jlm.DecisionKev {
		return fmt.Errorf("convert: a marker head's prompt needs its vocabulary's first token")
	}
	var p *jlm.MarkerPrompt
	var err error
	switch c.Decision {
	case jlm.DecisionLaya:
		if v.Sep < 0 || v.Mask < 0 || int(v.Mask) >= len(v.Tokens) {
			return fmt.Errorf("convert: a laya readout needs its vocabulary's [CLS], [SEP] and [MASK]")
		}
		if maxLen == 0 {
			maxLen = 512
		}
		p, err = layaPrompt(v.BOS, v.Sep, v.Mask, v.Tokens[v.Mask], int64(c.DecisionHeadTokens), maxLen)
	case jlm.DecisionD1Omni:
		var d [6]int32
		for i, name := range []string{"<|reserved_7|>", "<|reserved_8|>", "<|reserved_9|>", "<|reserved_10|>", "<|reserved_11|>", "<|mask|>"} {
			if d[i] = tokenOf(v, name); d[i] < 0 {
				return fmt.Errorf("convert: a d1-omni readout's vocabulary has no %s", name)
			}
		}
		if maxLen == 0 {
			maxLen = 16384
		}
		p, err = d1Prompt(v.BOS, d, maxLen)
	case jlm.DecisionKev:
		var d [5]int32
		for i, name := range []string{"<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"} {
			if d[i] = tokenOf(v, name); d[i] < 0 {
				return fmt.Errorf("convert: a kev readout's vocabulary has no %s", name)
			}
		}
		p, err = kevPrompt(d)
	default:
		return fmt.Errorf("convert: %v states head blocks and no prompt of its kind: %w", c.Decision, ErrNotImplemented)
	}
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
	b.p.Confidence = jlm.ConfidenceEntropy
	return &b.p, b.p.Valid()
}

// kevPrompt is Kev's systemone template as a program, d its delimiters
// (fim_prefix, fim_middle, box_start, box_end, fim_suffix):
//
//	<|fim_prefix|>state<|fim_middle|>instructions(<|box_start|>option<|box_end|>)*<|fim_suffix|>
//
// each piece encoded alone, as the template's specials part them; a choice's
// option is its key, a noul's no or yes (false then true), each with ": " and
// its description where it has one, a score's its description alone; a
// value printed as the template prints it, the state as llama.cpp's
// template engine prints an object, its JSON (Jinja2 prints Python's str()
// of it; Kev's goldens are llama.cpp's). An option's <|box_end|> is its
// marker, the prompt's last row the pointer's query.
func kevPrompt(d [5]int32) (*jlm.MarkerPrompt, error) {
	var b promptBuilder
	none := b.text("")
	desc := b.opInt(jlm.PromptOptionText, int64(jlm.PromptTemplate))
	empty := b.opInt(jlm.PromptOptionEmpty, int64(jlm.PromptFalsy))
	described := b.op(jlm.PromptSelect, empty, none, b.op(jlm.PromptConcat, b.text(": "), desc))
	choice := b.op(jlm.PromptConcat, b.op(jlm.PromptOptionKey), described)
	score := b.op(jlm.PromptSelect, empty, none, desc)
	noul := b.op(jlm.PromptConcat, b.op(jlm.PromptSelect, b.op(jlm.PromptLess, b.op(jlm.PromptOptionIndex), b.num(1)),
		b.text("no"), b.text("yes")), described)
	opt := b.op(jlm.PromptPick, b.op(jlm.PromptTypeIndex), choice, score, noul)
	opts := b.op(jlm.PromptCat, b.ids(d[2]), b.op(jlm.PromptEncode, opt), b.node(jlm.PromptNode{Op: jlm.PromptMarked, IDs: []int32{d[3]}}))
	state := b.op(jlm.PromptEncode, b.opInt(jlm.PromptState, int64(jlm.PromptJSON)))
	ins := b.op(jlm.PromptEncode, b.opInt(jlm.PromptInstructions, int64(jlm.PromptTemplate)))
	b.p.Out = b.op(jlm.PromptCut, b.op(jlm.PromptCat, b.ids(d[0]), state, b.ids(d[1]), ins, b.op(jlm.PromptFlatten, opts),
		b.ids(d[4])), b.op(jlm.PromptContext))
	return &b.p, b.p.Valid()
}

// d1Prompt is d1-omni's prompt.encode as a program, d its delimiters (the
// state's, the question's, an option's opening and closing, the decision's)
// and its marker:
//
//	<bos> <state> state <q> instructions (<opt> <mask> option </opt>)* <decide>
//
// the options' block budget = max(96, min(24k+32, maxLen/2)) tokens, each
// option's text cut to max(2, (budget-3k)/k); the question to max(16,
// budget); the state to the room maxLen leaves (or the instance, the
// smaller); the whole to it. Caller text has every <|name|> rewritten to
// <¦name¦> before it is encoded; the instructions are Python's str() of
// theirs, a description its string or its JSON, empty where None or "".
// Its confidence is the largest probability (prompt.answer).
func d1Prompt(bos int32, d [6]int32, maxLen int64) (*jlm.MarkerPrompt, error) {
	var b promptBuilder
	colon := b.text(": ")
	escape := func(t uint32) uint32 {
		return b.node(jlm.PromptNode{Op: jlm.PromptRegexp, Args: []uint32{t, b.text("<¦${1}¦>")}, Text: `<\|([A-Za-z0-9_]+)\|>`})
	}
	key := b.op(jlm.PromptOptionKey)
	desc := b.opInt(jlm.PromptOptionText, int64(jlm.PromptJSON))
	empty := b.opInt(jlm.PromptOptionEmpty, int64(jlm.PromptNoneOrEmpty))
	idx := b.op(jlm.PromptOptionIndex)
	choice := b.op(jlm.PromptSelect, empty, key, b.op(jlm.PromptConcat, key, colon, desc))
	score := b.op(jlm.PromptConcat, b.text("level "), b.op(jlm.PromptFormat, idx), colon, desc)
	noulDefault := b.op(jlm.PromptSelect, b.op(jlm.PromptLess, idx, b.num(1)), b.text("no, the statement does not hold"),
		b.text("yes, the statement holds"))
	noul := b.op(jlm.PromptConcat, key, colon, b.op(jlm.PromptSelect, empty, noulDefault, desc))
	opt := b.op(jlm.PromptPick, b.op(jlm.PromptTypeIndex), choice, score, noul)
	k := b.op(jlm.PromptOptions)
	ctx := b.op(jlm.PromptMin, b.op(jlm.PromptContext), b.num(maxLen))
	budget := b.op(jlm.PromptMax, b.num(96), b.op(jlm.PromptMin,
		b.op(jlm.PromptAdd, b.op(jlm.PromptMul, k, b.num(24)), b.num(32)), b.op(jlm.PromptDiv, ctx, b.num(2))))
	per := b.op(jlm.PromptMax, b.num(2), b.op(jlm.PromptDiv, b.op(jlm.PromptSub, budget, b.op(jlm.PromptMul, b.num(3), k)), k))
	text := b.op(jlm.PromptCut, b.op(jlm.PromptEncode, escape(b.op(jlm.PromptConcat, b.text(" "), opt))), per)
	opts := b.op(jlm.PromptCat, b.ids(d[2]), b.node(jlm.PromptNode{Op: jlm.PromptMarked, IDs: []int32{d[5]}}), text, b.ids(d[3]))
	ins := escape(b.opInt(jlm.PromptInstructions, int64(jlm.PromptPyStr)))
	question := b.op(jlm.PromptCut, b.op(jlm.PromptCat, b.ids(d[1]), b.op(jlm.PromptEncode, ins)),
		b.op(jlm.PromptMax, b.num(16), budget))
	question = b.op(jlm.PromptCat, question, b.op(jlm.PromptFlatten, opts), b.ids(d[4]))
	room := b.op(jlm.PromptMax, b.num(0), b.op(jlm.PromptSub, b.op(jlm.PromptSub, ctx, b.op(jlm.PromptLen, question)), b.num(2)))
	state := b.op(jlm.PromptCat, b.ids(d[0]),
		b.op(jlm.PromptCut, b.op(jlm.PromptEncode, escape(b.opInt(jlm.PromptState, int64(jlm.PromptJSON)))), room))
	b.p.Out = b.op(jlm.PromptCut, b.op(jlm.PromptCat, b.ids(bos), state, question), ctx)
	b.p.Confidence = jlm.ConfidencePMax
	return &b.p, b.p.Valid()
}

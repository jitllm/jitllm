package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// readoutPrompt is a marker head's prompt as the readout library states it:
// the tokens it names (a vocabulary special by its kind, or a token by its
// text), the cut its sequence takes where the file states none, the program
// -- let bindings in order, then out, each an expression of prompt
// operations -- and the confidence its answer states.
type readoutPrompt struct {
	MaxLen     int64                  `json:"max_len"`
	Tokens     map[string]promptToken `json:"tokens"`
	Let        [][2]string            `json:"let"`
	Out        string                 `json:"out"`
	Confidence string                 `json:"confidence"`
}

// promptToken is a token a prompt names: a vocabulary special (bos, sep or
// mask), or the token spelled Name.
type promptToken struct {
	Special string `json:"special"`
	Name    string `json:"name"`
}

// promptNullary are the prompt operations of no operand, by the name a
// statement calls them.
func promptNullary() map[string]jlm.PromptOp {
	return map[string]jlm.PromptOp{"context": jlm.PromptContext, "type": jlm.PromptType, "type_index": jlm.PromptTypeIndex,
		"option_key": jlm.PromptOptionKey, "option_index": jlm.PromptOptionIndex, "options": jlm.PromptOptions,
		"pictures": jlm.PromptPictures, "clips": jlm.PromptClips}
}

// promptOperands are the prompt operations over values, by name, with how
// many operands each takes (-1 for any).
func promptOperands() map[string]struct {
	op jlm.PromptOp
	n  int
} {
	type o = struct {
		op jlm.PromptOp
		n  int
	}
	return map[string]o{"concat": {jlm.PromptConcat, -1}, "cat": {jlm.PromptCat, -1}, "max": {jlm.PromptMax, 2},
		"min": {jlm.PromptMin, 2}, "pick": {jlm.PromptPick, -1}, "cut": {jlm.PromptCut, 2}, "encode": {jlm.PromptEncode, 1},
		"flatten": {jlm.PromptFlatten, 1}, "len": {jlm.PromptLen, 1}, "sum": {jlm.PromptSum, 1},
		"format": {jlm.PromptFormat, 1}, "replace": {jlm.PromptReplace, 3}}
}

// promptRendered are the operations that render a request value, by name,
// and the renderings and emptinesses they take, by the name a statement
// gives them.
func promptRendered() map[string]jlm.PromptOp {
	return map[string]jlm.PromptOp{"instructions": jlm.PromptInstructions, "state": jlm.PromptState,
		"option_text": jlm.PromptOptionText}
}

func promptRenders() map[string]int64 {
	return map[string]int64{"pystr": int64(jlm.PromptPyStr), "json": int64(jlm.PromptJSON),
		"json_ascii": int64(jlm.PromptJSONASCII), "template": int64(jlm.PromptTemplate),
		"json_or_empty": int64(jlm.PromptJSONOrEmpty)}
}

func promptEmpties() map[string]int64 {
	return map[string]int64{"falsy": int64(jlm.PromptFalsy), "none_or_empty": int64(jlm.PromptNoneOrEmpty)}
}

func promptArith() map[string]jlm.PromptOp {
	return map[string]jlm.PromptOp{"+": jlm.PromptAdd, "-": jlm.PromptSub, "*": jlm.PromptMul, "/": jlm.PromptDiv,
		"<": jlm.PromptLess}
}

// promptLowering writes a prompt statement's program node by node: each
// operation after its operands, left to right, a let's value once where it
// is bound, a literal or a fact (a number of the conversion's) a node of its
// own where it is read.
type promptLowering struct {
	b      promptBuilder
	lets   map[string]uint32
	tokens map[string]int32
	vocab  *jlm.Vocab
	facts  map[string]int64
}

func (pl *promptLowering) token(n *exprNode) (int32, error) {
	if n.op != "path" {
		return 0, fmt.Errorf("%s is not a token the prompt names", exprText(n))
	}
	id, ok := pl.tokens[n.str]
	if !ok {
		return 0, fmt.Errorf("%q is not a token the prompt names", n.str)
	}
	return id, nil
}

func (pl *promptLowering) lower(n *exprNode) (uint32, error) {
	switch n.op {
	case "num":
		if n.num != math.Trunc(n.num) || math.Abs(n.num) > 1<<53 {
			return 0, fmt.Errorf("%v is not an integer", n.num)
		}
		return pl.b.num(int64(n.num)), nil
	case "str":
		return pl.b.text(n.str), nil
	case "path":
		if id, ok := pl.lets[n.str]; ok {
			return id, nil
		}
		if v, ok := pl.facts[n.str]; ok {
			return pl.b.num(v), nil
		}
		return 0, fmt.Errorf("%q is no binding, and no fact of the conversion", n.str)
	case "?:":
		args, err := pl.lowerAll(n.args)
		if err != nil {
			return 0, err
		}
		return pl.b.op(jlm.PromptSelect, args...), nil
	case "call":
		return pl.call(n)
	}
	if op, ok := promptArith()[n.op]; ok && len(n.args) == 2 {
		args, err := pl.lowerAll(n.args)
		if err != nil {
			return 0, err
		}
		return pl.b.op(op, args...), nil
	}
	return 0, fmt.Errorf("%s is not in the prompt language", exprText(n))
}

func (pl *promptLowering) lowerAll(ns []*exprNode) ([]uint32, error) {
	out := make([]uint32, len(ns))
	for i, a := range ns {
		x, err := pl.lower(a)
		if err != nil {
			return nil, err
		}
		out[i] = x
	}
	return out, nil
}

func (pl *promptLowering) call(n *exprNode) (uint32, error) {
	name, args := n.str, n.args
	literal := func(table map[string]int64) (int64, error) {
		if len(args) != 1 || args[0].op != "str" {
			return 0, fmt.Errorf("%s takes the name of how it reads", name)
		}
		v, ok := table[args[0].str]
		if !ok {
			return 0, fmt.Errorf("%s: %q is no reading of it", name, args[0].str)
		}
		return v, nil
	}
	if op, ok := promptNullary()[name]; ok {
		if len(args) != 0 {
			return 0, fmt.Errorf("%s takes no operand", name)
		}
		return pl.b.op(op), nil
	}
	if op, ok := promptRendered()[name]; ok {
		r, err := literal(promptRenders())
		if err != nil {
			return 0, err
		}
		return pl.b.opInt(op, r), nil
	}
	switch name {
	case "option_empty":
		e, err := literal(promptEmpties())
		if err != nil {
			return 0, err
		}
		return pl.b.opInt(jlm.PromptOptionEmpty, e), nil
	case "ids", "marked":
		var ids []int32
		for _, a := range args {
			id, err := pl.token(a)
			if err != nil {
				return 0, err
			}
			ids = append(ids, id)
		}
		op := jlm.PromptIDs
		if name == "marked" {
			op = jlm.PromptMarked
		}
		return pl.b.node(jlm.PromptNode{Op: op, IDs: ids}), nil
	case "text_of":
		if len(args) != 1 {
			return 0, fmt.Errorf("text_of takes one token")
		}
		id, err := pl.token(args[0])
		if err != nil {
			return 0, err
		}
		return pl.b.text(pl.vocab.Tokens[id]), nil
	case "regexp":
		if len(args) != 3 || args[2].op != "str" {
			return 0, fmt.Errorf("regexp takes a text, its replacement and a pattern")
		}
		in, err := pl.lowerAll(args[:2])
		if err != nil {
			return 0, err
		}
		return pl.b.node(jlm.PromptNode{Op: jlm.PromptRegexp, Args: in, Text: args[2].str}), nil
	}
	o, ok := promptOperands()[name]
	if !ok {
		return 0, fmt.Errorf("%s is no prompt operation", name)
	}
	if o.n >= 0 && len(args) != o.n {
		return 0, fmt.Errorf("%s takes %d operand(s), not %d", name, o.n, len(args))
	}
	in, err := pl.lowerAll(args)
	if err != nil {
		return 0, err
	}
	return pl.b.op(o.op, in...), nil
}

// program is p's prompt for vocabulary v, its sequence cut to maxLen tokens
// (p.MaxLen where it is 0) and its head's budget headTokens.
func (p *readoutPrompt) program(name string, v *jlm.Vocab, maxLen, headTokens int64) (*jlm.MarkerPrompt, error) {
	if maxLen == 0 {
		maxLen = p.MaxLen
	}
	pl := &promptLowering{lets: map[string]uint32{}, tokens: map[string]int32{}, vocab: v,
		facts: map[string]int64{"max_len": maxLen, "head_tokens": headTokens}}
	for tn, t := range p.Tokens {
		id := int32(-1)
		switch t.Special {
		case "bos":
			id = v.BOS
		case "sep":
			id = v.Sep
		case "mask":
			id = v.Mask
		case "":
			id = tokenOf(v, t.Name)
		default:
			return nil, fmt.Errorf("convert: readouts.json: %s: %q is no vocabulary special", name, t.Special)
		}
		if id < 0 || int(id) >= len(v.Tokens) {
			return nil, fmt.Errorf("convert: a %s readout's vocabulary has no %s%s", name, t.Special, t.Name)
		}
		pl.tokens[tn] = id
	}
	bind := func(what, src string) (uint32, error) {
		x, err := parseExpr(src)
		if err != nil {
			return 0, fmt.Errorf("convert: readouts.json: %s: %s: %w", name, what, err)
		}
		id, err := pl.lower(x)
		if err != nil {
			return 0, fmt.Errorf("convert: readouts.json: %s: %s: %w", name, what, err)
		}
		return id, nil
	}
	for _, l := range p.Let {
		id, err := bind(l[0], l[1])
		if err != nil {
			return nil, err
		}
		pl.lets[l[0]] = id
	}
	out, err := bind("out", p.Out)
	if err != nil {
		return nil, err
	}
	pl.b.p.Out = out
	pl.b.p.Confidence = readoutConfidences()[p.Confidence]
	return &pl.b.p, pl.b.p.Valid()
}

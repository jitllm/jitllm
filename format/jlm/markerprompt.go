package jlm

import (
	"fmt"
	"regexp"
)

// MarkerPrompt is how a marker head's question becomes its prompt's ids and
// its options' marker rows (Flag2HeadBlocks): a program the converter writes
// from the model's own prompt code, nodes evaluated in order, each over
// earlier nodes' values. A value is an integer, a text or a token sequence
// whose marked tokens are the options' markers; a node over an option's
// field holds one value an option, and every node over such values is taken
// option by option, a scalar operand standing for every option. Out is the
// node whose token sequence is the prompt; its marked tokens, in order, are
// the options' markers. Confidence is how an answer's confidence is read.
type MarkerPrompt struct {
	Nodes      []PromptNode
	Out        uint32
	Confidence PromptConfidence
}

// PromptConfidence is how a marker head's answer states its confidence: as a
// label readout's question types do (ConfidenceOfType), one minus the
// distribution's entropy over log k (ConfidenceEntropy), or its largest
// probability (ConfidencePMax).
type PromptConfidence uint8

const (
	ConfidenceOfType PromptConfidence = iota
	ConfidenceEntropy
	ConfidencePMax
)

// PromptNode is one step of a MarkerPrompt: Op over the values of the nodes
// Args names, with Int, Text and IDs its literals.
type PromptNode struct {
	Op   PromptOp
	Args []uint32
	Int  int64
	Text string
	IDs  []int32
}

// PromptOp is a MarkerPrompt node's operation.
type PromptOp uint8

const (
	PromptInvalid PromptOp = iota
	// PromptInt, PromptText and PromptIDs are literals: Int, Text, IDs.
	PromptInt
	PromptText
	PromptIDs
	// PromptMarked is IDs, its first token an option's marker.
	PromptMarked
	// PromptContext is the rows the instance holds.
	PromptContext
	// PromptType is the question's type name (choice, score or noul) and
	// PromptTypeIndex its index in that order.
	PromptType
	PromptTypeIndex
	// PromptInstructions and PromptState are the question's instructions and
	// the request's state rendered as Int says (a PromptRender).
	PromptInstructions
	PromptState
	// PromptOptionKey, PromptOptionText and PromptOptionIndex are each
	// option's key, its description rendered as Int says and its index;
	// PromptOptionEmpty is 1 where its description is empty as Int says (a
	// PromptEmpty), 0 elsewhere; PromptOptions is how many options there are.
	PromptOptionKey
	PromptOptionText
	PromptOptionIndex
	PromptOptionEmpty
	PromptOptions
	// PromptConcat is its texts end to end; PromptFormat its integer in
	// decimal; PromptReplace Args[0] with every Args[1] replaced by Args[2];
	// PromptRegexp Args[0] with every match of Text replaced by Args[1]
	// ($1 its first group).
	PromptConcat
	PromptFormat
	PromptReplace
	PromptRegexp
	// PromptEncode is its text's ids, the tokenizer adding nothing.
	PromptEncode
	// PromptCat is its sequences end to end; PromptFlatten an option-wise
	// sequence's, option by option, end to end.
	PromptCat
	PromptFlatten
	// PromptCut is Args[0]'s first Args[1] tokens (all of them where fewer).
	PromptCut
	// PromptLen is a sequence's length; PromptSum an option-wise integer's
	// sum over the options.
	PromptLen
	PromptSum
	// PromptAdd .. PromptMax are integer arithmetic (PromptDiv the floor of
	// the quotient), PromptLess 1 where Args[0] < Args[1], 0 elsewhere.
	PromptAdd
	PromptSub
	PromptMul
	PromptDiv
	PromptMin
	PromptMax
	PromptLess
	// PromptSelect is Args[1] where Args[0] is not 0, Args[2] elsewhere;
	// PromptPick the Args[1+i] for Args[0] = i.
	PromptSelect
	PromptPick
	numPromptOps
)

// PromptRender is how a request value becomes text: its string where it is
// one, else Python's str() of it (PromptPyStr) or json.dumps() of it with
// the default separators, ensure_ascii off (PromptJSON) or on
// (PromptJSONASCII).
type PromptRender uint8

const (
	PromptPyStr PromptRender = iota
	PromptJSON
	PromptJSONASCII
)

// PromptEmpty is when a description counts as empty: falsy as Python reads
// it (PromptFalsy: None, False, 0, an empty string, list or object) or only
// absent or the empty string (PromptNoneOrEmpty).
type PromptEmpty uint8

const (
	PromptFalsy PromptEmpty = iota
	PromptNoneOrEmpty
)

// Valid reports why p is not a program this format defines: an unknown
// operation, an operand that is not an earlier node, an out-of-range Out, a
// pattern that does not compile.
func (p *MarkerPrompt) Valid() error {
	if len(p.Nodes) == 0 || int(p.Out) >= len(p.Nodes) {
		return fmt.Errorf("jlm: a marker prompt of %d nodes out at %d", len(p.Nodes), p.Out)
	}
	if p.Confidence > ConfidencePMax {
		return fmt.Errorf("jlm: a marker prompt's confidence %d", p.Confidence)
	}
	for i, n := range p.Nodes {
		if n.Op == PromptInvalid || n.Op >= numPromptOps {
			return fmt.Errorf("jlm: marker prompt node %d: operation %d", i, n.Op)
		}
		for _, a := range n.Args {
			if int(a) >= i {
				return fmt.Errorf("jlm: marker prompt node %d reads node %d", i, a)
			}
		}
		if n.Op == PromptRegexp {
			if _, err := regexp.Compile(n.Text); err != nil {
				return fmt.Errorf("jlm: marker prompt node %d: %w", i, err)
			}
		}
	}
	return nil
}

func encodeMarkerPrompt(w *wbuf, p *MarkerPrompt) {
	w.u32(uint32(len(p.Nodes)))
	for _, n := range p.Nodes {
		w.u8(uint8(n.Op))
		w.u32(uint32(len(n.Args)))
		for _, a := range n.Args {
			w.u32(a)
		}
		w.u64(uint64(n.Int))
		w.str(n.Text)
		w.u32(uint32(len(n.IDs)))
		for _, id := range n.IDs {
			w.i32(id)
		}
	}
	w.u32(p.Out)
	w.u8(uint8(p.Confidence))
}

func decodeMarkerPrompt(r *rbuf) *MarkerPrompt {
	p := &MarkerPrompt{}
	for range r.count(17) {
		n := PromptNode{Op: PromptOp(r.u8())}
		for range r.count(4) {
			n.Args = append(n.Args, r.u32())
		}
		n.Int, n.Text = int64(r.u64()), r.str()
		for range r.count(4) {
			n.IDs = append(n.IDs, r.i32())
		}
		p.Nodes = append(p.Nodes, n)
	}
	p.Out, p.Confidence = r.u32(), PromptConfidence(r.u8())
	if r.err == nil {
		if err := p.Valid(); err != nil {
			r.err = err
		}
	}
	return p
}

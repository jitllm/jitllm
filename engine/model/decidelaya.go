package model

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Laya's readout: one encoder pass per question over
//
//	[CLS] "<type> question: <instructions>" [SEP] ([MASK] " <option>")* [SEP] <state> [SEP]
//
// then the decision head and the scorer at each [MASK]. The sequence is built
// as the reference builds it (rl_common.build_sequence): each piece tokenized
// on its own, each option cut to 48 tokens, every option shrunk evenly when
// they leave the question under 16 of the head budget, the question cut to
// the rest, the state to what the sequence has left. llama.cpp renders one
// string from its template and re-cuts it, which is the same sequence when no
// piece's tokens merge across a boundary.

// layaOptionTokens is the cut on each option's text (build_sequence's [:48]).
const layaOptionTokens = 48

// layaMaxLen is the longest sequence the readout builds: the reference cuts
// the state to max_len (512 on laya, 1024 on its multilingual and
// typed-decisions checkpoints), which llama.cpp's GGUF does not carry, so the
// container's context bounds it (docs/design/decision-models.md section 8).
func (d *Decider) layaMaxLen() int { return d.m.Cfg.NCtx }

// layaSequence is one question's token ids and the position of each option's
// marker, in the question's option order.
func (d *Decider) layaSequence(state jinja.Value, q *DecisionQuestion) ([]int32, []int, error) {
	v := d.m.Vocab
	cv := d.m.container.Vocab()
	mask, sep, cls := cv.Mask, cv.Sep, cv.BOS
	if mask < 0 || sep < 0 || cls < 0 {
		return nil, nil, fmt.Errorf("the vocabulary has no [MASK], [SEP] or [CLS]")
	}
	maskText := v.Text(mask)
	enc := func(s string) []int32 { return v.EncodeSpecial(strings.ReplaceAll(s, maskText, " "), false) }
	ins := pyStr(q.Instructions, true)
	if q.Instructions.IsUndefined() || q.Instructions.IsNone() {
		ins = q.ID
	}
	head := enc(q.Type.String() + " question: " + ins)
	opts := make([][]int32, len(q.Options))
	sum := 0
	for i, o := range q.Options {
		ids := append([]int32{mask}, trim(enc(" "+layaOption(q.Type, i, o)), layaOptionTokens)...)
		opts[i] = ids
		sum += len(ids)
	}
	hm := int(d.cfg.DecisionHeadTokens)
	budget := hm - sum
	if budget < 16 {
		per := max(4, (hm-16)/max(1, len(opts)))
		sum = 0
		for i := range opts {
			opts[i] = trim(opts[i], per)
			sum += len(opts[i])
		}
		budget = hm - sum
	}
	head = trim(head, max(8, budget))
	ids := append([]int32{cls}, head...)
	ids = append(ids, sep)
	markers := make([]int, len(opts))
	for i, o := range opts {
		markers[i] = len(ids)
		ids = append(ids, o...)
	}
	ids = append(ids, sep)
	maxLen := d.layaMaxLen()
	room := max(0, maxLen-len(ids)-1)
	st := ""
	if state.IsString() {
		st = state.AsString()
	} else {
		st = state.JSON()
	}
	ids = append(ids, trim(enc(st), room)...)
	ids = append(ids, sep)
	if len(ids) > maxLen {
		return nil, nil, fmt.Errorf("the options do not fit in %d tokens", maxLen)
	}
	return ids, markers, nil
}

func trim(ids []int32, n int) []int32 {
	if len(ids) > n {
		return ids[:n]
	}
	return ids
}

// layaOption is an option's text (rl_common.render_options): "key" or
// "key: description" for a choice, "level i: description" for a score, and
// "false: ..." / "true: ..." for a noul, with its default wording.
func layaOption(t jlm.QuestionType, i int, o DecisionOption) string {
	desc := o.Description
	empty := desc.IsUndefined() || desc.IsNone() || (desc.IsString() && desc.AsString() == "")
	switch t {
	case jlm.QuestionChoice:
		if empty {
			return o.Key
		}
		return o.Key + ": " + pyStr(desc, false)
	case jlm.QuestionScore:
		return "level " + strconv.Itoa(i) + ": " + pyStr(desc, false)
	}
	if empty {
		if o.Key == "true" {
			return "true: yes, the statement holds"
		}
		return "false: no, the statement does not hold"
	}
	return o.Key + ": " + pyStr(desc, false)
}

// pyStr is a value as the reference's Python makes it text: a string as it is,
// anything else json.dumps (ensure_ascii, as RLAgent._to_internal writes an
// instruction) when asJSON, else str() -- Python's repr of a dict or list.
func pyStr(v jinja.Value, asJSON bool) string {
	if v.IsString() {
		return v.AsString()
	}
	if asJSON {
		return asciiJSON(v.JSON())
	}
	return pyRepr(v)
}

// asciiJSON escapes every non-ASCII rune of a JSON text as \uXXXX, surrogate
// pairs past the BMP, which is json.dumps's default.
func asciiJSON(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r < 0x80:
			b.WriteRune(r)
		case r < 0x10000:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			r -= 0x10000
			fmt.Fprintf(&b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
		}
	}
	return b.String()
}

// pyRepr is Python's repr of a JSON value: single-quoted strings unless the
// string holds a single quote and no double one, True/False/None.
func pyRepr(v jinja.Value) string {
	switch {
	case v.IsNone(), v.IsUndefined():
		return "None"
	case v.IsBool():
		if v.AsBool() {
			return "True"
		}
		return "False"
	case v.IsString():
		return pyQuote(v.AsString())
	case v.IsList():
		parts := make([]string, 0, v.AsList().Len())
		for _, x := range v.AsList().Items {
			parts = append(parts, pyRepr(x))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case v.IsDict():
		dd := v.AsDict()
		parts := make([]string, 0, dd.Len())
		for _, k := range dd.Keys {
			parts = append(parts, pyQuote(k)+": "+pyRepr(dd.Data[k]))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	}
	return v.String()
}

func pyQuote(s string) string {
	q := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		q = '"'
	}
	var b strings.Builder
	b.WriteByte(q)
	for _, r := range s {
		switch {
		case r == '\\' || r == rune(q):
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r == utf8.RuneError:
			b.WriteRune(r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(q)
	return b.String()
}

// layaForward is one question's scores, one per option in the question's
// order, into variant v's scratch: the encoder over ids and the scorer at
// each option's marker -- the engine's half of a question, which makes no
// heap allocation once warm (TestDecisionDoesNotAllocate).
func (d *Decider) layaForward(ids []int32, qt jlm.QuestionType, markers []int, v int) ([]float32, error) {
	if err := d.emb.mbEncode(ids); err != nil {
		return nil, err
	}
	out := d.variantScores(v, len(markers))
	if err := d.emb.layaScores(out, len(ids), qt, markers); err != nil {
		return nil, err
	}
	d.input += len(ids)
	return out, nil
}

// layaConfidence is Laya's own (rl_common.confidence_from_probs): one minus
// the answer's entropy h over log n. llama.cpp gives TypeSafe's formula instead.
func layaConfidence(h float64, n int) float64 {
	if n < 2 {
		return 1
	}
	return 1 - h/math.Log(float64(n))
}

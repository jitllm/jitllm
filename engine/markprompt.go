package engine

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf16"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// A marker head's prompt is the container's program (jlm.MarkerPrompt): this
// evaluates it over one question, the tokenizer encoding each text it names
// alone, and hands back the ids and the options' marker rows.

// promptValue is a MarkerPrompt node's value: one element, or one an option.
type promptValue struct {
	vec   bool
	ints  []int64
	texts []string
	toks  [][]int32
	marks [][]int
}

// at is element i's index into v: i for an option-wise value, 0 for a scalar.
func (v *promptValue) at(i int) int {
	if v.vec {
		return i
	}
	return 0
}

// markedPrompt evaluates p over question q about state: its ids and the rows
// of its marked tokens, in order. rows is the context the prompt is bounded
// by (jlm.PromptContext).
func markedPrompt(p *jlm.MarkerPrompt, encode func(string) []int32, state jinja.Value, q *DecisionQuestion, rows, pictures, clips int) ([]int32, []int, error) {
	vals := make([]promptValue, len(p.Nodes))
	k := len(q.Options)
	for ni := range p.Nodes {
		n := &p.Nodes[ni]
		args := make([]*promptValue, len(n.Args))
		width := 1
		vec := false
		for j, a := range n.Args {
			args[j] = &vals[a]
			if args[j].vec {
				vec, width = true, k
			}
		}
		v := &vals[ni]
		switch n.Op {
		case jlm.PromptInt:
			v.ints = []int64{n.Int}
		case jlm.PromptText:
			v.texts = []string{n.Text}
		case jlm.PromptIDs:
			v.toks, v.marks = [][]int32{n.IDs}, [][]int{nil}
		case jlm.PromptMarked:
			v.toks, v.marks = [][]int32{n.IDs}, [][]int{{0}}
		case jlm.PromptContext:
			v.ints = []int64{int64(rows)}
		case jlm.PromptPictures:
			v.ints = []int64{int64(pictures)}
		case jlm.PromptClips:
			v.ints = []int64{int64(clips)}
		case jlm.PromptType:
			v.texts = []string{q.Type.String()}
		case jlm.PromptTypeIndex:
			v.ints = []int64{int64(q.Type)}
		case jlm.PromptInstructions:
			v.texts = []string{renderPrompt(q.Instructions, jlm.PromptRender(n.Int))}
		case jlm.PromptState:
			v.texts = []string{renderPrompt(state, jlm.PromptRender(n.Int))}
		case jlm.PromptOptions:
			v.ints = []int64{int64(k)}
		case jlm.PromptOptionKey, jlm.PromptOptionText, jlm.PromptOptionIndex, jlm.PromptOptionEmpty:
			v.vec = true
			for i, o := range q.Options {
				switch n.Op {
				case jlm.PromptOptionKey:
					v.texts = append(v.texts, o.Key)
				case jlm.PromptOptionText:
					v.texts = append(v.texts, renderPrompt(o.Description, jlm.PromptRender(n.Int)))
				case jlm.PromptOptionIndex:
					v.ints = append(v.ints, int64(i))
				default:
					v.ints = append(v.ints, b2i(emptyPrompt(o.Description, jlm.PromptEmpty(n.Int))))
				}
			}
		case jlm.PromptFlatten:
			a := args[0]
			var ids []int32
			var marks []int
			for i := range len(a.toks) {
				for _, m := range a.marks[i] {
					marks = append(marks, len(ids)+m)
				}
				ids = append(ids, a.toks[i]...)
			}
			v.toks, v.marks = [][]int32{ids}, [][]int{marks}
		case jlm.PromptSum:
			var s int64
			for _, x := range args[0].ints {
				s += x
			}
			v.ints = []int64{s}
		default:
			v.vec = vec
			for i := range width {
				if err := promptStep(v, n, args, i, encode); err != nil {
					return nil, nil, fmt.Errorf("model: marker prompt node %d: %w", ni, err)
				}
			}
		}
	}
	out := &vals[p.Out]
	if out.vec || len(out.toks) != 1 {
		return nil, nil, fmt.Errorf("model: the marker prompt's out is not one sequence")
	}
	return out.toks[0], out.marks[0], nil
}

// promptStep is element i of node n over args into v.
func promptStep(v *promptValue, n *jlm.PromptNode, args []*promptValue, i int, encode func(string) []int32) error {
	text := func(j int) string { return args[j].texts[args[j].at(i)] }
	num := func(j int) int64 { return args[j].ints[args[j].at(i)] }
	switch n.Op {
	case jlm.PromptConcat:
		var b strings.Builder
		for j := range args {
			b.WriteString(text(j))
		}
		v.texts = append(v.texts, b.String())
	case jlm.PromptFormat:
		v.texts = append(v.texts, strconv.FormatInt(num(0), 10))
	case jlm.PromptReplace:
		v.texts = append(v.texts, strings.ReplaceAll(text(0), text(1), text(2)))
	case jlm.PromptRegexp:
		re, err := regexp.Compile(n.Text)
		if err != nil {
			return err
		}
		v.texts = append(v.texts, re.ReplaceAllString(text(0), text(1)))
	case jlm.PromptEncode:
		v.toks, v.marks = append(v.toks, encode(text(0))), append(v.marks, nil)
	case jlm.PromptCat:
		var ids []int32
		var marks []int
		for _, a := range args {
			e := a.at(i)
			for _, m := range a.marks[e] {
				marks = append(marks, len(ids)+m)
			}
			ids = append(ids, a.toks[e]...)
		}
		v.toks, v.marks = append(v.toks, ids), append(v.marks, marks)
	case jlm.PromptCut:
		a, e := args[0], args[0].at(i)
		m := int(max(num(1), 0))
		ids := a.toks[e][:min(m, len(a.toks[e]))]
		var marks []int
		for _, x := range a.marks[e] {
			if x < len(ids) {
				marks = append(marks, x)
			}
		}
		v.toks, v.marks = append(v.toks, ids), append(v.marks, marks)
	case jlm.PromptLen:
		v.ints = append(v.ints, int64(len(args[0].toks[args[0].at(i)])))
	case jlm.PromptAdd:
		v.ints = append(v.ints, num(0)+num(1))
	case jlm.PromptSub:
		v.ints = append(v.ints, num(0)-num(1))
	case jlm.PromptMul:
		v.ints = append(v.ints, num(0)*num(1))
	case jlm.PromptDiv:
		d := num(1)
		if d == 0 {
			return fmt.Errorf("a division by zero")
		}
		v.ints = append(v.ints, int64(math.Floor(float64(num(0))/float64(d))))
	case jlm.PromptMin:
		v.ints = append(v.ints, min(num(0), num(1)))
	case jlm.PromptMax:
		v.ints = append(v.ints, max(num(0), num(1)))
	case jlm.PromptLess:
		v.ints = append(v.ints, b2i(num(0) < num(1)))
	case jlm.PromptSelect, jlm.PromptPick:
		j := 2
		if n.Op == jlm.PromptSelect && num(0) != 0 {
			j = 1
		}
		if n.Op == jlm.PromptPick {
			j = 1 + int(num(0))
			if j < 1 || j >= len(args) {
				return fmt.Errorf("a pick of %d among %d", j-1, len(args)-1)
			}
		}
		a, e := args[j], args[j].at(i)
		switch {
		case a.ints != nil:
			v.ints = append(v.ints, a.ints[e])
		case a.texts != nil:
			v.texts = append(v.texts, a.texts[e])
		default:
			v.toks, v.marks = append(v.toks, a.toks[e]), append(v.marks, a.marks[e])
		}
	default:
		return fmt.Errorf("operation %d", n.Op)
	}
	return nil
}

func b2i(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// emptyPrompt says description d counts as empty under rule e.
func emptyPrompt(d jinja.Value, e jlm.PromptEmpty) bool {
	if d.IsUndefined() || d.IsNone() {
		return true
	}
	if e == jlm.PromptNoneOrEmpty {
		return d.IsString() && d.AsString() == ""
	}
	return !d.IsTruthy()
}

// renderPrompt is request value v as text: its string where it is one, else
// Python's str() or json.dumps() of it.
func renderPrompt(v jinja.Value, r jlm.PromptRender) string {
	if v.IsString() {
		return v.AsString()
	}
	if r == jlm.PromptTemplate {
		return jinja.Print(v)
	}
	if r == jlm.PromptJSONOrEmpty && (v.IsNone() || v.IsUndefined()) {
		return ""
	}
	var b strings.Builder
	if r == jlm.PromptPyStr {
		pyRepr(&b, v, true)
	} else {
		pyJSON(&b, v, r == jlm.PromptJSONASCII)
	}
	return b.String()
}

// pyRepr writes v as Python's repr() does (str() at the top for a string).
func pyRepr(b *strings.Builder, v jinja.Value, top bool) {
	switch {
	case v.IsUndefined() || v.IsNone():
		b.WriteString("None")
	case v.IsBool():
		if v.AsBool() {
			b.WriteString("True")
		} else {
			b.WriteString("False")
		}
	case v.IsInt():
		b.WriteString(strconv.FormatInt(v.AsInt(), 10))
	case v.IsFloat():
		b.WriteString(pyFloat(v.AsFloat()))
	case v.IsString():
		if top {
			b.WriteString(v.AsString())
		} else {
			pyQuote(b, v.AsString())
		}
	case v.IsList():
		b.WriteByte('[')
		for i, x := range v.AsList().Items {
			if i > 0 {
				b.WriteString(", ")
			}
			pyRepr(b, x, false)
		}
		b.WriteByte(']')
	case v.IsDict():
		d := v.AsDict()
		b.WriteByte('{')
		for i, k := range d.Keys {
			if i > 0 {
				b.WriteString(", ")
			}
			pyQuote(b, k)
			b.WriteString(": ")
			pyRepr(b, d.Data[k], false)
		}
		b.WriteByte('}')
	default:
		b.WriteString(v.String())
	}
}

// pyQuote writes s as Python's repr() of a string: single quotes unless it
// holds one and no double quote.
func pyQuote(b *strings.Builder, s string) {
	q := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		q = '"'
	}
	b.WriteByte(q)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == rune(q):
			b.WriteByte('\\')
			b.WriteByte(q)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(b, `\x%02x`, r)
		case !unicode.IsPrint(r) && r < 0x10000:
			fmt.Fprintf(b, `\u%04x`, r)
		case !unicode.IsPrint(r):
			fmt.Fprintf(b, `\U%08x`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(q)
}

// pyJSON writes v as Python's json.dumps(v, ensure_ascii=ascii) does.
func pyJSON(b *strings.Builder, v jinja.Value, ascii bool) {
	switch {
	case v.IsUndefined() || v.IsNone():
		b.WriteString("null")
	case v.IsBool():
		if v.AsBool() {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case v.IsInt():
		b.WriteString(strconv.FormatInt(v.AsInt(), 10))
	case v.IsFloat():
		f := v.AsFloat()
		switch {
		case math.IsNaN(f):
			b.WriteString("NaN")
		case math.IsInf(f, 1):
			b.WriteString("Infinity")
		case math.IsInf(f, -1):
			b.WriteString("-Infinity")
		default:
			b.WriteString(pyFloat(f))
		}
	case v.IsString():
		jsonQuote(b, v.AsString(), ascii)
	case v.IsList():
		b.WriteByte('[')
		for i, x := range v.AsList().Items {
			if i > 0 {
				b.WriteString(", ")
			}
			pyJSON(b, x, ascii)
		}
		b.WriteByte(']')
	case v.IsDict():
		d := v.AsDict()
		b.WriteByte('{')
		for i, k := range d.Keys {
			if i > 0 {
				b.WriteString(", ")
			}
			jsonQuote(b, k, ascii)
			b.WriteString(": ")
			pyJSON(b, d.Data[k], ascii)
		}
		b.WriteByte('}')
	default:
		jsonQuote(b, v.String(), ascii)
	}
}

// jsonQuote writes s as json.dumps does: with ascii, every rune past 0x7f
// as \uXXXX, one past the basic plane as its surrogate pair.
func jsonQuote(b *strings.Builder, s string, ascii bool) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			switch {
			case r < 0x20 || ascii && r >= 0x7f && r < 0x10000:
				fmt.Fprintf(b, `\u%04x`, r)
			case ascii && r >= 0x10000:
				r1, r2 := utf16.EncodeRune(r)
				fmt.Fprintf(b, `\u%04x\u%04x`, r1, r2)
			default:
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// pyFloat is f as Python's repr() writes a float: the shortest digits that
// read back, in positional form from 1e-4 up to 1e16 and with an exponent
// beyond, always with a fraction or an exponent.
func pyFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	s := strconv.FormatFloat(f, 'e', -1, 64)
	mant, exps, _ := strings.Cut(s, "e")
	exp, _ := strconv.Atoi(exps)
	if exp < -4 || exp >= 16 {
		sign := "+"
		if exp < 0 {
			sign, exp = "-", -exp
		}
		return fmt.Sprintf("%se%s%02d", mant, sign, exp)
	}
	out := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.ContainsAny(out, ".") {
		out += ".0"
	}
	return out
}

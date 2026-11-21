package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// mjVideo is how a tower reads a video as the library states it
// (jlm.Video): the rate a request of none is read at, the frames sampled
// (Count) and which (Index), whether a frame is taken once, the most frames,
// the frames a picture reads, the count the frames' resize reads (Sized),
// the frames' own steps (none: the picture's), their own entry, the stride
// that makes the groups one grid, the prompt it writes, and the chat
// template a conversation carrying one renders by where the reference's
// processor keeps its own. Each number is an expression over the tower's
// facts; each video expression one over the video's operands (videoOperands;
// a sized count's are sampled and filled).
type mjVideo struct {
	DefaultFps string       `json:"default_fps"`
	Count      mjVideoExpr  `json:"count"`
	Index      mjVideoExpr  `json:"index"`
	Once       bool         `json:"once"`
	MaxFrames  string       `json:"max_frames"`
	Group      string       `json:"group"`
	Sized      mjVideoExpr  `json:"sized"`
	Pre        []mjPreStep  `json:"pre"`
	Entry      []mjTowerOp  `json:"entry"`
	Stride     *mjVideoExpr `json:"stride"`
	Prompt     []mjPiece    `json:"prompt"`
	Template   string       `json:"template"`
	// If is the condition the tower reads a video under, and Inputs its
	// entry's frames (in, in1).
	If     string `json:"if"`
	Inputs int    `json:"inputs"`
}

// mjVideoExpr is a video expression: Let's values, each named and written in
// order, then Value, its result. A Let value is written once however often
// later ones read it.
type mjVideoExpr struct {
	Let   [][2]string `json:"let"`
	Value string      `json:"value"`
}

// mjPiece is one piece of a video's prompt: text, a value as a format writes
// it, the groups' pieces (Groups, with Between before every one but the
// first), a picture's rows (span), the tokens of marker pieces (ids), or the
// picture's span with its markers (picture).
type mjPiece struct {
	Text    string       `json:"text"`
	Value   *mjVideoExpr `json:"value"`
	Format  *mjFormat    `json:"format"`
	Groups  []mjPiece    `json:"groups"`
	Between []mjPiece    `json:"between"`
	Span    bool         `json:"span"`
	IDs     []string     `json:"ids"`
	Picture bool         `json:"picture"`
}

// mjFormat is how a value is written: decimal to Digits places (cut first
// where Trunc), a clock of Clock's widths with Sep between and Days' names,
// or a table of strings -- Words, an expression, its English cardinals from
// zero (cardinal).
type mjFormat struct {
	Kind   string    `json:"kind"`
	Digits int32     `json:"digits"`
	Trunc  bool      `json:"trunc"`
	Clock  [3]int32  `json:"clock"`
	Days   [2]string `json:"days"`
	Sep    string    `json:"sep"`
	Words  string    `json:"words"`
}

func videoOperands() map[string]uint32 {
	return map[string]uint32{"N": jlm.VideoN, "R": jlm.VideoRate, "K": jlm.VideoK, "at": jlm.VideoAt,
		"group": jlm.VideoGroup, "f0": jlm.VideoF0, "t0": jlm.VideoT0, "f1": jlm.VideoF1, "t1": jlm.VideoT1,
		"seconds": jlm.VideoSeconds, "sampled": 0, "filled": 1}
}

func videoFns() map[string]jlm.Fn {
	return map[string]jlm.Fn{"min": jlm.FnMin, "max": jlm.FnMax, "where": jlm.FnWhere, "trunc": jlm.FnTrunc,
		"floor": jlm.FnFloor, "ceil": jlm.FnCeil, "round": jlm.FnRound, "round_even": jlm.FnRoundEven, "abs": jlm.FnAbs,
		"sqrt": jlm.FnSqrt, "f32": jlm.FnF32}
}

// videoExpr lowers x to the container's terms, each term in the order a
// post-order walk reaches it, each Let value once: a value that is an
// operand or a constant alone is written as it plus zero, so the result is a
// term.
func videoExpr(x mjVideoExpr) (jlm.Expr, error) {
	var e jlm.ExprBuilder
	named := map[string]jlm.Arg{}
	ops := videoOperands()
	bin := map[string]jlm.Fn{"+": jlm.FnAdd, "-": jlm.FnSub, "*": jlm.FnMul, "/": jlm.FnDiv, "<": jlm.FnLess, "==": jlm.FnEqual}
	var lower func(n *exprNode) (jlm.Arg, error)
	lower = func(n *exprNode) (jlm.Arg, error) {
		switch n.op {
		case "num":
			return jlm.Num(float32(n.num)), nil
		case "neg":
			if a := n.args[0]; a.op == "num" {
				return jlm.Num(float32(-a.num)), nil
			}
			a, err := lower(n.args[0])
			if err != nil {
				return jlm.Arg{}, err
			}
			return e.Fn(jlm.FnSub, jlm.Num(0), a), nil
		case "path":
			if a, ok := named[n.str]; ok {
				return a, nil
			}
			if i, ok := ops[n.str]; ok {
				return jlm.Operand(i), nil
			}
			return jlm.Arg{}, fmt.Errorf("%q is no video operand and no earlier value", n.str)
		case "?:":
			var args [3]jlm.Arg
			for i, a := range n.args {
				v, err := lower(a)
				if err != nil {
					return jlm.Arg{}, err
				}
				args[i] = v
			}
			return e.Fn(jlm.FnWhere, args[0], args[1], args[2]), nil
		case "call":
			f, ok := videoFns()[n.str]
			if !ok {
				return jlm.Arg{}, fmt.Errorf("%s is no function a video expression calls", n.str)
			}
			args := make([]jlm.Arg, len(n.args))
			for i, a := range n.args {
				v, err := lower(a)
				if err != nil {
					return jlm.Arg{}, err
				}
				args[i] = v
			}
			return e.Fn(f, args...), nil
		case ">":
			l, err := lower(n.args[0])
			if err != nil {
				return jlm.Arg{}, err
			}
			r, err := lower(n.args[1])
			if err != nil {
				return jlm.Arg{}, err
			}
			return e.Fn(jlm.FnLess, r, l), nil
		}
		f, ok := bin[n.op]
		if !ok || len(n.args) != 2 {
			return jlm.Arg{}, fmt.Errorf("%q is no operation a video expression writes", n.op)
		}
		l, err := lower(n.args[0])
		if err != nil {
			return jlm.Arg{}, err
		}
		r, err := lower(n.args[1])
		if err != nil {
			return jlm.Arg{}, err
		}
		return e.Fn(f, l, r), nil
	}
	for _, l := range x.Let {
		n, err := parseExpr(l[1])
		if err != nil {
			return jlm.Expr{}, err
		}
		if named[l[0]], err = lower(n); err != nil {
			return jlm.Expr{}, fmt.Errorf("%s: %w", l[0], err)
		}
	}
	n, err := parseExpr(x.Value)
	if err != nil {
		return jlm.Expr{}, err
	}
	at := len(e.T)
	v, err := lower(n)
	if err != nil {
		return jlm.Expr{}, err
	}
	if len(e.T) == at {
		e.Fn(jlm.FnAdd, v, jlm.Num(0))
	}
	return jlm.Expr{Terms: e.T}, nil
}

// format is f as the container writes it.
func (f *mjFormat) format(num func(string) (float64, error)) (jlm.ValueFormat, error) {
	switch f.Kind {
	case "decimal":
		return jlm.ValueFormat{Kind: jlm.FormatDecimal, Digits: f.Digits, Trunc: f.Trunc}, nil
	case "clock":
		return jlm.ValueFormat{Kind: jlm.FormatClock, Clock: f.Clock, Sep: f.Sep, Days: f.Days}, nil
	case "words":
		n, err := num(f.Words)
		if err != nil {
			return jlm.ValueFormat{}, err
		}
		words := make([]string, int(n)+1)
		for i := range words {
			words[i] = cardinal(i)
		}
		return jlm.ValueFormat{Kind: jlm.FormatTable, Strings: words}, nil
	}
	return jlm.ValueFormat{}, fmt.Errorf("format %q is not decimal, clock or words", f.Kind)
}

// prompt is ps as the container writes a video's prompt, span the picture's.
func prompt(ps []mjPiece, s *jlm.Source, span *jlm.TowerSpan, num func(string) (float64, error)) ([]jlm.VideoPiece, error) {
	var out []jlm.VideoPiece
	for _, p := range ps {
		switch {
		case p.Text != "":
			out = append(out, textPiece(p.Text))
		case p.Value != nil:
			x, err := videoExpr(*p.Value)
			if err != nil {
				return nil, err
			}
			if p.Format == nil {
				return nil, fmt.Errorf("a video value states no format")
			}
			f, err := p.Format.format(num)
			if err != nil {
				return nil, err
			}
			out = append(out, valuePiece(x, f))
		case p.Groups != nil:
			body, err := prompt(p.Groups, s, span, num)
			if err != nil {
				return nil, err
			}
			between, err := prompt(p.Between, s, span, num)
			if err != nil {
				return nil, err
			}
			out = append(out, jlm.VideoPiece{Kind: jlm.PieceGroups, Body: body, Between: between})
		case p.Span:
			out = append(out, spanPiece())
		case p.IDs != nil:
			ids, err := markerIDs(s, p.IDs...)
			if err != nil {
				return nil, err
			}
			out = append(out, idsPiece(ids))
		case p.Picture:
			out = append(out, pictureSpan(span)...)
		default:
			return nil, fmt.Errorf("a video's prompt piece states nothing")
		}
	}
	return out, nil
}

// withVideoTemplate states the template a conversation carrying a video
// renders by, where the reference's processor keeps one of its own: named
// "video" beside the model's (model.ChatTemplateFor), the model's one
// unnamed template named "default" so the two are told apart. A model with
// no template, or one stating its own for a video, is left as it is.
func withVideoTemplate(v *jlm.Vocab, body string) {
	if v == nil || len(v.Templates) == 0 {
		return
	}
	for _, t := range v.Templates {
		if t.Name == "video" {
			return
		}
	}
	if len(v.Templates) == 1 && v.Templates[0].Name == "" {
		v.Templates[0].Name = "default"
	}
	v.Templates = append(v.Templates, jlm.ChatTemplate{Name: "video", Body: body})
}

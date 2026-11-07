package engine

import (
	"fmt"
	"slices"
	"sort"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
	"github.com/jitllm/jitllm/tok/jinja"
)

// A decision model answers typed questions about a state from one prompt's
// next-token logits and generates nothing (docs/design/decision-models.md).
// What its reference hardcodes -- the codes a question's options are shown
// with, the token groups that score them, the order they are shown in -- is
// the container's (jlm.DecisionLabels); the prompt is the container's
// systemone template. The logits are the label pick's (the head's
// projection of the groups' ids alone) and everything after them is one
// composed part (graph.DecisionScores), formed when the instance is made.

// DecisionOption is one allowed answer: its key (a choice's name, a score's
// level "0".."n-1", a noul's "true"/"false") and its description, a string or
// any JSON value, None when the request gave none.
type DecisionOption struct {
	Key         string
	Description jinja.Value
}

// DecisionQuestion is one named question of a request, its options in the
// order the request wrote them.
type DecisionQuestion struct {
	ID           string
	Type         jlm.QuestionType
	Instructions jinja.Value
	Options      []DecisionOption
}

// DecisionAnswer is one question's answer. Probs is per option, in the
// question's option order. Noul is P(true) for a noul; Choice, Score and
// Confidence are set by the types that have them.
type DecisionAnswer struct {
	Type       jlm.QuestionType
	Probs      []float64
	Noul       float64
	Choice     string
	Score      float64
	Confidence float64
}

// Decider asks a decision model's label readout questions through one
// instance, one request at a time.
type Decider struct {
	in   *Instance
	cfg  *jlm.Config
	lab  *jlm.DecisionLabels
	tpl  *jinja.Template
	bidi bool
	// The readout's inputs and outputs (graph.DecisionScores), and the
	// question's picked ids, N groups of G.
	ids                       []int32
	vals, keep, params, prior []float32
	probs, stats              []float32
	// input counts the prompt tokens the last Decide ran.
	input int
	// one is a bidirectional prompt as its one span, kept so a warm question
	// allocates nothing.
	one [1]model.Span
}

// NewDecider is a Decider over in, whose model states a label readout.
func NewDecider(in *Instance) (*Decider, error) {
	m := in.m
	cfg := m.Container.Config()
	lab := cfg.DecisionLabels
	if lab == nil {
		return nil, fmt.Errorf("model: %v states no label readout", cfg.Decision)
	}
	src, ok := m.ChatTemplate("systemone")
	if !ok {
		return nil, fmt.Errorf("model: the container carries no systemone template, the readout's prompt")
	}
	tpl, err := jinja.Compile(src)
	if err != nil {
		return nil, fmt.Errorf("model: the systemone template: %w", err)
	}
	const n, g = nn.DecisionOptions, nn.DecisionGroup
	return &Decider{in: in, cfg: cfg, lab: lab, tpl: tpl, bidi: m.Cfg.NonCausal,
		ids: make([]int32, n*g), vals: make([]float32, n*g), keep: make([]float32, n), params: make([]float32, 2+n),
		prior: make([]float32, n), probs: make([]float32, n), stats: make([]float32, graph.DecisionStats)}, nil
}

// InputTokens is how many prompt tokens the last Decide ran.
func (d *Decider) InputTokens() int { return d.input }

// maxOptions is the most options a question may have.
func (d *Decider) maxOptions() int {
	return min(int(d.lab.MaxOptions), nn.DecisionOptions)
}

// Prepare checks a request's questions against what the readout can ask.
func (d *Decider) Prepare(qs []DecisionQuestion) error {
	if len(qs) == 0 {
		return fmt.Errorf("questions must be a non-empty object")
	}
	for i := range qs {
		q := &qs[i]
		n := len(q.Options)
		switch q.Type {
		case jlm.QuestionChoice:
			if n == 0 {
				return fmt.Errorf("questions.%s: criteria must be a non-empty object", q.ID)
			}
		case jlm.QuestionScore:
			if n < 2 || n > 10 {
				return fmt.Errorf("questions.%s: criteria must be an array of 2 to 10 levels", q.ID)
			}
		case jlm.QuestionNoul:
			if n != 2 || q.Options[0].Key != "false" || q.Options[1].Key != "true" {
				return fmt.Errorf("questions.%s: a noul's options are false then true", q.ID)
			}
		default:
			return fmt.Errorf("questions.%s: type must be one of: choice, score, noul", q.ID)
		}
		if n > d.maxOptions() {
			return fmt.Errorf("questions.%s: too many options (%d), this model supports at most %d", q.ID, n, d.maxOptions())
		}
	}
	return nil
}

// Decide answers every question about state, one prompt per question and
// variant, the answers in the questions' order.
func (d *Decider) Decide(state jinja.Value, qs []DecisionQuestion) ([]DecisionAnswer, error) {
	if err := d.Prepare(qs); err != nil {
		return nil, err
	}
	d.input = 0
	out := make([]DecisionAnswer, len(qs))
	for i := range qs {
		a, err := d.answer(state, &qs[i])
		if err != nil {
			return nil, fmt.Errorf("questions.%s: %w", qs[i].ID, err)
		}
		out[i] = a
	}
	return out, nil
}

// shown is the question's options in the order its prompt shows them, and
// where each request option stands among them.
func (d *Decider) shown(q *DecisionQuestion) []DecisionOption {
	if q.Type == jlm.QuestionNoul && d.lab.NoulTrueFirst {
		return []DecisionOption{q.Options[1], q.Options[0]}
	}
	return q.Options
}

// variants is how many times a question is asked: twice where the readout
// shows a choice's options reversed as well.
func (d *Decider) variants(q *DecisionQuestion) int {
	if d.lab.Reversed && q.Type == jlm.QuestionChoice && len(q.Options) > 1 {
		return 2
	}
	return 1
}

// ratings is how many levels a noul is read at, 0 where it reads its two
// options.
func (d *Decider) ratings(q *DecisionQuestion) int {
	if q.Type == jlm.QuestionNoul && d.lab.NoulRatings > 0 {
		return int(d.lab.NoulRatings)
	}
	return 0
}

// answer asks q and reads its answer from the readout's outputs.
func (d *Decider) answer(state jinja.Value, q *DecisionQuestion) (DecisionAnswer, error) {
	shown := d.shown(q)
	labels, groups, err := d.groups(q, shown)
	if err != nil {
		return DecisionAnswer{}, err
	}
	// The readout's options: the request's, or a noul's rating levels, each
	// at its index (a level at its share of the top one, so the expected
	// value is a noul's probability of true).
	n, r := len(q.Options), d.ratings(q)
	k := n
	if r > 0 {
		k = r
	}
	clear(d.keep)
	for i := range k {
		d.keep[i] = 1
		d.params[2+i] = float32(i)
		if r > 1 {
			d.params[2+i] = float32(i) / float32(r-1)
		}
	}
	d.params[0] = 1 / d.cfg.Temperature(q.Type, n)
	for v := range d.variants(q) {
		// Slot j holds the group read for request option j (or level j):
		// shown at n-1-j in the reversed variant, at its place in shown
		// otherwise.
		clear(d.ids)
		for j := range k {
			at := j
			switch {
			case r > 0:
			case v == 1:
				at = n - 1 - j
			case len(shown) == 2 && shown[0].Key != q.Options[0].Key:
				at = 1 - j
			}
			g := groups[at]
			for x := range nn.DecisionGroup {
				d.ids[j*nn.DecisionGroup+x] = g[min(x, len(g)-1)]
			}
		}
		text, err := d.render(state, q, shown, labels, v)
		if err != nil {
			return DecisionAnswer{}, err
		}
		ids, err := d.encode(text)
		if err != nil {
			return DecisionAnswer{}, err
		}
		if err := d.ask(ids); err != nil {
			return DecisionAnswer{}, err
		}
		// The second variant averages with the first.
		d.params[1] = float32(v)
		if v == 1 {
			copy(d.prior, d.probs)
		}
		if !d.in.decideScores(d.vals, d.keep, d.params, d.prior, d.probs, d.stats) {
			return DecisionAnswer{}, d.in.sampleDeclined("the label readout")
		}
	}
	if p := d.stats[graph.DecisionPMax]; p != p {
		return DecisionAnswer{}, fmt.Errorf("the model could not evaluate the decision (a score is NaN)")
	}
	a := DecisionAnswer{Type: q.Type}
	if q.Type == jlm.QuestionNoul {
		a.Noul = float64(d.probs[1])
		if r > 0 {
			a.Noul = float64(d.stats[graph.DecisionExpected])
		}
		return a, nil
	}
	a.Probs = make([]float64, n)
	for i := range n {
		a.Probs[i] = float64(d.probs[i])
	}
	conf := graph.DecisionChoiceConf
	if q.Type == jlm.QuestionChoice {
		a.Choice = q.Options[int(d.stats[graph.DecisionMode])].Key
	} else {
		a.Score, conf = float64(d.stats[graph.DecisionExpected]), graph.DecisionScoreConf
	}
	if d.lab.EntropyConfidence {
		conf = graph.DecisionEntropyConf
	}
	a.Confidence = float64(d.stats[conf])
	return a, nil
}

// encode is a rendered prompt's ids.
func (d *Decider) encode(text string) ([]int32, error) {
	ids := d.in.m.Vocab.EncodeSpecial(text, false)
	if len(ids) == 0 {
		return nil, fmt.Errorf("the prompt is empty")
	}
	if len(ids) > d.in.MaxSeq() {
		return nil, fmt.Errorf("the prompt is %d tokens and this instance holds %d", len(ids), d.in.MaxSeq())
	}
	return ids, nil
}

// ask runs a prompt's ids and leaves its label pick in vals: the head's
// projection of the question's ids alone, on whichever device holds the
// head. It is the engine's half of a question, which makes no heap
// allocation once warm (TestDecisionDoesNotAllocate).
func (d *Decider) ask(ids []int32) error {
	s := d.in
	s.Reset()
	s.PickIDs, s.PickVals, s.Picked = d.ids, d.vals, false
	if h := s.head; h != nil {
		h.Pick, h.PickVals, h.Picked = d.ids, d.vals, false
	}
	defer func() {
		s.PickIDs, s.PickVals = nil, nil
		if h := s.head; h != nil {
			h.Pick, h.PickVals = nil, nil
		}
	}()
	var err error
	if d.bidi {
		d.one[0] = model.Span{Tokens: ids, Bidir: true}
		_, err = s.PrefillMixed(d.one[:]...)
		d.one[0] = model.Span{}
	} else {
		_, err = s.Prefill(ids)
	}
	if err != nil {
		return err
	}
	d.input += len(ids)
	if !s.Picked && (s.head == nil || !s.head.Picked) {
		return fmt.Errorf("the prompt's labels were not picked: %v", s.prepErr())
	}
	return nil
}

// groups are the codes shown beside each option (nil where the template
// writes its own) and the token groups that score each shown option -- or,
// for a noul read at rating levels, each level.
func (d *Decider) groups(q *DecisionQuestion, shown []DecisionOption) ([]string, [][]int32, error) {
	l := d.lab
	single := func(forms ...string) []int32 {
		var out []int32
		for _, f := range forms {
			if ids := d.in.m.Vocab.Encode(f, false); len(ids) == 1 && !slices.Contains(out, ids[0]) {
				out = append(out, ids[0])
			}
		}
		return out
	}
	byKey := func(group func(o DecisionOption) []int32) ([]string, [][]int32, error) {
		var texts []string
		var groups [][]int32
		for _, o := range shown {
			g := group(o)
			if len(g) == 0 {
				return nil, nil, fmt.Errorf("decision label %q is not a single token", o.Key)
			}
			texts, groups = append(texts, o.Key), append(groups, g)
		}
		return texts, groups, nil
	}
	switch {
	case q.Type == jlm.QuestionNoul && len(l.NoulTrue) > 0:
		return byKey(func(o DecisionOption) []int32 {
			if o.Key == "true" {
				return single(l.NoulTrue...)
			}
			return single(l.NoulFalse...)
		})
	case q.Type == jlm.QuestionScore && l.ScoreByKey:
		return byKey(func(o DecisionOption) []int32 { return single(o.Key) })
	}
	n := max(len(shown), d.ratings(q))
	letters := l.KeyCodes
	for _, o := range shown {
		r := []rune(o.Key)
		letters = letters && len(o.Key) == 1 && len(r) == 1 && (r[0] >= 'a' && r[0] <= 'z' || r[0] >= 'A' && r[0] <= 'Z')
	}
	var used []int32
	var texts []string
	var groups [][]int32
	take := func(code string) bool {
		ids := single(code)
		if len(ids) == 0 || slices.Contains(used, ids[0]) {
			return false
		}
		used = append(used, ids[0])
		g := ids
		if l.Spaced {
			for _, x := range single(" " + code) {
				if x != g[0] {
					g = append(g, x)
				}
			}
		}
		texts, groups = append(texts, code), append(groups, g)
		return true
	}
	for i := range n {
		want := ""
		switch {
		case letters && i < len(shown):
			want = shown[i].Key
		case n <= len(l.Codes):
			want = l.Codes[i]
		case i < len(l.WideCodes):
			want = l.WideCodes[i]
		}
		taken := want != "" && take(want)
		for _, c := range l.Codes {
			if taken {
				break
			}
			taken = take(c)
		}
		for _, c := range l.Pool {
			if taken {
				break
			}
			taken = take(c)
		}
		if !taken {
			return nil, nil, fmt.Errorf("no single-token label left for %d options", n)
		}
	}
	if !l.ShowCodes {
		texts = nil
	}
	return texts, groups, nil
}

// render is the systemone template over one question, given the inputs
// llama.cpp's server gives it: id, type, instructions, state and the
// options, each with its key, description and label, and an empty image
// list. Variant 1 shows the options reversed, the labels in their order.
func (d *Decider) render(state jinja.Value, q *DecisionQuestion, opts []DecisionOption, labels []string, v int) (string, error) {
	n := len(opts)
	list := make([]jinja.Value, n)
	for i := range opts {
		o := opts[i]
		if v == 1 {
			o = opts[n-1-i]
		}
		od := jinja.NewDict()
		od.AsDict().Set("key", jinja.NewString(o.Key))
		desc := o.Description
		if desc.IsUndefined() {
			desc = jinja.None()
		}
		od.AsDict().Set("description", desc)
		if labels != nil {
			od.AsDict().Set("label", jinja.NewString(labels[i]))
		}
		list[i] = od
	}
	ins := q.Instructions
	if ins.IsUndefined() {
		ins = jinja.None()
	}
	if ins.IsNone() && d.lab.IDInstructions {
		ins = jinja.NewString(q.ID)
	}
	in := jinja.NewDict()
	dict := in.AsDict()
	dict.Set("id", jinja.NewString(q.ID))
	dict.Set("type", jinja.NewString(q.Type.String()))
	dict.Set("instructions", ins)
	dict.Set("state", state)
	dict.Set("options", jinja.NewList(list))
	if d.lab.SortKeys {
		in = sortKeys(in)
		dict = in.AsDict()
	}
	data := map[string]any{"images": jinja.NewList(nil)}
	for _, k := range dict.Keys {
		data[k] = dict.Data[k]
	}
	return d.tpl.Render(data)
}

// sortKeys is v with every object's keys in sorted order.
func sortKeys(v jinja.Value) jinja.Value {
	switch {
	case v.IsList():
		items := v.AsList().Items
		out := make([]jinja.Value, len(items))
		for i, x := range items {
			out[i] = sortKeys(x)
		}
		return jinja.NewList(out)
	case v.IsDict():
		src := v.AsDict()
		keys := append([]string(nil), src.Keys...)
		sort.Strings(keys)
		out := jinja.NewDict()
		for _, k := range keys {
			out.AsDict().Set(k, sortKeys(src.Data[k]))
		}
		return out
	}
	return v
}

// ParseDecisionQuestions reads TypeSafe's questions object, in its key order.
func ParseDecisionQuestions(v jinja.Value) ([]DecisionQuestion, error) {
	if !v.IsDict() || v.AsDict().Len() == 0 {
		return nil, fmt.Errorf("questions must be a non-empty object")
	}
	var out []DecisionQuestion
	qd := v.AsDict()
	for _, id := range qd.Keys {
		x := qd.Data[id]
		if !x.IsDict() {
			return nil, fmt.Errorf("questions.%s: must be an object", id)
		}
		qx := x.AsDict()
		q := DecisionQuestion{ID: id, Instructions: jinja.None()}
		if ins, ok := qx.Get("instructions"); ok {
			q.Instructions = ins
		}
		tv, _ := qx.Get("type")
		typ := ""
		if tv.IsString() {
			typ = tv.AsString()
		}
		crit, hasCrit := qx.Get("criteria")
		switch typ {
		case "choice":
			q.Type = jlm.QuestionChoice
			switch {
			case hasCrit && crit.IsList():
				// A list of names for undescribed options.
				for _, k := range crit.AsList().Items {
					if !k.IsString() {
						return nil, fmt.Errorf("questions.%s: a criteria list holds option names", id)
					}
					q.Options = append(q.Options, DecisionOption{Key: k.AsString(), Description: jinja.None()})
				}
			case hasCrit && crit.IsDict():
				cd := crit.AsDict()
				for _, k := range cd.Keys {
					q.Options = append(q.Options, DecisionOption{Key: k, Description: cd.Data[k]})
				}
			default:
				return nil, fmt.Errorf("questions.%s: criteria must be a non-empty object", id)
			}
		case "score":
			q.Type = jlm.QuestionScore
			if !hasCrit || !crit.IsList() {
				return nil, fmt.Errorf("questions.%s: criteria must be an array of 2 to 10 levels", id)
			}
			for i, c := range crit.AsList().Items {
				q.Options = append(q.Options, DecisionOption{Key: fmt.Sprint(i), Description: c})
			}
		case "noul":
			q.Type = jlm.QuestionNoul
			if hasCrit && !crit.IsNone() && !crit.IsDict() {
				return nil, fmt.Errorf("questions.%s: criteria must be an object", id)
			}
			for _, k := range []string{"false", "true"} {
				desc := jinja.None()
				if hasCrit && crit.IsDict() {
					if dv, ok := crit.AsDict().Get(k); ok {
						desc = dv
					}
				}
				q.Options = append(q.Options, DecisionOption{Key: k, Description: desc})
			}
		default:
			return nil, fmt.Errorf("questions.%s: type must be one of: choice, score, noul", id)
		}
		out = append(out, q)
	}
	return out, nil
}

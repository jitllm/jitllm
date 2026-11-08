package engine

import (
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
	"github.com/jitllm/jitllm/tok/jinja"
)

// A joint head's readout (jlm.JointHead; model.jointGraph): every question
// of a request in one prompt, the head reading each question's instructions
// and each option's semantics as spans of its rows, one logit an option.

// JointQuestion is one question of a joint request: its row of the type
// table, its instructions' span of the prompt and each option's, a span its
// first row then one past its last.
type JointQuestion struct {
	Type    int
	Span    [2]int
	Options [][2]int
}

// JointScores runs the prompt spans as one joint request of the questions
// qs and writes each option's logit into out, the questions' options in
// order. A span's rows are counted as Span.N counts them, a picture's
// among them.
func (s *Instance) JointScores(spans []model.Span, qs []JointQuestion, out []float32) error {
	n, nq, no := 0, len(qs), 0
	for i := range spans {
		n += spans[i].N(s.c.NEmbd)
	}
	for i := range qs {
		no += len(qs[i].Options)
	}
	switch {
	case !s.c.Joint:
		return fmt.Errorf("model: %s states no joint head", s.c.Arch)
	case n == 0 || n > s.maxSeq:
		return fmt.Errorf("model: %d tokens, and this instance holds %d", n, s.maxSeq)
	case nq == 0 || no == 0 || nq > n || no > n || len(out) < no:
		return fmt.Errorf("model: %d questions of %d options into %d logits over %d tokens", nq, no, len(out), n)
	}
	// The slots are each n words; a form's rows past n read up to its own
	// rows past the last slot's first word.
	words := composer.RequestRows + nn.JointSlots*s.maxSeq
	if cap(s.req) < words {
		s.req = make([]uint32, words)
	}
	req := s.req[:words]
	clear(req)
	slot := func(k, r int) []uint32 { return req[composer.RequestRows+k*n+r:] }
	span := func(k, r int, sp [2]int) error {
		if sp[0] < 0 || sp[1] <= sp[0] || sp[1] > n {
			return fmt.Errorf("model: a span of rows %d to %d of %d", sp[0], sp[1], n)
		}
		slot(k, r)[0], slot(k+1, r)[0] = uint32(sp[0]), uint32(sp[1])
		return nil
	}
	req[composer.RequestCount] = uint32(no)
	o := 0
	for i := range qs {
		q := &qs[i]
		if err := span(nn.JointQuestionSpans, i, q.Span); err != nil {
			return err
		}
		if len(q.Options) == 0 {
			return fmt.Errorf("model: question %d has no options", i)
		}
		slot(nn.JointQuestionTypes, i)[0] = uint32(q.Type)
		slot(nn.JointQuestionOptions, i)[0], slot(nn.JointQuestionOptions+1, i)[0] = uint32(o), uint32(o+len(q.Options))
		for _, sp := range q.Options {
			if err := span(nn.JointOptionSpans, o, sp); err != nil {
				return err
			}
			slot(nn.JointOptionQuestion, o)[0] = uint32(i)
			o++
		}
	}
	// The rows past the questions and the options read the first's, so
	// every row's spans hold rows.
	for r := nq; r < n; r++ {
		for _, k := range []int{nn.JointQuestionSpans, nn.JointQuestionSpans + 1, nn.JointQuestionTypes,
			nn.JointQuestionOptions, nn.JointQuestionOptions + 1} {
			slot(k, r)[0] = slot(k, 0)[0]
		}
	}
	for r := no; r < n; r++ {
		for _, k := range []int{nn.JointOptionSpans, nn.JointOptionSpans + 1, nn.JointOptionQuestion} {
			slot(k, r)[0] = slot(k, 0)[0]
		}
	}
	// A row's token is its id, 0 for a row no token is (a picture's).
	r := 0
	for i := range spans {
		k := spans[i].N(s.c.NEmbd)
		if len(spans[i].Embd) == 0 && len(spans[i].Tokens) == k {
			for j, id := range spans[i].Tokens {
				slot(nn.JointTokens, r+j)[0] = uint32(id)
			}
		}
		r += k
	}
	for r := range n {
		slot(nn.JointFields, r)[0], slot(nn.JointFields+1, r)[0] = 0, uint32(nq)
	}
	s.Reset()
	s.setRequest(req)
	defer s.setRequest(nil)
	s.mark = embedMode{bidir: s.c.NonCausal, whole: true, total: n, out: out[:no], req: req, wins: s.mark.wins}
	s.emb = &s.mark
	defer func() { s.emb = nil }()
	var err error
	if len(spans) == 1 && len(spans[0].Tokens) == n && !spans[0].Bidir {
		_, err = s.Prefill(spans[0].Tokens)
	} else {
		_, err = s.PrefillMixed(spans...)
	}
	return err
}

// The sentinels a joint head's template is rendered with: sep parts its
// pieces, each encoded alone; a piece led by markQuestion is a question's
// instructions' span, by markOption an option's.
const (
	jointSep          = "\x00jitllm.sep\x00"
	jointMarkQuestion = "\x00jitllm.question\x00"
	jointMarkOption   = "\x00jitllm.option\x00"
)

// decideJoint answers every question of qs about state from one prompt
// (jointPrompt), the head's logits taken per question as its options'
// scores.
func (d *Decider) decideJoint(state jinja.Value, qs []DecisionQuestion) ([]DecisionAnswer, error) {
	spans, jq, shown, err := d.jointPrompt(state, qs)
	if err != nil {
		return nil, err
	}
	no := 0
	for i := range qs {
		no += len(qs[i].Options)
	}
	if cap(d.scores) < no {
		d.scores = make([]float32, no)
	}
	logits := d.scores[:no]
	if err := d.in.JointScores(spans, jq, logits); err != nil {
		return nil, err
	}
	d.input = 0
	for i := range spans {
		d.input += spans[i].N(d.in.c.NEmbd)
	}
	out := make([]DecisionAnswer, len(qs))
	at := 0
	for i := range qs {
		q := &qs[i]
		n := len(q.Options)
		clear(d.keep)
		clear(d.vals)
		for k, oi := range shown[i] {
			d.keep[oi], d.params[2+oi] = 1, float32(oi)
			for x := range nn.DecisionGroup {
				d.vals[oi*nn.DecisionGroup+x] = logits[at+k]
			}
		}
		at += n
		d.params[0], d.params[1] = 1/d.cfg.Temperature(q.Type, n), 0
		if !d.in.decideScores(d.vals, d.keep, d.params, d.prior, d.probs, d.stats) {
			return nil, d.in.sampleDeclined("the joint readout")
		}
		a, err := d.result(q, 0, d.joint.Confidence)
		if err != nil {
			return nil, fmt.Errorf("questions.%s: %w", q.ID, err)
		}
		out[i] = a
	}
	return out, nil
}

// jointPrompt is qs about state as one prompt's spans: the systemone
// template rendered with the sentinels, each piece encoded alone and a piece
// holding pictures cut into their spans (Model.PromptSpans); the questions'
// spans of its rows; and shown[i], question i's options in the order the
// prompt shows them, each its index among the request's.
func (d *Decider) jointPrompt(state jinja.Value, qs []DecisionQuestion) ([]model.Span, []JointQuestion, [][]int, error) {
	j := d.joint
	// shown[i] is question i's options in the order the prompt shows them,
	// each its index among the request's.
	shown := make([][]int, len(qs))
	list := make([]jinja.Value, len(qs))
	for i := range qs {
		q := &qs[i]
		order := make([]int, len(q.Options))
		for k := range order {
			order[k] = k
		}
		switch {
		case q.Type == jlm.QuestionNoul && j.NoulTrueFirst:
			slices.Reverse(order)
		case q.Type == jlm.QuestionChoice && j.SortChoices:
			slices.SortStableFunc(order, func(a, b int) int { return strings.Compare(q.Options[a].Key, q.Options[b].Key) })
		}
		shown[i] = order
		opts := make([]jinja.Value, len(order))
		for k, oi := range order {
			o := q.Options[oi]
			desc := o.Description
			if desc.IsUndefined() {
				desc = jinja.None()
			}
			od := jinja.NewDict()
			od.AsDict().Set("key", jinja.NewString(o.Key))
			od.AsDict().Set("description", desc)
			opts[k] = od
		}
		ins := q.Instructions
		if ins.IsUndefined() || ins.IsNone() || ins.IsString() && ins.AsString() == "" {
			ins = jinja.None()
			if j.IDInstructions {
				ins = jinja.NewString(q.ID)
			}
		}
		qd := jinja.NewDict()
		qd.AsDict().Set("id", jinja.NewString(q.ID))
		qd.AsDict().Set("type", jinja.NewString(q.Type.String()))
		qd.AsDict().Set("instructions", ins)
		qd.AsDict().Set("options", jinja.NewList(opts))
		list[i] = qd
	}
	questions := jinja.NewList(list)
	if j.SortKeys {
		state, questions = sortKeys(state), sortKeys(questions)
	}
	text, err := d.tpl.Render(map[string]any{"images": d.images(), "state": state, "questions": questions,
		"sep": jinja.NewString(jointSep), "mark_question": jinja.NewString(jointMarkQuestion),
		"mark_option": jinja.NewString(jointMarkOption)})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("model: the systemone template: %w", err)
	}
	// prompt is the spans before ids, the tokens since the last picture;
	// rows counts both.
	var prompt []model.Span
	var ids []int32
	var spans [][2]int
	rows, used := 0, 0
	for _, piece := range strings.Split(text, jointSep) {
		if k := strings.Count(piece, decisionPicture); k > 0 {
			if used+k > len(d.pics) {
				return nil, nil, nil, fmt.Errorf("model: the template wrote %d pictures for %d", used+k, len(d.pics))
			}
			ps, err := d.in.m.PromptSpans(piece, decisionPicture, d.pics[used:used+k])
			if err != nil {
				return nil, nil, nil, err
			}
			used += k
			if len(ids) > 0 {
				prompt = append(prompt, model.Span{Tokens: ids})
				ids = nil
			}
			for i := range ps {
				rows += ps[i].N(d.in.c.NEmbd)
			}
			prompt = append(prompt, ps...)
			continue
		}
		lead := strings.HasPrefix(piece, jointMarkQuestion) || strings.HasPrefix(piece, jointMarkOption)
		piece = strings.TrimPrefix(strings.TrimPrefix(piece, jointMarkQuestion), jointMarkOption)
		at := rows
		tok := d.encodePlain(piece)
		ids = append(ids, tok...)
		rows += len(tok)
		if lead {
			spans = append(spans, [2]int{at, rows})
		}
	}
	if used != len(d.pics) {
		return nil, nil, nil, fmt.Errorf("model: the template wrote %d of %d pictures", used, len(d.pics))
	}
	if len(ids) > 0 {
		prompt = append(prompt, model.Span{Tokens: ids})
	}
	jq := make([]JointQuestion, len(qs))
	no := 0
	for i := range qs {
		no += len(qs[i].Options)
	}
	if len(spans) != len(qs)+no {
		return nil, nil, nil, fmt.Errorf("model: %d questions of %d options, and the prompt marks %d spans", len(qs), no, len(spans))
	}
	at := 0
	for i := range qs {
		jq[i] = JointQuestion{Type: int(j.TypeRows[qs[i].Type]), Span: spans[at], Options: spans[at+1 : at+1+len(qs[i].Options)]}
		at += 1 + len(qs[i].Options)
	}
	return prompt, jq, shown, nil
}

package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// clefAnswers splits a Clef pass's scores -- question by question, each in
// the order its prompt showed the options -- into the answers: a choice's and
// a score's back in the request's order, a noul's kept true first, the order
// answer reads it in (Decider.order).
func (d *Decider) clefAnswers(qs []DecisionQuestion, s []float32) ([]DecisionAnswer, error) {
	out := make([]DecisionAnswer, len(qs))
	at := 0
	for qi := range qs {
		q := &qs[qi]
		k := len(q.Options)
		if at+k > len(s) {
			return nil, fmt.Errorf("the head gave %d scores for more options", len(s))
		}
		row := make([]float32, k)
		for i, oi := range d.clefOrder(q) {
			if q.Type == jlm.QuestionNoul {
				row[i] = s[at+i]
			} else {
				row[oi] = s[at+i]
			}
		}
		at += k
		a, err := d.answer(q, [][]float32{row})
		if err != nil {
			return nil, fmt.Errorf("questions.%s: %w", q.ID, err)
		}
		out[qi] = a
	}
	return out, nil
}

// clefJob writes a request's prompt into a batch job.
func (d *Decider) clefJob(j *batchJob, state jinja.Value, qs []DecisionQuestion) error {
	p, err := d.clefPrompt(state, qs)
	if err != nil {
		return err
	}
	j.clef, j.ids = p, p.ids
	return nil
}

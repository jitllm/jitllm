package engine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// ParseDecisionRequest reads a /v1/systemone body: the model it names ("" for
// none), the state every question is about and the questions, every object's
// keys in the order the body wrote them.
func ParseDecisionRequest(body []byte) (string, jinja.Value, []DecisionQuestion, error) {
	v, err := jinja.FromJSON(body)
	if err != nil {
		return "", jinja.None(), nil, fmt.Errorf("the request is not JSON: %w", err)
	}
	if !v.IsDict() {
		return "", jinja.None(), nil, fmt.Errorf("the request must be a JSON object")
	}
	req := v.AsDict()
	name := ""
	if mv, ok := req.Get("model"); ok && mv.IsString() {
		name = mv.AsString()
	}
	state, ok := req.Get("state")
	if !ok {
		return "", jinja.None(), nil, fmt.Errorf("state must be provided")
	}
	qv, _ := req.Get("questions")
	qs, err := ParseDecisionQuestions(qv)
	if err != nil {
		return "", jinja.None(), nil, err
	}
	return name, state, qs, nil
}

// DecisionResponse is TypeSafe's response body: the answers keyed by question
// id in request order, each in TypeSafe's shape. A score's legend carries
// each level's description as the request gave it, or, where the readout
// states SortedLegend, a description that is not text as its compact JSON
// with every object's keys sorted, a string.
func (d *Decider) DecisionResponse(modelID string, qs []DecisionQuestion, answers []DecisionAnswer) []byte {
	var b strings.Builder
	b.WriteString(`{"model":`)
	b.WriteString(strconv.Quote(modelID))
	b.WriteString(`,"answers":{`)
	for i, q := range qs {
		if i > 0 {
			b.WriteByte(',')
		}
		a := answers[i]
		b.WriteString(jsonString(q.ID))
		b.WriteString(`:{"type":`)
		b.WriteString(jsonString(q.Type.String()))
		switch q.Type {
		case jlm.QuestionNoul:
			b.WriteString(`,"noul":`)
			b.WriteString(jsonFloat(a.Noul))
		case jlm.QuestionChoice:
			b.WriteString(`,"choice":`)
			b.WriteString(jsonString(a.Choice))
			b.WriteString(`,"confidence":`)
			b.WriteString(jsonFloat(a.Confidence))
			writeProbs(&b, q, a)
		case jlm.QuestionScore:
			b.WriteString(`,"score":`)
			b.WriteString(jsonFloat(a.Score))
			b.WriteString(`,"confidence":`)
			b.WriteString(jsonFloat(a.Confidence))
			b.WriteString(`,"legend":{`)
			for j, o := range q.Options {
				if j > 0 {
					b.WriteByte(',')
				}
				b.WriteString(jsonString(o.Key))
				b.WriteByte(':')
				desc := o.Description
				if desc.IsUndefined() {
					desc = jinja.None()
				}
				if d.lab.SortedLegend && !desc.IsString() {
					src := sortKeys(desc).JSON()
					var c bytes.Buffer
					if json.Compact(&c, []byte(src)) == nil {
						src = c.String()
					}
					b.WriteString(jsonString(src))
					continue
				}
				b.WriteString(desc.JSON())
			}
			b.WriteByte('}')
			writeProbs(&b, q, a)
		}
		b.WriteByte('}')
	}
	b.WriteString(`},"usage":{"input_tokens":`)
	b.WriteString(strconv.Itoa(d.input))
	b.WriteString(`,"output_tokens":0}}`)
	return []byte(b.String())
}

func writeProbs(b *strings.Builder, q DecisionQuestion, a DecisionAnswer) {
	b.WriteString(`,"probabilities":{`)
	for j, o := range q.Options {
		if j > 0 {
			b.WriteByte(',')
		}
		b.WriteString(jsonString(o.Key))
		b.WriteByte(':')
		b.WriteString(jsonFloat(a.Probs[j]))
	}
	b.WriteByte('}')
}

func jsonString(s string) string { return jinja.NewString(s).JSON() }

func jsonFloat(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }

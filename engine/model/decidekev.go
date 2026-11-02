package model

import (
	"fmt"
	"math"
	"regexp"
	"strings"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Kev's pointer head (jaredpalmer/kev; llama.cpp's server-decision.cpp and
// the qwen35 graph's cls_out): the backbone's final hidden state, normed by
// the output norm, projected by cls.output to 2P values; an option's score is
// the question's query -- the first P of the prompt's last row -- dotted with
// the option's key -- the last P of the row at the <|box_end|> that ends it --
// over sqrt(P). The prompt is the systemone template over the request flattened
// to text (kevText).

// kevHead is cls.output: P = rows/2.
type kevHead struct {
	w tensor
	b []float32
}

// kevLoad reads Kev's head from the container, once per model.
func (m *Model) kevLoad() (*kevHead, error) {
	m.bind.Lock()
	defer m.bind.Unlock()
	if m.kev != nil {
		return m.kev, nil
	}
	c := m.container
	d := m.Cfg.NEmbd
	var e *jlm.Entry
	for _, idx := range []int32{-1, 0, 1} {
		if x, ok := c.Find(jlm.RoleScorerOut, jlm.DenseBlock, idx); ok {
			e = x
			break
		}
	}
	if e == nil {
		return nil, fmt.Errorf("model: kev head: the container carries no cls.output")
	}
	typ, ok := jlm.SourceType(e.Type)
	if !ok || !quant.Dequantable(typ) || e.NDim != 2 || int(e.Dims[0]) != d || e.Dims[1]%2 != 0 {
		return nil, fmt.Errorf("model: kev head: cls.output is %v %v, want a %d-wide matrix of an even row count",
			e.Type, e.Dims[:e.NDim], d)
	}
	h := &kevHead{w: tensor{e: e, typ: typ, rows: int(e.Dims[1]), k: d}}
	bindOne(c, &h.w)
	h.b = make([]float32, h.w.rows)
	if be, ok := c.Find(jlm.RoleScorerOutBias, e.Block, e.Index); ok {
		bt, ok := jlm.SourceType(be.Type)
		if !ok || int(be.Dims[0]) != h.w.rows {
			return nil, fmt.Errorf("model: kev head: cls.output.bias is %v %v, want %d", be.Type, be.Dims[:be.NDim], h.w.rows)
		}
		qs, _, _ := c.Span(be)
		if err := quant.Dequant32(bt, qs, h.b); err != nil {
			return nil, err
		}
	}
	m.kev = h
	return h, nil
}

// kevDevHead is the head as a device reads it, built once per model: the
// query and key halves of cls.output f32, row by row through the generated
// row decoders.
func (m *Model) kevDevHead() *nn.KevHead {
	m.bind.Lock()
	defer m.bind.Unlock()
	if m.kevDev != nil {
		return m.kevDev
	}
	h := m.kev
	t, d, p := h.w, m.Cfg.NEmbd, h.w.rows/2
	all := make([]float32, t.rows*d)
	for r := 0; r < t.rows; r++ {
		row := all[r*d : (r+1)*d]
		if t.packed != nil && len(t.packed.QS) != 0 {
			nn.RowPacked32JIT(row, t.typ, t.packed, r, t.rows, d)
		} else {
			nn.Row32JIT(row, t.typ, t.data, r, d)
		}
	}
	m.kevDev = &nn.KevHead{D: d, P: p, RMSEps: float32(m.Cfg.RMSEps), OutNorm: m.outNorm,
		WQ: all[:p*d], WK: all[p*d:], BQ: h.b[:p], BK: h.b[p:]}
	return m.kevDev
}

// kevOnDevice runs the head on the instance's device when its output projection
// is there: the pointer row and the option rows go up, the scores come back.
// false: the host's generated head runs it. A device that has no room for
// the head declines it as it declines a block (kevDevWhy says why); the
// every-block gate holds the device to running it.
func (d *Decider) kevOnDevice(ids []int32, out []float32) bool {
	s, r := d.st, d.kev
	dev, isDev := s.ld.(nn.KevHeadDevice)
	if !isDev || s.head == nil || d.violation != "" || d.m.outNormB != nil || d.m.Cfg.LayerNorm {
		return false
	}
	n := d.m.Cfg.NEmbd
	if cap(r.keys) < len(r.mk)*n {
		r.keys = make([]float32, len(r.mk)*n)
	}
	keys := r.keys[:len(r.mk)*n]
	for i, at := range r.mk {
		copy(keys[i*n:(i+1)*n], r.hid[at*n:(at+1)*n])
	}
	last := len(ids) - 1
	if !dev.KevHead(d.m.kevDevHead(), r.hid[last*n:(last+1)*n], keys, out) {
		d.kevDevWhy = "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
			d.kevDevWhy = e.Err()
		}
		return false
	}
	d.kevDevRuns++
	return true
}

// kevRun is a Decider's scratch for Kev's head.
type kevRun struct {
	head *kevHead
	tap  func(base int, rows []float32)
	hid  []float32 // every row's final residual
	rows int
	norm []float32 // one normed row
	q, k []float32 // the question's projection, one option's
	mk   []int     // the <|box_end|> positions
	keys []float32 // the option rows, for a device
	box  int32     // the <|box_end|> token
}

func (d *Decider) kevInit() error {
	h, err := d.m.kevLoad()
	if err != nil {
		return err
	}
	id, ok := d.single("<|box_end|>")
	if !ok {
		if ids := d.m.Vocab.EncodeSpecial("<|box_end|>", false); len(ids) == 1 {
			id, ok = ids[0], true
		}
	}
	if !ok {
		return fmt.Errorf("model: kev: the vocabulary has no <|box_end|> token")
	}
	r := &kevRun{head: h, box: id}
	r.norm = make([]float32, d.m.Cfg.NEmbd)
	r.q, r.k = make([]float32, h.w.rows), make([]float32, h.w.rows)
	r.tap = d.kevTap
	d.kev = r
	return nil
}

func (d *Decider) kevTap(_ int, rows []float32) {
	r := d.kev
	n := d.m.Cfg.NEmbd
	if need := (r.rows*n + len(rows)); cap(r.hid) < need {
		grown := make([]float32, need, 2*need)
		copy(grown, r.hid[:r.rows*n])
		r.hid = grown
	}
	r.hid = r.hid[:r.rows*n+len(rows)]
	copy(r.hid[r.rows*n:], rows)
	r.rows += len(rows) / n
}

// kevForward runs q's prompt and leaves one score an option, in the order
// the prompt shows them, in variant v's scratch.
func (d *Decider) kevForward(state jinja.Value, q *DecisionQuestion, v int) ([]float32, error) {
	ids, mk, err := d.kevPrompt(state, q, v)
	if err != nil {
		return nil, err
	}
	return d.kevRows(ids, mk, v)
}

// kevPrompt is q's prompt ids and the positions of the <|box_end|> that ends
// each option: the text half of a question.
func (d *Decider) kevPrompt(state jinja.Value, q *DecisionQuestion, v int) ([]int32, []int, error) {
	prompt, _, err := d.promptText(state, q, v)
	if err != nil {
		return nil, nil, err
	}
	ids := d.m.Vocab.EncodeSpecial(prompt, false)
	var mk []int
	for i, id := range ids {
		if id == d.kev.box {
			mk = append(mk, i)
		}
	}
	if len(mk) != len(q.Options) {
		return nil, nil, fmt.Errorf("the prompt holds %d option ends for %d options", len(mk), len(q.Options))
	}
	return ids, mk, nil
}

// kevRows runs a prompt and the head over its rows: the engine half, which
// makes no heap allocation once warm (TestDecisionDoesNotAllocate).
func (d *Decider) kevRows(ids []int32, mk []int, v int) ([]float32, error) {
	if len(ids) > d.st.MaxSeq() {
		return nil, fmt.Errorf("the prompt is %d tokens and this decider holds %d", len(ids), d.st.MaxSeq())
	}
	d.kevBegin(mk)
	_, err := d.st.Prefill(ids)
	d.kevEnd()
	if err != nil {
		return nil, err
	}
	d.input += len(ids)
	return d.kevHead(ids, v)
}

// kevBegin readies the instance for a prompt whose option ends are at mk: every
// row it prefills, in as many calls as it likes, is kept for the head.
func (d *Decider) kevBegin(mk []int) {
	r, s := d.kev, d.st
	r.mk = mk
	if d.violation != "no-reset" {
		s.Reset()
	}
	r.rows = 0
	s.emb, s.rowTap = &d.clefEmb, r.tap
}

// kevEnd stops keeping rows.
func (d *Decider) kevEnd() {
	d.kevRuns++
	d.st.emb, d.st.rowTap = nil, nil
}

// kevHead is the pointer head over the rows kevBegin kept, one score an
// option in variant v's scratch.
func (d *Decider) kevHead(ids []int32, v int) ([]float32, error) {
	r, s := d.kev, d.st
	out := d.variantScores(v, len(r.mk))
	if d.kevOnDevice(ids, out) {
		return out, nil
	}
	n, p := d.m.Cfg.NEmbd, r.head.w.rows/2
	project := func(out []float32, row int) error {
		s.norm(r.norm, r.hid[row*n:(row+1)*n], d.m.outNorm, d.m.outNormB)
		// The activation cache is keyed by the buffer, which every row
		// reuses: a new input each row.
		s.jit.NewInput()
		if err := s.mv(out, r.head.w, r.norm); err != nil {
			return err
		}
		nn.Add32JIT(out, r.head.b)
		return nil
	}
	pointer := len(ids) - 1
	if err := project(r.q, pointer); err != nil {
		return nil, err
	}
	for i, at := range r.mk {
		if d.violation == "key-before-end" {
			at-- // a gate's: each key read a row before its option's end
		}
		if err := project(r.k, at); err != nil {
			return nil, err
		}
		key := r.k[p:]
		if d.violation == "same-half" {
			key = r.k[:p] // a gate's: the key read from the query's half
		}
		dot, err := d.dot(r.q[:p], key)
		if err != nil {
			return nil, err
		}
		out[i] = dot
	}
	nn.Scale32JIT(out, float32(1/math.Sqrt(float64(p))))
	return out, nil
}

// kevSpecial is the form a special token takes in request text: Kev's
// server writes <|name|> as <¦name¦>, so text cannot fake a marker.
var kevSpecial = regexp.MustCompile(`<\|([A-Za-z0-9_]+)\|>`)

// kevText is v as Kev's text (kev/api.py render, as llama.cpp's server
// writes it): an object's keys as labels, a list as dashed lines, nested
// values indented, booleans as Python writes them, special tokens escaped.
func kevText(v jinja.Value) jinja.Value {
	return jinja.NewString(kevSpecial.ReplaceAllString(kevRender(v, 0), "<¦$1¦>"))
}

func kevRender(v jinja.Value, indent int) string {
	pad := strings.Repeat("  ", indent)
	switch {
	case v.IsUndefined() || v.IsNone():
		return ""
	case v.IsString():
		return v.AsString()
	case v.IsBool():
		if v.AsBool() {
			return "True"
		}
		return "False"
	case v.IsList():
		var b strings.Builder
		for _, item := range v.AsList().Items {
			text := strings.TrimLeft(kevRender(item, indent+1), " \t\n\r")
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(pad + "- " + text)
		}
		return b.String()
	case v.IsDict():
		// In the request's order: llama.cpp's server's json keeps insertion
		// order, as kev/api.py's dict does.
		var b strings.Builder
		dct := v.AsDict()
		for _, key := range dct.Keys {
			item := dct.Data[key]
			nested := item.IsDict() || item.IsList()
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			if nested {
				b.WriteString(pad + key + ":\n" + kevRender(item, indent+1))
			} else {
				b.WriteString(pad + key + ": " + kevRender(item, 0))
			}
		}
		return b.String()
	}
	return v.JSON()
}

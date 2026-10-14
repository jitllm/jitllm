package model

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Clef's readout (jlm.DecisionClef): one prompt per request holding the
// state and the whole schema, the backbone's final hidden states over every
// row of it, and the joint schema head (joint_schema_model.py's
// JointSchemaHead in Cloudflare/clef-flash) reading them -- every question and
// every option mean-pooled over its span, routing blocks in which the options
// read the prompt, joint blocks in which the questions read each other and
// the prompt, and one logit per option of every question in one pass.
//
// The prompt is built as the reference builds it (encode_record): each piece
// tokenized on its own, JSON values compact with sorted keys, a choice's
// options sorted by key, a noul's true first. llama.cpp's server renders the
// GGUF's systemone template with separators between the same pieces, which is
// the same sequence.
//
// Every operation over a row is a generated kernel: the norms, the matmuls,
// the attention's scores, softmax and accumulate, the GELU, the row means
// (axpy), the elementwise products and the L2 norms (an RMSNorm and a scale).
// What is Go is the loops over rows and spans and a few setup scalars.

const clefSystem = "Read the complete state and schema. Decide every field jointly. Each answer " +
	"must be exactly one of that field's allowed options."

// clefTypes is the reference's QUESTION_TYPES: the row of the type table each
// question type reads.
var clefTypes = [...]int32{jlm.QuestionNoul: 0, jlm.QuestionChoice: 1, jlm.QuestionScore: 2}

// clefBlock is one of the head's blocks: a routing block (kvNorm set, no self
// attention) or a joint block (self attention, the memory read raw).
type clefBlock struct {
	selfW, selfB       []float32
	sq, sk, sv, so     tensor
	sbq, sbk, sbv, sbo []float32
	crossW, crossB     []float32
	kvW, kvB           []float32
	cq, ck, cv, co     tensor
	cbq, cbk, cbv, cbo []float32
	ffnW, ffnB         []float32
	up, down           tensor
	bUp, bDown         []float32
}

func (b *clefBlock) weights() []*tensor {
	return []*tensor{&b.sq, &b.sk, &b.sv, &b.so, &b.cq, &b.ck, &b.cv, &b.co, &b.up, &b.down}
}

// clefHead is the joint schema head's weights.
type clefHead struct {
	hidW, hidB                  []float32
	pMem, pQ, pOQ, pG, pOC, pOL tensor
	sumW, sumB, fieldW, fieldB  []float32
	optW, optB                  []float32
	scales                      []float32 // prior, joint, gate: the values used
	types                       []float32 // three rows of the head's width
	sc                          tensor
	scB, scOut, scOutB          []float32
	blocks                      []clefBlock
	routing, width, heads, ffn  int
	eps                         float64
}

func (h *clefHead) weights() []*tensor {
	ws := []*tensor{&h.pMem, &h.pQ, &h.pOQ, &h.pG, &h.pOC, &h.pOL, &h.sc}
	for i := range h.blocks {
		ws = append(ws, h.blocks[i].weights()...)
	}
	return ws
}

// clefLoad reads Clef's head from the container, once per model.
func (m *Model) clefLoad() (*clefHead, error) {
	m.bind.Lock()
	defer m.bind.Unlock()
	if m.clef != nil {
		return m.clef, nil
	}
	c := m.container
	jc := c.Config()
	d := m.Cfg.NEmbd
	var err error
	get := func(role jlm.Role, index int32, rows, k int) tensor {
		if err != nil {
			return tensor{}
		}
		e, ok := c.Find(role, jlm.DenseBlock, index)
		if !ok {
			err = fmt.Errorf("model: clef head: missing %v (head block %d)", role, index)
			return tensor{}
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok || !quant.Dequantable(typ) || e.NDim != 2 {
			err = fmt.Errorf("model: clef head: %v (head block %d) is %v in %d dimensions", role, index, e.Type, e.NDim)
			return tensor{}
		}
		t := tensor{e: e, typ: typ, rows: int(e.Dims[1]), k: int(e.Dims[0])}
		if (rows > 0 && t.rows != rows) || (k > 0 && t.k != k) {
			err = fmt.Errorf("model: clef head: %v (head block %d) is %dx%d, want %dx%d", role, index, t.rows, t.k, rows, k)
			return tensor{}
		}
		bindOne(c, &t)
		return t
	}
	vec := func(role jlm.Role, index int32, want int) []float32 {
		if err != nil {
			return nil
		}
		e, ok := c.Find(role, jlm.DenseBlock, index)
		if !ok {
			err = fmt.Errorf("model: clef head: missing %v (head block %d)", role, index)
			return nil
		}
		typ, ok := jlm.SourceType(e.Type)
		if !ok {
			err = fmt.Errorf("model: clef head: %v is %v, which has no dequantizer", role, e.Type)
			return nil
		}
		n := 1
		for i := 0; i < int(e.NDim); i++ {
			n *= int(e.Dims[i])
		}
		if n != want {
			err = fmt.Errorf("model: clef head: %v (head block %d) has %d values, want %d", role, index, n, want)
			return nil
		}
		qs, _, _ := c.Span(e)
		out := make([]float32, n)
		if e := quant.Dequant32(typ, qs, out); e != nil {
			err = fmt.Errorf("model: clef head: %v: %w", role, e)
		}
		return out
	}
	h := &clefHead{routing: int(jc.DecisionRouting), heads: int(jc.DecisionHeads), eps: float64(jc.DecisionEps)}
	h.pMem = get(jlm.RoleDecProjMemory, -1, 0, d)
	w := h.pMem.rows
	if err != nil {
		return nil, err
	}
	if w == 0 || h.heads == 0 || w%h.heads != 0 {
		return nil, fmt.Errorf("model: clef head: %d heads do not divide a %d-wide head", h.heads, w)
	}
	h.width = w
	h.hidW, h.hidB = vec(jlm.RoleDecHiddenNorm, -1, d), vec(jlm.RoleDecHiddenNormBias, -1, d)
	h.pQ = get(jlm.RoleDecProjQuestion, -1, w, d)
	h.pOQ = get(jlm.RoleDecProjOptQ, -1, w, d)
	h.pG = get(jlm.RoleDecProjGlobal, -1, w, d)
	h.pOC = get(jlm.RoleDecProjOptCtx, -1, w, d)
	h.pOL = get(jlm.RoleDecProjOptLex, -1, w, d)
	h.sumW, h.sumB = vec(jlm.RoleDecOptSumNorm, -1, w), vec(jlm.RoleDecOptSumNormBias, -1, w)
	h.fieldW, h.fieldB = vec(jlm.RoleDecFieldNorm, -1, w), vec(jlm.RoleDecFieldNormBias, -1, w)
	h.optW, h.optB = vec(jlm.RoleDecOptNorm, -1, w), vec(jlm.RoleDecOptNormBias, -1, w)
	h.scales = vec(jlm.RoleDecScales, -1, 3)
	h.types = vec(jlm.RoleTokenTypes, jlm.DenseBlock, 3*w)
	h.sc = get(jlm.RoleScorer, -1, w, 4*w)
	h.scB = vec(jlm.RoleScorerBias, -1, w)
	h.scOut = vec(jlm.RoleScorerOut, -1, w)
	h.scOutB = vec(jlm.RoleScorerOutBias, -1, 1)
	h.blocks = make([]clefBlock, int(jc.DecisionRouting+jc.DecisionBlocks))
	for i := range h.blocks {
		b, bi := &h.blocks[i], int32(i)
		if i >= h.routing {
			b.selfW, b.selfB = vec(jlm.RoleDecSelfNorm, bi, w), vec(jlm.RoleDecSelfNormBias, bi, w)
			b.sq, b.sk = get(jlm.RoleDecSelfQ, bi, w, w), get(jlm.RoleDecSelfK, bi, w, w)
			b.sv, b.so = get(jlm.RoleDecSelfV, bi, w, w), get(jlm.RoleDecSelfOut, bi, w, w)
			b.sbq, b.sbk = vec(jlm.RoleDecSelfQBias, bi, w), vec(jlm.RoleDecSelfKBias, bi, w)
			b.sbv, b.sbo = vec(jlm.RoleDecSelfVBias, bi, w), vec(jlm.RoleDecSelfOutBias, bi, w)
		} else {
			b.kvW, b.kvB = vec(jlm.RoleDecCrossKVNorm, bi, w), vec(jlm.RoleDecCrossKVNormBias, bi, w)
		}
		b.crossW, b.crossB = vec(jlm.RoleDecCrossNorm, bi, w), vec(jlm.RoleDecCrossNormBias, bi, w)
		b.cq, b.ck = get(jlm.RoleDecCrossQ, bi, w, w), get(jlm.RoleDecCrossK, bi, w, w)
		b.cv, b.co = get(jlm.RoleDecCrossV, bi, w, w), get(jlm.RoleDecCrossOut, bi, w, w)
		b.cbq, b.cbk = vec(jlm.RoleDecCrossQBias, bi, w), vec(jlm.RoleDecCrossKBias, bi, w)
		b.cbv, b.cbo = vec(jlm.RoleDecCrossVBias, bi, w), vec(jlm.RoleDecCrossOutBias, bi, w)
		b.ffnW, b.ffnB = vec(jlm.RoleDecFFNNorm, bi, w), vec(jlm.RoleDecFFNNormBias, bi, w)
		b.up = get(jlm.RoleDecFFNUp, bi, 0, w)
		if err != nil {
			return nil, err
		}
		if i == 0 {
			h.ffn = b.up.rows
		} else if b.up.rows != h.ffn {
			return nil, fmt.Errorf("model: clef head blocks 0 and %d differ in MLP width", i)
		}
		b.down = get(jlm.RoleDecFFNDown, bi, w, h.ffn)
		b.bUp, b.bDown = vec(jlm.RoleDecFFNUpBias, bi, h.ffn), vec(jlm.RoleDecFFNDownBias, bi, w)
	}
	if err != nil {
		return nil, err
	}
	if m.output.e == nil || m.output.k != d {
		return nil, fmt.Errorf("model: clef head reads the output embedding, and the model has none %d wide", d)
	}
	m.clef = h
	return h, nil
}

// clefSpan is one question's or option's rows of the prompt, [lo, hi).
type clefSpan struct{ lo, hi int }

// clefPrompt is a request's prompt: its ids, each question's span and type
// row, each option's span and question, in the order the prompt shows them.
type clefPrompt struct {
	// ids is the prompt's token per row, a picture's rows standing as its
	// processor's pad token; spans is the prompt as it is prefilled when it
	// carries pictures (nil when it does not).
	ids    []int32
	spans  []Span
	qs     []clefSpan
	qtype  []int32
	opts   []clefSpan
	owner  []int32
	nopts  []int // options per question
	pieces [][]int32
}

// clefOrder is the order a question's options are shown in, as indexes into
// q.Options: a noul's true first, a choice's by key, a score's by level.
func (d *Decider) clefOrder(q *DecisionQuestion) []int {
	idx := make([]int, len(q.Options))
	for i := range idx {
		idx[i] = i
	}
	switch q.Type {
	case jlm.QuestionNoul:
		if d.violation != "false-first" {
			idx[0], idx[1] = 1, 0
		}
	case jlm.QuestionChoice:
		if d.violation != "unsorted" {
			sort.SliceStable(idx, func(a, b int) bool { return q.Options[idx[a]].Key < q.Options[idx[b]].Key })
		}
	}
	return idx
}

// clefRender is the reference's render: a string as it is, any other value
// as compact JSON with sorted keys and no ASCII escaping.
func clefRender(v jinja.Value) string {
	if v.IsString() {
		return v.AsString()
	}
	var b strings.Builder
	clefJSON(&b, v)
	return b.String()
}

// clefJSON is json.dumps(v, ensure_ascii=False, separators=(",", ":"),
// sort_keys=True).
func clefJSON(b *strings.Builder, v jinja.Value) { pyJSON(b, v, ",", ":", true) }

// pyJSON is json.dumps(v, ensure_ascii=False, separators=(comma, colon),
// sort_keys=sortKeys).
func pyJSON(b *strings.Builder, v jinja.Value, comma, colon string, sortKeys bool) {
	switch {
	case v.IsString():
		pyJSONString(b, v.AsString())
	case v.IsList():
		b.WriteByte('[')
		for i, x := range v.AsList().Items {
			if i > 0 {
				b.WriteString(comma)
			}
			pyJSON(b, x, comma, colon, sortKeys)
		}
		b.WriteByte(']')
	case v.IsDict():
		dd := v.AsDict()
		keys := dd.Keys
		if sortKeys {
			keys = append([]string(nil), dd.Keys...)
			sort.Strings(keys)
		}
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteString(comma)
			}
			pyJSONString(b, k)
			b.WriteString(colon)
			pyJSON(b, dd.Data[k], comma, colon, sortKeys)
		}
		b.WriteByte('}')
	default:
		b.WriteString(v.JSON())
	}
}

// pyJSONString is a JSON string as Python's json.dumps writes it with
// ensure_ascii off: the two-character escapes, \u00XX below 0x20, every other
// rune as it is.
func pyJSONString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for i := 0; i < len(s); {
		r, n := utf8.DecodeRuneInString(s[i:])
		i += n
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
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// clefNoul is a noul option's description where the request gives none
// (question_options' defaults).
func clefNoul(key string) string {
	if key == "true" {
		return "The proposition is true or the answer is yes."
	}
	return "The proposition is false or the answer is no."
}

// clefPrompt builds a request's prompt (encode_record): the system prompt and
// the state, the schema with each question's instruction and each option's
// semantics as spans, and the assistant's opening. The state is cut to what
// the Decider holds after everything else.
func (d *Decider) clefPrompt(state jinja.Value, qs []DecisionQuestion) (*clefPrompt, error) {
	enc := func(s string) []int32 { return d.m.Vocab.EncodeSpecial(s, false) }
	p := &clefPrompt{}
	var schema []int32
	schema = append(schema, enc("\n\nSCHEMA FIELDS:\n")...)
	for qi := range qs {
		q := &qs[qi]
		schema = append(schema, enc(fmt.Sprintf("\nFIELD %d\nID: %s\nTYPE: %s\nINSTRUCTION: ", qi+1, q.ID, q.Type))...)
		ins := q.Instructions
		if ins.IsUndefined() || ins.IsNone() || (ins.IsString() && ins.AsString() == "") {
			ins = jinja.NewString(q.ID)
		}
		lo := len(schema)
		schema = append(schema, enc(clefRender(ins))...)
		if len(schema) == lo {
			return nil, fmt.Errorf("questions.%s: the instructions tokenize to nothing", q.ID)
		}
		p.qs = append(p.qs, clefSpan{lo, len(schema)})
		p.qtype = append(p.qtype, clefTypes[q.Type])
		schema = append(schema, enc("\nALLOWED OPTIONS:\n")...)
		order := d.clefOrder(q)
		for i, oi := range order {
			o := q.Options[oi]
			schema = append(schema, enc("OPTION "+strconv.Itoa(i+1)+": ")...)
			sem := jinja.NewDict()
			desc := o.Description
			if q.Type == jlm.QuestionNoul && (desc.IsUndefined() || desc.IsNone()) {
				desc = jinja.NewString(clefNoul(o.Key))
			}
			if !desc.IsUndefined() && !desc.IsNone() {
				sem.AsDict().Set("description", desc)
			}
			sem.AsDict().Set("option_id", jinja.NewString(o.Key))
			lo := len(schema)
			schema = append(schema, enc(clefRender(sem))...)
			p.opts = append(p.opts, clefSpan{lo, len(schema)})
			p.owner = append(p.owner, int32(qi))
			schema = append(schema, enc("\n")...)
		}
		p.nopts = append(p.nopts, len(order))
		schema = append(schema, enc("END FIELD\n")...)
	}
	prefix := enc("<|im_start|>system\n" + clefSystem + "<|im_end|>\n<|im_start|>user\nSTATE:\n")
	// The pictures (encode_record's _encode_media): each one's markers and
	// rows, then a newline, between the prefix and the state.
	// The pictures, then the videos (each pair of frames a Picture whose
	// rows stand as <|video_pad|>).
	var media []Span
	var mediaIDs []int32
	if len(d.pics) > 0 || len(d.vids) > 0 {
		pad, ok1 := d.m.Vocab.ID("<|image_pad|>")
		vpad, ok2 := d.m.Vocab.ID("<|video_pad|>")
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("the vocabulary has no <|image_pad|> and <|video_pad|>")
		}
		for _, run := range d.pics {
			media = append(media, run...)
		}
		for _, run := range d.vids {
			media = append(media, run...)
		}
		media = append(media, Span{Tokens: enc("\n")})
		for _, sp := range media {
			if sp.Picture != nil {
				id := pad
				if sp.Picture.Pixels2 != nil {
					id = vpad
				}
				for range sp.Picture.Rows {
					mediaIDs = append(mediaIDs, id)
				}
				continue
			}
			mediaIDs = append(mediaIDs, sp.Tokens...)
		}
		prefix = append(prefix[:len(prefix):len(prefix)], mediaIDs...)
	}
	suffix := enc("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
	st := enc(clefRender(state))
	fixed := len(prefix) + len(schema) + len(suffix)
	room := d.st.MaxSeq()
	if fixed > room {
		return nil, fmt.Errorf("the schema needs %d tokens before the state, and this decider holds %d", fixed, room)
	}
	st = trim(st, room-fixed)
	off := len(prefix) + len(st)
	for i := range p.qs {
		p.qs[i].lo += off
		p.qs[i].hi += off
	}
	for i := range p.opts {
		p.opts[i].lo += off
		p.opts[i].hi += off
	}
	p.ids = append(append(append(append(make([]int32, 0, fixed+len(st)), prefix...), st...), schema...), suffix...)
	if len(media) > 0 {
		head := len(prefix) - len(mediaIDs)
		p.spans = append([]Span{{Tokens: p.ids[:head]}}, media...)
		p.spans = append(p.spans, Span{Tokens: p.ids[len(prefix):]})
		p.spans = mergeTokenSpans(p.spans)
	}
	return p, nil
}

// clefRun is the head's working memory, grown to the largest request and
// kept: a warm request allocates nothing in the engine's half.
type clefRun struct {
	hid, tmp           []float32 // n rows of the backbone's width
	mem, kv, k, v      []float32 // n rows of the head's width
	qv, gq             []float32 // a question's pooled row, projected
	ctx, lex           []float32 // an option's pooled rows
	opts, oq, a, t, ff []float32
	fields             []float32
	feat               []float32 // an option's four scorer features
	anchor, ln, ones   []float32
	att                []float32 // the heads' score rows
	acc                []float32
	prior, cos, res    []float32 // one per option
	w                  []float32 // a question's option weights
	owner, qtype       []int32
	glob               []float32
	rows               int
	attn               *nn.AttnSet
	tap                func(int, []float32)
	at                 clefAttnRegion
}

// clefAttnRegion is the head's attention over heads: nq query rows over nk
// key rows, rows of the head's width (clefAttend).
type clefAttnRegion struct {
	out, q, k, v []float32
	nq, nk, hd   int
	stride       int
	scale        float32
	fn           func(lo, hi int)
}

func growF(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n)
	}
	return b[:n]
}

// clefGrow sizes the run for a prompt of n rows, nq questions and no options.
func (d *Decider) clefGrow(n, nq, no int) {
	r, h, D := d.clef, d.clefH, d.m.Cfg.NEmbd
	W := h.width
	r.hid, r.tmp = growF(r.hid, n*D), growF(r.tmp, n*D)
	r.mem, r.kv, r.k, r.v = growF(r.mem, n*W), growF(r.kv, n*W), growF(r.k, n*W), growF(r.v, n*W)
	rows := max(nq, no)
	r.qv, r.gq = growF(r.qv, nq*D), growF(r.gq, nq*W)
	r.ctx, r.lex = growF(r.ctx, no*D), growF(r.lex, no*D)
	r.opts, r.oq, r.a = growF(r.opts, no*W), growF(r.oq, no*W), growF(r.a, rows*W)
	r.t, r.ff = growF(r.t, max(rows, 2)*W), growF(r.ff, rows*max(h.ffn, W))
	r.fields = growF(r.fields, nq*W)
	r.feat = growF(r.feat, no*4*W)
	r.anchor, r.ln = growF(r.anchor, nq*D), growF(r.ln, max(D, W))
	if len(r.ones) < max(D, W) {
		r.ones = make([]float32, max(D, W))
		for i := range r.ones {
			r.ones[i] = 1
		}
	}
	r.att, r.acc = growF(r.att, h.heads*max(n, rows)), growF(r.acc, W)
	r.prior, r.cos, r.res = growF(r.prior, no), growF(r.cos, no), growF(r.res, no)
	r.w = growF(r.w, no)
	r.glob = growF(r.glob, W)
	if cap(r.owner) < no {
		r.owner = make([]int32, no)
	}
	if cap(r.qtype) < nq {
		r.qtype = make([]int32, nq)
	}
	r.owner, r.qtype = r.owner[:no], r.qtype[:nq]
}

// clefInit builds the head's working state for a Decider.
func (d *Decider) clefInit() error {
	h, err := d.m.clefLoad()
	if err != nil {
		return err
	}
	d.clefH = h
	d.clef = &clefRun{}
	hd := h.width / h.heads
	d.clef.attn = d.st.jit.AttnSetFor(hd, hd, h.width, cpu.KVF32)
	d.clef.at.fn = d.clefHeads
	// The tap as a value made once: a method value per prompt is a heap
	// object per prompt.
	d.clef.tap = d.clefTap
	return nil
}

// clefForward runs a request's prompt and the head, and leaves one score per
// option of every question in out, question by question, each in the order
// the prompt shows it.
func (d *Decider) clefForward(p *clefPrompt, out []float32) error {
	// 1. The backbone over the prompt, every row's final residual kept: the
	// prefill stops at the residual (no output projection) and hands each
	// chunk's rows over as they come out.
	d.clefBegin(p)
	var err error
	if p.spans != nil {
		_, err = d.st.PrefillMixed(p.spans...)
	} else {
		_, err = d.st.Prefill(p.ids)
	}
	d.clefEnd()
	if err != nil {
		return err
	}
	return d.clefHead(p, out)
}

// clefBegin readies the State to keep every row of p's prefill (one call or
// several, as a relocation mid-prompt runs it) and clefEnd stops it.
func (d *Decider) clefBegin(p *clefPrompt) {
	r, s := d.clef, d.st
	d.clefGrow(len(p.ids), len(p.qs), len(p.opts))
	copy(r.owner, p.owner)
	copy(r.qtype, p.qtype)
	if d.violation != "no-reset" {
		s.Reset()
	}
	r.rows = 0
	s.emb = &d.clefEmb
	s.rowTap = r.tap
}

func (d *Decider) clefEnd() { d.st.emb, d.st.rowTap = nil, nil }

// clefHead is the head over p's rows, kept by the prefill since clefBegin.
func (d *Decider) clefHead(p *clefPrompt, out []float32) error {
	r, h, s, c := d.clef, d.clefH, d.st, d.m.Cfg
	D, W := c.NEmbd, h.width
	n, nq, no := len(p.ids), len(p.qs), len(p.opts)
	if r.rows != n {
		return fmt.Errorf("model: the prompt's %d rows gave %d hidden rows", n, r.rows)
	}
	d.input += n
	// Where the backbone's head is on a device, the decision head runs there
	// over the rows (decideclefdev.go); a device that declines leaves them
	// for the host below.
	if ok, err := d.clefOnDevice(p, out); ok || err != nil {
		return err
	}
	hid := r.hid[:n*D]
	// The backbone's output norm, then the head's LayerNorm over it.
	s.batchNormB(r.tmp[:n*D], hid, s.outNorm, d.m.outNormB, D, n)
	s.normRows(hid, r.tmp[:n*D], h.hidW, h.hidB, D, n, true, h.eps, 1)
	if d.violation == "no-hidden-norm" {
		copy(hid, r.tmp[:n*D])
	}

	// 2. The memory, the pooled spans and the global row.
	mem := r.mem[:n*W]
	if err := d.clefMM(mem, h.pMem, hid, n); err != nil {
		return err
	}
	global := hid[(n-1)*D : n*D]
	for i, sp := range p.qs {
		d.clefMean(r.qv[i*D:(i+1)*D], hid, D, sp)
	}
	for i, sp := range p.opts {
		d.clefMean(r.ctx[i*D:(i+1)*D], hid, D, sp)
		if err := d.clefLexical(r.lex[i*D:(i+1)*D], p.ids[sp.lo:sp.hi]); err != nil {
			return err
		}
	}

	// 3. One query per option: its context, its words and its question.
	opts := r.opts[:no*W]
	if err := d.clefMM(opts, h.pOC, r.ctx[:no*D], no); err != nil {
		return err
	}
	if d.violation != "no-lexical" {
		if err := d.clefMM(r.oq[:no*W], h.pOL, r.lex[:no*D], no); err != nil {
			return err
		}
		nn.Axpy32JIT(opts, r.oq[:no*W], 1)
	}
	if err := d.clefMM(r.gq[:nq*W], h.pOQ, r.qv[:nq*D], nq); err != nil {
		return err
	}
	s.axpyRows(opts, r.gq[:nq*W], r.owner, W, no, 1, 1)

	// 4. The routing blocks: the options read the normed memory.
	for bi := 0; bi < h.routing; bi++ {
		b := &h.blocks[bi]
		kv := r.kv[:n*W]
		s.normRows(kv, mem, b.kvW, b.kvB, W, n, true, h.eps, 1)
		q := r.t[:no*W]
		s.normRows(q, opts, b.crossW, b.crossB, W, no, true, h.eps, 1)
		if err := d.clefAttn(r.a[:no*W], q, kv, no, n, b.cq, b.ck, b.cv, b.co, b.cbq, b.cbk, b.cbv, b.cbo); err != nil {
			return err
		}
		nn.Axpy32JIT(opts, r.a[:no*W], 1)
		if err := d.clefFFN(opts, b, no); err != nil {
			return err
		}
	}

	// 5. One field per question: its text, a summary of its options weighted
	// by how they meet it, the end of the prompt and its type.
	fields := r.fields[:nq*W]
	if err := d.clefMM(fields, h.pQ, r.qv[:nq*D], nq); err != nil {
		return err
	}
	sum := r.a[:nq*W]
	inv := float32(1 / math.Sqrt(float64(W)))
	at := 0
	for qi := 0; qi < nq; qi++ {
		k := p.nopts[qi]
		w := r.w[:k]
		if err := d.clefRows(w, opts[at*W:(at+k)*W], fields[qi*W:(qi+1)*W], k, W); err != nil {
			return err
		}
		nn.Scale32JIT(w, inv)
		nn.Softmax32JIT(w, k)
		row := sum[qi*W : (qi+1)*W]
		clear(row)
		for j := 0; j < k; j++ {
			nn.Axpy32JIT(row, opts[(at+j)*W:(at+j+1)*W], w[j])
		}
		at += k
	}
	s.normRows(r.t[:nq*W], sum, h.sumW, h.sumB, W, nq, true, h.eps, 1)
	copy(sum, r.t[:nq*W])
	nn.Axpy32JIT(fields, sum, 1)
	if err := d.clefMM(r.glob, h.pG, global, 1); err != nil {
		return err
	}
	s.axpyRows(fields, r.glob, nil, W, nq, 1, 1)
	if d.violation != "no-type" {
		s.axpyRows(fields, h.types, r.qtype, W, nq, 1, 1)
	}

	// 6. The joint blocks: the fields read each other, then the raw memory.
	for bi := h.routing; bi < len(h.blocks); bi++ {
		b := &h.blocks[bi]
		x := r.t[:nq*W]
		s.normRows(x, fields, b.selfW, b.selfB, W, nq, true, h.eps, 1)
		if d.violation != "no-self" {
			if err := d.clefAttn(r.a[:nq*W], x, x, nq, nq, b.sq, b.sk, b.sv, b.so, b.sbq, b.sbk, b.sbv, b.sbo); err != nil {
				return err
			}
			nn.Axpy32JIT(fields, r.a[:nq*W], 1)
		}
		s.normRows(x, fields, b.crossW, b.crossB, W, nq, true, h.eps, 1)
		if err := d.clefAttn(r.a[:nq*W], x, mem, nq, n, b.cq, b.ck, b.cv, b.co, b.cbq, b.cbk, b.cbv, b.cbo); err != nil {
			return err
		}
		nn.Axpy32JIT(fields, r.a[:nq*W], 1)
		if err := d.clefFFN(fields, b, nq); err != nil {
			return err
		}
	}
	s.normRows(r.t[:nq*W], fields, h.fieldW, h.fieldB, W, nq, true, h.eps, 1)
	copy(fields, r.t[:nq*W])

	// 7. The prior: each option's words against its question and the end of
	// the prompt, both L2-normed.
	for qi := 0; qi < nq; qi++ {
		a := r.anchor[qi*D : (qi+1)*D]
		copy(a, r.qv[qi*D:(qi+1)*D])
		nn.Axpy32JIT(a, global, 1)
		d.clefL2(a)
	}
	for o := 0; o < no; o++ {
		x := r.lex[o*D : (o+1)*D]
		d.clefL2(x)
		qi := int(r.owner[o])
		if err := d.clefRows(r.prior[o:o+1], x, r.anchor[qi*D:(qi+1)*D], 1, D); err != nil {
			return err
		}
	}

	// 8. The joint score: each normed option against its question's field,
	// as a cosine and through the scorer over [f, o, f*o, |f-o|].
	on := r.oq[:no*W]
	s.normRows(on, opts, h.optW, h.optB, W, no, true, h.eps, 1)
	for o := 0; o < no; o++ {
		qi := int(r.owner[o])
		f, x := fields[qi*W:(qi+1)*W], on[o*W:(o+1)*W]
		ft := r.feat[o*4*W : (o+1)*4*W]
		copy(ft[:W], f)
		copy(ft[W:2*W], x)
		prod, diff := ft[2*W:3*W], ft[3*W:]
		copy(prod, f)
		nn.ActMul32JIT(prod, x, nn.ActIdentity)
		// |f - x| as relu(f - x) + relu(x - f).
		copy(diff, f)
		nn.Axpy32JIT(diff, x, -1)
		neg := r.ln[:W]
		copy(neg, diff)
		nn.Scale32JIT(neg, -1)
		nn.Act32JIT(diff, nn.ActReLU)
		nn.Act32JIT(neg, nn.ActReLU)
		nn.Axpy32JIT(diff, neg, 1)
		// The cosine of the two, L2-normed in the scorer's own copies.
		fl, xl := r.t[:W], r.t[W:2*W]
		copy(fl, f)
		copy(xl, x)
		d.clefL2(fl)
		d.clefL2(xl)
		if err := d.clefRows(r.cos[o:o+1], fl, xl, 1, W); err != nil {
			return err
		}
	}
	hs := r.a[:no*W]
	if err := d.clefMM(hs, h.sc, r.feat[:no*4*W], no); err != nil {
		return err
	}
	s.axpyRows(hs, h.scB, nil, W, no, 1, 1)
	nn.Act32JIT(hs, nn.ActGELUErf)
	if err := d.clefRows(r.res[:no], hs, h.scOut, no, W); err != nil {
		return err
	}
	// logit = scale0 * prior + scale2 * (scale1 * cos + residual): the
	// residual's bias first, then the scales, each a generated op over the
	// options.
	s.axpyRows(r.res[:no], h.scOutB, nil, 1, no, 1, 1)
	if d.violation != "no-cosine" {
		nn.Axpy32JIT(r.res[:no], r.cos[:no], h.scales[1])
	}
	out = out[:no]
	copy(out, r.prior[:no])
	nn.Scale32JIT(out, h.scales[0])
	if d.violation == "no-prior" {
		clear(out)
	}
	nn.Axpy32JIT(out, r.res[:no], h.scales[2])
	return nil
}

// clefTap keeps a prefill chunk's final residual rows.
func (d *Decider) clefTap(_ int, rows []float32) {
	r := d.clef
	copy(r.hid[r.rows*d.m.Cfg.NEmbd:], rows)
	r.rows += len(rows) / d.m.Cfg.NEmbd
}

// clefMean is dst = the mean of rows sp of x (rows of width w).
func (d *Decider) clefMean(dst, x []float32, w int, sp clefSpan) {
	clear(dst)
	inv := 1 / float32(sp.hi-sp.lo)
	if d.violation == "first-row" {
		copy(dst, x[sp.lo*w:(sp.lo+1)*w])
		return
	}
	for i := sp.lo; i < sp.hi; i++ {
		nn.Axpy32JIT(dst, x[i*w:(i+1)*w], inv)
	}
}

// clefLexical is dst = the mean of the output embedding's rows for ids.
func (d *Decider) clefLexical(dst []float32, ids []int32) error {
	o := &d.m.output
	clear(dst)
	row := d.clef.tmp[:len(dst)]
	inv := 1 / float32(len(ids))
	for _, id := range ids {
		if id < 0 || int(id) >= o.rows {
			return fmt.Errorf("model: token %d is outside the output's %d rows", id, o.rows)
		}
		if o.packed != nil && len(o.packed.QS) != 0 {
			nn.RowPacked32JIT(row, o.typ, o.packed, int(id), o.rows, o.k)
		} else {
			nn.Row32JIT(row, o.typ, o.data, int(id), o.k)
		}
		nn.Axpy32JIT(dst, row, inv)
	}
	return nil
}

// clefL2 is x / ||x|| in place: an RMSNorm with unit weight, which divides by
// sqrt(sum/n), then the scale by 1/sqrt(n).
func (d *Decider) clefL2(x []float32) {
	n := len(x)
	r := d.clef
	nn.RMSNorm32JIT(r.ln[:n], x, r.ones[:n], 1e-24/float64(n))
	copy(x, r.ln[:n])
	nn.Scale32JIT(x, float32(1/math.Sqrt(float64(n))))
}

// clefRows is out[i] = rows[i] . x for k rows of width w, on the generated
// f32 matvec.
func (d *Decider) clefRows(out, rows, x []float32, k, w int) error {
	if !d.st.jit.MatVecHost(out[:k], quant.F32, f32Bytes(rows[:k*w]), x[:w], k, w) {
		return fmt.Errorf("model: no generated f32 matvec of %dx%d on this host", k, w)
	}
	return nil
}

// clefMM is out = W x for n rows of x.
func (d *Decider) clefMM(out []float32, w tensor, x []float32, n int) error {
	j := d.st.jit
	if w.packed != nil {
		if n > 1 && j.MatMulPacked(out, w.typ, w.packed, x, w.rows, w.k, n) {
			return nil
		}
		for i := 0; i < n; i++ {
			j.NewInput()
			if !j.MatVecPacked(out[i*w.rows:(i+1)*w.rows], w.typ, w.packed, x[i*w.k:(i+1)*w.k], w.rows, w.k) {
				return fmt.Errorf("model: no host kernel reads %s (%s)", w.name(), w.typ)
			}
		}
		return nil
	}
	if n > 1 && j.MatMulFloat(out, w.typ, w.data, x, w.rows, w.k, n) {
		return nil
	}
	for i := 0; i < n; i++ {
		if !j.MatVecHost(out[i*w.rows:(i+1)*w.rows], w.typ, w.data, x[i*w.k:(i+1)*w.k], w.rows, w.k) {
			return fmt.Errorf("model: no host kernel reads %s (%s, %dx%d)", w.name(), w.typ, w.rows, w.k)
		}
	}
	return nil
}

// clefAttn is torch's MultiheadAttention: out = o(attn(q(xq), k(xkv),
// v(xkv))) + bias, nq query rows over nk key rows, no mask.
func (d *Decider) clefAttn(out, xq, xkv []float32, nq, nk int, wq, wk, wv, wo tensor, bq, bk, bv, bo []float32) error {
	r, h, s := d.clef, d.clefH, d.st
	W := h.width
	q := r.ff[:nq*W] // the MLP's scratch is free during the attention
	k, v := r.k[:nk*W], r.v[:nk*W]
	for _, p := range []struct {
		w   tensor
		b   []float32
		x   []float32
		dst []float32
		n   int
	}{{wq, bq, xq, q, nq}, {wk, bk, xkv, k, nk}, {wv, bv, xkv, v, nk}} {
		if err := d.clefMM(p.dst, p.w, p.x, p.n); err != nil {
			return err
		}
		s.axpyRows(p.dst, p.b, nil, W, p.n, 1, 1)
	}
	hd := W / h.heads
	j := &r.at
	j.out, j.q, j.k, j.v, j.nq, j.nk, j.hd, j.stride = out, q, k, v, nq, nk, hd, max(nq, nk)
	j.scale = float32(1 / math.Sqrt(float64(hd)))
	s.jit.Parallel(h.heads, 1, j.fn)
	j.out, j.q, j.k, j.v = nil, nil, nil, nil
	// The heads' rows are in out; the projection goes through r.t, which held
	// the normed queries every caller passes and is free once q is made.
	t := r.t[:nq*W]
	if err := d.clefMM(t, wo, out[:nq*W], nq); err != nil {
		return err
	}
	copy(out[:nq*W], t)
	s.axpyRows(out, bo, nil, W, nq, 1, 1)
	return nil
}

// clefHeads is clefAttn's heads lo..hi-1.
func (d *Decider) clefHeads(lo, hi int) {
	r := d.clef
	j := &r.at
	W := d.clefH.width
	for h := lo; h < hi; h++ {
		off := h * j.hd
		att := r.att[h*j.stride : h*j.stride+j.nk]
		for i := 0; i < j.nq; i++ {
			r.attn.AttnScores(att, j.k[off:], j.q[i*W+off:], j.nk)
			nn.Scale32JIT(att, j.scale)
			nn.Softmax32JIT(att, j.nk)
			r.attn.AttnAcc(j.out[i*W+off:i*W+off+j.hd], j.v[off:], att, j.nk)
		}
	}
}

// clefFFN is x += down(gelu(up(LN(x)))) for n rows, biases on both.
func (d *Decider) clefFFN(x []float32, b *clefBlock, n int) error {
	r, h, s := d.clef, d.clefH, d.st
	W := h.width
	ln := r.t[:n*W]
	s.normRows(ln, x, b.ffnW, b.ffnB, W, n, true, h.eps, 1)
	u := r.ff[:n*h.ffn]
	if err := d.clefMM(u, b.up, ln, n); err != nil {
		return err
	}
	s.axpyRows(u, b.bUp, nil, h.ffn, n, 1, 1)
	act := nn.ActGELUErf
	if d.violation == "tanh-gelu" {
		act = nn.ActGELU
	}
	nn.Act32JIT(u, act)
	dn := r.a[:n*W]
	if err := d.clefMM(dn, b.down, u, n); err != nil {
		return err
	}
	s.axpyRows(dn, b.bDown, nil, W, n, 1, 1)
	nn.Axpy32JIT(x, dn, 1)
	return nil
}

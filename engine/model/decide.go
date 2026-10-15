package model

import (
	"fmt"
	"image"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Decisions: a decision model answers typed questions about a state in one
// forward pass and generates nothing (docs/design/decision-models.md). The
// request is TypeSafe's /v1/systemone; this file is the engine half of it --
// the prompt each question is asked with, the scores read from the forward
// pass, and the calibrated answer. The readout is the container's
// (jlm.DecisionKind): what the reference hardcodes per model, stated at
// conversion.
//
// The label readouts (Lev, d1) are the backbone's own forward pass and logits
// read at a handful of ids, so a decision runs wherever that backbone runs, on
// every tier, through Prefill.

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

// Decision is the model's readout, DecisionNone for a model that is not a
// decision model.
func (m *Model) Decision() jlm.DecisionKind {
	if m.container == nil {
		return jlm.DecisionNone
	}
	return m.container.Config().Decision
}

// levRatings is the scale lev reads a noul from: 0 = certainly no, 8 =
// certainly yes, read at the first nine codes.
const levRatings = 9

// Decider asks a decision model questions. It holds one State and is one
// request at a time; build one per concurrent caller.
type Decider struct {
	m    *Model
	cfg  *jlm.Config
	kind jlm.DecisionKind
	st   *State
	tpl  *jinja.Template

	// lev: the single-token codes A..Z, AA..ZZ, in order, and their ids.
	labels     []int32
	labelTexts []string
	maxOptions int

	scores []float32
	input  int
	// pick is a question's label ids, the groups flattened, and pickVals
	// their logits: picked on the device where the head is there
	// (nn.Head.Pick), gathered from the row where it is not. picked counts
	// the prompts whose labels the device picked.
	pick     []int32
	pickVals []float32
	picked   int

	// The answer's scratch: the averaged probabilities, one variant's, a
	// working row, and the ramp 0..n-1 (answer).
	acc, prob, row, ramp []float32

	// emb is the encoder's working memory, for a readout that runs on an
	// encoder (Laya); st is nil then.
	emb *Embedder

	// one is forward's prompt as one span, kept so a text question
	// allocates nothing.
	one [1]Span

	// vids is the request in flight's videos, each its run of the prompt
	// (cutVideos); nil between requests.
	vids [][]Span

	// pics is the request in flight's pictures, each its run of the prompt
	// (cutPictures); nil between requests and for a text request.
	pics [][]Span
	// omniPre is d1-omni's picture prefix for the request in flight, rows of
	// NEmbd (none for text), and omniVis the vision segment that encodes it.
	omniPre []float32
	// omniSpoken says the prefix is a voice clip's (DecideAudio), and
	// omniClip is the clip as the tower reads it.
	omniSpoken bool
	omniClip   []float32
	omniVis    *State
	// vscores is each variant's scores (variantScores): a question's
	// variants are answered together, so each keeps its own.
	vscores [][]float32

	// batch is the session DecideBatch runs a decoder readout's prompts in,
	// placed where dev and devMax (SetDevice's) say.
	batch  *Scheduler
	dev    nn.Device
	devMax int
	// jobs is DecideBatch's prompts, kept from batch to batch (batchJob).
	jobs []*batchJob
	// batchN is how many of jobs the last batch ran.
	batchN int
	// packIDs and packMarks are a packed encoder pass's ids and marker rows
	// (layaPacked), and packs counts the passes run.
	packIDs   []int32
	packMarks []int
	packs     int

	// clef is Clef's head and its working memory (decideclef.go), and
	// clefEmb the prefill mode its prompt runs under: the residual of every
	// row, no output projection.
	clefH   *clefHead
	clef    *clefRun
	clefEmb embedMode
	// clefIn is the device head's input, kept; clefDevRuns counts the
	// requests whose head ran on a device.
	clefIn      nn.ClefHeadInput
	clefDevRuns int

	// violation names one feature of the readout a test removes, to show the
	// gate sees it (RULE 10). Empty in every caller but those tests.
	violation string
}

// NewDecider builds a Decider whose prompts may run to maxSeq tokens.
func (m *Model) NewDecider(maxSeq int) (*Decider, error) {
	kind := m.Decision()
	if kind == jlm.DecisionNone {
		return nil, fmt.Errorf("model: %s is not a decision model: its container states no decision readout", m.Cfg.Arch)
	}
	if m.Vocab == nil {
		return nil, fmt.Errorf("model: a decision model needs its tokenizer: %v", m.TokErr)
	}
	src, ok := m.ChatTemplate("systemone")
	if !ok {
		return nil, fmt.Errorf("model: decision model carries no systemone template")
	}
	tpl, err := jinja.Compile(src)
	if err != nil {
		return nil, fmt.Errorf("model: the systemone template: %w", err)
	}
	d := &Decider{m: m, cfg: m.container.Config(), kind: kind, tpl: tpl, maxOptions: 255}
	switch kind {
	case jlm.DecisionLev, jlm.DecisionPplxDecider:
		// Codes A..Z then AA..ZZ; only those that are one token are used, up
		// to 255 (lev's serving code, ADR-028).
		var codes []string
		for a := 'A'; a <= 'Z'; a++ {
			codes = append(codes, string(a))
		}
		for a := 'A'; a <= 'Z'; a++ {
			for b := 'A'; b <= 'Z'; b++ {
				codes = append(codes, string([]rune{a, b}))
			}
		}
		for _, c := range codes {
			if id, ok := d.single(c); ok && len(d.labels) < 255 {
				d.labels = append(d.labels, id)
				d.labelTexts = append(d.labelTexts, c)
			}
		}
		if len(d.labels) < levRatings {
			return nil, fmt.Errorf("model: lev needs at least %d single-token codes, the vocabulary has %d", levRatings, len(d.labels))
		}
		d.maxOptions = len(d.labels)
	case jlm.DecisionLFM2D1:
	case jlm.DecisionOpenJev:
		// One letter per option, each of which must be one token
		// (server-decision.cpp).
		for _, c := range "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" {
			id, ok := d.single(string(c))
			if !ok {
				return nil, fmt.Errorf("model: openjev's label %q is not one token", c)
			}
			d.labels = append(d.labels, id)
		}
		d.maxOptions = len(d.labels)
	case jlm.DecisionClef:
		d.st = m.NewState(maxSeq)
		if err := d.clefInit(); err != nil {
			d.st.Close()
			return nil, err
		}
		return d, nil
	case jlm.DecisionLaya, jlm.DecisionD1Omni:
		if m.encMB() == nil || m.encMB().head == nil {
			return nil, fmt.Errorf("model: a %v readout wants an encoder with its head", kind)
		}
		d.emb = &Embedder{m: m}
		if err := d.emb.encoderJIT(); err != nil {
			return nil, err
		}
		m.mbBind()
		return d, nil
	default:
		return nil, fmt.Errorf("model: decision readout %v has no implementation", kind)
	}
	d.st = m.NewState(maxSeq)
	return d, nil
}

// Close releases the Decider's State.
func (d *Decider) Close() error {
	if d.emb != nil {
		if d.omniVis != nil {
			d.omniVis.Close()
		}
		d.emb.Close()
		return nil
	}
	if d.batch != nil {
		d.batch.Close()
	}
	return d.st.Close()
}

// SetDevice places the Decider's blocks on d through the one placement path
// (State.SetDeviceLayers): max caps how many are offered, -1 offers every
// block and lets the device's memory decide, and nil detaches. A decoder
// readout's prompts then prefill there and its label logits come from the
// device's head; an encoder readout's blocks are placed the same way
// (Embedder.SetDevice).
func (d *Decider) SetDevice(dev nn.Device, max int) error {
	if d.emb != nil {
		if err := d.emb.SetDeviceLayers(dev, max); err != nil {
			return err
		}
		// An audio tower's blocks go where the encoder's do, every one the
		// device takes (d1-omni's).
		if max != 0 {
			return d.emb.setAudioDevice(dev)
		}
		return d.emb.setAudioDevice(nil)
	}
	if err := d.st.SetDeviceLayers(dev, max); err != nil {
		return err
	}
	return d.setBatchDevice(dev, max)
}

// DeviceBlocks is how many of the readout's blocks run on a device.
func (d *Decider) DeviceBlocks() int {
	if d.emb != nil {
		return d.emb.DeviceBlocks()
	}
	return d.st.GPULayers()
}

// single is the id text encodes to when it is exactly one token.
func (d *Decider) single(text string) (int32, bool) {
	ids := d.m.Vocab.Encode(text, false)
	if len(ids) != 1 {
		return 0, false
	}
	return ids[0], true
}

// InputTokens is how many tokens the last Decide ran.
func (d *Decider) InputTokens() int { return d.input }

// Prepare checks a request's questions against what this readout can ask and
// puts each in the order the model was trained to see it: a noul true first on
// d1. It is called by Decide; a caller that wants the refusal before queueing
// calls it first.
func (d *Decider) Prepare(state jinja.Value, qs []DecisionQuestion, images int) error {
	if err := d.checkImages(images); err != nil {
		return err
	}
	if len(qs) == 0 {
		return fmt.Errorf("questions must be a non-empty object")
	}
	// d1, d1-omni and Clef read a null state (pictures alone): d1's runner,
	// d1-omni's system_one ("" for None) and
	// llama.cpp allow it, and Clef's systemone renders it as JSON null.
	if (state.IsNone() || state.IsUndefined()) && d.kind != jlm.DecisionLFM2D1 && d.kind != jlm.DecisionClef &&
		d.kind != jlm.DecisionD1Omni {
		return fmt.Errorf("state must be provided")
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
		if n > d.maxOptions {
			return fmt.Errorf("questions.%s: too many options (%d), this model supports at most %d", q.ID, n, d.maxOptions)
		}
	}
	return nil
}

// Decide answers every question about state and imgs, one forward pass per
// question (two for a lev choice, its options shown in both orders). qs is
// the request's questions in its order; the answers come back in the same
// order. imgs are the request's pictures, in its order (d1 only).
func (d *Decider) Decide(state jinja.Value, qs []DecisionQuestion, imgs []image.Image) ([]DecisionAnswer, error) {
	return d.DecideMedia(state, qs, imgs, nil)
}

// DecideMedia is Decide for a request that carries videos too (Clef's).
func (d *Decider) DecideMedia(state jinja.Value, qs []DecisionQuestion, imgs []image.Image,
	vids []VideoFrames) ([]DecisionAnswer, error) {
	if err := d.Prepare(state, qs, len(imgs)); err != nil {
		return nil, err
	}
	if err := d.checkVideos(len(vids), len(imgs)); err != nil {
		return nil, err
	}
	d.input = 0
	d.pics, d.vids = nil, nil
	if len(imgs) > 0 && d.kind == jlm.DecisionD1Omni {
		if err := d.d1OmniPrefix(imgs); err != nil {
			return nil, err
		}
		defer func() { d.omniPre = d.omniPre[:0] }()
	} else if len(imgs) > 0 {
		pics, err := d.cutPictures(imgs)
		if err != nil {
			return nil, err
		}
		d.pics = pics
		defer func() { d.pics = nil }()
		state = replaceText(state, decisionMarker, " ")
	}
	if len(vids) > 0 {
		v, err := d.cutVideos(vids)
		if err != nil {
			return nil, err
		}
		d.vids = v
		defer func() { d.vids = nil }()
	}
	out := make([]DecisionAnswer, len(qs))
	if d.kind == jlm.DecisionClef {
		// One pass answers every question jointly.
		p, err := d.clefPrompt(state, qs)
		if err != nil {
			return nil, err
		}
		s := d.variantScores(0, len(p.opts))
		if err := d.clefForward(p, s); err != nil {
			return nil, err
		}
		return d.clefAnswers(qs, s)
	}
	for i := range qs {
		q := &qs[i]
		nv := d.variants(q)
		var all [][]float32
		for v := 0; v < nv; v++ {
			s, err := d.run(state, q, v)
			if err != nil {
				return nil, fmt.Errorf("questions.%s: %w", q.ID, err)
			}
			all = append(all, s)
		}
		a, err := d.answer(q, all)
		if err != nil {
			return nil, fmt.Errorf("questions.%s: %w", q.ID, err)
		}
		out[i] = a
	}
	return out, nil
}

// variants is how many times a question is asked: lev shows a choice's options
// in two orders, to cancel its preference for the first label.
func (d *Decider) variants(q *DecisionQuestion) int {
	if d.kind == jlm.DecisionLev && q.Type == jlm.QuestionChoice && len(q.Options) > 1 && d.violation != "one-order" {
		return 2
	}
	return 1
}

// order is the question's options in the order the prompt shows them: d1
// shows a noul true first.
func (d *Decider) order(q *DecisionQuestion) []DecisionOption {
	if (d.kind == jlm.DecisionLFM2D1 || d.kind == jlm.DecisionClef || d.kind == jlm.DecisionOpenJev) && q.Type == jlm.QuestionNoul && d.violation != "false-first" {
		return []DecisionOption{q.Options[1], q.Options[0]}
	}
	return q.Options
}

// run renders variant v of q, runs it, and returns one score per output: per
// option in the shown order (reversed for variant 1), or lev's nine ratings
// for a noul. The scores are the Decider's, good until the next run of the
// same variant.
func (d *Decider) run(state jinja.Value, q *DecisionQuestion, v int) ([]float32, error) {
	if d.kind == jlm.DecisionClef {
		// A request of this one question, its scores in the prompt's order.
		p, err := d.clefPrompt(state, []DecisionQuestion{*q})
		if err != nil {
			return nil, err
		}
		out := d.variantScores(v, len(p.opts))
		return out, d.clefForward(p, out)
	}
	if d.kind == jlm.DecisionLaya {
		ids, markers, err := d.layaSequence(state, q)
		if err != nil {
			return nil, err
		}
		return d.layaForward(ids, q.Type, markers, v)
	}
	if d.kind == jlm.DecisionD1Omni {
		ids, markers, err := d.d1OmniSequence(state, q, d.d1OmniMaxLen())
		if err != nil {
			return nil, err
		}
		return d.d1OmniForward(ids, q.Type, markers, v)
	}
	if len(d.pics) > 0 {
		spans, groups, err := d.promptSpans(state, q, v)
		if err != nil {
			return nil, err
		}
		return d.forwardSpans(spans, groups, v)
	}
	ids, groups, err := d.prompt(state, q, v)
	if err != nil {
		return nil, err
	}
	return d.forward(ids, groups, v)
}

// forward runs a decoder readout's prompt ids and reads its label groups'
// scores into variant v's scratch: the engine's half of a question, which
// makes no heap allocation once warm (TestDecisionDoesNotAllocate). Writing
// the prompt is text work before it.
func (d *Decider) forward(ids []int32, groups [][]int32, v int) ([]float32, error) {
	d.one[0] = Span{Tokens: ids}
	return d.forwardSpans(d.one[:], groups, v)
}

// forwardSpans is forward over a prompt of spans: a request's pictures
// stand in it as Picture spans (decideimages.go).
func (d *Decider) forwardSpans(spans []Span, groups [][]int32, v int) ([]float32, error) {
	d.st.Reset()
	// The labels' logits: where the head is on a device, gathered there and
	// those alone read back (nn.Head.Pick) -- not under a head bias or a
	// logit scale, which the host adds after the head.
	flat := d.pick[:0]
	for _, g := range groups {
		flat = append(flat, g...)
	}
	d.pick = flat
	if cap(d.pickVals) < len(flat) {
		d.pickVals = make([]float32, len(flat))
	}
	h := d.st.head
	if h != nil && d.m.outB == nil && d.st.c.LogitScale == 1 && d.violation != "no-pick" {
		h.Pick, h.PickVals, h.Picked = flat, d.pickVals[:len(flat)], false
		if d.violation == "pick-off-by-one" {
			h.Pick = make([]int32, len(flat))
			for i, id := range flat {
				h.Pick[i] = id + 1
			}
		}
		defer func() { h.Pick, h.PickVals = nil, nil }()
	}
	// Where the head runs on the host, the same pick there: the labels'
	// rows alone (State.pickOnHost).
	if d.violation != "no-pick" {
		if cap(d.pickVals) < len(flat) {
			d.pickVals = make([]float32, len(flat))
		}
		d.st.HostPick, d.st.HostPickVals = flat, d.pickVals[:len(flat)]
		if d.violation == "pick-off-by-one" {
			d.st.HostPick = make([]int32, len(flat))
			for i, id := range flat {
				d.st.HostPick[i] = id + 1
			}
		}
		defer func() { d.st.HostPick, d.st.HostPickVals, d.st.HostPicked = nil, nil, false }()
	}
	var lg []float32
	var err error
	if len(spans) == 1 && spans[0].Picture == nil && spans[0].Embd == nil {
		lg, err = d.st.Prefill(spans[0].Tokens)
	} else {
		lg, err = d.st.PrefillMixed(spans...)
	}
	if err != nil {
		return nil, err
	}
	d.input += SpanPositions(spans, d.m.Cfg.NEmbd)
	if h != nil && h.Picked {
		d.picked++
		return d.labelScoresOf(h.PickVals, groups, v)
	}
	if d.st.HostPicked {
		d.picked++
		return d.labelScoresOf(d.st.HostPickVals, groups, v)
	}
	vals := d.pickVals[:0]
	for _, id := range flat {
		if int(id) >= len(lg) {
			return nil, fmt.Errorf("label id %d is past the %d logits", id, len(lg))
		}
		vals = append(vals, lg[id])
	}
	d.pickVals = vals
	return d.labelScoresOf(vals, groups, v)
}

// variantScores is variant v's score scratch, n wide.
func (d *Decider) variantScores(v, n int) []float32 {
	for len(d.vscores) <= v {
		d.vscores = append(d.vscores, nil)
	}
	if cap(d.vscores[v]) < n {
		d.vscores[v] = make([]float32, n)
	}
	return d.vscores[v][:n]
}

// prompt is variant v of q as a decoder readout asks it: the prompt's spans
// (its ids alone without pictures), and the label ids each output reads, a
// group per option (or per rating).
func (d *Decider) prompt(state jinja.Value, q *DecisionQuestion, v int) ([]int32, [][]int32, error) {
	text, groups, err := d.promptText(state, q, v)
	if err != nil {
		return nil, nil, err
	}
	ids := d.m.Vocab.EncodeSpecial(text, false)
	if len(ids) == 0 {
		return nil, nil, fmt.Errorf("the prompt is empty")
	}
	if len(ids) > d.st.MaxSeq() {
		return nil, nil, fmt.Errorf("the prompt is %d tokens and this decider holds %d", len(ids), d.st.MaxSeq())
	}
	return ids, groups, nil
}

// promptSpans is prompt for a request with pictures: the text between the
// template's markers as tokens, each picture's run where its marker stood.
func (d *Decider) promptSpans(state jinja.Value, q *DecisionQuestion, v int) ([]Span, [][]int32, error) {
	text, groups, err := d.promptText(state, q, v)
	if err != nil {
		return nil, nil, err
	}
	spans, err := d.pictureSpans(text, d.pics)
	if err != nil {
		return nil, nil, err
	}
	if n := SpanPositions(spans, d.m.Cfg.NEmbd); n > d.st.MaxSeq() {
		return nil, nil, fmt.Errorf("the prompt is %d positions and this decider holds %d", n, d.st.MaxSeq())
	}
	return spans, groups, nil
}

// promptText is variant v of q rendered, and the label ids each output
// reads.
func (d *Decider) promptText(state jinja.Value, q *DecisionQuestion, v int) (string, [][]int32, error) {
	opts := d.order(q)
	var groups [][]int32
	var labels []string
	switch d.kind {
	case jlm.DecisionLev:
		labels = d.labelTexts
		n := len(opts)
		if q.Type == jlm.QuestionNoul {
			n = levRatings
		}
		for _, id := range d.labels[:n] {
			groups = append(groups, []int32{id})
		}
	case jlm.DecisionPplxDecider:
		labels = d.labelTexts
		ids := d.labels[:len(opts)]
		if d.violation == "codes-shifted" {
			ids = d.labels[1 : len(opts)+1] // a gate's: each option read at the next code
		}
		for _, id := range ids {
			groups = append(groups, []int32{id})
		}
	case jlm.DecisionOpenJev:
		// The template writes the letters itself.
		for _, id := range d.labels[:len(opts)] {
			groups = append(groups, []int32{id})
		}
	case jlm.DecisionLFM2D1:
		var err error
		if labels, groups, err = d.d1Labels(q.Type, opts); err != nil {
			return "", nil, err
		}
	}
	prompt, err := d.render(state, q, opts, labels, v)
	if err != nil {
		return "", nil, err
	}
	return prompt, groups, nil
}

// labelScores is one score per label group of the prompt's last logits lg.
func (d *Decider) labelScores(lg []float32, groups [][]int32) ([]float32, error) {
	var vals []float32
	for _, g := range groups {
		for _, id := range g {
			if int(id) >= len(lg) {
				return nil, fmt.Errorf("label id %d is past the %d logits", id, len(lg))
			}
			vals = append(vals, lg[id])
		}
	}
	s, err := d.labelScoresOf(vals, groups, 0)
	return slices.Clone(s), err
}

// labelScoresOf is one score per label group from the groups' logits, vals,
// in the groups' order: a group scores by its largest logit (d1's code and
// " "+code), which the generated argmax picks.
func (d *Decider) labelScoresOf(vals []float32, groups [][]int32, v int) ([]float32, error) {
	s := d.variantScores(v, len(groups))
	at := 0
	for i, g := range groups {
		gv := vals[at : at+len(g)]
		at += len(g)
		best, ok := nn.Argmax32JIT(gv)
		if !ok {
			return nil, fmt.Errorf("model: no generated argmax on this host")
		}
		s[i] = gv[best]
	}
	return s, nil
}

// render is the systemone template over one question, given the inputs
// llama.cpp's server gives it (server-decision.cpp: render): id, type,
// instructions, state and the options, each with its key, description and
// label, and an empty image list.
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
	// Instructions are optional in TypeSafe's protocol, and the reference
	// models that read an absent one ask by the question's id (Clef's
	// encode_record, lev's template); llama.cpp refuses the request instead.
	ins := q.Instructions
	if ins.IsUndefined() || ins.IsNone() {
		ins = jinja.NewString(q.ID)
	}
	in := jinja.NewDict()
	dict := in.AsDict()
	dict.Set("id", jinja.NewString(q.ID))
	dict.Set("type", jinja.NewString(q.Type.String()))
	dict.Set("instructions", ins)
	dict.Set("state", state)
	dict.Set("options", jinja.NewList(list))
	if d.kind == jlm.DecisionLev && d.violation != "unsorted" {
		// lev was trained with every object's keys sorted.
		in = sortKeys(in)
		dict = in.AsDict()
	}
	// One marker per picture, which prompt turns into the picture's run.
	marks := make([]jinja.Value, len(d.pics))
	for i := range marks {
		marks[i] = jinja.NewString(decisionMarker)
	}
	data := map[string]any{"images": jinja.NewList(marks)}
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

// d1Labels are d1's option codes and the token group each is scored by
// (prompt.py of the model repo, as llama.cpp's server-decision.cpp ports it):
// a noul reads yes/Yes/YES against no/No/NO, a score its level digits, and a
// choice its one-letter keys when every key is one letter, else A, B, ... or
// 00..99, each code taken only when it is one token not used before, and
// otherwise the next free one from a pool.
func (d *Decider) d1Labels(t jlm.QuestionType, opts []DecisionOption) ([]string, [][]int32, error) {
	singles := func(forms ...string) []int32 {
		var out []int32
		for _, f := range forms {
			if id, ok := d.single(f); ok && !containsID(out, id) {
				out = append(out, id)
			}
		}
		return out
	}
	var texts []string
	var groups [][]int32
	if t != jlm.QuestionChoice {
		for _, o := range opts {
			var g []int32
			switch {
			case t == jlm.QuestionScore:
				g = singles(o.Key)
			case o.Key == "true":
				g = singles("yes", "Yes", "YES")
			default:
				g = singles("no", "No", "NO")
			}
			if len(g) == 0 {
				return nil, nil, fmt.Errorf("decision label %q is not a single token", o.Key)
			}
			texts = append(texts, o.Key)
			groups = append(groups, g)
		}
		return texts, groups, nil
	}
	letters := true
	for _, o := range opts {
		r := []rune(o.Key)
		letters = letters && len(o.Key) == 1 && len(r) == 1 && (r[0] >= 'a' && r[0] <= 'z' || r[0] >= 'A' && r[0] <= 'Z')
	}
	n := len(opts)
	codes := make([]string, n)
	for i := range opts {
		switch {
		case letters:
			codes[i] = opts[i].Key
		case n <= 26:
			codes[i] = string(rune('A' + i))
		default:
			codes[i] = fmt.Sprintf("%02d", i)
		}
	}
	var pool []string
	for c := 'A'; c <= 'Z'; c++ {
		pool = append(pool, string(c))
	}
	for i := 0; i < 100; i++ {
		pool = append(pool, fmt.Sprintf("%02d", i))
	}
	for c := 'a'; c <= 'z'; c++ {
		pool = append(pool, string(c))
	}
	for i := 0; i < 200; i++ {
		pool = append(pool, fmt.Sprintf("#%d", i))
	}
	for a := 'A'; a <= 'Z'; a++ {
		for b := 'A'; b <= 'Z'; b++ {
			pool = append(pool, string([]rune{a, b}))
		}
	}
	var used []int32
	take := func(code string) bool {
		id, ok := d.single(code)
		if !ok || containsID(used, id) {
			return false
		}
		used = append(used, id)
		g := []int32{id}
		for _, x := range singles(" " + code) {
			if d.violation == "code-only" {
				break
			}
			if x != id {
				g = append(g, x)
			}
		}
		texts = append(texts, code)
		groups = append(groups, g)
		return true
	}
	for _, code := range codes {
		taken := take(code)
		for i := 0; !taken && i < len(pool); i++ {
			taken = take(pool[i])
		}
		if !taken {
			return nil, nil, fmt.Errorf("no single-token label left for %d options", n)
		}
	}
	return texts, groups, nil
}

func containsID(s []int32, id int32) bool {
	for _, x := range s {
		if x == id {
			return true
		}
	}
	return false
}

// answer calibrates and averages a question's variants and forms the answer
// (server-decision.cpp: format_answer, which is TypeSafe's arithmetic). Every
// operation over the option scores is generated code: the scale and softmax,
// the variants' mean (axpy), the expected level and the distance to the mode
// (f32 matvecs against a ramp and a distance row, tables of whole numbers),
// the mode (argmax) and Laya's entropy (log-softmax and a matvec). What is
// left in Go is a handful of scalars per question.
func (d *Decider) answer(q *DecisionQuestion, variants [][]float32) (DecisionAnswer, error) {
	shown := d.order(q)
	n := len(variants[0])
	t := d.cfg.Temperature(q.Type, len(q.Options))
	if d.kind == jlm.DecisionD1Omni && len(d.omniPre) > 0 {
		t = 1 // the reference calibrates text alone (calibrate, modeling_d1.py)
	}
	if d.violation == "uncalibrated" {
		t = 1
	}
	d.growAnswer(n)
	acc, p, row := d.acc[:n], d.prob[:n], d.row[:n]
	clear(acc)
	for v, s := range variants {
		if len(s) != n {
			return DecisionAnswer{}, fmt.Errorf("variant %d has %d scores, want %d", v, len(s), n)
		}
		for _, x := range s {
			if x != x {
				return DecisionAnswer{}, fmt.Errorf("the model could not evaluate the decision (a score is NaN)")
			}
		}
		// The second variant shows the options reversed: its scores go back
		// to the request's order before they are averaged in.
		if v == 1 && d.violation != "unreversed" {
			for i := range p {
				p[i] = s[n-1-i]
			}
		} else {
			copy(p, s)
		}
		nn.Scale32JIT(p, 1/t)
		if d.kind == jlm.DecisionLaya {
			copy(row, p) // the scaled scores, for the entropy below
		}
		nn.Softmax32JIT(p, n)
		w := 1 / float32(len(variants))
		if d.violation == "summed" {
			w = 1
		}
		nn.Axpy32JIT(acc, p, w)
	}
	a := DecisionAnswer{Type: q.Type}
	if q.Type == jlm.QuestionNoul {
		if d.kind == jlm.DecisionLev {
			// The expected rating over 0..n-1, as a probability of yes.
			e, err := d.dot(acc, d.ramp[:n])
			if err != nil {
				return a, err
			}
			a.Noul = float64(e) / float64(n-1)
			return a, nil
		}
		for i, o := range shown {
			if o.Key == "true" {
				a.Noul = float64(acc[i])
			}
		}
		return a, nil
	}
	// Choice and score options were shown in the request's order.
	a.Probs = make([]float64, n)
	for i, x := range acc {
		a.Probs[i] = float64(x)
	}
	best, ok := nn.Argmax32JIT(acc)
	if !ok {
		return a, fmt.Errorf("model: no generated argmax on this host")
	}
	top := float64(acc[best])
	if q.Type == jlm.QuestionChoice {
		a.Choice = q.Options[best].Key
		a.Confidence = confidenceChoice(top, n)
	} else {
		e, err := d.dot(acc, d.ramp[:n])
		if err != nil {
			return a, err
		}
		a.Score = float64(e)
		// The mean distance to the mode, against a uniform distribution's
		// mean distance to its centre: sum |i - (n-1)/2| / n, which is n/4
		// for an even count and (n*n-1)/(4n) for an odd one.
		// p is free once the variants are in acc; row keeps Laya's scores.
		for i := range p {
			p[i] = float32(abs(i - int(best)))
			if d.violation == "from-zero" {
				p[i] = float32(i)
			}
		}
		dist, err := d.dot(acc, p)
		if err != nil {
			return a, err
		}
		uni := float64(n) / 4
		if n%2 == 1 {
			uni = float64(n*n-1) / float64(4*n)
		}
		a.Confidence = confidenceScore(float64(dist), uni, n)
	}
	if (d.kind == jlm.DecisionClef || d.kind == jlm.DecisionD1Omni) && d.violation != "typesafe-confidence" {
		// Clef's own (systemone_answer): the chosen option's probability, and
		// a score's largest; llama.cpp gives TypeSafe's formula.
		a.Confidence = top
	}
	if d.kind == jlm.DecisionLaya {
		// One minus the entropy over log n: -sum p log p, with log p the
		// log-softmax of the same scaled scores (one variant on Laya).
		nn.LogSoftmax32JIT(row, n)
		h, err := d.dot(acc, row)
		if err != nil {
			return a, err
		}
		a.Confidence = layaConfidence(-float64(h), n)
	}
	return a, nil
}

func abs(i int) int {
	if i < 0 {
		return -i
	}
	return i
}

// growAnswer sizes the answer's scratch for n options, and the ramp 0..n-1
// a level's expectation is read against.
func (d *Decider) growAnswer(n int) {
	if len(d.acc) >= n {
		return
	}
	d.acc, d.prob, d.row = make([]float32, n), make([]float32, n), make([]float32, n)
	d.ramp = make([]float32, n)
	for i := range d.ramp {
		d.ramp[i] = float32(i)
	}
}

// dot is x . w over len(x) on the generated f32 matvec, w a one-row weight.
func (d *Decider) dot(x, w []float32) (float32, error) {
	j := d.jit()
	var out [1]float32
	b := unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), 4*len(w))
	if !j.MatVecHost(out[:], quant.F32, b, x, 1, len(x)) {
		return 0, fmt.Errorf("model: no generated f32 matvec of width %d on this host", len(x))
	}
	return out[0], nil
}

// jit is the code generator the Decider's model runs on.
func (d *Decider) jit() *nn.JIT {
	if d.emb != nil {
		return d.emb.jit
	}
	return d.st.jit
}

// confidenceChoice is TypeSafe's choice confidence: how far the top
// probability stands above uniform, on [0, 1].
func confidenceChoice(top float64, n int) float64 {
	if n < 2 {
		return 1
	}
	u := 1 / float64(n)
	return max(0, (top-u)/(1-u))
}

// confidenceScore is TypeSafe's score confidence: one minus the mean distance
// to the mode, relative to that of a uniform distribution about its centre.
func confidenceScore(dist, uni float64, n int) float64 {
	if n < 2 {
		return 1
	}
	return max(0, 1-dist/uni)
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
			if hasCrit && crit.IsList() {
				// TypeSafe clients send a list of names for undescribed
				// options (lev and laya accept it).
				for _, k := range crit.AsList().Items {
					if !k.IsString() {
						return nil, fmt.Errorf("questions.%s: a criteria list holds option names", id)
					}
					q.Options = append(q.Options, DecisionOption{Key: k.AsString(), Description: jinja.None()})
				}
			} else if hasCrit && crit.IsDict() {
				cd := crit.AsDict()
				for _, k := range cd.Keys {
					q.Options = append(q.Options, DecisionOption{Key: k, Description: cd.Data[k]})
				}
			} else {
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

// DecisionResponse is TypeSafe's response body: the answers keyed by question id in request order,
// each in TypeSafe's shape. A score's legend carries each level's description
// as the request gave it.
func DecisionResponse(modelID string, qs []DecisionQuestion, answers []DecisionAnswer, inputTokens int) []byte {
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
				b.WriteString(o.Description.JSON())
			}
			b.WriteByte('}')
			writeProbs(&b, q, a)
		}
		b.WriteByte('}')
	}
	b.WriteString(`},"usage":{"input_tokens":`)
	b.WriteString(strconv.Itoa(inputTokens))
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

// DecisionBody is a /v1/systemone body as ParseDecisionRequest reads it.
type DecisionBody struct {
	// Model is the model it names, "" when none.
	Model     string
	State     jinja.Value
	Questions []DecisionQuestion
	// Images are the pictures' data URLs in request order (DecisionMedia,
	// which takes them out of the state); Videos the videos'
	// (DecisionVideos).
	Images []string
	Videos []VideoURLs
	// Audio is the voice clips' data URLs (data:audio/...): at most one, and
	// never beside a picture (d1-omni's).
	Audio []string
}

// ParseDecisionRequest reads a /v1/systemone body, every object's keys in the
// order the body wrote them.
func ParseDecisionRequest(body []byte) (DecisionBody, error) {
	var out DecisionBody
	var err error
	out.Model, out.State, out.Questions, out.Images, err = parseDecisionRequest(body)
	if err != nil {
		return out, err
	}
	// A voice clip rides "files" beside the pictures (llama.cpp's server
	// tells them apart by content; here by the URL's media type).
	pics := out.Images[:0]
	for _, u := range out.Images {
		if strings.HasPrefix(u, "data:audio/") {
			out.Audio = append(out.Audio, u)
		} else {
			pics = append(pics, u)
		}
	}
	out.Images = pics
	if len(out.Audio) > 1 || len(out.Audio) == 1 && len(out.Images) > 0 {
		return out, fmt.Errorf("a request has images or one audio clip, not both")
	}
	v, _ := jinja.FromJSON(body)
	out.Videos, err = DecisionVideos(v.AsDict())
	return out, err
}

func parseDecisionRequest(body []byte) (string, jinja.Value, []DecisionQuestion, []string, error) {
	v, err := jinja.FromJSON(body)
	if err != nil {
		return "", jinja.None(), nil, nil, fmt.Errorf("the request is not JSON: %w", err)
	}
	if !v.IsDict() {
		return "", jinja.None(), nil, nil, fmt.Errorf("the request must be a JSON object")
	}
	req := v.AsDict()
	name := ""
	if mv, ok := req.Get("model"); ok && mv.IsString() {
		name = mv.AsString()
	}
	state, ok := req.Get("state")
	if !ok {
		return "", jinja.None(), nil, nil, fmt.Errorf("state must be provided")
	}
	qv, _ := req.Get("questions")
	qs, err := ParseDecisionQuestions(qv)
	if err != nil {
		return "", jinja.None(), nil, nil, err
	}
	state, urls, err := DecisionMedia(req, state)
	if err != nil {
		return "", jinja.None(), nil, nil, err
	}
	return name, state, qs, urls, nil
}

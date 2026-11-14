package server

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/grammar"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/tok"
)

// Structured output: a generate constrained to a grammar (GBNF, or a JSON
// Schema compiled to it). Each step the tokens the grammar does not allow go
// to -inf in a copy of the logits before the sampler reads them, and the
// sampled token advances the grammar. A grammar is compiled, and its states'
// allowed sets kept, once per (model, grammar text): JSON revisits its states,
// so after the first few fields a step is a lookup.

// grammars is a model's compiled grammars, by their text. It is bounded: a
// server fed endless distinct schemas drops them all and starts again rather
// than growing.
type grammars struct {
	mu sync.Mutex
	m  map[string]*grammar.Matcher
	// toks is the model's tokenizer indexed for the masks, built on the
	// first grammar and shared by every one.
	toks *grammar.Tokens
	// toolToks is toks with the model's tool-call markers (toolMatcher).
	toolToks *grammar.Tokens
	// toolForced counts the generates a forced tool call constrained.
	toolForced atomic.Int64
}

// maxGrammars is how many compiled grammars a model keeps.
const maxGrammars = 64

func (lm *LoadedModel) matcher(src string) (*grammar.Matcher, error) {
	g := &lm.grammars
	g.mu.Lock()
	defer g.mu.Unlock()
	if m, ok := g.m[src]; ok {
		return m, nil
	}
	gr, err := grammar.Parse(src)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	if g.m == nil || len(g.m) >= maxGrammars {
		g.m = map[string]*grammar.Matcher{}
	}
	if g.toks == nil {
		g.toks = grammar.IndexTokens(lm.m.Vocab)
	}
	m := grammar.NewMatcher(gr, g.toks)
	g.m[src] = m
	return m, nil
}

// constraint is one generate's place in its grammar.
type constraint struct {
	m  *grammar.Matcher
	st grammar.State
	// lg is the masked copy of the step's logits, bias the row it adds: 0
	// at the tokens the grammar allows while a mask is taken, -inf elsewhere
	// and between.
	lg, bias []float32
	// on is ignore_eos: the reply does not end while the grammar lets it go
	// on (grammar.Matcher.AllowOn).
	on bool
	// ub is the request's logit bias over the vocabulary (logitBias), nil
	// for none, and biased the logits plus it, which the grammar's mask then
	// reads.
	ub, biased []float32
}

// mask is logits plus the request's logit bias, with every token the
// grammar does not allow at -inf, in the constraint's own copies, through
// in's composed mask. m is nil for a bias alone.
func (c *constraint) mask(in *engine.Instance, logits []float32) ([]float32, error) {
	if c.ub != nil {
		if err := in.MaskRow(c.biased, logits, c.ub); err != nil {
			return nil, err
		}
		logits = c.biased
	}
	if c.m == nil {
		return logits, nil
	}
	n := len(logits)
	if cap(c.lg) < n {
		c.lg, c.bias = make([]float32, n), make([]float32, n)
		for i := range c.bias {
			c.bias[i] = float32(math.Inf(-1))
		}
	}
	c.lg, c.bias = c.lg[:n], c.bias[:n]
	if c.on {
		c.m.AllowOn(c.st, c.bias, 0)
	} else {
		c.m.Allow(c.st, c.bias, 0)
	}
	err := in.MaskRow(c.lg, logits, c.bias)
	if c.on {
		c.m.AllowOn(c.st, c.bias, float32(math.Inf(-1)))
	} else {
		c.m.Allow(c.st, c.bias, float32(math.Inf(-1)))
	}
	return c.lg, err
}

// specConstraint is a generate's constraint as its Speculator reads it
// (engine.SpecConstraint): states[i] the grammar's state after the round's
// first i pushed tokens, states[0] the committed one, which the generate's
// constraint holds once the round commits.
type specConstraint struct {
	c      *constraint
	in     *engine.Instance
	states []grammar.State
}

// spec is c as a Speculator on in reads it.
func (c *constraint) spec(in *engine.Instance) *specConstraint {
	return &specConstraint{c: c, in: in, states: []grammar.State{c.st}}
}

// Mask is the logits the constraint allows after the round's first depth
// pushes, through in's composed mask.
func (s *specConstraint) Mask(depth int, logits []float32) ([]float32, error) {
	st := s.c.st
	s.c.st = s.states[depth]
	lg, err := s.c.mask(s.in, logits)
	s.c.st = st
	return lg, err
}

// Push advances the round past tok, false where the grammar does not allow
// it; a bias alone allows every token.
func (s *specConstraint) Push(tok int32) bool {
	at := s.states[len(s.states)-1]
	if s.c.m != nil {
		var ok bool
		if at, ok = s.c.m.Accept(at, tok); !ok {
			return false
		}
	}
	s.states = append(s.states, at)
	return true
}

// Commit keeps the round's first n pushes: the generate's constraint
// advances past them.
func (s *specConstraint) Commit(n int) {
	s.c.st = s.states[n]
	s.states = append(s.states[:0], s.c.st)
}

// accept advances past tok; false where the grammar does not allow it.
func (c *constraint) accept(tok int32) bool {
	if c.m == nil {
		return true
	}
	st, ok := c.m.Accept(c.st, tok)
	if ok {
		c.st = st
	}
	return ok
}

// oaResponseFormat is OpenAI's response_format as a grammar's text: none for
// "text" or an absent field, any JSON object for "json_object", and the
// schema's grammar for "json_schema". A schema the converter does not build
// is an ErrInvalid naming what.
func oaResponseFormat(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var rf struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
			Strict *bool           `json:"strict"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(raw, &rf); err != nil {
		return "", fmt.Errorf("%w: response_format: %v", ErrInvalid, err)
	}
	switch rf.Type {
	case "text":
		return "", nil
	case "json_object":
		return grammar.JSONObject(), nil
	case "json_schema":
		if rf.JSONSchema == nil || len(rf.JSONSchema.Schema) == 0 {
			return "", fmt.Errorf("%w: response_format json_schema needs json_schema.schema", ErrInvalid)
		}
		src, err := grammar.FromJSONSchema(rf.JSONSchema.Schema)
		if err != nil {
			return "", fmt.Errorf("%w: response_format: %v", ErrInvalid, err)
		}
		return src, nil
	}
	return "", fmt.Errorf("%w: response_format type %q: text, json_object or json_schema", ErrInvalid, rf.Type)
}

// toolStream reads a generate's tool calls, when its chat declares tools: in
// the form the model's container states, or refused where it states none.
func (lm *LoadedModel) toolStream(o GenerateOptions) (*model.ToolStream, error) {
	ch := o.Prompt.Chat
	if o.Prompt.Kind != PromptChat && o.Prompt.Kind != PromptSpans || ch == nil || len(ch.Tools) == 0 || lm.m.Vocab == nil {
		return nil, nil
	}
	ts, err := lm.m.NewToolStream(model.ParseToolSet(ch.Tools))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	return ts, nil
}

// logitBias is a request's logit bias as a row over the model's logits:
// each entry's bias at its token, or at every token its text encodes to
// without specials, summed where entries meet, as llama.cpp's sampler adds
// them one after another. A token outside the vocabulary is refused.
func (lm *LoadedModel) logitBias(bs []TokenBias) ([]float32, error) {
	n := lm.m.Cfg.NVocab
	row := make([]float32, n)
	for _, b := range bs {
		ids := []int32{b.ID}
		if b.Text != "" {
			ids = lm.m.Vocab.Encode(b.Text, false)
		}
		for _, id := range ids {
			if id < 0 || int(id) >= n {
				return nil, fmt.Errorf("%w: logit_bias token id %d is outside the vocabulary of %d", ErrInvalid, id, n)
			}
			row[id] += b.Bias
		}
	}
	return row, nil
}

// toolConstraint is the constraint a chat whose tool_choice forces a call
// generates under (model.ToolGrammar): the model's own tool-call syntax
// around the called tool's schema, through the same matcher and mask as
// structured output. nil when the choice forces nothing.
func (lm *LoadedModel) toolConstraint(o GenerateOptions, ids []int32) (*constraint, error) {
	ch := o.Prompt.Chat
	if o.Prompt.Kind != PromptChat && o.Prompt.Kind != PromptSpans || ch == nil || len(ch.Tools) == 0 {
		return nil, nil
	}
	if !ch.ToolChoice.Forces() {
		// A model whose template reads no tools has every reply held to
		// the generic form's object; a caller's own grammar replaces it.
		if syn, err := lm.m.ToolSyntax(); err != nil || o.Grammar != "" || !syn.ConstrainsReply(ch.ToolChoice) {
			return nil, nil
		}
	}
	if o.Grammar != "" {
		return nil, fmt.Errorf("%w: a forced tool call and response_format are one constraint each; give one",
			ErrInvalid)
	}
	// The prompt's end, as raw text, says whether the reply starts inside
	// a reasoning block.
	var tail strings.Builder
	for _, id := range ids[max(0, len(ids)-32):] {
		tail.WriteString(lm.m.Vocab.Literal(id))
	}
	src, err := lm.m.ToolGrammar(model.ParseToolSet(ch.Tools), ch.ToolChoice, tail.String())
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	m, err := lm.toolMatcher(src)
	if err != nil {
		return nil, err
	}
	lm.grammars.toolForced.Add(1)
	return &constraint{m: m, st: m.Start()}, nil
}

// toolMatcher is matcher over the tool tokens: the model's tokenizer with
// its syntax's control-token markers offered by their literal text, which
// a structured-output grammar never sees (a JSON string must not be able to
// spell <|call|>).
func (lm *LoadedModel) toolMatcher(src string) (*grammar.Matcher, error) {
	g := &lm.grammars
	g.mu.Lock()
	defer g.mu.Unlock()
	key := "tools\x00" + src
	if m, ok := g.m[key]; ok {
		return m, nil
	}
	gr, err := grammar.Parse(src)
	if err != nil {
		return nil, fmt.Errorf("server: a tool-call grammar does not parse: %v", err)
	}
	if g.m == nil || len(g.m) >= maxGrammars {
		g.m = map[string]*grammar.Matcher{}
	}
	if g.toolToks == nil {
		syn, err := lm.m.ToolSyntax()
		if err != nil {
			return nil, err
		}
		g.toolToks = grammar.IndexTokens(markedVocab{lm.m.Vocab, markers(lm.m.Vocab, syn)})
	}
	m := grammar.NewMatcher(gr, g.toolToks)
	g.m[key] = m
	return m, nil
}

// markedVocab is a tokenizer whose marker control tokens have a piece: their
// literal text.
type markedVocab struct {
	v     *tok.Vocab
	marks map[int32]string
}

func (x markedVocab) Size() int { return x.v.Size() }

func (x markedVocab) Piece(id int32) (string, bool) {
	if s, ok := x.marks[id]; ok {
		return s, true
	}
	return x.v.Piece(id)
}

func (x markedVocab) IsEOG(id int32) bool { return x.v.IsEOG(id) }

// markers are syn's marker texts that v holds as single tokens with no
// piece of their own.
func markers(v *tok.Vocab, syn model.ToolSyntax) map[int32]string {
	out := map[int32]string{}
	for _, s := range syn.ToolMarkers() {
		id, ok := v.ID(s)
		if !ok || v.Literal(id) != s {
			continue
		}
		if _, plain := v.Piece(id); !plain {
			out[id] = s
		}
	}
	return out
}

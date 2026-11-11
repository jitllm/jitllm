package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"strings"
	"time"

	"github.com/jitllm/jitllm/grammar"
	"github.com/jitllm/jitllm/model"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The Connect side of the chat features the OpenAI routes have: tools,
// response_format, logprobs, n, usage. Nothing here is a second
// implementation. A request is turned into the OpenAI route's own JSON
// fragments and read by the OpenAI route's own functions (oaTools,
// oaToolChoice, oaResponseFormat, oaChoiceSeeds, logprobRequestOf,
// samplingFields.validate), so a value one surface accepts the other accepts,
// and one it refuses the other refuses in the same words; and a generate's
// events are read by the same toolText the OpenAI routes feed, so the two
// agree on what is a tool call.

// errInvalid is a request refusal as the Connect surface reports it: the
// message is the one the OpenAI surface sends, unprefixed.
func errInvalid(err error) error {
	if err == nil {
		return nil
	}
	return invalid("%s", err.Error())
}

// samplingOf is a request's sampling values as the fields both surfaces check.
func samplingOf(p *v1.SamplingParams) samplingFields {
	if p == nil {
		return samplingFields{}
	}
	f64 := func(x *float32) *float64 {
		if x == nil {
			return nil
		}
		v := float64(*x)
		return &v
	}
	q := samplingFields{
		Temperature:      f64(p.Temperature),
		TopP:             f64(p.TopP),
		MinP:             f64(p.MinP),
		PresencePenalty:  f64(p.PresencePenalty),
		FrequencyPenalty: f64(p.FrequencyPenalty),
	}
	// repeat_penalty is the OpenAI surface's repetition_penalty. On this
	// surface 0 has always meant off, as 1 does, and front ends send the
	// session's zero; only a negative value is refused.
	if p.RepeatPenalty != nil && *p.RepeatPenalty != 0 {
		q.RepetitionPenalty = f64(p.RepeatPenalty)
	}
	if p.TopK != nil {
		k := int(*p.TopK)
		q.TopK = &k
	}
	return q
}

// pbToolsJSON is a request's tools in the OpenAI shape, the bytes the chat
// template receives: [{"type": "function", "function": {...}}].
func pbToolsJSON(ts []*v1.Tool) (json.RawMessage, error) {
	if len(ts) == 0 {
		return nil, nil
	}
	type function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
		Strict      *bool           `json:"strict,omitempty"`
	}
	type tool struct {
		Type     string   `json:"type"`
		Function function `json:"function"`
	}
	out := make([]tool, len(ts))
	for i, t := range ts {
		f := function{Name: t.GetName(), Description: t.GetDescription(), Strict: t.Strict}
		if p := t.GetParametersJson(); p != "" {
			if !json.Valid([]byte(p)) {
				return nil, fmt.Errorf("tools[%d].parameters_json is not JSON", i)
			}
			f.Parameters = json.RawMessage(p)
		}
		out[i] = tool{Type: "function", Function: f}
	}
	return json.Marshal(out)
}

// pbToolChoiceJSON is a request's tool_choice and parallel_tool_calls as the
// OpenAI route reads them.
func pbToolChoiceJSON(c *v1.ToolChoice) (json.RawMessage, *bool) {
	if c == nil {
		return nil, nil
	}
	var parallel *bool
	if c.SingleCall {
		f := false
		parallel = &f
	}
	var v any
	switch c.Mode {
	case v1.ToolChoiceMode_TOOL_CHOICE_MODE_NONE:
		v = "none"
	case v1.ToolChoiceMode_TOOL_CHOICE_MODE_AUTO:
		v = "auto"
	case v1.ToolChoiceMode_TOOL_CHOICE_MODE_REQUIRED:
		v = "required"
	case v1.ToolChoiceMode_TOOL_CHOICE_MODE_FUNCTION:
		v = map[string]any{"type": "function", "function": map[string]any{"name": c.Name}}
	default:
		return nil, parallel
	}
	b, _ := json.Marshal(v)
	return b, parallel
}

// pbResponseFormatJSON is a request's response_format as the OpenAI route
// reads it; nil for none.
func pbResponseFormatJSON(r *v1.ResponseFormat) (json.RawMessage, error) {
	type schema struct {
		Name   string          `json:"name,omitempty"`
		Schema json.RawMessage `json:"schema,omitempty"`
		Strict *bool           `json:"strict,omitempty"`
	}
	rf := struct {
		Type       string  `json:"type"`
		JSONSchema *schema `json:"json_schema,omitempty"`
	}{}
	switch r.GetType() {
	case v1.ResponseFormat_TYPE_TEXT:
		rf.Type = "text"
	case v1.ResponseFormat_TYPE_JSON_OBJECT:
		rf.Type = "json_object"
	case v1.ResponseFormat_TYPE_JSON_SCHEMA:
		rf.Type = "json_schema"
		rf.JSONSchema = &schema{Name: r.GetName(), Strict: r.Strict}
		if s := r.GetJsonSchema(); s != "" {
			if !json.Valid([]byte(s)) {
				return nil, errors.New("response_format: json_schema is not JSON")
			}
			rf.JSONSchema.Schema = json.RawMessage(s)
		}
	default:
		return nil, nil
	}
	return json.Marshal(rf)
}

// generateOptionsWith turns the wire request into the engine's one request
// shape, the same struct the HTTP shims build.
func generateOptionsWith(ctx context.Context, m *v1.GenerateRequest, p ImagePolicy) (GenerateOptions, error) {
	p = p.withDefaults()
	o := GenerateOptions{
		SessionID:    m.SessionId,
		ModelID:      m.ModelId,
		MaxTokens:    int(m.MaxTokens),
		Sampling:     pbSampling(m.Sampling),
		Stop:         m.Stop,
		Continue:     m.ContinueSession,
		Echo:         m.EchoPrompt,
		QueueTimeout: time.Duration(m.QueueTimeoutMillis) * time.Millisecond,
		IgnoreEOS:    m.IgnoreEos,
		Speculation:  pbSpeculation(m.Speculation),
		Priority:     m.Priority,
	}
	if err := samplingOf(m.Sampling).validate(); err != nil {
		return o, errInvalid(err)
	}
	var seed *int64
	if m.Sampling != nil && m.Sampling.Seed != nil {
		v := int64(*m.Sampling.Seed)
		seed = &v
	}
	var n *int
	if m.N != 0 {
		v := int(m.N)
		n = &v
	}
	var err error
	if o.Seeds, err = oaChoiceSeeds(n, seed); err != nil {
		return o, errInvalid(err)
	}
	var top *int
	if m.TopLogprobs != nil {
		v := int(*m.TopLogprobs)
		top = &v
	}
	lpr, err := logprobRequestOf(m.Logprobs, top)
	if err != nil {
		return o, errInvalid(err)
	}
	lpr.apply(&o)

	rfJSON, err := pbResponseFormatJSON(m.ResponseFormat)
	if err != nil {
		return o, errInvalid(err)
	}
	rf, err := oaResponseFormat(rfJSON)
	if err != nil {
		return o, connectErr(err)
	}
	switch {
	case m.Grammar != "" && m.JsonSchema != "":
		return o, invalid("generate: grammar and json_schema are one constraint each; give one")
	case rf != "" && (m.Grammar != "" || m.JsonSchema != ""):
		return o, invalid("generate: response_format, grammar and json_schema are one constraint each; give one")
	case rf != "":
		o.Grammar = rf
	case m.JsonSchema != "":
		g, err := grammar.FromJSONSchema([]byte(m.JsonSchema))
		if err != nil {
			return o, invalid("generate: json_schema: %v", err)
		}
		o.Grammar = g
	default:
		o.Grammar = m.Grammar
	}
	switch in := m.Prompt.GetInput().(type) {
	case *v1.PromptInput_Text:
		o.Prompt = Prompt{Kind: PromptText, Text: in.Text}
	case *v1.PromptInput_TokenIds:
		o.Prompt = Prompt{Kind: PromptIDs, IDs: in.TokenIds.GetIds()}
	case *v1.PromptInput_Chat:
		c, err := chatInput(ctx, in.Chat, p)
		if err != nil {
			return o, err
		}
		o.Prompt = Prompt{Kind: PromptChat, Chat: c}
	case nil:
		if !m.ContinueSession {
			return o, invalid("generate: prompt is required")
		}
	}
	return o, nil
}

// chatInput is a chat prompt as the engine takes it: its messages, pictures
// and tools.
func chatInput(ctx context.Context, in *v1.ChatPrompt, p ImagePolicy) (*ChatInput, error) {
	c := &ChatInput{
		AddGenerationPrompt: in.GetAddGenerationPrompt(),
		TemplateName:        in.GetTemplateName(),
	}
	if in.System != nil {
		c.System, c.HasSystem = in.GetSystem(), true
	}
	pics := newPictures(p)
	for i, msg := range in.GetMessages() {
		cm := chatMessage(msg.GetRole(), msg.GetContent())
		cm.ToolCallID, cm.Name = msg.GetToolCallId(), msg.GetName()
		for _, tc := range msg.GetToolCalls() {
			cm.ToolCalls = append(cm.ToolCalls, pbToolCall(tc))
		}
		for j, img := range msg.GetImages() {
			part := fmt.Sprintf("prompt.chat.messages[%d].images[%d]", i, j)
			err := pics.add(part, func() (image.Image, error) {
				if img.GetUrl() != "" {
					if len(img.GetData()) > 0 {
						return nil, errors.New("carries both data and url; give one")
					}
					return pics.p.imageURL(ctx, img.GetUrl())
				}
				mt := ""
				if img.GetMediaType() != "" {
					var ok bool
					if mt, ok = imageMediaType(img.GetMediaType()); !ok {
						return nil, fmt.Errorf("has media_type %q; the pictures accepted are image/png, "+
							"image/jpeg, image/webp and image/gif", img.GetMediaType())
					}
				}
				return pics.p.decodeImage(img.GetData(), mt)
			})
			if err != nil {
				return nil, invalid("generate: %v", err)
			}
			cm.Images++
		}
		c.Messages = append(c.Messages, cm)
	}
	c.Images = pics.imgs

	raw, err := pbToolsJSON(in.GetTools())
	if err != nil {
		return nil, errInvalid(err)
	}
	choice, parallel := pbToolChoiceJSON(in.GetToolChoice())
	t, err := oaTools(raw, choice)
	if err != nil {
		return nil, errInvalid(err)
	}
	if c.Tools = t; t != nil {
		if c.ToolChoice, err = oaToolChoice(choice, parallel, toolNames(t)); err != nil {
			return nil, errInvalid(err)
		}
	}
	return c, nil
}

// genChoice is one continuation's state while its events are read.
type genChoice struct {
	tt      *toolText
	started *Started
	fin     *Finished
	// raw is every token's text as the engine sent it, tool markup included;
	// content is the reply without its calls, read once it is finished.
	raw     strings.Builder
	content string
	// calls are the calls sent so far, in order.
	calls []*v1.ToolCall
}

// genRun reads a generate's events into the Connect surface's: the one
// translation Generate streams and Complete collects.
type genRun struct {
	b  Backend
	o  GenerateOptions
	cs []*genChoice
}

func newGenRun(b Backend, o GenerateOptions) *genRun {
	var tools []byte
	if o.Prompt.Chat != nil {
		tools = o.Prompt.Chat.Tools
	}
	r := &genRun{b: b, o: o, cs: make([]*genChoice, max(len(o.Seeds), 1))}
	for i := range r.cs {
		r.cs[i] = &genChoice{tt: newToolText(tools)}
	}
	return r
}

func (r *genRun) choice(ev Event) (*genChoice, int32) {
	if ev.Choice < 0 || ev.Choice >= len(r.cs) {
		return r.cs[0], 0
	}
	return r.cs[ev.Choice], int32(ev.Choice)
}

func pbToolCall(c *v1.ToolCall) model.ToolCall {
	return model.ToolCall{ID: c.GetId(), Name: c.GetName(), Arguments: c.GetArguments()}
}

// handle turns one engine event into zero or more responses.
func (r *genRun) handle(ev Event, send func(*v1.GenerateResponse) error) error {
	c, idx := r.choice(ev)
	switch ev.Kind {
	case EventStarted:
		c.started = ev.Started
		c.tt.start(ev.Started)
		out := pbEvent(ev)
		out.Choice = idx
		return send(out)
	case EventToken:
		tok := ev.Token
		c.raw.WriteString(tok.Text)
		text := tok.Text
		var calls []*v1.ToolCall
		if c.tt.on {
			if c.tt.ids && tok.ID < 0 {
				// Text the engine held and flushed: the tool reader decodes
				// the ids itself, so it has it already.
				return nil
			}
			var cs []model.ToolCall
			text, cs = c.tt.push(tok)
			calls = r.record(c, cs)
		}
		msg := &v1.GenerateToken{TokenId: tok.ID, Text: text, Index: int32(tok.Index)}
		if tok.Logprob != nil {
			msg.Logprob = pbLogprob(tok.ID, tok.Logprob)
		}
		if err := send(&v1.GenerateResponse{Choice: idx,
			Event: &v1.GenerateResponse_Token{Token: msg}}); err != nil {
			return err
		}
		return r.sendCalls(c, idx, calls, send)
	default:
		f := *ev.Finished
		c.fin = &f
		reason := pbFinish(f.Reason)
		if c.tt.on {
			// What the tool holdback kept back goes out now: the calls, or
			// the text that turned out not to be one.
			tail, cs, content := c.tt.finish(f.StopMatched)
			c.content = content
			if tail != "" {
				if err := send(&v1.GenerateResponse{Choice: idx, Event: &v1.GenerateResponse_Token{
					Token: &v1.GenerateToken{TokenId: -1, Text: tail, Index: -1}}}); err != nil {
					return err
				}
			}
			if err := r.sendCalls(c, idx, r.record(c, cs), send); err != nil {
				return err
			}
			if len(c.calls) > 0 {
				reason = v1.FinishReason_FINISH_REASON_TOOL_CALLS
			}
		}
		out := pbFinished(&f)
		out.Reason = reason
		return send(&v1.GenerateResponse{Choice: idx, Event: &v1.GenerateResponse_Finished{Finished: out}})
	}
}

// record numbers calls the reply made, minting each an id as the OpenAI
// surface does.
func (r *genRun) record(c *genChoice, calls []model.ToolCall) []*v1.ToolCall {
	var out []*v1.ToolCall
	for _, tc := range calls {
		p := &v1.ToolCall{Id: "call_" + r.b.NextID("tc"), Name: tc.Name, Arguments: tc.Arguments}
		c.calls = append(c.calls, p)
		out = append(out, p)
	}
	return out
}

func (r *genRun) sendCalls(c *genChoice, idx int32, calls []*v1.ToolCall, send func(*v1.GenerateResponse) error) error {
	if len(calls) == 0 {
		return nil
	}
	return send(&v1.GenerateResponse{Choice: idx, Event: &v1.GenerateResponse_ToolCalls{
		ToolCalls: &v1.GenerateToolCalls{Calls: calls, FirstIndex: int32(len(c.calls) - len(calls))}}})
}

// done checks every choice finished, which a Backend that returned nil
// without a Finished event did not.
func (r *genRun) done() error {
	for _, c := range r.cs {
		if c.fin == nil {
			return errors.New("the generate finished without a finished event")
		}
	}
	return nil
}

// usage counts the request as the OpenAI surface does: the prompt once, for
// every choice after the first restored it rather than ran it.
func (r *genRun) usage() *v1.Usage {
	u := &v1.Usage{}
	for i, c := range r.cs {
		if c.fin == nil {
			continue
		}
		if i == 0 {
			u.PromptTokens = int32(c.fin.PromptTokens)
			u.CachedTokens = int32(c.fin.Restored)
		}
		u.CompletionTokens += int32(c.fin.CompletionTokens)
	}
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	return u
}

func pbLogprob(id int32, l *TokenLogprob) *v1.TokenLogprob {
	out := &v1.TokenLogprob{TokenId: id, Token: l.Text, Logprob: l.Logprob}
	for _, a := range l.Top {
		out.Top = append(out.Top, &v1.TopLogprob{TokenId: a.ID, Token: a.Text, Logprob: a.Logprob})
	}
	return out
}

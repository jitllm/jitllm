package server

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/model"
)

// toolText sits between the engine's token stream and a shim's output when a
// request declares tools. Text that is not, and cannot become, a tool call
// streams as it arrives; a call goes out as soon as its markup closes. Both
// shims use it, so they agree on what is a call.
//
// The calls are read by the generate's model.ToolStream (Started.Tools) from
// the token ids, in the model's own syntax -- which may be written in control
// tokens the text drops. A backend that sends no stream (one that has only
// text) is read as text in the default syntax.
type toolText struct {
	on    bool
	tools []byte
	ts    *model.ToolStream
	ids   bool
	// calls are every call read so far.
	calls []model.ToolCall
}

func newToolText(tools []byte) *toolText {
	t := &toolText{on: len(tools) > 0, tools: tools}
	if t.on {
		t.ts = model.NewTextToolStream(model.ToolSyntaxFor(convert.DefaultToolForm()), model.ParseToolSet(tools))
	}
	return t
}

// start takes the generate's own stream, when the backend made one.
func (t *toolText) start(s *Started) {
	if t.on && s != nil && s.Tools != nil {
		t.ts, t.ids = s.Tools, true
	}
}

// push takes one token and returns the text safe to emit now and the calls
// that closed with it.
func (t *toolText) push(tok *Token) (string, []model.ToolCall) {
	if !t.on {
		return tok.Text, nil
	}
	var s string
	var calls []model.ToolCall
	switch {
	case t.ids && tok.ID < 0:
		// Text the engine held and flushed: the stream decodes the ids
		// itself, so it has it already.
		return "", nil
	case t.ids:
		s, calls = t.ts.Push(tok.ID)
	default:
		s, calls = t.ts.PushText(tok.Text)
	}
	t.calls = append(t.calls, calls...)
	return s, calls
}

// generic is whether the stream reads the generic form, whose reply is a JSON
// object: what the client is sent as content is the answer in it, and the
// reply's own text is not.
func (t *toolText) generic() bool { return t.ts != nil && t.ts.Syntax().Generic() }

// finish returns the text a stream still owes, the calls not yet returned,
// and the whole reply's text without its calls. stop is the stop string that
// ended the reply, if one did: the ids carry it and the reply does not.
func (t *toolText) finish(stop string) (tail string, calls []model.ToolCall, content string) {
	if !t.on {
		return "", nil, ""
	}
	tail, calls, content = t.ts.Finish()
	t.calls = append(t.calls, calls...)
	if stop != "" && t.ids {
		if i := strings.Index(tail, stop); i >= 0 {
			tail = tail[:i]
		}
		if i := strings.Index(content, stop); i >= 0 {
			content = strings.TrimSpace(content[:i])
		}
	}
	return tail, calls, content
}

// toolNames reads the declared function names out of an OpenAI-shaped list.
func toolNames(tools []byte) []string { return model.ParseToolSet(tools).Names() }

// oaToolChoice reads OpenAI's tool_choice and parallel_tool_calls: "none",
// "auto", "required", or {"type": "function", "function": {"name": X}}. A
// named tool must be one of names.
func oaToolChoice(raw json.RawMessage, parallel *bool, names []string) (model.ToolChoice, error) {
	var c model.ToolChoice
	c.Single = parallel != nil && !*parallel
	if len(raw) == 0 || string(raw) == "null" {
		return c, nil
	}
	var mode string
	if json.Unmarshal(raw, &mode) == nil {
		switch mode {
		case "none", "auto", "required":
			c.Mode = model.ToolChoiceMode(mode)
			return c, nil
		}
		return c, fmt.Errorf("tool_choice %q: none, auto, required, or a named function", mode)
	}
	var named struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &named); err != nil || named.Type != "function" || named.Function.Name == "" {
		return c, fmt.Errorf("tool_choice must be none, auto, required or " +
			`{"type": "function", "function": {"name": ...}}`)
	}
	c.Mode, c.Name = model.ToolChoiceFunction, named.Function.Name
	return c, declared(c, names)
}

// anToolChoice reads Anthropic's tool_choice: {"type": "auto" | "any" |
// "tool" (with name) | "none", "disable_parallel_tool_use"}.
func anToolChoice(raw json.RawMessage, names []string) (model.ToolChoice, error) {
	var c model.ToolChoice
	if len(raw) == 0 || string(raw) == "null" {
		return c, nil
	}
	var x struct {
		Type                   string `json:"type"`
		Name                   string `json:"name"`
		DisableParallelToolUse bool   `json:"disable_parallel_tool_use"`
	}
	if err := json.Unmarshal(raw, &x); err != nil {
		return c, fmt.Errorf("tool_choice: %v", err)
	}
	c.Single = x.DisableParallelToolUse
	switch x.Type {
	case "auto", "":
		c.Mode = model.ToolChoiceAuto
	case "none":
		c.Mode = model.ToolChoiceNone
	case "any":
		c.Mode = model.ToolChoiceRequired
	case "tool":
		if x.Name == "" {
			return c, fmt.Errorf(`tool_choice {"type": "tool"} needs a name`)
		}
		c.Mode, c.Name = model.ToolChoiceFunction, x.Name
		return c, declared(c, names)
	default:
		return c, fmt.Errorf("tool_choice type %q: auto, any, tool or none", x.Type)
	}
	return c, nil
}

// declared refuses a choice naming a tool the request did not declare.
func declared(c model.ToolChoice, names []string) error {
	for _, n := range names {
		if n == c.Name {
			return nil
		}
	}
	return fmt.Errorf("tool_choice names %q, which is not among the request's tools", c.Name)
}

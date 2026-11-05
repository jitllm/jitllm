package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/tok/jinja"
)

// ToolCall is one function call an assistant turn made. Arguments is the JSON
// object text, compact, which is what the OpenAI API carries as a string and
// what the Anthropic API carries as an object.
type ToolCall struct {
	ID, Name  string
	Arguments string
}

// toolFields adds a message's tool calls and tool-result identity in the shape
// HuggingFace templates read: tool_calls as [{id, type, function: {name,
// arguments}}] with arguments a MAPPING (Llama-3.1 pipes it through tojson,
// Qwen3 checks `is string` first), and a "tool" turn's tool_call_id and name.
// stringArgs hands the arguments as the JSON text the request carried instead,
// for a template that only reads that (renderChatTemplate).
func toolFields(mm map[string]any, x ChatMessage, stringArgs bool) error {
	if x.ToolCallID != "" {
		mm["tool_call_id"] = x.ToolCallID
	}
	if x.Name != "" {
		mm["name"] = x.Name
	}
	if len(x.ToolCalls) == 0 {
		return nil
	}
	calls := make([]any, len(x.ToolCalls))
	for i, c := range x.ToolCalls {
		var args any = c.Arguments
		if a := strings.TrimSpace(c.Arguments); a == "" {
			args = map[string]any{}
			if stringArgs {
				args = "{}"
			}
		} else if stringArgs {
			args = c.Arguments
		} else if v, err := jinja.FromJSON([]byte(a)); err == nil && v.IsDict() {
			args = v
		} else if err != nil {
			return fmt.Errorf("tool call %q: arguments are not JSON: %w", c.Name, err)
		}
		calls[i] = map[string]any{
			"id":       c.ID,
			"type":     "function",
			"function": map[string]any{"name": c.Name, "arguments": args},
		}
	}
	mm["tool_calls"] = calls
	return nil
}

// ParseToolCalls reads the tool calls out of a completion in the default
// syntax (ToolSyntaxHermes: Hermes <tool_call> blocks, or a reply that opens
// with a bare JSON call as Llama 3.x and Mistral v0.3 write one), returning
// the text that is not part of any call and the calls. names are the
// declared tools; a call to anything else is not a call and stays text. nil
// names accepts any. A model's own syntax is ToolSyntax.Parse.
//
// The declared-name check is what keeps a model asked to answer IN JSON from
// having its answer taken for a call.
func ParseToolCalls(text string, names []string) (string, []ToolCall) {
	return ToolSyntaxHermes.Parse(text, ToolSet{names: names, any: names == nil})
}

// decodeCall reads {"name", "arguments"|"parameters"}; arguments may be an
// object or a string holding one (Qwen emits both).
func decodeCall(b []byte) (ToolCall, bool) {
	var c struct {
		Name       string          `json:"name"`
		Arguments  json.RawMessage `json:"arguments"`
		Parameters json.RawMessage `json:"parameters"`
	}
	if json.Unmarshal(b, &c) != nil || c.Name == "" {
		return ToolCall{}, false
	}
	a := c.Arguments
	if len(a) == 0 {
		a = c.Parameters
	}
	var s string
	if json.Unmarshal(a, &s) == nil {
		a = json.RawMessage(s)
	}
	if len(bytes.TrimSpace(a)) == 0 {
		a = json.RawMessage("{}")
	}
	var buf bytes.Buffer
	if json.Compact(&buf, a) != nil || buf.Bytes()[0] != '{' {
		return ToolCall{}, false
	}
	return ToolCall{Name: c.Name, Arguments: buf.String()}, true
}

// ToolCallHold is ToolSyntaxHermes.Hold: where a stream must stop emitting
// text because a tool call has begun, or may be beginning, at that byte of
// text; -1 when nothing needs holding.
func ToolCallHold(text string) int { return ToolSyntaxHermes.Hold(text) }

package model

import (
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

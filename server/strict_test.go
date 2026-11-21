package server

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// TestStrictIsOpenAIsStrict: a strict response_format, or a strict function
// tool's parameters, is held to OpenAI's strict mode -- every object closed
// (additionalProperties false) and requiring every property it lists, at
// every depth -- and refused as the caller's input naming where; the same
// schema sent unstrict, and a strict one that closes every object, compile.
func TestStrictIsOpenAIsStrict(t *testing.T) {
	closed := `{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"array","items":{"type":"object",` +
		`"properties":{"c":{"type":"string"}},"required":["c"],"additionalProperties":false}}},"required":["a","b"],` +
		`"additionalProperties":false}`
	for _, c := range []struct {
		name, schema, where string
	}{
		{"closed", closed, ""},
		{"no additionalProperties", `{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}`, "the schema states no additionalProperties"},
		{"additionalProperties true", `{"type":"object","properties":{"a":{}},"required":["a"],"additionalProperties":true}`, "the schema states no additionalProperties"},
		{"an optional property", `{"type":"object","properties":{"a":{},"b":{}},"required":["a"],"additionalProperties":false}`, `does not require "b"`},
		{"an open item", strings.Replace(closed, `"required":["c"],"additionalProperties":false`, `"required":["c"]`, 1), "the schema.properties.b.items states"},
		{"an open definition", `{"$defs":{"p":{"type":"object","properties":{"x":{}},"required":["x"]}},"$ref":"#/$defs/p"}`, "the schema.$defs.p states"},
		{"an open branch", `{"anyOf":[{"type":"null"},{"type":["object","null"],"properties":{}}]}`, "the schema.anyOf[1] states"},
	} {
		rf := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":` + c.schema + `}}`)
		_, err := oaResponseFormat(rf)
		if c.where == "" {
			if err != nil {
				t.Errorf("%s: a strict schema OpenAI accepts was refused: %v", c.name, err)
			}
			continue
		}
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), c.where) {
			t.Errorf("%s: %v, want the caller's input naming %q", c.name, err, c.where)
		}
		loose := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"x","schema":` + c.schema + `}}`)
		if _, err := oaResponseFormat(loose); err != nil {
			t.Errorf("%s: the schema sent unstrict was refused: %v", c.name, err)
		}
		tool := `[{"type":"function","function":{"name":"f","strict":true,"parameters":` + c.schema + `}}]`
		if _, err := oaTools(json.RawMessage(tool), nil); err == nil || !strings.Contains(err.Error(), c.where) {
			t.Errorf("%s: a strict tool's parameters: %v, want a refusal naming %q", c.name, err, c.where)
		}
	}
	if _, err := oaTools(json.RawMessage(`[{"type":"function","function":{"name":"f","strict":true}}]`), nil); err == nil {
		t.Error("a strict tool stating no parameters was taken")
	}
	ok := `[{"type":"function","function":{"name":"f","strict":true,"parameters":` + closed + `}}]`
	if _, err := oaTools(json.RawMessage(ok), nil); err != nil {
		t.Errorf("a strict tool whose parameters close every object was refused: %v", err)
	}
}

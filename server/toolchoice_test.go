package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/schemacheck"
	"github.com/jitllm/jitllm/model"
)

// tool_choice: "required" and a named tool hold the reply to a call in the
// model's own syntax. The shims read the choice into ChatInput; the engine
// turns a forcing one into a constraint over the model's tool grammar.

// TestToolChoiceReachesTheEngine: each API's spelling of each choice arrives
// as the one model.ToolChoice, and a name the request did not declare is a
// 400.
func TestToolChoiceReachesTheEngine(t *testing.T) {
	for _, c := range []struct {
		path, body string
		want       model.ToolChoice
	}{
		{"/v1/chat/completions", strings.Replace(oaToolReq, "%s", `,"tool_choice":"required"`, 1),
			model.ToolChoice{Mode: model.ToolChoiceRequired}},
		{"/v1/chat/completions", strings.Replace(oaToolReq, "%s",
			`,"tool_choice":{"type":"function","function":{"name":"get_weather"}},"parallel_tool_calls":false`, 1),
			model.ToolChoice{Mode: model.ToolChoiceFunction, Name: "get_weather", Single: true}},
		{"/v1/chat/completions", strings.Replace(oaToolReq, "%s", `,"tool_choice":"auto"`, 1),
			model.ToolChoice{Mode: model.ToolChoiceAuto}},
		{"/v1/messages", strings.Replace(anToolReq, "%s", `,"tool_choice":{"type":"any"}`, 1),
			model.ToolChoice{Mode: model.ToolChoiceRequired}},
		{"/v1/messages", strings.Replace(anToolReq, "%s",
			`,"tool_choice":{"type":"tool","name":"get_weather","disable_parallel_tool_use":true}`, 1),
			model.ToolChoice{Mode: model.ToolChoiceFunction, Name: "get_weather", Single: true}},
		{"/v1/messages", strings.Replace(anToolReq, "%s", `,"tool_choice":{"type":"auto"}`, 1),
			model.ToolChoice{Mode: model.ToolChoiceAuto}},
	} {
		f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
		r := post(t, serve(t, f), c.path, c.body)
		r.Body.Close()
		if got := f.opts().Prompt.Chat.ToolChoice; got != c.want {
			t.Errorf("%s %s: tool choice %+v, want %+v", c.path, c.body[len(c.body)-90:], got, c.want)
		}
	}
	for _, c := range []struct{ path, body string }{
		{"/v1/chat/completions", strings.Replace(oaToolReq, "%s",
			`,"tool_choice":{"type":"function","function":{"name":"get_time"}}`, 1)},
		{"/v1/messages", strings.Replace(anToolReq, "%s", `,"tool_choice":{"type":"tool","name":"get_time"}`, 1)},
		{"/v1/chat/completions", strings.Replace(oaToolReq, "%s", `,"tool_choice":"sometimes"`, 1)},
	} {
		f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
		r := post(t, serve(t, f), c.path, c.body)
		r.Body.Close()
		if r.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400 for a choice naming no declared tool", c.path, r.StatusCode)
		}
	}
}

// TestToolCallsStreamAsTheyClose: a reply with two calls streams the first
// as soon as its markup closes, before the second is written.
func TestToolCallsStreamAsTheyClose(t *testing.T) {
	two := []string{"<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>",
		"\n<tool_call>\n{\"name\": \"get_weather\", ", "\"arguments\": {\"location\": \"Rome\"}}\n</tool_call>"}
	f := &fakeBackend{tokens: two, reason: FinishEOS}
	r := post(t, serve(t, f), "/v1/chat/completions", strings.Replace(oaToolReq, "%s", `,"stream":true`, 1))
	defer r.Body.Close()
	fr := newFrameReader(r.Body)
	var order []string
	for {
		x, ok := fr.next()
		if !ok || x.data == "[DONE]" {
			break
		}
		var c oaChatResponse
		if err := json.Unmarshal([]byte(x.data), &c); err != nil {
			t.Fatal(err)
		}
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			order = append(order, fmt.Sprintf("%d:%s", *tc.Index, tc.Function.Arguments))
		}
	}
	if len(order) != 2 || order[0] != `0:{"location":"Paris"}` || order[1] != `1:{"location":"Rome"}` {
		t.Fatalf("streamed calls %v", order)
	}
}

// toolModels are the real models the forced-call gate runs: each family's
// syntax on a model small enough for the laptop. JITLLM_TOOL_MODELS names
// others (file names, comma-separated), for a card or a bigger box.
func forcedToolModels() []string {
	if s := os.Getenv("JITLLM_TOOL_MODELS"); s != "" {
		return strings.Split(s, ",")
	}
	return []string{"Qwen3-0.6B-Q8_0.jlm", "Llama-3.2-1B-Instruct-Q4_K_M.jlm"}
}

const forcedTools = `[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city",` +
	`"parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},` +
	`"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}},` +
	`{"type":"function","function":{"name":"get_time","description":"Get the time in a time zone",` +
	`"parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]}}},` +
	`{"type":"function","function":{"name":"get_stock_price","description":"Get a stock's last price",` +
	`"parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}}]`

// forcedPrompt is a request no model answers with a call of its own accord.
const forcedPrompt = `[{"role":"system","content":"You are a poet. Call a tool only when the user asks for the weather or the time."},{"role":"user","content":"Write one short sentence about the sea."}]`

// TestToolChoiceForcesACall: on a real model, a prompt the model answers in
// text (tool_choice auto -- the violation: no constraint, no call) becomes a
// call under "required" and under a named tool, through both APIs, streaming
// and not; the arguments validate against the tool's schema, and the forced
// generates are counted.
func TestToolChoiceForcesACall(t *testing.T) {
	for _, file := range forcedToolModels() {
		t.Run(file, func(t *testing.T) {
			_, lm, c := loadedEngine(t, file, "m", LoadOptions{})
			syn, err := lm.m.ToolSyntax()
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%s: tool syntax %s", file, syn)
			oa := func(extra string, stream bool) (string, []oaToolCall, string) {
				body := fmt.Sprintf(`{"model":"m","messages":%s,"tools":%s,"max_tokens":1024,"temperature":0%s,"stream":%v}`,
					forcedPrompt, forcedTools, extra, stream)
				if !stream {
					var out oaChatResponse
					status, b := postRaw(t, c.url+"/v1/chat/completions", body)
					if status != 200 || json.Unmarshal([]byte(b), &out) != nil {
						t.Fatalf("%d %s", status, b)
					}
					return out.Choices[0].Message.Content, out.Choices[0].Message.ToolCalls, *out.Choices[0].FinishReason
				}
				resp, err := http.Post(c.url+"/v1/chat/completions", "application/json", strings.NewReader(body))
				if err != nil {
					t.Fatal(err)
				}
				defer resp.Body.Close()
				fr := newFrameReader(resp.Body)
				var text strings.Builder
				var calls []oaToolCall
				finish := ""
				for {
					x, ok := fr.next()
					if !ok || x.data == "[DONE]" {
						break
					}
					var ch oaChatResponse
					if err := json.Unmarshal([]byte(x.data), &ch); err != nil {
						t.Fatalf("%s: %v", x.data, err)
					}
					if d := ch.Choices[0].Delta; d != nil {
						text.WriteString(d.Content)
						calls = append(calls, d.ToolCalls...)
					}
					if ch.Choices[0].FinishReason != nil {
						finish = *ch.Choices[0].FinishReason
					}
				}
				return text.String(), calls, finish
			}
			// The violation arm: nothing forces a call. The named arms below
			// force a tool whose call the unforced reply does not already
			// match: one it did not call, or, when it called every tool (Llama
			// 3.2 1B calls each tool it is given whatever it is asked), one
			// tool alone where it called several -- the named check holds
			// every call to that one name. An unforced reply of exactly the
			// forced call would leave the gate nothing to prove.
			txt, calls, _ := oa(`,"tool_choice":"auto"`, false)
			called := map[string]bool{}
			for _, tc := range calls {
				called[tc.Function.Name] = true
			}
			forced := ""
			for _, name := range []string{"get_time", "get_stock_price"} {
				if !called[name] {
					forced = name
					break
				}
			}
			if forced == "" {
				forced = "get_time"
				if len(called) < 2 {
					t.Fatalf("unforced, the model called %+v: this prompt cannot show the constraint forcing %s", calls, forced)
				}
			}
			t.Logf("auto: %q calls %+v", txt, calls)
			before := lm.grammars.toolForced.Load()
			check := func(arm string, calls []oaToolCall, finish, content, want string) {
				t.Helper()
				if finish != "tool_calls" || len(calls) == 0 {
					t.Fatalf("%s: finish %q calls %+v content %q: no call", arm, finish, calls, content)
				}
				for _, tc := range calls {
					if want != "" && tc.Function.Name != want {
						t.Fatalf("%s: called %q, want %q", arm, tc.Function.Name, want)
					}
					validArgs(t, arm, tc.Function.Name, tc.Function.Arguments)
				}
				if strings.Contains(content, "<tool_call>") || strings.Contains(content, "\"name\"") {
					t.Fatalf("%s: the call's markup is in the content %q", arm, content)
				}
				t.Logf("%s: %s(%s) content %q", arm, calls[0].Function.Name, calls[0].Function.Arguments, content)
			}
			txt, calls, fin := oa(`,"tool_choice":"required"`, false)
			check("required", calls, fin, txt, "")
			txt, calls, fin = oa(`,"tool_choice":{"type":"function","function":{"name":"`+forced+`"}}`, false)
			check("named "+forced, calls, fin, txt, forced)
			txt, calls, fin = oa(`,"tool_choice":{"type":"function","function":{"name":"get_weather"}}`, true)
			check("named get_weather, streamed", calls, fin, txt, "get_weather")
			for i, tc := range calls {
				if tc.Index == nil || *tc.Index != i {
					t.Fatalf("streamed call %d carries index %v", i, tc.Index)
				}
			}

			// Anthropic: any, and a named tool streamed.
			an := fmt.Sprintf(`{"model":"m","max_tokens":1024,"temperature":0,"messages":%s,"tools":[`+
				`{"name":"get_weather","description":"Get the current weather for a city","input_schema":`+
				`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}},`+
				`{"name":"get_time","description":"Get the time in a time zone","input_schema":`+
				`{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]}},`+
				`{"name":"get_stock_price","description":"Get a stock's last price","input_schema":`+
				`{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}]`, forcedPrompt)
			status, b := postRaw(t, c.url+"/v1/messages", an+`,"tool_choice":{"type":"any"}}`)
			var ar anResponse
			if status != 200 || json.Unmarshal([]byte(b), &ar) != nil {
				t.Fatalf("anthropic any: %d %s", status, b)
			}
			var used int
			for _, blk := range ar.Content {
				if blk.Type == "tool_use" {
					used++
					validArgs(t, "anthropic any", blk.Name, string(blk.Input))
				}
			}
			if *ar.StopReason != "tool_use" || used == 0 {
				t.Fatalf("anthropic any: %s", b)
			}
			resp, err := http.Post(c.url+"/v1/messages", "application/json",
				strings.NewReader(an+`,"stream":true,"tool_choice":{"type":"tool","name":"`+forced+`"}}`))
			if err != nil {
				t.Fatal(err)
			}
			fr := newFrameReader(resp.Body)
			var name, js, stop string
			for {
				x, ok := fr.next()
				if !ok {
					break
				}
				var raw map[string]json.RawMessage
				json.Unmarshal([]byte(x.data), &raw)
				var cb struct {
					Type, Name string
				}
				json.Unmarshal(raw["content_block"], &cb)
				if cb.Type == "tool_use" {
					name = cb.Name
				}
				var delta map[string]any
				json.Unmarshal(raw["delta"], &delta)
				if delta["type"] == "input_json_delta" {
					js += delta["partial_json"].(string)
				}
				if s, ok := delta["stop_reason"].(string); ok {
					stop = s
				}
			}
			resp.Body.Close()
			if name != forced || stop != "tool_use" {
				t.Fatalf("anthropic named, streamed: tool %q stop %q", name, stop)
			}
			validArgs(t, "anthropic named", name, js)

			// Every forced arm went through the constraint, and the auto arm
			// did not.
			if n := lm.grammars.toolForced.Load() - before; n != 5 {
				t.Fatalf("%d generates were constrained to a call, want 5", n)
			}
		})
	}
}

// validArgs holds a call's arguments to its tool's schema.
func validArgs(t *testing.T, arm, name, args string) {
	t.Helper()
	var tools []struct {
		Function struct {
			Name       string          `json:"name"`
			Parameters json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	json.Unmarshal([]byte(forcedTools), &tools)
	for _, x := range tools {
		if x.Function.Name != name {
			continue
		}
		var schema, doc any
		json.Unmarshal(x.Function.Parameters, &schema)
		if err := schemacheck.Parse(args, &doc); err != nil {
			t.Fatalf("%s: arguments %q: %v", arm, args, err)
		}
		if err := schemacheck.Validate(schema, schema, doc); err != nil {
			t.Fatalf("%s: arguments %s: %v", arm, args, err)
		}
		return
	}
	t.Fatalf("%s: called %q, which was not declared", arm, name)
}

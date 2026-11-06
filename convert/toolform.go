package convert

import (
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// stateToolForm states how the model writes a tool call: the form of the
// template a request with tools renders (the one named tool_use, else the
// default), read from that template and the vocabulary (ToolFormOf).
func stateToolForm(s *jlm.Source) {
	v := s.Vocab
	if v == nil || len(v.Templates) == 0 {
		return
	}
	src, ok := v.Template("tool_use")
	if !ok {
		src, _ = v.Template("")
	}
	v.Tools = ToolFormOf(src, harmonyTokens(v))
}

// harmonyTokens is whether v frames messages as gpt-oss's harmony does: it
// holds every mark that framing writes.
func harmonyTokens(v *jlm.Vocab) bool {
	need := map[string]bool{"<|start|>": true, "<|channel|>": true, "<|message|>": true, "<|end|>": true, "<|return|>": true, "<|call|>": true}
	for _, t := range v.Tokens {
		delete(need, t)
	}
	return len(need) == 0
}

// ToolFormOf is the tool-call form a chat template trains its model to write
// a call in, read from the markup the template writes or describes: harmony
// where the vocabulary frames messages that way. The checks are ordered: a
// template naming several markers is the family whose marker is most
// specific (Qwen3.5's writes <tool_call> around <function=, GLM's around
// <arg_key>). A template naming none is read as Hermes calls.
func ToolFormOf(template string, harmony bool) jlm.ToolForm {
	has := func(s ...string) bool {
		for _, x := range s {
			if !strings.Contains(template, x) {
				return false
			}
		}
		return true
	}
	switch {
	case harmony || has("<|channel|>", "to="):
		return toolForm("harmony")
	case has("｜DSML｜"):
		return toolForm("dsml")
	case has("<｜tool▁calls▁begin｜>", "```json"):
		return toolForm("deepseek-v3")
	case has("<｜tool▁calls▁begin｜>"):
		return toolForm("deepseek-v3.1")
	case has("<|tool_calls_section_begin|>"):
		return toolForm("kimi")
	case has("<|open|>", "'tools'"):
		return toolForm("kimi-xtml")
	case has("[ARGS]"):
		return toolForm("mistral-args")
	case has("[TOOL_CALLS]"):
		return toolForm("mistral")
	case has("<minimax:tool_call>"):
		return toolForm("minimax")
	case has("<seed:tool_call>"):
		return toolForm("seed-xml")
	case has("<function="):
		return toolForm("qwen-xml")
	case has("<arg_key>"):
		return toolForm("glm")
	case has("<|tool_call>"):
		return toolForm("gemma4")
	case has("<|START_ACTION|>"):
		return toolForm("command-r7b")
	case has("Action:", "```json", "tool_name"):
		return toolForm("command-r")
	case has("<TOOLCALL>"):
		return toolForm("nemotron")
	case has("<|tool_call|>"):
		return toolForm("granite")
	case has("<|tools_prefix|>"):
		return toolForm("apertus")
	case has("<|tool_list_start|>") || has("<|tool_call_start|>") || has("List of tools:"):
		return toolForm("lfm2")
	case has("<|tool|>", "<|end|>") && !has("<tool_call>"):
		return toolForm("phi4-mini")
	case has("<tool_calls>", "```json"):
		return toolForm("hunyuan")
	case has("<tool_call>"):
		return toolForm("hermes")
	case has("<|header_start|>", "tool_calls"):
		return toolForm("pythonic")
	case has("<|start_header_id|>", "ipython"):
		return toolForm("json")
	}
	return toolForm("hermes")
}

// DefaultToolForm is the form a reply is read in where no model states one:
// Hermes calls, the most common.
func DefaultToolForm() jlm.ToolForm { return toolForm("hermes") }

// toolForm is the named family's form: the texts its template writes a call
// in, and what a forced call writes.
func toolForm(name string) jlm.ToolForm {
	jsonN := jlm.ToolNotation{Quotes: []string{`"`}, Escapes: true, True: []string{"true"}, False: []string{"false"}, None: []string{"null"}}
	pyN := jlm.ToolNotation{Quotes: []string{`"`, `'`}, Escapes: true, True: []string{"True", "true"},
		False: []string{"False", "false"}, None: []string{"None", "null"}}
	think := []string{"<think>", "</think>"}
	call := func(f jlm.ToolForm) jlm.ToolForm {
		f.Name, f.Notation, f.Several = name, jsonN, true
		if f.NameKey == "" && !f.Keyed {
			f.NameKey, f.ArgsKey, f.AltArgsKey = "name", "arguments", "parameters"
		}
		return f
	}
	obj := `{"name": $QNAME, "arguments": $ARGS}`
	list := func(calls, wOpen, wClose string, markers []string) jlm.ToolForm {
		return call(jlm.ToolForm{Calls: []string{calls}, Markers: markers,
			Write: jlm.ToolWrite{Call: obj, Open: wOpen, Sep: ", ", Close: wClose}})
	}
	ds := []string{"<｜tool▁call▁begin｜>function<｜tool▁sep｜>$NAME(?```json?)$ARGS(?```?)<｜tool▁call▁end｜>",
		"<｜tool▁call▁begin｜>$NAME<｜tool▁sep｜>(?```json?)$ARGS(?```?)<｜tool▁call▁end｜>"}
	dsMarks := []string{"<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>", "<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>", "<｜tool▁sep｜>", "<think>", "</think>"}
	mistral := call(jlm.ToolForm{Calls: []string{"[TOOL_CALLS]$CALLS", "[TOOL_CALLS]$NAME(?[CALL_ID]$ID?)[ARGS]$ARGS", "$CALLS"},
		Bare: true, IDDigits: 9, IDAlnum: true, ThinkOpen: "[THINK]", ThinkClose: "[/THINK]",
		Markers: []string{"[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "[THINK]", "[/THINK]"}})
	xmlParams := []string{"<parameter=$KEY>\n$VALUE\n</parameter>"}
	py := func(open, close, wOpen, wClose string, markers []string) jlm.ToolForm {
		f := call(jlm.ToolForm{Open: open, Close: close, Sep: ",", Calls: []string{"$NAME($PARAMS)"}, Params: []string{"$KEY=$VALUE"},
			ParamSep: ",", Markers: markers,
			Write: jlm.ToolWrite{Call: "$NAME($PARAMS)", Params: []string{"$KEY=$VALUE"}, ParamSep: ", ", Open: wOpen, Sep: ", ", Close: wClose}})
		f.Notation = pyN
		return f
	}
	switch name {
	case "hermes":
		return call(jlm.ToolForm{Calls: []string{"<tool_call>$CALL</tool_call>", "[TOOL_CALLS]$CALLS", "$CALLS"}, Bare: true, Sep: ";",
			Markers: append([]string{"<tool_call>", "</tool_call>"}, think...),
			Write:   jlm.ToolWrite{Call: "<tool_call>\n" + obj + "\n</tool_call>", Sep: "\n"}})
	case "json":
		return call(jlm.ToolForm{Calls: []string{"(?<|python_tag|>?)$CALLS"}, Bare: true, Leads: true, Sep: ";",
			Write: jlm.ToolWrite{Call: `{"name": $QNAME, "parameters": $ARGS}`, Sep: "; "}})
	case "qwen-xml":
		return call(jlm.ToolForm{Calls: []string{"<tool_call>\n<function=$NAME>\n$PARAMS</function>\n</tool_call>"}, Params: xmlParams,
			TextValues: true, GuessJSON: true, Markers: append([]string{"<tool_call>", "</tool_call>"}, think...),
			Write: jlm.ToolWrite{Call: "<function=$NAME>\n$PARAMS</function>\n", Params: []string{"<parameter=$KEY>\n$VALUE\n</parameter>\n"},
				Section: [2]string{"<tool_call>\n", "</tool_call>"}, Sep: "\n"}})
	case "seed-xml":
		return call(jlm.ToolForm{Calls: []string{"<seed:tool_call>\n<function=$NAME>\n$PARAMS</function>\n</seed:tool_call>"}, Params: xmlParams,
			TextValues: true, GuessJSON: true, ThinkOpen: "<seed:think>", ThinkClose: "</seed:think>",
			Markers: []string{"<seed:tool_call>", "</seed:tool_call>", "<seed:think>", "</seed:think>"},
			Write: jlm.ToolWrite{Call: "<function=$NAME>\n$PARAMS</function>\n", Params: []string{"<parameter=$KEY>$VALUE</parameter>\n"},
				Section: [2]string{"<seed:tool_call>\n", "</seed:tool_call>"}, Sep: "\n"}})
	case "minimax":
		return call(jlm.ToolForm{Open: "<minimax:tool_call>", Close: "</minimax:tool_call>", Calls: []string{"<invoke name=\"$NAME\">\n$PARAMS</invoke>"},
			Params: []string{`<parameter name="$KEY">$VALUE</parameter>`}, TextValues: true, GuessJSON: true,
			Markers: append([]string{"<minimax:tool_call>", "</minimax:tool_call>"}, think...),
			Write: jlm.ToolWrite{Call: "<invoke name=\"$NAME\">\n$PARAMS</invoke>\n", Params: []string{"<parameter name=\"$KEY\">$VALUE</parameter>\n"},
				Open: "<minimax:tool_call>\n", Close: "</minimax:tool_call>"}})
	case "dsml":
		return call(jlm.ToolForm{Open: "<｜DSML｜tool_calls>", Close: "</｜DSML｜tool_calls>", Calls: []string{`<｜DSML｜invoke name="$NAME">$PARAMS</｜DSML｜invoke>`},
			Params: []string{`<｜DSML｜parameter name="$KEY" string="$TYPE">$VALUE</｜DSML｜parameter>`}, TextValues: true,
			Types: map[string]string{"true": "string", "false": "json"}, Markers: think,
			Write: jlm.ToolWrite{Call: "<｜DSML｜invoke name=\"$NAME\">\n$PARAMS</｜DSML｜invoke>\n",
				Params: []string{"<｜DSML｜parameter name=\"$KEY\" string=\"true\">$VALUE</｜DSML｜parameter>\n",
					"<｜DSML｜parameter name=\"$KEY\" string=\"false\">$VALUE</｜DSML｜parameter>\n"},
				Open: "(?\n\n?)<｜DSML｜tool_calls>\n", Close: "</｜DSML｜tool_calls>"}})
	case "glm":
		return call(jlm.ToolForm{Calls: []string{"<tool_call>$NAME\n$PARAMS</tool_call>"}, Params: []string{"<arg_key>$KEY</arg_key>\n<arg_value>$VALUE</arg_value>"},
			TextValues: true, GuessJSON: true, Markers: append([]string{"<tool_call>", "</tool_call>"}, think...),
			Write: jlm.ToolWrite{Call: "<tool_call>$NAME\n$PARAMS</tool_call>", Params: []string{"<arg_key>$KEY</arg_key>\n<arg_value>$VALUE</arg_value>\n"}, Sep: "\n"}})
	case "deepseek-v3":
		return call(jlm.ToolForm{Open: "<｜tool▁calls▁begin｜>", Close: "<｜tool▁calls▁end｜>", Calls: ds, Markers: dsMarks,
			Write: jlm.ToolWrite{Call: "<｜tool▁call▁begin｜>function<｜tool▁sep｜>$NAME\n```json\n$ARGS\n```<｜tool▁call▁end｜>",
				Open: "<｜tool▁calls▁begin｜>", Sep: "\n", Close: "<｜tool▁calls▁end｜>"}})
	case "deepseek-v3.1":
		return call(jlm.ToolForm{Open: "<｜tool▁calls▁begin｜>", Close: "<｜tool▁calls▁end｜>", Calls: ds, Markers: dsMarks,
			Write: jlm.ToolWrite{Call: "<｜tool▁call▁begin｜>$NAME<｜tool▁sep｜>$ARGS<｜tool▁call▁end｜>",
				Open: "<｜tool▁calls▁begin｜>", Sep: "\n", Close: "<｜tool▁calls▁end｜>"}})
	case "kimi":
		return call(jlm.ToolForm{Open: "<|tool_calls_section_begin|>", Close: "<|tool_calls_section_end|>",
			Calls: []string{"<|tool_call_begin|>functions.$NAME:$ID<|tool_call_argument_begin|>$ARGS<|tool_call_end|>",
				"<|tool_call_begin|>$NAME(?:$ID?)<|tool_call_argument_begin|>$ARGS<|tool_call_end|>"}, IDDigits: 3,
			Markers: []string{"<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>",
				"<|tool_call_argument_begin|>", "<|tool_call_end|>", "<think>", "</think>"},
			Write: jlm.ToolWrite{Call: "<|tool_call_begin|>functions.$NAME:$ID<|tool_call_argument_begin|>$ARGS<|tool_call_end|>",
				Open: "<|tool_calls_section_begin|>", Close: "<|tool_calls_section_end|>"}})
	case "kimi-xtml":
		return call(jlm.ToolForm{Open: "<|open|>tools<|sep|>", Close: "<|close|>tools<|sep|>",
			Calls: []string{`<|open|>call tool="$XNAME"(? index="$ID"?)<|sep|><|open|>json(? type="object"?)<|sep|>$ARGS<|close|>json<|sep|><|close|>call<|sep|>`,
				`<|open|>call tool="$XNAME"(? index="$ID"?)<|sep|>$PARAMS<|close|>call<|sep|>`},
			Params:     []string{`<|open|>argument key="$KEY"(? type="$TYPE"?)<|sep|>$VALUE<|close|>argument<|sep|>`},
			TextValues: true, IDDigits: 3, IDMin: 1,
			ThinkOpen: "<|open|>think<|sep|>", ThinkClose: "<|close|>think<|sep|>", ThinkSeen: "'think'",
			Markers: []string{"<|open|>", "<|close|>", "<|sep|>"},
			Write: jlm.ToolWrite{Call: `<|open|>call tool="$XNAME" index="$ID"<|sep|><|open|>json type="object"<|sep|>$ARGS<|close|>json<|sep|><|close|>call<|sep|>`,
				Open: "<|open|>tools<|sep|>", Close: "<|close|>tools<|sep|>"}})
	case "harmony":
		f := call(jlm.ToolForm{Calls: []string{
			"<|start|>assistant to=functions.$NAME<|channel|>commentary(? json?)(? <|constrain|>json?)<|message|>$ARGS(?<|call|>?)(?<|end|>?)",
			"<|start|>assistant<|channel|>commentary to=functions.$NAME(? <|constrain|>json?)<|message|>$ARGS(?<|call|>?)(?<|end|>?)",
			"<|channel|>commentary to=functions.$NAME(? <|constrain|>json?)<|message|>$ARGS(?<|call|>?)(?<|end|>?)"},
			ThinkOpen: "<|channel|>analysis<|message|>", ThinkClose: "<|end|>", ThinkAlways: true, ThinkThen: "<|start|>assistant",
			Markers: []string{"<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|constrain|>"},
			Write:   jlm.ToolWrite{Call: "<|channel|>commentary to=functions.$NAME <|constrain|>json<|message|>$ARGS"}})
		f.Several = false
		return f
	case "mistral":
		f := mistral
		f.Write = jlm.ToolWrite{Call: obj, Open: "[TOOL_CALLS][", Sep: ", ", Close: "]"}
		return f
	case "mistral-args":
		f := mistral
		f.Write = jlm.ToolWrite{Call: "[TOOL_CALLS]$NAME(?[CALL_ID]$ID?)[ARGS]$ARGS"}
		return f
	case "granite":
		return list("<|tool_call|>$CALLS", "<|tool_call|>[", "]", []string{"<|tool_call|>"})
	case "nemotron":
		return list("<TOOLCALL>$CALLS</TOOLCALL>", "<TOOLCALL>[", "]</TOOLCALL>", append([]string{"<TOOLCALL>", "</TOOLCALL>"}, think...))
	case "phi4-mini":
		return list("functools$CALLS", "functools[", "]", nil)
	case "command-r7b":
		return call(jlm.ToolForm{Calls: []string{"<|START_ACTION|>$CALLS<|END_ACTION|>"}, NameKey: "tool_name", ArgsKey: "parameters", IDDigits: 3,
			ThinkOpen: "<|START_THINKING|>", ThinkClose: "<|END_THINKING|>",
			Markers: []string{"<|START_ACTION|>", "<|END_ACTION|>", "<|START_THINKING|>", "<|END_THINKING|>"},
			Write: jlm.ToolWrite{Call: `{"tool_call_id": "$ID", "tool_name": $QNAME, "parameters": $ARGS}`,
				Open: "<|START_ACTION|>[\n    ", Sep: ",\n    ", Close: "\n]<|END_ACTION|>"}})
	case "command-r":
		return call(jlm.ToolForm{Calls: []string{"Action:(?```json?)$CALLS(?```?)"}, NameKey: "tool_name", ArgsKey: "parameters",
			Write: jlm.ToolWrite{Call: `{"tool_name": $QNAME, "parameters": $ARGS}`, Open: "Action: ```json\n[\n    ", Sep: ",\n    ", Close: "\n]\n```"}})
	case "apertus":
		return call(jlm.ToolForm{Calls: []string{"<|tools_prefix|>$CALLS<|tools_suffix|>"}, Keyed: true,
			Markers: []string{"<|tools_prefix|>", "<|tools_suffix|>"},
			Write:   jlm.ToolWrite{Call: "{$QNAME: $ARGS}", Open: "<|tools_prefix|>[", Sep: ", ", Close: "]<|tools_suffix|>"}})
	case "pythonic":
		f := py("(?<|python_start|>?)[", "](?<|python_end|>?)", "[", "]", []string{"<|python_start|>", "<|python_end|>"})
		f.Leads = true
		return f
	case "lfm2":
		return py("<|tool_call_start|>[", "]<|tool_call_end|>", "<|tool_call_start|>[", "]<|tool_call_end|>",
			append([]string{"<|tool_call_start|>", "<|tool_call_end|>"}, think...))
	case "hunyuan":
		return call(jlm.ToolForm{Open: "<tool_calls>", Close: "</tool_calls>", Calls: []string{"<tool_call>$NAME(?```json?)$ARGS(?```?)</tool_call>"},
			Markers: append([]string{"<tool_calls>", "</tool_calls>", "<tool_call>", "</tool_call>"}, think...),
			Write:   jlm.ToolWrite{Call: "<tool_call>$NAME\n```json\n$ARGS\n```</tool_call>", Open: "<tool_calls>", Close: "</tool_calls>"}})
	case "gemma4":
		f := call(jlm.ToolForm{Calls: []string{"<|tool_call>call:$NAME{$PARAMS}<tool_call|>"}, Params: []string{"$KEY:$VALUE"}, ParamSep: ",",
			Markers: []string{"<|tool_call>", "<tool_call|>", `<|"|>`},
			Write:   jlm.ToolWrite{Call: "<|tool_call>call:$NAME{$PARAMS}<tool_call|>", Params: []string{"$KEY:$VALUE"}, ParamSep: ",", Sorted: true}})
		f.Notation = jlm.ToolNotation{Quotes: []string{`<|"|>`}, BareKeys: true, True: []string{"true"}, False: []string{"false"}, None: []string{"null"}}
		return f
	}
	return jlm.ToolForm{}
}

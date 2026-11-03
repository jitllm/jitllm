package jinja_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/tok/jinja"
)

// chatGoldDate is the date strftime_now prints, pinned as scripts/chatgold.py
// pins it, so a golden made on one day still matches on the next.
func chatGoldDate() time.Time {
	return time.Date(2024, 7, 26, 0, 0, 0, 0, time.UTC)
}

// A templateEntry is one row of testdata/templates/manifest.json: a supported
// model family's chat template, where it was read from, and the BOS and EOS
// text it interpolates. Three fields say how the family's own shapes differ
// from the engine's defaults: Image, for a vision template, is "parts" for
// typed content parts or the placeholder text a string-content template is
// handed in the picture's place; Tools "cohere" is Command-R's
// parameter_definitions schema in place of OpenAI's; Arguments "string" is a
// tool call's arguments as JSON text, which DeepSeek-V3.2 concatenates; Content
// "parts" is every message's text as a typed part, for a template that reads
// nothing else (SmolVLM's, Janus-Pro's), as engine/model hands it one.
type templateEntry struct {
	File      string            `json:"file"`
	Family    string            `json:"family"`
	Source    map[string]string `json:"source"`
	Image     string            `json:"image"`
	Content   string            `json:"content"`
	Tools     string            `json:"tools"`
	Arguments string            `json:"arguments"`
	BOS       string            `json:"bos"`
	EOS       string            `json:"eos"`
}

func loadManifest(t testing.TB) []templateEntry {
	b, err := os.ReadFile(filepath.Join("testdata", "templates", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m []templateEntry
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("manifest.json: %v", err)
	}
	return m
}

func readTemplate(t testing.TB, file string) string {
	b, err := os.ReadFile(filepath.Join("testdata", "templates", file))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// conversations is testdata/conversations.json: the cases every template is
// rendered over, the tool-call arguments, and the tools in each family's shape.
// Messages stay raw JSON so key order survives into the template, as a
// request's does.
type conversations struct {
	Arguments json.RawMessage            `json:"arguments"`
	Tools     map[string]json.RawMessage `json:"tools"`
	Documents json.RawMessage            `json:"documents"`
	Cases     []chatCase                 `json:"cases"`
}

// A chatCase is one conversation. $ARGS in its messages stands for the call's
// arguments in the family's shape, and Image marks the vision case, built per
// template from its manifest row.
type chatCase struct {
	Name     string          `json:"name"`
	Gen      bool            `json:"add_generation_prompt"`
	Tools    bool            `json:"tools"`
	Image    bool            `json:"image"`
	Docs     bool            `json:"documents"`
	Messages json.RawMessage `json:"messages"`
}

func loadConversations(t testing.TB) conversations {
	b, err := os.ReadFile(filepath.Join("testdata", "conversations.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c conversations
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatalf("conversations.json: %v", err)
	}
	return c
}

// jsonString is s as a JSON string literal.
func jsonString(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// imageMessages is the vision case in the shape e reads it.
func imageMessages(e templateEntry) string {
	const q = "What is in this picture?"
	if e.Image == "parts" {
		return `[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "` + q + `"}]}]`
	}
	return `[{"role": "user", "content": ` + jsonString(e.Image+"\n"+q) + `}]`
}

// chatContext is the JSON of the render context for c through e: what
// engine/model binds and apply_chat_template binds alike, tools and documents
// null when the request carries none. It is what the golden script renders, so
// both engines see the same bytes; the golden records its hash.
func chatContext(e templateEntry, conv conversations, c chatCase) string {
	msgs := string(c.Messages)
	if c.Image {
		msgs = imageMessages(e)
	}
	args := string(conv.Arguments)
	if e.Arguments == "string" {
		args = jsonString(args)
	}
	msgs = strings.ReplaceAll(msgs, `"$ARGS"`, args)
	if e.Content == "parts" {
		msgs = textParts(msgs)
	}
	tools := "null"
	if c.Tools {
		shape := "openai"
		if e.Tools != "" {
			shape = e.Tools
		}
		tools = string(conv.Tools[shape])
	}
	var b strings.Builder
	docs := "null"
	if c.Docs {
		docs = string(conv.Documents)
	}
	b.WriteString(`{"messages": ` + msgs + `, "tools": ` + tools + `, "documents": ` + docs)
	b.WriteString(`, "add_generation_prompt": ` + strconv.FormatBool(c.Gen))
	b.WriteString(`, "bos_token": ` + jsonString(e.BOS) + `, "eos_token": ` + jsonString(e.EOS) + `}`)
	return b.String()
}

// textParts rewrites every string content in msgs, a JSON array of messages,
// as one text part, [{"type": "text", "text": ...}], keeping every key's place.
func textParts(msgs string) string {
	var in []json.RawMessage
	if err := json.Unmarshal([]byte(msgs), &in); err != nil {
		panic(err)
	}
	var b strings.Builder
	b.WriteString("[")
	for i, m := range in {
		if i > 0 {
			b.WriteString(", ")
		}
		d := json.NewDecoder(strings.NewReader(string(m)))
		if _, err := d.Token(); err != nil {
			panic(err)
		}
		b.WriteString("{")
		for n := 0; d.More(); n++ {
			k, err := d.Token()
			if err != nil {
				panic(err)
			}
			var v json.RawMessage
			if err := d.Decode(&v); err != nil {
				panic(err)
			}
			if k == "content" && len(v) > 0 && v[0] == '"' {
				v = json.RawMessage(`[{"type": "text", "text": ` + string(v) + `}]`)
			}
			if n > 0 {
				b.WriteString(", ")
			}
			b.WriteString(jsonString(k.(string)) + ": " + string(v))
		}
		b.WriteString("}")
	}
	b.WriteString("]")
	return b.String()
}

// goldenCase is one rendering by transformers: the text, or the exception it
// raised and that exception's type -- "TemplateError" is the template's own
// raise_exception, whose message the engine must carry too.
type goldenCase struct {
	Ctx   string `json:"ctx"`
	Out   string `json:"out"`
	Raise string `json:"raise"`
	By    string `json:"by"`
}

// TestChatTemplatesMatchTransformers renders every supported family's chat
// template (testdata/templates/manifest.json) over the conversations in
// testdata/conversations.json and holds the output to transformers' own rendering byte for byte,
// recorded by scripts/chatgold.py from the same context JSON. A template that
// raises in transformers must raise here, with the template's message when the
// template raised it.
//
// JITLLM_CHATGOLD=<path> writes the contexts for the script instead.
func TestChatTemplatesMatchTransformers(t *testing.T) {
	entries, conv := loadManifest(t), loadConversations(t)
	if p := os.Getenv("JITLLM_CHATGOLD"); p != "" {
		dumpChatContexts(t, entries, conv, p)
		return
	}
	compared, raised := 0, 0
	for _, e := range entries {
		tpl, err := jinja.Compile(readTemplate(t, e.File))
		if err != nil {
			t.Errorf("%s: compile: %v", e.File, err)
			continue
		}
		var golden map[string]goldenCase
		if b, err := os.ReadFile(filepath.Join("testdata", "golden", strings.TrimSuffix(e.File, ".jinja")+".json")); err == nil {
			if err := json.Unmarshal(b, &golden); err != nil {
				t.Fatalf("%s: golden: %v", e.File, err)
			}
		}
		for _, c := range conv.Cases {
			if c.Image && e.Image == "" {
				continue
			}
			ctxJSON := chatContext(e, conv, c)
			sum := sha256.Sum256([]byte(ctxJSON))
			want, ok := golden[c.Name]
			if !ok || want.Ctx != hex.EncodeToString(sum[:8]) {
				t.Errorf("%s/%s: no golden for this context; regenerate with scripts/chatgold.py", e.File, c.Name)
				continue
			}
			v, err := jinja.FromJSON([]byte(ctxJSON))
			if err != nil {
				t.Fatalf("%s/%s: context: %v", e.File, c.Name, err)
			}
			data := make(map[string]any, v.AsDict().Len()+1)
			for k, x := range v.AsDict().Data {
				data[k] = x
			}
			data["strftime_now"] = jinja.StrftimeNow(func() time.Time { return chatGoldDate() })
			got, err := tpl.Render(data)
			compared++
			switch {
			case want.Raise != "":
				raised++
				if err == nil {
					t.Errorf("%s/%s: transformers raises %q and this rendered %d bytes", e.File, c.Name, want.Raise, len(got))
				} else if want.By == "TemplateError" && !strings.Contains(err.Error(), want.Raise) {
					t.Errorf("%s/%s: the template raises %q; this failed with %v", e.File, c.Name, want.Raise, err)
				}
			case err != nil:
				t.Errorf("%s/%s: render: %v", e.File, c.Name, err)
			case got != want.Out:
				t.Errorf("%s/%s: diverges from transformers\n%s", e.File, c.Name, diffSummary(got, want.Out))
			}
		}
		// A template that renders none of the conversations is compared only
		// on its failures, which proves nothing about its output. Command-R's
		// tool_use and rag render only with tools and with documents.
		rendered := false
		for _, g := range golden {
			rendered = rendered || g.Out != ""
		}
		if !rendered {
			t.Errorf("%s: transformers renders none of the conversations", e.File)
		}
	}
	// A golden or a template the manifest no longer names is a family that
	// left the set without its files, or one that joined without a row.
	listed := map[string]bool{"manifest.json": true}
	for _, e := range entries {
		if e.Source["repo"] == "" {
			t.Errorf("%s: the manifest names no source repo, so it cannot be refreshed", e.File)
		}
		listed[e.File] = true
		listed[strings.TrimSuffix(e.File, ".jinja")+".json"] = true
	}
	for _, dir := range []string{"templates", "golden"} {
		files, err := os.ReadDir(filepath.Join("testdata", dir))
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range files {
			if !listed[f.Name()] {
				t.Errorf("testdata/%s/%s is not in the manifest", dir, f.Name())
			}
		}
	}
	if len(entries) < 40 || compared < 6*len(entries) {
		t.Fatalf("%d templates, %d comparisons: the manifest or the cases have shrunk", len(entries), compared)
	}
	t.Logf("%d templates, %d renderings compared, %d of them raising in both engines", len(entries), compared, raised)
}

// dumpChatContexts writes every (template, case) context for
// scripts/chatgold.py to render, and nothing is compared: it is a tool, not a
// pass, so it ends in a skip that names what it wrote.
func dumpChatContexts(t *testing.T, entries []templateEntry, conv conversations, path string) {
	type dumped struct {
		Template string `json:"template"`
		Case     string `json:"case"`
		Ctx      string `json:"ctx"`
	}
	var dump []dumped
	for _, e := range entries {
		for _, c := range conv.Cases {
			if !c.Image || e.Image != "" {
				dump = append(dump, dumped{e.File, c.Name, chatContext(e, conv, c)})
			}
		}
	}
	b, err := json.MarshalIndent(dump, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Skipf("wrote %d contexts to %s; render them with scripts/chatgold.py", len(dump), path)
}

// diffSummary says where two renderings first part: the byte, its line and
// column, and a window of each side.
func diffSummary(got, want string) string {
	i := 0
	for i < len(got) && i < len(want) && got[i] == want[i] {
		i++
	}
	line := strings.Count(got[:i], "\n") + 1
	col := i - strings.LastIndex(got[:i], "\n")
	window := func(s string) string {
		return s[max(i-60, 0):min(i+60, len(s))]
	}
	return "first difference at byte " + strconv.Itoa(i) + " (line " + strconv.Itoa(line) + ", col " +
		strconv.Itoa(col) + "); got " + strconv.Itoa(len(got)) + " B, want " + strconv.Itoa(len(want)) + " B\n" +
		"  got:  " + strconv.Quote(window(got)) + "\n  want: " + strconv.Quote(window(want))
}

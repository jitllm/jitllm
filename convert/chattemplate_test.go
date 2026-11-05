package convert

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// stories260KGGUF is the committed fixture: a GGUF with a vocabulary and no
// chat template, so whatever template its container carries came from the
// option.
const stories260KGGUF = "../testdata/models/stories260K.gguf"

const (
	tplA = `{% for m in messages %}A {{ m['content'] }}{% endfor %}`
	tplB = `{% for m in messages %}B {{ m['content'] }}{% endfor %}`
	tplC = `{% for m in messages %}C {{ m['content'] }}{% endfor %}`
)

func writeFiles(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// Every form -chat-template takes reads the template the safetensors path
// would read from the same files.
func TestChatTemplatesFromEveryForm(t *testing.T) {
	named := `{"chat_template": [{"name": "tool_use", "template": "` + jsonEsc(tplB) +
		`"}, {"name": "default", "template": "` + jsonEsc(tplA) + `"}]}`
	cases := []struct {
		name  string
		files map[string]string
		arg   string // inside the directory; "" is the directory itself
		want  []jlm.ChatTemplate
	}{
		{"tokenizer_config string", map[string]string{"tokenizer_config.json": `{"chat_template": "` + jsonEsc(tplA) + `"}`},
			"tokenizer_config.json", []jlm.ChatTemplate{{Body: tplA}}},
		{"tokenizer_config named list", map[string]string{"tokenizer_config.json": named},
			"tokenizer_config.json", []jlm.ChatTemplate{{Name: "default", Body: tplA}, {Name: "tool_use", Body: tplB}}},
		{"tokenizer_config beside chat_template.jinja", map[string]string{
			"tokenizer_config.json": `{"bos_token": "<s>"}`, "chat_template.jinja": tplC},
			"tokenizer_config.json", []jlm.ChatTemplate{{Body: tplC}}},
		{"chat_template.json", map[string]string{"chat_template.json": `{"chat_template": "` + jsonEsc(tplB) + `"}`},
			"chat_template.json", []jlm.ChatTemplate{{Body: tplB}}},
		{"a .jinja file", map[string]string{"mine.jinja": tplC}, "mine.jinja", []jlm.ChatTemplate{{Body: tplC}}},
		{"a model directory", map[string]string{"tokenizer_config.json": named},
			"", []jlm.ChatTemplate{{Name: "default", Body: tplA}, {Name: "tool_use", Body: tplB}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := writeFiles(t, c.files)
			got, err := ChatTemplatesFrom(filepath.Join(dir, c.arg))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("got %+v, want %+v", got, c.want)
			}
			// The safetensors path's reader on the same directory: the two
			// inputs must not disagree about one model's templates.
			if c.files["tokenizer_config.json"] != "" {
				d := dirFiles(dir)
				tc, err := readHFTokenizerConfig(d)
				if err != nil {
					t.Fatal(err)
				}
				st, err := hfTemplates(d, tc.ChatTemplate)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, st) {
					t.Errorf("the option read %+v and the safetensors path %+v", got, st)
				}
			}
		})
	}
}

// A file that cannot supply a template is refused by name, at conversion.
func TestChatTemplatesFromRefusesABadFile(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		arg   string
		say   string
	}{
		// The file's name: the words are the OS's ("no such file", "File not
		// found"), and that it is the OS's not-exist error is checked below.
		{"missing", nil, "nothere.json", "nothere.json"},
		{"not JSON", map[string]string{"tokenizer_config.json": `{"chat_template": `}, "tokenizer_config.json", "unexpected end"},
		{"no template", map[string]string{"tokenizer_config.json": `{"bos_token": "<s>"}`}, "tokenizer_config.json", "carries no chat template"},
		{"empty jinja", map[string]string{"t.jinja": ""}, "t.jinja", "carries no chat template"},
		{"blank template", map[string]string{"t.jinja": " \n "}, "t.jinja", "is empty"},
		{"does not compile", map[string]string{"t.jinja": "{% for m in messages %}"}, "t.jinja", "does not compile"},
		{"neither string nor list", map[string]string{"tokenizer_config.json": `{"chat_template": 7}`}, "tokenizer_config.json", "neither a string nor a named list"},
		{"another kind of file", map[string]string{"t.txt": tplA}, "t.txt", "is not a tokenizer_config.json"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := writeFiles(t, c.files)
			_, err := ChatTemplatesFrom(filepath.Join(dir, c.arg))
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), c.say) {
				t.Errorf("error %q does not say %q", err, c.say)
			}
		})
	}
	if _, err := ChatTemplatesFrom(filepath.Join(t.TempDir(), "nothere.json")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("a missing file gave %v, which does not wrap fs.ErrNotExist", err)
	}
	// And the conversion stops before writing anything.
	dir := writeFiles(t, map[string]string{"t.jinja": "{% if %}"})
	dst := filepath.Join(dir, "out.jlm")
	if _, err := FromGGUF(stories260KGGUF, dst, jlm.Fingerprint{}, WithChatTemplate(filepath.Join(dir, "t.jinja"))); err == nil {
		t.Fatal("a GGUF converted with a template that does not compile")
	}
	if _, err := os.Stat(dst); err == nil {
		t.Error("a refused conversion left a container behind")
	}
}

// The stored template is what the container reports, and the option replaces
// the source's rather than adding to it.
func TestConvertStoresTheChatTemplate(t *testing.T) {
	named := `{"chat_template": [{"name": "default", "template": "` + jsonEsc(tplA) +
		`"}, {"name": "tool_use", "template": "` + jsonEsc(tplB) + `"}]}`
	dir := writeFiles(t, map[string]string{"tokenizer_config.json": named})
	read := func(opts ...Option) []jlm.ChatTemplate {
		t.Helper()
		dst := filepath.Join(t.TempDir(), "out.jlm")
		if _, err := FromGGUF(stories260KGGUF, dst, jlm.Fingerprint{}, opts...); err != nil {
			t.Fatal(err)
		}
		c, err := jlm.Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		return c.Vocab().Templates
	}
	if got := read(); len(got) != 0 {
		t.Fatalf("the fixture already carries %d template(s); this gate would prove nothing", len(got))
	}
	want := []jlm.ChatTemplate{{Name: "default", Body: tplA}, {Name: "tool_use", Body: tplB}}
	got := read(WithChatTemplate(filepath.Join(dir, "tokenizer_config.json")))
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stored %+v, want %+v", got, want)
	}
	v := jlm.Vocab{Templates: got}
	if b, _ := v.Template(""); b != tplA {
		t.Errorf("the default is %q, want the one named default", b)
	}
	if b, _ := v.Template("tool_use"); b != tplB {
		t.Errorf("tool_use is %q", b)
	}
	// "default" is chosen by name, not by sorting first.
	v = jlm.Vocab{Templates: []jlm.ChatTemplate{{Name: "agent", Body: tplB}, {Name: "default", Body: tplA}}}
	if b, _ := v.Template(""); b != tplA {
		t.Errorf("the default of [agent default] is %q, want the one named default", b)
	}

	// The safetensors path takes the option too, over the directory's own.
	hfDirPath := testmodels.Path("qwen3")
	if _, err := os.Stat(filepath.Join(hfDirPath, "config.json")); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- the safetensors half proved nothing", err)
	} else {
		dst := filepath.Join(t.TempDir(), "st.jlm")
		if _, err := FromSafetensors(hfDirPath, dst, jlm.Fingerprint{},
			WithChatTemplate(filepath.Join(dir, "tokenizer_config.json"))); err != nil {
			t.Fatal(err)
		}
		c, err := jlm.Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(c.Vocab().Templates, want) {
			t.Errorf("safetensors stored %+v, want %+v", c.Vocab().Templates, want)
		}
		c.Close()
	}

	// A GGUF that has one is replaced, not appended to.
	one := writeFiles(t, map[string]string{"x.jinja": tplC})
	src := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	f, err := gguf.Open(src)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- the replacement half proved nothing", err)
	}
	s, err := sourceOf(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Vocab.Templates) == 0 {
		t.Fatal("Llama-3.2-1B-Instruct's GGUF carries no template; the replacement half proved nothing")
	}
	o := options{chatTemplate: filepath.Join(one, "x.jinja")}
	if err := o.applyChatTemplate(s); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s.Vocab.Templates, []jlm.ChatTemplate{{Body: tplC}}) {
		t.Errorf("after the option: %+v", s.Vocab.Templates)
	}
}

// A GGUF's named templates are stored as the tokenizer_config list they came
// from is: llama.cpp's converter writes the default to tokenizer.chat_template
// and each other one to tokenizer.chat_template.<name>.
func TestGGUFNamedTemplatesMatchTheList(t *testing.T) {
	kv := map[string]meta.Value{
		"tokenizer.chat_template":          meta.MakeString(tplA),
		"tokenizer.chat_template.tool_use": meta.MakeString(tplB),
		"tokenizer.chat_template.rag":      meta.MakeString(tplC),
		"tokenizer.chat_templates":         meta.MakeStrings([]string{"tool_use", "rag"}),
	}
	got := ggufTemplates(kv)
	dir := writeFiles(t, map[string]string{"tokenizer_config.json": `{"chat_template": [` +
		`{"name": "tool_use", "template": "` + jsonEsc(tplB) + `"},` +
		`{"name": "default", "template": "` + jsonEsc(tplA) + `"},` +
		`{"name": "rag", "template": "` + jsonEsc(tplC) + `"}]}`})
	want, err := ChatTemplatesFrom(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("gguf %+v, list %+v", got, want)
	}
	// A single template stays unnamed, as every container before named
	// templates were read has it.
	if got := ggufTemplates(map[string]meta.Value{"tokenizer.chat_template": meta.MakeString(tplA)}); !reflect.DeepEqual(got, []jlm.ChatTemplate{{Body: tplA}}) {
		t.Errorf("one template: %+v", got)
	}
	if got := ggufTemplates(map[string]meta.Value{}); got != nil {
		t.Errorf("no template: %+v", got)
	}

	// A real GGUF written by llama.cpp's converter from a named list.
	src := testmodels.Path("synth-commandr.gguf")
	f, err := gguf.Open(src)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- the real-file half proved nothing", err)
	}
	defer f.Close()
	var names []string
	for _, x := range ggufTemplates(f.KV) {
		names = append(names, x.Name)
	}
	if len(names) < 2 || names[0] != "default" {
		t.Errorf("synth-commandr's templates: %v, want default and its named ones", names)
	}
	t.Logf("synth-commandr carries %v", names)
}

func jsonEsc(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return r.Replace(s)
}

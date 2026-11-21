package convert

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// WithChatTemplate stores the chat template(s) path carries in place of the
// source's own.
//
// A GGUF's tokenizer.chat_template is whatever its converter copied when it
// ran, and publishers revise theirs: Mistral-7B-Instruct-v0.3's GGUF has no
// system role, where Mistral's tokenizer_config.json puts the system text
// before the last user turn. The publisher's template is the authority
// (RULE 7m), and the caller choosing one outranks both.
//
// path is a tokenizer_config.json, a chat_template.json, a .jinja file, or a
// HuggingFace model directory (its tokenizer_config.json). Every form is read
// by hfTemplates, the safetensors path's reader, so a GGUF given its model's
// tokenizer_config.json stores the list that model's safetensors conversion
// stores. A file that carries no template, or one that does not compile, is
// refused.
func WithChatTemplate(path string) Option {
	return func(o *options) { o.chatTemplate = path }
}

// ChatTemplatesFrom reads the chat templates of path, in any form
// WithChatTemplate takes.
func ChatTemplatesFrom(path string) ([]jlm.ChatTemplate, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("convert: chat template: %w", err)
	}
	var ts []jlm.ChatTemplate
	switch base := filepath.Base(path); {
	case fi.IsDir():
		// The model directory: what the safetensors path reads, the same way.
		dir := dirFiles(path)
		tc, err := readHFTokenizerConfig(dir)
		if err != nil {
			return nil, err
		}
		if ts, err = hfTemplates(dir, tc.ChatTemplate); err != nil {
			return nil, err
		}
	case strings.HasSuffix(base, ".jinja"):
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("convert: chat template: %w", err)
		}
		if len(b) > 0 {
			ts = []jlm.ChatTemplate{{Body: string(b)}}
		}
	case strings.HasSuffix(base, ".json"):
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("convert: chat template: %w", err)
		}
		// tokenizer_config.json and chat_template.json both keep it under
		// chat_template; a repo that moved it to chat_template.jinja leaves
		// the key out, and hfTemplates looks beside the file for that.
		var tc hfTokenizerConfig
		if err := json.Unmarshal(b, &tc); err != nil {
			return nil, fmt.Errorf("convert: chat template: %s: %w", path, err)
		}
		if ts, err = hfTemplates(dirFiles(filepath.Dir(path)), tc.ChatTemplate); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("convert: chat template: %s is not a tokenizer_config.json, "+
			"a chat_template.json, a .jinja file or a model directory", path)
	}
	if len(ts) == 0 {
		return nil, fmt.Errorf("convert: chat template: %s carries no chat template", path)
	}
	for _, t := range ts {
		if strings.TrimSpace(t.Body) == "" {
			return nil, fmt.Errorf("convert: chat template: %s: template %q is empty", path, t.Name)
		}
		// Refused here, where there is a file in front of a person, rather
		// than on the first chat request against the container.
		if _, err := jinja.Compile(t.Body); err != nil {
			return nil, fmt.Errorf("convert: chat template: %s: template %q does not compile: %w",
				path, t.Name, err)
		}
	}
	return ts, nil
}

// applyChatTemplate replaces s's templates with the option's, when one was
// given.
func (o *options) applyChatTemplate(s *jlm.Source) error {
	if o.chatTemplate == "" {
		return nil
	}
	ts, err := ChatTemplatesFrom(o.chatTemplate)
	if err != nil {
		return err
	}
	if s.Vocab == nil {
		return fmt.Errorf("convert: chat template: this model carries no vocabulary to keep one with")
	}
	s.Vocab.Templates = ts
	return nil
}

// ggufTemplates reads the chat templates a GGUF carries: a
// single template under tokenizer.chat_template, or, from a tokenizer_config
// list, the "default" entry there and every other one under
// tokenizer.chat_template.<name>. A named set is stored as hfTemplates stores
// the list it came from -- the default under its name, sorted -- so a model
// converted from its GGUF and from its directory carries the same templates.
func ggufTemplates(kv map[string]meta.Value) []jlm.ChatTemplate {
	const key = "tokenizer.chat_template"
	var out []jlm.ChatTemplate
	for k, v := range kv {
		name, ok := strings.CutPrefix(k, key+".")
		if !ok || name == "" {
			continue
		}
		if t, ok := v.String(); ok && t != "" {
			out = append(out, jlm.ChatTemplate{Name: name, Body: t})
		}
	}
	t, ok := kv[key].String()
	switch {
	case len(out) == 0:
		if ok && t != "" {
			return []jlm.ChatTemplate{{Body: t}}
		}
		return nil
	case ok && t != "":
		out = append(out, jlm.ChatTemplate{Name: "default", Body: t})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

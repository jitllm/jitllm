package convert

import (
	"embed"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// stateToolForm states how the model writes a tool call: the form of the
// template a request with tools renders (the one named tool_use, else the
// default), read from that template and the vocabulary (ToolFormOf), the
// form added to taken where that is not nil.
func stateToolForm(s *jlm.Source, taken map[string]bool) {
	v := s.Vocab
	if v == nil || len(v.Templates) == 0 {
		return
	}
	src, ok := v.Template("tool_use")
	if !ok {
		src, _ = v.Template("")
	}
	v.Tools = ToolFormOf(src, v.Tokens)
	if v.Tools.Name != "" {
		take(taken, "tool form "+v.Tools.Name)
	}
}

//go:embed toolforms.json
var toolFormsFS embed.FS

// toolForms is toolforms.json: each tool-call form, when it applies, and the
// form a template that matches none is read in.
type toolForms struct {
	Default string `json:"default"`
	Forms   []struct {
		Name string `json:"name"`
		// When is the form's tests, any one of which picks it: every one of
		// All in the template and none of None, or every one of Vocab among
		// the vocabulary's tokens.
		When []struct {
			All, None, Vocab []string
		} `json:"when"`
		Form jlm.ToolForm `json:"form"`
	} `json:"forms"`
}

func readToolForms() (*toolForms, error) {
	b, err := toolFormsFS.ReadFile("toolforms.json")
	if err != nil {
		return nil, err
	}
	var t toolForms
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, fmt.Errorf("convert: toolforms.json: %w", err)
	}
	return &t, nil
}

// ToolFormOf is the tool-call form a chat template trains its model to write
// a call in, read from the markup the template writes or describes and the
// vocabulary's tokens: the first of toolforms.json's forms whose test holds,
// the file's default where none does. The order is the file's: a template
// naming several markers is read as the form whose marker is most specific.
func ToolFormOf(template string, vocab []string) jlm.ToolForm {
	t, err := readToolForms()
	if err != nil {
		return jlm.ToolForm{}
	}
	var tokens map[string]bool
	holds := func(all, none, inVocab []string) bool {
		for _, x := range all {
			if !strings.Contains(template, x) {
				return false
			}
		}
		for _, x := range none {
			if strings.Contains(template, x) {
				return false
			}
		}
		if len(inVocab) == 0 {
			return true
		}
		if tokens == nil {
			tokens = make(map[string]bool, len(vocab))
			for _, tok := range vocab {
				tokens[tok] = true
			}
		}
		return !slices.ContainsFunc(inVocab, func(x string) bool { return !tokens[x] })
	}
	for _, f := range t.Forms {
		for _, w := range f.When {
			if holds(w.All, w.None, w.Vocab) {
				return f.Form
			}
		}
	}
	return named(t, t.Default)
}

// DefaultToolForm is the form a reply is read in where no model states one:
// toolforms.json's default.
func DefaultToolForm() jlm.ToolForm {
	t, err := readToolForms()
	if err != nil {
		return jlm.ToolForm{}
	}
	return named(t, t.Default)
}

func named(t *toolForms, name string) jlm.ToolForm {
	for _, f := range t.Forms {
		if f.Name == name {
			return f.Form
		}
	}
	return jlm.ToolForm{}
}

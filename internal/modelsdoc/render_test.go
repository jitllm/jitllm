package modelsdoc

import (
	"maps"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
)

// TestEveryNameNeedsALine runs the page against its violations: a name the
// code has and the library does not describe, and a line for a name the code
// lacks.
// Each must refuse to render, naming the culprit.
func TestEveryNameNeedsALine(t *testing.T) {
	if _, err := Render(); err != nil {
		t.Fatalf("the shipped lines do not render: %v", err)
	}
	cases := []struct {
		name string
		edit func(*lines)
		want string
	}{
		{"a GGUF name with no line", func(l *lines) { delete(l.gguf, "qwen3next") }, "qwen3next"},
		{"a class with no line", func(l *lines) { delete(l.hf, "KimiLinearForCausalLM") }, "KimiLinearForCausalLM"},
		{"a projector with no line", func(l *lines) { delete(l.projector, "kimivl") }, "kimivl"},
		{"an empty line", func(l *lines) { l.gguf["llama"] = " " }, "llama"},
		{"a line naming nothing", func(l *lines) { l.gguf["llama9"] = "Llama 9" }, "llama9"},
	}
	for _, c := range cases {
		d, derr := convert.Descriptions()
		if derr != nil {
			t.Fatal(derr)
		}
		l := lines{gguf: maps.Clone(d.GGUF), hf: maps.Clone(d.Classes), projector: maps.Clone(d.Projectors), audio: maps.Clone(d.Audio)}
		c.edit(&l)
		_, err := render(l)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: render gave %v, want an error naming %s", c.name, err, c.want)
		}
	}
}

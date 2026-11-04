package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/tok/jinja"
)

// TestImagePartsReachABothShapeTemplate: a template that reads `content` as
// a string or as typed parts (Gemma 3's shape) is handed a message carrying
// an image as parts, so it writes its image marker, while a text-only
// message still renders as the string it is; a template that reads only a
// string carries the tower's mark before the text for each image, and with
// no mark refuses the image rather than dropping it.
func TestImagePartsReachABothShapeTemplate(t *testing.T) {
	const both = `{%- for m in messages -%}<turn>{%- if m['content'] is string -%}{{ m['content'] }}` +
		`{%- else -%}{%- for p in m['content'] -%}{%- if p['type'] == 'image' -%}<img>` +
		`{%- elif p['type'] == 'text' -%}{{ p['text'] }}{%- endif -%}{%- endfor -%}{%- endif -%}</turn>{%- endfor -%}`
	const str = `{%- for m in messages -%}<turn>{{ m['content'] }}</turn>{%- endfor -%}`
	render := func(src string, images int, mark string) (string, error) {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatal(err)
		}
		return renderCompiled(tpl, src, "", "", nil, []ChatMessage{{Role: "user", Content: "hi", Images: images}},
			nil, ToolChoice{}, false, true, mark)
	}
	for _, c := range []struct {
		src    string
		images int
		mark   string
		want   string
	}{
		{both, 1, "", "<turn><img>hi</turn>"},
		{both, 1, "<image>\n", "<turn><img>hi</turn>"},
		{both, 0, "", "<turn>hi</turn>"},
		{str, 0, "", "<turn>hi</turn>"},
		{str, 2, "<image>\n", "<turn><image>\n<image>\nhi</turn>"},
	} {
		if got, err := render(c.src, c.images, c.mark); err != nil || got != c.want {
			t.Errorf("%d image(s), mark %q: rendered %q, %v; want %q", c.images, c.mark, got, err, c.want)
		}
	}
	if _, err := render(str, 1, ""); err == nil || !strings.Contains(err.Error(), "reads a message as a string") {
		t.Errorf("a string-only template took an image: %v", err)
	}
}

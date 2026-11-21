package jinja_test

import (
	"testing"

	"github.com/jitllm/jitllm/tok/jinja"
)

// TestFormatFilter holds format to Python's % operator, as jinja2 applies it;
// each want is what transformers' environment renders.
func TestFormatFilter(t *testing.T) {
	for src, want := range map[string]string{
		`{{ '%s|%5.2f|%-4d|%r|%%' | format('a', 3.14159, 7, 'q') }}`: `a| 3.14|7   |'q'|%`,
		`{{ '%(a)s-%(b)s' | format(a=1, b=none) }}`:                  `1-None`,
		`{{ '%s' | format([1, 'x', none, true]) }}`:                  `[1, 'x', None, True]`,
	} {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		got, err := tpl.Render(nil)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		if got != want {
			t.Errorf("%s: got %q, want %q", src, got, want)
		}
	}
}

// TestMembershipIsEquality: in compares a list's items as == does, so a
// float is in a list of the integer it equals; each want is jinja2's.
func TestMembershipIsEquality(t *testing.T) {
	for src, want := range map[string]string{
		`{{ 1.0 in [0, 1] }}`:     `True`,
		`{{ 1 in [0.0, 1.0] }}`:   `True`,
		`{{ 2.0 not in [0, 1] }}`: `True`,
		`{{ 0.5 in [0, 1] }}`:     `False`,
		`{{ 'a' in ['a', 1] }}`:   `True`,
	} {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		got, err := tpl.Render(nil)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		if got != want {
			t.Errorf("%s: got %q, want %q", src, got, want)
		}
	}
}

// TestReadingAnUndefinedValueFails: an attribute, item or slice of an
// undefined value raises in jinja2, where a missing key of a defined value is
// undefined -- so the first three fail and the last two render.
func TestReadingAnUndefinedValueFails(t *testing.T) {
	for src, fails := range map[string]bool{
		`{{ x.y is defined }}`: true,
		`{{ x[0] }}`:           true,
		`{{ x[1:] }}`:          true,
		`{{ (x or {}).y }}`:    false,
		`{{ m.a is defined }}`: false,
	} {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		_, err = tpl.Render(map[string]any{"m": map[string]any{}})
		if (err != nil) != fails {
			t.Errorf("%s: error %v, want failure %v", src, err, fails)
		}
	}
}

// TestBlockTrimmingTouchesOnlyAdjacentText: trim_blocks and lstrip_blocks
// strip the text directly beside a block tag and nothing past an expression,
// so the newline after {{ d }} is output. Command-R's rag template writes each
// document's number that way. Each want is what transformers renders.
func TestBlockTrimmingTouchesOnlyAdjacentText(t *testing.T) {
	for src, want := range map[string]string{
		"{% for d in [1, 2] %}{{ d }}\n{% if true %}x{% endif %}\n{% endfor %}": "1\nx2\nx",
		"a {{ 'b' }}  {% if true %}c{% endif %}":                                "a b  c",
	} {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatalf("%q: %v", src, err)
		}
		got, err := tpl.Render(nil)
		if err != nil {
			t.Fatalf("%q: %v", src, err)
		}
		if got != want {
			t.Errorf("%q: got %q, want %q", src, got, want)
		}
	}
}

// TestIteratingAScalarFails: a for loop over None or a number is a TypeError in
// Python, and over an undefined value it is empty.
func TestIteratingAScalarFails(t *testing.T) {
	for src, fails := range map[string]bool{
		`{% for x in n %}{{ x }}{% endfor %}`: true,
		`{% for x in 3 %}{% endfor %}`:        true,
		`{% for x in u %}{{ x }}{% endfor %}`: false,
	} {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		out, err := tpl.Render(map[string]any{"n": nil})
		if (err != nil) != fails || out != "" {
			t.Errorf("%s: %q, error %v, want failure %v", src, out, err, fails)
		}
	}
}

// TestPlusJoinsOnlyStrings: + between a string and anything else is a
// TypeError in Python, where ~ writes both sides as text.
func TestPlusJoinsOnlyStrings(t *testing.T) {
	for src, fails := range map[string]bool{
		`{{ 'a' + 'b' }}`:  false,
		`{{ 'a' ~ d }}`:    false,
		`{{ 1 + 2.5 }}`:    false,
		`{{ 'a' + d }}`:    true,
		`{{ 'a' + 1 }}`:    true,
		`{{ none + 'a' }}`: true,
	} {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		_, err = tpl.Render(map[string]any{"d": map[string]any{"k": 1}})
		if (err != nil) != fails {
			t.Errorf("%s: error %v, want failure %v", src, err, fails)
		}
	}
}

// TestFloatsPrintAsPythonDoes: a float is written as Python's repr wherever a
// template turns it into text -- output, a container's repr, tojson, ~, string
// and format -- so 18.0 is "18.0", not "18". Each want is what transformers'
// environment renders with x = 18.0 and y the JSON {"t": 18.0, "n": 1e5}.
func TestFloatsPrintAsPythonDoes(t *testing.T) {
	y, err := jinja.FromJSON([]byte(`{"t": 18.0, "n": 1e5}`))
	if err != nil {
		t.Fatal(err)
	}
	data := map[string]any{"x": 18.0, "y": y}
	for src, want := range map[string]string{
		`{{ 18.0 }}`:                    `18.0`,
		`{{ x }}`:                       `18.0`,
		`{{ [x, 2.5, 1e16, 1.5e-07] }}`: `[18.0, 2.5, 1e+16, 1.5e-07]`,
		`{{ x|tojson }}`:                `18.0`,
		`{{ {'a': x}|tojson }}`:         `{"a": 18.0}`,
		`{{ 4 / 2 }}`:                   `2.0`,
		`{{ 7 / 2 }}`:                   `3.5`,
		`{{ 2.5|round }}`:               `2.0`,
		`{{ 'v=' ~ x }}`:                `v=18.0`,
		`{{ x|string }}`:                `18.0`,
		`{{ '%s' | format(x) }}`:        `18.0`,
		`{{ 0.1 + 0.2 }}`:               `0.30000000000000004`,
		`{{ -0.0 }}`:                    `-0.0`,
		`{{ 1000000000000000.0 }}`:      `1000000000000000.0`,
		`{{ 0.00001 }}`:                 `1e-05`,
		`{{ 3 * 1.0 }}`:                 `3.0`,
		`{{ y }}`:                       `{'t': 18.0, 'n': 100000.0}`,
		`{{ y|tojson }}`:                `{"t": 18.0, "n": 100000.0}`,
	} {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		got, err := tpl.Render(data)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		if got != want {
			t.Errorf("%s: got %q, want %q", src, got, want)
		}
	}
}

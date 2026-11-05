package grammar

import "testing"

// textMatcher is src over a vocabulary of nothing: driven by AcceptText.
func textMatcher(t *testing.T, src string) *Matcher {
	t.Helper()
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("%v\n%s", err, src)
	}
	return NewMatcher(g, IndexTokens(oneToken{}))
}

type oneToken struct{}

func (oneToken) Size() int                     { return 1 }
func (oneToken) Piece(id int32) (string, bool) { return "x", true }
func (oneToken) IsEOG(id int32) bool           { return false }

func takes(m *Matcher, s string) bool {
	st, ok := m.AcceptText(m.Start(), s)
	return ok && m.Done(st)
}

// TestUntilStopsAtItsMarker: Until's text runs to the marker and never
// through it, including a marker that starts with a character the text has
// just used ("</p" inside "a</pb</p>").
func TestUntilStopsAtItsMarker(t *testing.T) {
	b := NewBuilder()
	u := b.Until("v", "</p>")
	m := textMatcher(t, b.Text(Literal("<p>")+" "+u+" "+Literal("</p>")))
	for s, want := range map[string]bool{
		"<p>hello</p>":         true,
		"<p></p>":              true,
		"<p>a < b</p>":         true,
		"<p>a</pb and</p>":     true,
		"<p>x</p>y</p>":        false, // the text cannot run through the marker
		"<p>unclosed":          false,
		"<p>Zürich, 3°</p>":    true,
		"<p>line\nline</p>":    true,
		"<p>a<</p>":            false, // the narrower language: '<' then the marker
		"<p>tail <x</p>":       true,
		"<p>half </ p</p>":     true,
		"<p>full </p></p>":     false,
		"<p>nested <p></p>":    true,
		"<p>bracket ]-^\\</p>": true,
	} {
		if got := takes(m, s); got != want {
			t.Errorf("%q: accepted %v, want %v", s, got, want)
		}
	}
}

// TestBuilderComposesSchemas: two schemas in one grammar, each with its own
// $defs, behind literal wrappers; Properties gives an object's arguments one
// by one in the schema's order.
func TestBuilderComposesSchemas(t *testing.T) {
	b := NewBuilder()
	a, err := b.Schema([]byte(`{"type":"object","properties":{"n":{"$ref":"#/$defs/x"}},"required":["n"],"$defs":{"x":{"type":"integer"}}}`), "a")
	if err != nil {
		t.Fatal(err)
	}
	c, err := b.Schema([]byte(`{"type":"object","properties":{"s":{"$ref":"#/$defs/x"}},"required":["s"],"$defs":{"x":{"type":"string"}}}`), "c")
	if err != nil {
		t.Fatal(err)
	}
	m := textMatcher(t, b.Text("("+Literal("A")+" "+a+" | "+Literal("C")+" "+c+")"))
	for s, want := range map[string]bool{
		`A{"n": 3}`:   true,
		`C{"s": "x"}`: true,
		`A{"n": "x"}`: false,
		`C{"s": 3}`:   false,
	} {
		if got := takes(m, s); got != want {
			t.Errorf("%q: accepted %v, want %v", s, got, want)
		}
	}
	ps, err := b.Properties([]byte(`{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"integer"}},"required":["a"]}`), "p")
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 2 || ps[0].Key != "z" || ps[0].Type != "string" || ps[0].Required ||
		ps[1].Key != "a" || ps[1].Type != "integer" || !ps[1].Required {
		t.Fatalf("properties %+v", ps)
	}
}

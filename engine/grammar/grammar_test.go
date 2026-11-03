package grammar

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/schemacheck"
)

// toyVocab is a vocabulary of every single byte, a few multi-byte pieces and
// one end-of-generation token, the last.
type toyVocab struct{ pieces []string }

func newToyVocab(extra ...string) *toyVocab {
	v := &toyVocab{}
	for b := range 256 {
		v.pieces = append(v.pieces, string([]byte{byte(b)}))
	}
	v.pieces = append(v.pieces, extra...)
	v.pieces = append(v.pieces, "") // the end token
	return v
}

func (v *toyVocab) Size() int { return len(v.pieces) }
func (v *toyVocab) Piece(id int32) (string, bool) {
	if int(id) >= len(v.pieces)-1 {
		return "", false
	}
	return v.pieces[id], true
}
func (v *toyVocab) IsEOG(id int32) bool { return int(id) == len(v.pieces)-1 }
func (v *toyVocab) eog() int32          { return int32(len(v.pieces) - 1) }

func mustParse(t *testing.T, src string) *Grammar {
	t.Helper()
	g, err := Parse(src)
	if err != nil {
		t.Fatalf("%v\n%s", err, src)
	}
	return g
}

// accepts is whether the whole of s is a sentence of the grammar.
func accepts(m *Matcher, s string) bool {
	st, ok := m.AcceptText(m.Start(), s)
	return ok && m.Done(st)
}

func TestGBNFAcceptsItsLanguage(t *testing.T) {
	v := newToyVocab()
	cases := []struct {
		src     string
		yes, no []string
	}{
		{`root ::= "a" | "bc"`, []string{"a", "bc"}, []string{"", "b", "abc", "ac"}},
		{`root ::= [a-c]+ "!"`, []string{"a!", "abcab!"}, []string{"!", "ad!", "a"}},
		{`root ::= x{2,3}
x ::= "ab"`, []string{"abab", "ababab"}, []string{"ab", "abababab"}},
		{`root ::= [^"]* "\""`, []string{`"`, `héllo"`}, []string{`a"b"`}},
		{`root ::= ("(" root ")")?`, []string{"", "()", "(())"}, []string{"(", "())"}},
		{`root ::= "é" [一-鿿] "\x41"`, []string{"é中A"}, []string{"e中A", "éaA"}},
		{"root ::= a |\n  b\na ::= \"x\" # a comment\nb ::= \"y\"", []string{"x", "y"}, []string{"xy"}},
		{`root ::= . . `, []string{"ab", "日本"}, []string{"a"}},
	}
	for _, c := range cases {
		m := NewMatcher(mustParse(t, c.src), IndexTokens(v))
		for _, s := range c.yes {
			if !accepts(m, s) {
				t.Errorf("%q refuses %q", c.src, s)
			}
		}
		for _, s := range c.no {
			if accepts(m, s) {
				t.Errorf("%q accepts %q", c.src, s)
			}
		}
	}
}

func TestGBNFRefusesWhatItCannotRun(t *testing.T) {
	for _, src := range []string{
		`root ::= root "a" | "b"`,
		`root ::= x
x ::= y? root`,
		`root ::= undefined`,
		`notroot ::= "a"`,
		`root ::= "a`,
		`root ::= [b-a]`,
		`root ::= "a"{3,1}`,
	} {
		if _, err := Parse(src); err == nil {
			t.Errorf("%q compiled", src)
		}
	}
}

// TestTokenBoundaries: a token that spans two terminals, a code point split
// across byte tokens, and a partial code point the grammar can never finish.
func TestTokenBoundaries(t *testing.T) {
	v := newToyVocab(`":`, `"a":`, "\xc3", "é", `{"`)
	id := func(p string) int32 { return int32(slices.Index(v.pieces, p)) }
	m := NewMatcher(mustParse(t, `root ::= "{" "\"" [a-z]+ "\"" ":" "é" "}"`), IndexTokens(v))
	st := m.Start()
	step := func(p string, want bool) {
		t.Helper()
		toks, _ := m.Allowed(st)
		_, inMask := slices.BinarySearch(toks, id(p))
		next, ok := m.Accept(st, id(p))
		if ok != want || inMask != want {
			t.Fatalf("token %q at %v: accepted %v, in the mask %v, want %v", p, st, ok, inMask, want)
		}
		if ok {
			st = next
		}
	}
	step(`"a":`, false)
	step(`{"`, true) // two terminals in one token
	step(`":`, false)
	step("a", true)
	step(`"a":`, false)
	step(`":`, true) // the string's end and the colon
	step("\xc3", true)
	step("\xa8", false) // è: the code point finishes as one the grammar refuses
	step("\xa9", true)  // é, its second byte in a token of its own
	step("}", true)
	if !m.Done(st) || !m.Final(st) {
		t.Fatal("the sentence is complete and nothing more is allowed")
	}
	if _, ok := m.Accept(st, v.eog()); !ok {
		t.Fatal("the end token is refused at the end")
	}
	// A lead byte whose every completion is refused is not in the mask.
	m2 := NewMatcher(mustParse(t, `root ::= [a-z]`), IndexTokens(v))
	toks, _ := m2.Allowed(m2.Start())
	if _, in := slices.BinarySearch(toks, id("\xc3")); in {
		t.Fatal("a lead byte nothing can complete is allowed")
	}
}

func TestMaskSendsTheRestToMinusInfinity(t *testing.T) {
	v := newToyVocab()
	m := NewMatcher(mustParse(t, `root ::= [ab] "c"?`), IndexTokens(v))
	lg := make([]float32, v.Size()+3) // a padded head
	for i := range lg {
		lg[i] = float32(i % 7)
	}
	want := slices.Clone(lg)
	st := m.Start()
	m.Mask(st, lg)
	for i, x := range lg {
		ok := i == 'a' || i == 'b'
		if ok && x != want[i] || !ok && !math.IsInf(float64(x), -1) {
			t.Fatalf("logit %d is %v after the mask (was %v)", i, x, want[i])
		}
	}
	st, _ = m.Accept(st, 'a')
	copy(lg, want)
	m.Mask(st, lg)
	for i, x := range lg {
		ok := i == 'c' || int32(i) == v.eog()
		if ok == math.IsInf(float64(x), -1) {
			t.Fatalf("after a: logit %d is %v", i, x)
		}
	}
	if a := testing.AllocsPerRun(50, func() { copy(lg, want); m.Mask(st, lg) }); a != 0 {
		t.Fatalf("a mask of a visited state allocates %.0f times", a)
	}
}

// schemaCases are schemas with outputs of bounded length, a document each
// must accept and one each must refuse.
type schemaCase struct {
	schema  string
	yes, no string
}

func schemaCases() []schemaCase {
	return []schemaCase{
		{`{"type":"object","properties":{"name":{"type":"string","maxLength":12},"age":{"type":"integer"},` +
			`"tags":{"type":"array","items":{"enum":["a","b",3,null]},"maxItems":3}},"required":["name","age"]}`,
			`{"name": "bo", "age": -4, "tags": ["a", 3]}`, `{"age": 4, "name": "bo"}`},
		{`{"type":"array","items":{"type":"boolean"},"minItems":1,"maxItems":4}`, `[true,false]`, `[]`},
		{`{"anyOf":[{"type":"number"},{"type":"string","minLength":2,"maxLength":3}]}`, `1.5e3`, `"a"`},
		{`{"$defs":{"pt":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},` +
			`"required":["x"],"additionalProperties":false}},"type":"array","items":{"$ref":"#/$defs/pt"},"maxItems":2}`,
			`[{"x":1,"y":2},{"x":3}]`, `[{"y":2}]`},
		{`{"type":"object","properties":{"a":{"const":"k"},"b":{"type":"null"},"c":{"type":"boolean"}}}`,
			`{"b":null}`, `{"c":true,"a":"k"}`},
		{`{"type":["string","null"],"maxLength":5}`, `null`, `"toolong"`},
		{`{"prefixItems":[{"type":"integer"},{"type":"string","maxLength":2}],"items":false}`, `[1,"ab"]`, `[1]`},
	}
}

func TestSchemaGrammarAcceptsAndRefuses(t *testing.T) {
	v := newToyVocab()
	for _, c := range schemaCases() {
		src, err := FromJSONSchema([]byte(c.schema))
		if err != nil {
			t.Fatalf("%s: %v", c.schema, err)
		}
		m := NewMatcher(mustParse(t, src), IndexTokens(v))
		if !accepts(m, c.yes) {
			t.Errorf("%s refuses %s\n%s", c.schema, c.yes, src)
		}
		if accepts(m, c.no) {
			t.Errorf("%s accepts %s\n%s", c.schema, c.no, src)
		}
	}
}

// TestSchemaRandomWalksValidate draws outputs by choosing uniformly among the
// tokens each state allows -- the masked sampler at its widest -- and holds
// every one to the schema: it parses, and validates.
func TestSchemaRandomWalksValidate(t *testing.T) {
	v := newToyVocab(`": `, `", "`, `{"`, `"}`, "true", "null", ", ", "é", "\xc3", "\xe4\xb8", "\xad")
	rng := rand.New(rand.NewPCG(1, 2))
	for _, c := range schemaCases() {
		src, err := FromJSONSchema([]byte(c.schema))
		if err != nil {
			t.Fatal(err)
		}
		m := NewMatcher(mustParse(t, src), IndexTokens(v))
		var schema any
		if err := json.Unmarshal([]byte(c.schema), &schema); err != nil {
			t.Fatal(err)
		}
		for range 200 {
			out := walk(t, m, v, rng, 4000)
			var doc any
			if err := schemacheck.Parse(out, &doc); err != nil {
				t.Fatalf("%s: a walk wrote %q, which does not parse: %v", c.schema, out, err)
			}
			if err := schemacheck.Validate(schema, schema, doc); err != nil {
				t.Fatalf("%s: a walk wrote %s: %v", c.schema, out, err)
			}
		}
	}
}

// walk draws one output; the end token is taken with even odds against the
// rest where it is allowed.
func walk(t *testing.T, m *Matcher, v *toyVocab, rng *rand.Rand, limit int) string {
	t.Helper()
	st := m.Start()
	var b strings.Builder
	for range limit {
		toks, end := m.Allowed(st)
		if end && (len(toks) == 0 || rng.IntN(2) == 0) {
			return b.String()
		}
		if len(toks) == 0 {
			t.Fatalf("a dead end after %q", b.String())
		}
		id := toks[rng.IntN(len(toks))]
		next, ok := m.Accept(st, id)
		if !ok {
			t.Fatalf("token %q is in the mask and refused by Accept after %q", v.pieces[id], b.String())
		}
		st = next
		b.WriteString(v.pieces[id])
	}
	t.Fatalf("no end after %d tokens: %q", limit, b.String())
	return ""
}

func TestSchemaRefusesByName(t *testing.T) {
	for kw, schema := range map[string]string{
		"pattern":              `{"type":"string","pattern":"^a$"}`,
		"minimum":              `{"type":"integer","minimum":3}`,
		"allOf":                `{"allOf":[{"type":"string"}]}`,
		"format":               `{"type":"string","format":"date"}`,
		"$ref":                 `{"$ref":"https://example.com/s.json"}`,
		"additionalProperties": `{"type":"object","properties":{"a":{}},"additionalProperties":true}`,
	} {
		_, err := FromJSONSchema([]byte(schema))
		var u ErrUnsupported
		if !errors.As(err, &u) || !strings.Contains(err.Error(), kw) {
			t.Errorf("%s: %v, want an ErrUnsupported naming %s", schema, err, kw)
		}
	}
}

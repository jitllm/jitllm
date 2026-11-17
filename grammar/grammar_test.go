package grammar

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"regexp"
	"runtime"
	"slices"
	"strconv"
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

func TestParseAcceptsItsLanguage(t *testing.T) {
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

func TestParseRefusesWhatItCannotRun(t *testing.T) {
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

func TestAllowOpensWhatTheGrammarAllows(t *testing.T) {
	v := newToyVocab()
	m := NewMatcher(mustParse(t, `root ::= [ab] "c"?`), IndexTokens(v))
	ninf := float32(math.Inf(-1))
	bias := make([]float32, v.Size()+3) // a padded head
	for i := range bias {
		bias[i] = ninf
	}
	check := func(st State, allowed func(i int) bool, what string) {
		t.Helper()
		if n := m.Allow(st, bias, 0); n == 0 {
			t.Fatalf("%s: nothing allowed", what)
		}
		for i, x := range bias {
			if allowed(i) != (x == 0) || !allowed(i) && x != ninf {
				t.Fatalf("%s: bias %d is %v", what, i, x)
			}
		}
		m.Allow(st, bias, ninf)
		for i, x := range bias {
			if x != ninf {
				t.Fatalf("%s: bias %d is %v after the close", what, i, x)
			}
		}
	}
	st := m.Start()
	check(st, func(i int) bool { return i == 'a' || i == 'b' }, "at the start")
	st, _ = m.Accept(st, 'a')
	check(st, func(i int) bool { return i == 'c' || int32(i) == v.eog() }, "after a")
	if a := testing.AllocsPerRun(50, func() { m.Allow(st, bias, 0); m.Allow(st, bias, ninf) }); a != 0 {
		t.Fatalf("an allow of a visited state allocates %.0f times", a)
	}
	// AllowOn keeps the end out where the grammar may go on, and lets it in
	// where nothing else is allowed.
	m.AllowOn(st, bias, 0)
	for i, x := range bias {
		if (i == 'c') != (x == 0) {
			t.Fatalf("on after a: bias %d is %v", i, x)
		}
	}
	m.AllowOn(st, bias, ninf)
	end, _ := m.Accept(st, 'c')
	m.AllowOn(end, bias, 0)
	for i, x := range bias {
		if (int32(i) == v.eog()) != (x == 0) {
			t.Fatalf("on at the end: bias %d is %v", i, x)
		}
	}
	m.AllowOn(end, bias, ninf)
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
			`{"name": "bo", "age": -4, "tags": ["a", 3]}`, `{"name": "bo"}`},
		{`{"type":"array","items":{"type":"boolean"},"minItems":1,"maxItems":4}`, `[true,false]`, `[]`},
		{`{"anyOf":[{"type":"number"},{"type":"string","minLength":2,"maxLength":3}]}`, `1.5e3`, `"a"`},
		{`{"$defs":{"pt":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},` +
			`"required":["x"],"additionalProperties":false}},"type":"array","items":{"$ref":"#/$defs/pt"},"maxItems":2}`,
			`[{"x":1,"y":2},{"x":3}]`, `[{"y":2}]`},
		{`{"type":"object","properties":{"a":{"const":"k"},"b":{"type":"null"},"c":{"type":"boolean"}}}`,
			`{"b":null}`, `{"a":"j"}`},
		{`{"type":["string","null"],"maxLength":5}`, `null`, `"toolong"`},
		// Each part's additionalProperties holds the keys the other parts
		// name, and a false property forbids its key.
		{`{"allOf":[{"type":"object","properties":{"a":{"type":"string"}}},{"additionalProperties":{"type":"integer"}}]}`, `{"b": 1}`, `{"a": "x"}`},
		{`{"allOf":[{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false},{"properties":{"b":{"type":"string"}}}]}`, `{"a": "x"}`, `{"b": "x"}`},
		{`{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false,"anyOf":[{"required":["a"]},{"properties":{"c":{"type":"null"}}}]}`, `{"a": 1}`, `{"c": null}`},
		{`{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false,"oneOf":[{"required":["a"],"type":"object"},{"properties":{"c":{"type":"null"}},"type":"string"}]}`, `{"a": 1}`, `{"c": null}`},
		{`{"type":"object","properties":{"a":false,"b":{"type":"integer"}}}`, `{"b": 1}`, `{"a": 1}`},
		{`{"type":"string","maxLength":5000}`, `"` + strings.Repeat("x", 5000) + `"`, `"` + strings.Repeat("x", 5001) + `"`},
		{`{"prefixItems":[{"type":"integer"},{"type":"string","maxLength":2}],"items":false}`, `[1]`, `[1,"ab",2]`},
		{`{"prefixItems":[{"type":"integer"},{"type":"boolean"}]}`, `[1,true,"x",{}]`, `[true]`},
		{`{"prefixItems":[{"type":"integer"},{"type":"boolean"}],"minItems":2,"maxItems":3}`, `[1,true,null]`, `[1]`},
		{`{"type":"integer","minimum":-12,"exclusiveMaximum":250}`, `-12`, `250`},
		{`{"type":"array","items":{"type":"integer","exclusiveMinimum":0.5,"maximum":1e3}}`, `[1,1000]`, `[0]`},
		{`{"type":"object","properties":{"d":{"type":"string","format":"date"},"t":{"type":"string","format":"date-time"},` +
			`"c":{"type":"string","format":"time"},"u":{"type":"string","format":"uuid"},"h":{"type":"string","format":"hostname"}},` +
			`"required":["d","t","u"]}`,
			`{"d":"2024-02-28","t":"2024-11-10T15:04:05.123+01:00","u":"123e4567-e89b-12d3-a456-426614174000","h":"x"}`,
			`{"d":"2024-02-30","t":"2024-11-10T15:04:05Z","u":"123e4567-e89b-12d3-a456-426614174000"}`},
		{`{"prefixItems":[{"type":"integer","maximum":3}],"items":{"type":"boolean"},"maxItems":3}`, `[2,true,false]`, `[2,true,false,true]`},
		{`{"type":"object","properties":{"a":{"type":"integer"}},"required":["a","b"]}`, `{"a":1,"b":[1,"x"]}`, `{"a":1}`},
		{`{"type":"object","required":["k"]}`, `{"k":null}`, `{}`},
		{`{"type":"string","pattern":"^[A-Z]{2}-\\d{3}$"}`, `"AB-123"`, `"AB-12"`},
		{`{"type":"string","pattern":"^(cat|dog)s?$"}`, `"dogs"`, `"hot dogs"`},
		{`{"type":"string","pattern":"^[^a-z\"]+$"}`, `"AB 1"`, `"Ab"`},
		{`{"type":"string","pattern":"^a.c$"}`, `"a\u0007c"`, `"a\nc"`},
		{`{"allOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},` +
			`{"properties":{"b":{"type":"string","maxLength":3}},"required":["b"]}]}`, `{"a":1,"b":"x"}`, `{"a":1}`},
		{`{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":{"type":"boolean"}}`,
			`{"a":1,"x":true}`, `{"a":1,"x":2}`},
		{`{"type":"object","properties":{"k":{"type":"null"}},"additionalProperties":true,"required":["k"]}`,
			`{"k":null,"z":[1]}`, `{"z":1}`},
		// RFC 8259 and 2020-12 where a schema states no bound: absent
		// additionalProperties takes other keys, numerals and whitespace
		// runs of any length, a type-less schema's keywords hold only their
		// own type.
		{`{"type":"object","properties":{"a":{"type":"integer"}}}`, `{"a":1,"b":"x"}`, `{"a":"x"}`},
		{`{"type":"integer","minimum":5}`, `123456789012345678901234567890`, `4`},
		{`{"type":"integer","maximum":-3}`, `-123456789012345678901234567890`, `-2`},
		{`{"type":"number"}`, `-0.123456789012345678901234567890e0010`, `01`},
		{`{"type":"array","items":{"type":"integer"}}`, "\r\n\t[ 1 ,\n" + strings.Repeat("\t", 16) + "2\r\n]", `[1,]`},
		{`{"properties":{"a":{"type":"integer"}},"minLength":2}`, `5`, `{"a":"x"}`},
		{`{"const":{"a":[1,"<"]}}`, `{ "a" : [ 1 , "<" ] }`, `{"a":["<",1]}`},
		{`{"type":"string","format":"date-time"}`, `"2024-02-29t23:59:59.123456789012z"`, `"2023-02-29T00:00:00Z"`},
		{`{"type":"string","format":"date"}`, `"2000-02-29"`, `"1900-02-29"`},
		{`{"type":"string","minLength":2}`, `"` + esc("d83d") + esc("de00") + esc("0041") + `"`, `"` + esc("d83d") + esc("de00") + `"`},
		{`{"type":"string","maxLength":1}`, `"` + esc("D83D") + esc("DE00") + `"`, `"` + esc("de00") + `"`},
		{`{"type":"string","maxLength":1}`, "\"\x7f\"", `"\ud800"`},
		// Every keyword of a schema holds at once: the ones beside an enum,
		// a const, an anyOf and a oneOf, and every part of an allOf.
		{`{"type":"string","enum":["a",1,"b",null]}`, `"b"`, `1`},
		{`{"type":"integer","enum":[1.0,1.5,"1"]}`, `1.0`, `1.5`},
		{`{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"additionalProperties":false,` +
			`"anyOf":[{"required":["a"]},{"required":["b"]}]}`, `{"b": "x"}`, `{}`},
		{`{"type":"string","anyOf":[{"type":"integer"},{"maxLength":2}]}`, `"ab"`, `1`},
		{`{"type":["string","null"],"oneOf":[{"type":"string","maxLength":1},{"type":"integer"},{"type":"null"}]}`, `null`, `"ab"`},
		{`{"allOf":[{"type":["integer","string"]},{"type":"number"}]}`, `-7`, `"a"`},
		// A oneOf whose branches no value satisfies two of: by type, by
		// value, and by a required property's value.
		{`{"oneOf":[{"type":"integer"},{"type":"string","maxLength":2}]}`, `-3`, `"abc"`},
		{`{"oneOf":[{"const":1},{"enum":[2,"1"]}]}`, `"1"`, `3`},
		{`{"type":"object","required":["kind"],"oneOf":[` +
			`{"properties":{"kind":{"const":"a"},"n":{"type":"integer"}},"additionalProperties":false},` +
			`{"properties":{"kind":{"const":"b"},"s":{"type":"string"}},"additionalProperties":false}]}`,
			`{"kind": "b", "s": "x"}`, `{"kind": "a", "s": "x"}`},
	}
}

// esc is JSON's escape of the UTF-16 code unit of hex digits h.
func esc(h string) string { return `\` + "u" + h }

// TestWhitespaceIsHeldToALine: a gap between two tokens is nothing, a space,
// or a line break and up to maxIndent spaces and tabs, and no more -- the
// bound schema.go's head measures.
func TestWhitespaceIsHeldToALine(t *testing.T) {
	src, err := FromJSONSchema([]byte(`{"type":"array","items":{"type":"integer"}}`))
	if err != nil {
		t.Fatal(err)
	}
	m := NewMatcher(mustParse(t, src), IndexTokens(newToyVocab()))
	run := func(n int) string { return strings.Repeat(" \t", n)[:n] }
	for s, want := range map[string]bool{
		"[1, 2]":                            true,
		"[1,2]":                             true,
		"[1,\r\n" + run(maxIndent) + "2\n]": true,
		"\n" + run(maxIndent) + "[1]":       true,
		"[1,\n" + run(maxIndent+1) + "2]":   false,
		"[1,  2]":                           false,
		"[1, \n2]":                          false,
		"[1,\t2]":                           false,
		"[1,\n\n2]":                         false,
		"[1,\r\r\n2]":                       false,
		"  [1]":                             false,
	} {
		if accepts(m, s) != want {
			t.Errorf("%q taken %v, want %v", s, !want, want)
		}
	}
}

// TestTheReplyEndsWithItsValue: whitespace before a value is taken, none
// after it, so the grammar is done once the value closes.
func TestTheReplyEndsWithItsValue(t *testing.T) {
	v := newToyVocab()
	for _, schema := range []string{`{"type":"object"}`, `{"type":"integer","minimum":3}`, `{"enum":["a",1]}`} {
		src, err := FromJSONSchema([]byte(schema))
		if err != nil {
			t.Fatal(err)
		}
		m := NewMatcher(mustParse(t, src), IndexTokens(v))
		doc := map[string]string{`{"type":"object"}`: `{"k": [1, 2]}`, `{"type":"integer","minimum":3}`: `40`, `{"enum":["a",1]}`: `"a"`}[schema]
		st, ok := m.AcceptText(m.Start(), "\n \t"+doc)
		if !ok || !m.Done(st) {
			t.Fatalf("%s: %q after leading whitespace refused", schema, doc)
		}
		if toks, end := m.Allowed(st); end && len(toks) > 0 && schema != `{"type":"integer","minimum":3}` {
			t.Errorf("%s: %d tokens allowed after the value closed", schema, len(toks))
		}
		if _, ok := m.AcceptText(st, " "); ok {
			t.Errorf("%s: whitespace after the value was taken", schema)
		}
	}
	m := NewMatcher(mustParse(t, JSONObject()), IndexTokens(v))
	if !accepts(m, "\n{ }") || accepts(m, "{} ") {
		t.Error("json_object: leading whitespace refused or trailing taken")
	}
}

// TestAdditionalKeysRepeatNoProperty holds the keys an object writes beside
// its properties to the names none of them has, decoded: a listed name is
// refused there however it is spelled.
func TestAdditionalKeysRepeatNoProperty(t *testing.T) {
	src, err := FromJSONSchema([]byte(`{"type":"object","properties":{"ab":{"type":"integer"},"a":{"type":"integer"}},` +
		`"additionalProperties":{"type":"boolean"}}`))
	if err != nil {
		t.Fatal(err)
	}
	m := NewMatcher(mustParse(t, src), IndexTokens(newToyVocab()))
	for s, want := range map[string]bool{
		`{"ab":1}`: true, `{"abc":true}`: true, `{"b":true}`: true, `{"":true}`: true, `{"a":1,"x":false,"y":true}`: true,
		`{"a":true}`: false, `{"ab":true}`: false, `{"ab":1,"ab":true}`: false, `{"\u0061":true}`: false,
		`{"a\u0062":true}`: false, `{"x":true,"a":1}`: false,
	} {
		if accepts(m, s) != want {
			t.Errorf("%s taken %v, want %v", s, !want, want)
		}
	}
}

// TestPatternsMatchTheirLanguage holds a pattern's grammar to Go's regexp,
// which agrees with ECMA-262 on every pattern and alphabet here: every
// string of up to four characters from the alphabet, written as JSON writes
// it, is taken exactly where the regexp finds a match in it.
func TestPatternsMatchTheirLanguage(t *testing.T) {
	v := newToyVocab()
	alphabet := []rune{'a', 'b', 'c', 'A', '-', '1', '"', '\\', ' ', '\n', 'é'}
	var strs []string
	var grow func(prefix []rune)
	grow = func(prefix []rune) {
		strs = append(strs, string(prefix))
		if len(prefix) == 4 {
			return
		}
		for _, r := range alphabet {
			grow(append(slices.Clone(prefix), r))
		}
	}
	grow(nil)
	for _, pt := range []string{`^a*b$`, `ab|c`, `^(a|b)+$`, `[^a]`, `^[a-c]{2,3}$`, `^\d+-?$`, `"`, `^\\`, `a.c`, `^$`,
		`^(?:ab)?c$`, `[\w-]+`, `^[^\n]*$`, `é`, `^[\s"]{2}`, `^a|b$`, `^(a|)c$`, `[^-a-c1]$`, `\x41`, `^\u00e9?a`,
		`^\p{L}+$`, `^\p{Lu}`, `[\p{N}\p{Pd}]`, `^\P{L}*$`, `^\p{Script=Latin}+$`, `\p{gc=Nd}`, `^[^\p{Ll}\s]+$`,
		`^\u{e9}+$`, `[\u{41}-\u{43}]`} {
		// Go's regexp spells ECMA-262's \u escape \x{}, and a script by its
		// name alone.
		goRe := pt
		if g, ok := map[string]string{`^\u00e9?a`: `^\x{e9}?a`, `^\p{Script=Latin}+$`: `^\p{Latin}+$`,
			`\p{gc=Nd}`: `\p{Nd}`,
			`^\u{e9}+$`: `^\x{e9}+$`, `[\u{41}-\u{43}]`: `[\x{41}-\x{43}]`}[pt]; ok {
			goRe = g
		}
		re := regexp.MustCompile(goRe)
		schema, err := json.Marshal(map[string]any{"type": "string", "pattern": pt})
		if err != nil {
			t.Fatal(err)
		}
		src, err := FromJSONSchema(schema)
		if err != nil {
			t.Fatalf("%s: %v", pt, err)
		}
		m := NewMatcher(mustParse(t, src), IndexTokens(v))
		for _, s := range strs {
			var b strings.Builder
			e := json.NewEncoder(&b)
			e.SetEscapeHTML(false)
			if err := e.Encode(s); err != nil {
				t.Fatal(err)
			}
			if got, want := accepts(m, strings.TrimSuffix(b.String(), "\n")), re.MatchString(s); got != want {
				t.Fatalf("pattern %s: %q taken %v, the regexp %v\n%s", pt, s, got, want, src)
			}
		}
	}
}

// TestIntegerRangesAreExact holds an integer's bounds to the integers they
// admit, each one written as RFC 8259's int writes it: every integer around
// each range is taken exactly where it lies within, across lengths and
// signs, 0 as -0 too; a side no keyword bounds has no end.
func TestIntegerRangesAreExact(t *testing.T) {
	v := newToyVocab()
	for _, rg := range [][2]int{{0, 0}, {0, 9}, {7, 7}, {10, 19}, {5, 123}, {-17, 4}, {-250, -3}, {99, 1000},
		{-1000, -999}, {0, 1000}, {-1, 1}, {101, 1099}, {-9001, 87}} {
		src, err := FromJSONSchema([]byte(fmt.Sprintf(`{"type":"integer","minimum":%d,"maximum":%d}`, rg[0], rg[1])))
		if err != nil {
			t.Fatal(err)
		}
		m := NewMatcher(mustParse(t, src), IndexTokens(v))
		for x := rg[0] - 120; x <= rg[1]+120; x++ {
			if got, want := accepts(m, strconv.Itoa(x)), rg[0] <= x && x <= rg[1]; got != want {
				t.Fatalf("[%d, %d]: %d accepted %v\n%s", rg[0], rg[1], x, got, src)
			}
		}
		if accepts(m, "0"+strconv.Itoa(rg[1])) || accepts(m, "-00") {
			t.Fatalf("[%d, %d]: a leading zero is taken", rg[0], rg[1])
		}
		if got, want := accepts(m, "-0"), rg[0] <= 0 && 0 <= rg[1]; got != want {
			t.Fatalf("[%d, %d]: -0 accepted %v", rg[0], rg[1], got)
		}
	}
	long := strings.Repeat("9", 40)
	for schema, cases := range map[string]map[string]bool{
		`{"type":"integer","exclusiveMinimum":4}`: {"4": false, "5": true, "9999999999999999": true, "10000000000000000": true,
			long: true, "1" + long: true, "-5": false, "-0": false, "05": false},
		`{"type":"integer","maximum":-12}`: {"-12": true, "-11": false, "-" + long: true, "-100": true, "0": false, "-0": false},
		`{"type":"integer","minimum":-3}`:  {"-3": true, "-4": false, "-0": true, "0": true, long: true},
		`{"type":"integer"}`:               {"-" + long: true, long: true, "-0": true, "1.0": false, "1e2": false},
	} {
		src, err := FromJSONSchema([]byte(schema))
		if err != nil {
			t.Fatal(err)
		}
		m := NewMatcher(mustParse(t, src), IndexTokens(v))
		for s, want := range cases {
			if accepts(m, s) != want {
				t.Errorf("%s: %s accepted %v", schema, s, !want)
			}
		}
	}
	if _, err := FromJSONSchema([]byte(`{"type":"integer","minimum":3.5,"maximum":3.9}`)); err == nil {
		t.Error("a range no integer lies in was built")
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
			out := walk(t, m, v, rng, 20000)
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
		"lookaround":                       `{"type":"string","pattern":"^(?=a)a$"}`,
		"backreference":                    `{"type":"string","pattern":"(a)\\1"}`,
		"word boundary":                    `{"type":"string","pattern":"\\bx"}`,
		"minimum":                          `{"type":"number","minimum":3}`,
		"allOf":                            `{"allOf":[{"type":"string","maxLength":2},{"maxLength":3}]}`,
		"uniqueItems":                      `{"type":"array","uniqueItems":true}`,
		"$ref":                             `{"$ref":"https://example.com/s.json"}`,
		"maxLength":                        `{"type":"string","maxLength":2097153}`,
		"minItems":                         `{"type":"array","minItems":1e9}`,
		"enum beside maxLength":            `{"enum":["a"],"maxLength":3}`,
		"const beside properties":          `{"const":{},"properties":{}}`,
		"share no type":                    `{"allOf":[{"type":"string"},{"type":"integer"}]}`,
		"anyOf whose every branch":         `{"type":"string","anyOf":[{"type":"integer"},false]}`,
		"enum none of whose values":        `{"type":"boolean","enum":[0,"true"]}`,
		"folded into itself":               `{"$defs":{"a":{"allOf":[{"$ref":"#/$defs/a"}]}},"$ref":"#/$defs/a"}`,
		"into itself":                      `{"$defs":{"a":{"type":"object","anyOf":[{"$ref":"#/$defs/a"}]}},"$ref":"#/$defs/a"}`,
		"oneOf whose branches may overlap": `{"oneOf":[{"type":"integer"},{"type":"number"}]}`,
		"may overlap":                      `{"type":"object","oneOf":[{"properties":{"k":{"const":1.0}},"required":["k"]},{"properties":{"k":{"enum":[10e-1,2]}},"required":["k"]}]}`,
		"overlap":                          `{"oneOf":[{"type":"object","properties":{"k":{"const":1}},"required":["k"]},{"type":"object","properties":{"k":{"const":2}}}]}`,
	} {
		_, err := FromJSONSchema([]byte(schema))
		var u ErrUnsupported
		if !errors.As(err, &u) || !strings.Contains(err.Error(), kw) {
			t.Errorf("%s: %v, want an ErrUnsupported naming %s", schema, err, kw)
		}
	}
}

// nestedSchema is depth objects, each holding the next as property x beside
// an anyOf of the branch maps fold: each branch folds the keywords beside it
// in, the property among them.
func nestedSchema(depth int, branches string) string {
	s := `{"type":"integer"}`
	for range depth {
		s = `{"type":"object","properties":{"x":` + s + `},"anyOf":` + branches + `}`
	}
	return s
}

// schemaSpent is the work FromJSONSchema does on schema: the rule text it
// writes and the visits it makes, as maxSchemaText counts them.
func schemaSpent(t *testing.T, schema string) (int, error) {
	t.Helper()
	d := json.NewDecoder(strings.NewReader(schema))
	d.UseNumber()
	root, err := decodeOrdered(d)
	if err != nil {
		t.Fatal(err)
	}
	c := newConverter(root)
	_, err = c.visit(root, "schema")
	return c.spent, err
}

// TestSchemaCompileIsBounded: folding the keywords beside an anyOf into each
// branch builds what they hold once, not once per branch at every level, and
// a fold that would still multiply the text level by level is refused by name
// once it passes maxSchemaText, its work held there. The work is counted, not
// timed: the text and visits are the time and the memory.
func TestSchemaCompileIsBounded(t *testing.T) {
	required := `[{"required":["x"]},{"required":["y"]}]`
	at := map[int]int{}
	for _, depth := range []int{10, 20, 60} {
		spent, err := schemaSpent(t, nestedSchema(depth, required))
		if err != nil {
			t.Fatalf("depth %d: %v", depth, err)
		}
		at[depth] = spent
	}
	// A rule's name carries its path, so the text grows as the square of
	// the depth: twice that is the bound, where a fold per level doubles it.
	if at[20] > 2*4*at[10] || at[60] > 2*9*at[20] {
		t.Errorf("the work grows faster than the square of the nesting: %v", at)
	}
	src, err := FromJSONSchema([]byte(nestedSchema(60, required)))
	if err != nil {
		t.Fatal(err)
	}
	if len(src) > 1<<20 {
		t.Errorf("depth 60 compiles to %d bytes", len(src))
	}
	v := newToyVocab()
	m := NewMatcher(mustParse(t, src), IndexTokens(v))
	if !accepts(m, `{"x": {"x": {"y": 1}}}`) || accepts(m, `{"x": {"x": {"z": 1}}}`) {
		t.Errorf("depth 60's grammar does not hold each level to its anyOf")
	}

	// Two branches each closing the object differently: each folds the
	// property below with its own additionalProperties, a schema of its own
	// at every level.
	closing := `[{"additionalProperties":{"type":"object"}},{"additionalProperties":{"type":"object","required":["z"]}}]`
	spent, err := schemaSpent(t, nestedSchema(40, closing))
	var u ErrUnsupported
	if !errors.As(err, &u) || !strings.Contains(err.Error(), "passes") {
		t.Fatalf("depth 40, closing branches: %v, want an ErrUnsupported naming the bound", err)
	}
	if spent > maxSchemaText+1<<16 {
		t.Errorf("refused after %d bytes of work, past the %d bound", spent, maxSchemaText)
	}
	if _, err := FromJSONSchema([]byte(nestedSchema(3, closing))); err != nil {
		t.Errorf("depth 3, closing branches: %v", err)
	}
}

// TestParseHoldsItsSize: a grammar's counted repetitions are written out, so
// one past maxPositions is refused by name before it is (a repetition that
// alone would pass it allocates nothing of itself), the repetitions of one
// grammar counted together, and one within them parses.
func TestParseHoldsItsSize(t *testing.T) {
	lit := `"` + strings.Repeat("a", 600) + `"`
	for _, c := range []struct {
		src   string
		alone bool
	}{
		{`root ::= ` + lit + `{4096}`, true},
		{`root ::= "a"{0,2000000}`, true},
		{`root ::= ` + lit + `{1800} ` + lit + `{1800}`, false},
	} {
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		_, err := Parse(c.src)
		runtime.ReadMemStats(&after)
		if err == nil || !strings.Contains(err.Error(), "positions") {
			t.Errorf("%.40q: %v, want a refusal naming the positions", c.src, err)
		}
		if n := after.TotalAlloc - before.TotalAlloc; c.alone && n > 1<<20 {
			t.Errorf("%.40q: refused after allocating %d bytes", c.src, n)
		}
	}
	g, err := Parse(`root ::= "a"{5000} ` + lit + `{100}`)
	if err != nil {
		t.Fatal(err)
	}
	if !accepts(NewMatcher(g, IndexTokens(newToyVocab())), strings.Repeat("a", 5000+600*100)) {
		t.Error("a grammar within its positions refuses its sentence")
	}
}

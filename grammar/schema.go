package grammar

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"reflect"
	"slices"
	"strings"
)

// JSON Schema 2020-12 to grammar rules over RFC 8259's JSON text: the
// keywords below, each read as the specification reads it, and every other
// validation keyword refused by name (ErrUnsupported) rather than ignored,
// since ignoring one would let through output the schema forbids.
//
// Supported: type (one or a list), properties (a false one forbidding its
// key), required, additionalProperties (false, true or a schema: beside
// properties, the keys none of them name; absent, any key and value, as
// 2020-12 reads it), allOf over object schemas (schemaobj.go; each part's
// additionalProperties holds the keys the other parts name), items,
// prefixItems (followed by items or not), minItems, maxItems, minLength,
// maxLength, an integer's minimum, maximum, exclusiveMinimum and
// exclusiveMaximum, format (date, time and date-time as RFC 3339 writes them,
// uuid as RFC 4122 does; any other is an annotation, as 2020-12 reads it),
// pattern (pattern.go), enum and const (beside a type and each other; beside
// any other keyword refused by name), anyOf and oneOf (each branch held to
// the keywords beside it, a branch nothing then satisfies left out), $ref
// into #/$defs or #/definitions, and the annotations. A key required and
// given no schema is written with any value.
//
// No value is bounded where the schema bounds none: a string's length, a
// numeral's digits, an array's items and an object's keys. What the rules do
// not write, and why:
//
//   - A gap between two tokens is nothing, one space, or one line break and
//     at most maxIndent spaces and tabs of indentation: RFC 8259's ws, held to
//     the shapes a JSON writer formats with. It bounds formatting, never a
//     value. Unbounded, a sampled reply does not end; the measurement is in
//     docs/engineering-history/model-correctness.md, "grammar".
//   - An object's properties come in the schema's order, the keys none of
//     them names after them: a grammar admitting every order of n keys, each
//     once, needs a rule for each of their 2^n subsets.
//   - An integer is written as RFC 8259's int, with no fraction or exponent.
//     2020-12 counts 1.0 and 1.5e1 as integers too, but which numerals with
//     a fraction and an exponent are integral compares the exponent's value
//     with the fraction's length, which no context-free grammar holds. A
//     schema with bounds and no type writes its numbers as integers.
//   - A oneOf is written only where no value can satisfy two of its
//     branches (schemaobj.go, disjoint), and refused by name otherwise: the
//     union of the branches cannot exclude a value two of them both match.
//   - A number's minimum and maximum are refused by name, not built yet: the
//     numerals without an exponent between two bounds are a regular language,
//     as an integer range is (docs/open-questions.md).
//   - The reply ends where its value does: RFC 8259's JSON-text allows
//     whitespace after the value, which adds nothing to it, and ending at the
//     value is what lets a constrained reply end without the model choosing
//     to (a base model rarely ends its own text). The ws before the value is
//     a gap like any other.
//   - A \u escape names a Unicode scalar value, a surrogate pair together:
//     a lone surrogate's escape decodes to no character (RFC 8259, 8.2), and
//     a pair counts as one character against minLength and maxLength.
//   - A date-time's second is 00 to 59: a leap second's 60 is valid only at
//     the instants the IERS has announced, a table rather than a rule.
//   - A schema is sized by its request, so its grammar has a ceiling: rule
//     text past maxSchemaText is refused by name as it is written, and a
//     grammar past maxPositions (bnf.go) once its counted repetitions are.

// ErrUnsupported is a schema keyword or form the converter does not build.
type ErrUnsupported struct{ What string }

func (e ErrUnsupported) Error() string {
	return fmt.Sprintf("grammar: JSON Schema %s is not supported", e.What)
}

// noValue is a schema nothing satisfies: refused alone, as an
// ErrUnsupported, and left out as one branch of an anyOf or oneOf.
type noValue struct{ ErrUnsupported }

func (e noValue) Unwrap() error { return e.ErrUnsupported }

func nothing(what string) error {
	return noValue{ErrUnsupported{what + ", which nothing satisfies"}}
}

// maxIndent bounds the spaces and tabs after a gap's line break (the head of
// this file says why).
const maxIndent = 16

// maxSchemaText is the rule text a schema may compile to, and visitCost what
// each schema visited counts for against it, so a fold that writes nothing (a
// branch left out) is held to it too. Folding the keywords beside an anyOf
// into each branch multiplies the text with each level that nests one where
// the branches change what the level below reads; past this the schema is
// refused by name rather than its text built.
const (
	maxSchemaText = 4 << 20
	visitCost     = 64
)

// The primitives: RFC 8259's JSON grammar, each value written bare and its
// ws standing beside the structural characters inside it (sep); "" for a
// name that is not one.
func primitive(n string) string {
	switch n {
	case "space":
		return fmt.Sprintf(`( " " | "\r"? "\n" [ \t]{0,%d} )?`, maxIndent)
	case "boolean":
		return `"true" | "false"`
	case "null":
		return `"null"`
	case "integral-part":
		return `"0" | [1-9] [0-9]*`
	case "digits":
		return `[0-9]+`
	case "integer":
		return `"-"? integral-part`
	case "number":
		return `"-"? integral-part ( "." digits )? ( [eE] [-+]? digits )?`
	case "char":
		return `[^"\\\x00-\x1F] | [\\] ( ["\\/bfnrt] | "u" ( [0-9a-cA-Ce-fE-F] [0-9a-fA-F]{3} | [dD] [0-7] [0-9a-fA-F]{2} | ` +
			`[dD] [89abAB] [0-9a-fA-F]{2} [\\] "u" [dD] [c-fC-F] [0-9a-fA-F]{2} ) )`
	case "string":
		return `"\"" char* "\""`
	case "value":
		return `object | array | string | number | boolean | null`
	case "member":
		return `string space ":" space value`
	case "object":
		return `"{" space ( member ( space "," space member )* space )? "}"`
	case "array":
		return `"[" space ( value ( space "," space value )* space )? "]"`
	}
	return ""
}

// primDeps is what each primitive refers to.
func primDeps(n string) []string {
	switch n {
	case "integer":
		return []string{"integral-part"}
	case "number":
		return []string{"integral-part", "digits"}
	case "string":
		return []string{"char"}
	case "value":
		return []string{"object", "array", "string", "number", "boolean", "null"}
	case "member":
		return []string{"string", "space", "value"}
	case "object":
		return []string{"member", "space"}
	case "array":
		return []string{"value", "space"}
	}
	return nil
}

// sep is a separator between two tokens of a value: the structural
// character s with RFC 8259's ws on each side.
func (c *converter) sep(s string) string {
	sp := c.prim("space")
	return sp + " " + literal(s) + " " + sp
}

// list is items written in brackets open and close: RFC 8259's ws after
// open and before close, and between items their separator. body is the
// items and their separators, empty for none; empty says whether the list
// may hold none at all.
func (c *converter) list(open, close, body string, empty bool) string {
	sp := c.prim("space")
	if body == "" {
		return literal(open) + " " + sp + " " + literal(close)
	}
	inner := body + " " + sp
	if empty {
		inner = "( " + inner + " )?"
	}
	return literal(open) + " " + sp + " " + inner + " " + literal(close)
}

// JSONObject is the grammar of any JSON object, after RFC 8259's leading
// ws: the OpenAI API's json_object.
func JSONObject() string {
	c := newConverter(nil)
	c.rules["root"] = c.prim("space") + " " + c.prim("object")
	c.order = append(c.order, "root")
	return c.text()
}

// FromJSONSchema compiles a JSON Schema to the grammar text Parse reads: its
// sentences are the values valid under the schema, after RFC 8259's leading
// ws.
func FromJSONSchema(schema []byte) (string, error) {
	d := json.NewDecoder(bytes.NewReader(schema))
	d.UseNumber()
	root, err := decodeOrdered(d)
	if err != nil {
		return "", fmt.Errorf("grammar: the schema is not JSON: %v", err)
	}
	if _, err := d.Token(); err == nil {
		return "", fmt.Errorf("grammar: the schema has text after its value")
	}
	c := newConverter(root)
	body, err := c.visit(root, "schema")
	if err != nil {
		return "", err
	}
	c.add("root", c.prim("space")+" "+body)
	return c.text(), nil
}

type converter struct {
	root  any
	rules map[string]string
	order []string
	refs  map[string]string // $ref -> rule name
	// merging are the $refs folded into the value being built (mergeAll).
	merging []string
	// built is each schema object visited at a value of its own, by
	// identity: a fold reaches the keywords beside an anyOf once per branch,
	// and building what they hold again each time is exponential in their
	// nesting. The schema is held so that its address stays its own.
	built map[uintptr]builtSchema
	// spent is the rule text written and the visits made (maxSchemaText).
	spent int
}

type builtSchema struct {
	s    map[string]any
	expr string
	err  error
}

func newConverter(root any) *converter {
	return &converter{root: root, rules: map[string]string{}, refs: map[string]string{}, built: map[uintptr]builtSchema{}}
}

func (c *converter) text() string {
	var b strings.Builder
	for _, n := range c.order {
		fmt.Fprintf(&b, "%s ::= %s\n", n, c.rules[n])
	}
	return b.String()
}

func (c *converter) prim(n string) string {
	if _, ok := c.rules[n]; ok {
		return n
	}
	c.rules[n] = primitive(n)
	c.order = append(c.order, n)
	for _, d := range primDeps(n) {
		c.prim(d)
	}
	return n
}

// ruleName is name with every run of characters outside [a-zA-Z0-9-] made one
// dash, and the dashes at its ends trimmed.
func ruleName(name string) string {
	var b strings.Builder
	dash := false
	for i := 0; i < len(name); i++ {
		c := name[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' {
			b.WriteByte(c)
			dash = false
			continue
		}
		if !dash {
			b.WriteByte('-')
			dash = true
		}
	}
	return strings.Trim(b.String(), "-")
}

// add names body: name itself where free, else name with a number.
func (c *converter) add(name, body string) string {
	name = ruleName(name)
	if name == "" {
		name = "r"
	}
	n := name
	for i := 1; ; i++ {
		if _, ok := c.rules[n]; !ok {
			break
		}
		if primitive(n) == "" && c.rules[n] == body {
			return n
		}
		n = fmt.Sprintf("%s-%d", name, i)
	}
	c.rules[n] = body
	c.order = append(c.order, n)
	c.spent += len(n) + len(body)
	return n
}

// literal is the rule text for the exact text s.
func literal(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&b, `\x%02X`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

// jsonLiteral is the rule text for the value v, an object's keys in the
// schema's order and RFC 8259's ws between its tokens.
func (c *converter) jsonLiteral(v any) (string, error) {
	var parts []string
	switch x := v.(type) {
	case map[string]any:
		for _, k := range keysOf(x) {
			kj, err := jsonText(k)
			if err != nil {
				return "", err
			}
			e, err := c.jsonLiteral(x[k])
			if err != nil {
				return "", err
			}
			parts = append(parts, literal(kj)+" "+c.sep(":")+" "+e)
		}
		return c.list("{", "}", strings.Join(parts, " "+c.sep(",")+" "), false), nil
	case []any:
		for _, e := range x {
			lit, err := c.jsonLiteral(e)
			if err != nil {
				return "", err
			}
			parts = append(parts, lit)
		}
		return c.list("[", "]", strings.Join(parts, " "+c.sep(",")+" "), false), nil
	}
	j, err := jsonText(v)
	if err != nil {
		return "", err
	}
	return literal(j), nil
}

// jsonText is the scalar v as JSON writes it: <, > and & bare, which RFC
// 8259 does not escape.
func jsonText(v any) (string, error) {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(v); err != nil {
		return "", err
	}
	return strings.TrimSuffix(b.String(), "\n"), nil
}

// plain is v without the key orders decodeOrdered keeps.
func plain(v any) any {
	switch x := v.(type) {
	case map[string]any:
		m := make(map[string]any, len(x))
		for k, e := range x {
			if k != orderKey {
				m[k] = plain(e)
			}
		}
		return m
	case []any:
		a := make([]any, len(x))
		for i, e := range x {
			a[i] = plain(e)
		}
		return a
	}
	return v
}

// annotations are keywords that say nothing about which values are valid.
func annotation(k string) bool {
	switch k {
	case "title", "description", "default", "examples", "$schema", "$id", "$comment", "deprecated", "readOnly", "writeOnly", "$defs", "definitions",
		"contentEncoding", "contentMediaType", "contentSchema":
		return true
	}
	return false
}

// handled are the validation keywords the converter builds.
func handled(k string) bool {
	switch k {
	case "type", "properties", "required", "additionalProperties", "items", "prefixItems", "minItems", "maxItems", "minLength", "maxLength", "enum", "const", "anyOf", "oneOf", "$ref",
		"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "format", "pattern", "allOf":
		return true
	}
	return false
}

// visit is a rule expression for schema s at a value of its own; name is
// what a rule made for it is called. The $refs folded in around it are not
// held there: a value inside another consumes text before it.
func (c *converter) visit(s any, name string) (string, error) {
	m, isObj := s.(map[string]any)
	var key uintptr
	if isObj {
		key = reflect.ValueOf(m).Pointer()
		if b, ok := c.built[key]; ok {
			return b.expr, b.err
		}
	}
	held := c.merging
	c.merging = nil
	e, err := c.visitInline(s, name)
	c.merging = held
	if isObj {
		c.built[key] = builtSchema{m, e, err}
	}
	return e, err
}

// visitInline is visit where s stands for the value its caller is building
// (a branch, a folded allOf), the $refs folded in so far still held.
func (c *converter) visitInline(s any, name string) (string, error) {
	c.spent += visitCost
	if err := c.within(); err != nil {
		return "", err
	}
	switch v := s.(type) {
	case bool:
		if !v {
			return "", nothing("false")
		}
		return c.prim("value"), nil
	case map[string]any:
		e, err := c.visitObject(v, name)
		if err != nil {
			return "", err
		}
		c.spent += len(e)
		return e, c.within()
	}
	return "", fmt.Errorf("grammar: a schema must be an object or a boolean, not %T", s)
}

// within refuses a schema whose grammar has passed maxSchemaText.
func (c *converter) within() error {
	if c.spent > maxSchemaText {
		return ErrUnsupported{fmt.Sprintf("a schema whose grammar passes %d bytes", maxSchemaText)}
	}
	return nil
}

func (c *converter) visitObject(s map[string]any, name string) (string, error) {
	for _, k := range keysOf(s) {
		if !annotation(k) && !handled(k) {
			return "", ErrUnsupported{fmt.Sprintf("keyword %q", k)}
		}
	}
	if r, ok := s["$ref"]; ok {
		if !onlyAnnotations(s, "$ref") {
			return "", ErrUnsupported{"$ref beside other keywords"}
		}
		ref, _ := r.(string)
		return c.ref(ref)
	}
	if _, ok := s["allOf"]; ok {
		held := len(c.merging)
		merged, refs, err := c.mergeAll(s)
		if err != nil {
			return "", err
		}
		c.merging = append(c.merging, refs...)
		e, err := c.visitObject(merged, name)
		c.merging = c.merging[:held]
		return e, err
	}
	if v, ok := s["const"]; ok {
		return c.values(s, []any{v}, "const")
	}
	if v, ok := s["enum"]; ok {
		vs, ok := v.([]any)
		if !ok || len(vs) == 0 {
			return "", fmt.Errorf("grammar: enum must be a non-empty array")
		}
		return c.values(s, vs, "enum")
	}
	for _, k := range []string{"anyOf", "oneOf"} {
		if _, ok := s[k]; ok {
			return c.branches(s, k, name)
		}
	}
	var types []string
	if t, ok := s["type"]; ok {
		ts, err := typeNames(t)
		if err != nil {
			return "", err
		}
		types = ts
	} else {
		// Every type, each held to the keywords that apply to it: an
		// object's keywords say nothing about a string. A number within
		// bounds is written as an integer there, the bounds built for
		// integers alone.
		if onlyAnnotations(s, "") {
			return c.prim("value"), nil
		}
		num := "number"
		if hasBounds(s) {
			num = "integer"
		}
		types = []string{"object", "array", "string", num, "boolean", "null"}
	}
	if len(types) == 0 {
		return "", nothing("type []")
	}
	alts := make([]string, 0, len(types))
	for _, t := range types {
		e, err := c.typed(s, t, name)
		if err != nil {
			return "", err
		}
		alts = append(alts, e)
	}
	if len(alts) == 1 {
		return alts[0], nil
	}
	return "(" + strings.Join(alts, " | ") + ")", nil
}

// typeNames is a type keyword's names, one or a list.
func typeNames(t any) ([]string, error) {
	switch v := t.(type) {
	case string:
		return []string{v}, nil
	case []any:
		ns := make([]string, 0, len(v))
		for _, x := range v {
			n, ok := x.(string)
			if !ok {
				return nil, fmt.Errorf("grammar: type lists names")
			}
			ns = append(ns, n)
		}
		return ns, nil
	}
	return nil, fmt.Errorf("grammar: type is a name or a list of names")
}

// isType is whether the decoded value v is of the JSON type t: an integer
// is a number whose value is integral, 1.0 among them.
func isType(v any, t string) bool {
	switch x := v.(type) {
	case map[string]any:
		return t == "object"
	case []any:
		return t == "array"
	case string:
		return t == "string"
	case bool:
		return t == "boolean"
	case nil:
		return t == "null"
	case json.Number:
		if t != "integer" {
			return t == "number"
		}
		if !strings.ContainsAny(string(x), ".eE") {
			return true
		}
		r, ok := new(big.Rat).SetString(string(x))
		return ok && r.IsInt()
	}
	return false
}

// values is the rule for the values const or enum (kw) lists, each one
// written only where the keywords beside it admit it: its type and the
// other list. Any other keyword beside them is refused by name.
func (c *converter) values(s map[string]any, vs []any, kw string) (string, error) {
	var alts []string
	for _, x := range vs {
		ok, err := admits(s, x, kw)
		if err != nil {
			return "", err
		}
		if !ok {
			continue
		}
		lit, err := c.jsonLiteral(x)
		if err != nil {
			return "", err
		}
		alts = append(alts, lit)
	}
	if len(alts) == 0 {
		return "", nothing(kw + " none of whose values the type or list beside it admits")
	}
	return "(" + strings.Join(alts, " | ") + ")", nil
}

// admits is whether the value x, listed by keyword kw, satisfies the type,
// const and enum of schema s.
func admits(s map[string]any, x any, kw string) (bool, error) {
	for _, k := range keysOf(s) {
		switch k {
		case "const":
			if !sameJSON(s[k], x) {
				return false, nil
			}
		case "enum":
			vs, ok := s[k].([]any)
			if !ok || len(vs) == 0 {
				return false, fmt.Errorf("grammar: enum must be a non-empty array")
			}
			if !containsJSON(vs, x) {
				return false, nil
			}
		case "type":
			ts, err := typeNames(s[k])
			if err != nil {
				return false, err
			}
			if !slices.ContainsFunc(ts, func(t string) bool { return isType(x, t) }) {
				return false, nil
			}
		default:
			if !annotation(k) {
				return false, ErrUnsupported{fmt.Sprintf("%s beside %s", kw, k)}
			}
		}
	}
	return true, nil
}

// branches is the rule for an anyOf or oneOf (kw): each branch held to the
// keywords beside it as well, since every keyword of a schema holds at once,
// and a branch nothing then satisfies left out. A oneOf is written as the
// union of its branches only where no two of them can hold one value.
func (c *converter) branches(s map[string]any, kw, name string) (string, error) {
	vs, ok := s[kw].([]any)
	if !ok || len(vs) == 0 {
		return "", fmt.Errorf("grammar: %s must be a non-empty array", kw)
	}
	rest := without(s, kw)
	alone := onlyAnnotations(rest, "")
	var alts []string
	var kept []any
	for i, x := range vs {
		b := x
		if !alone {
			b = map[string]any{"allOf": []any{rest, x}, orderKey: []string{"allOf"}}
		}
		e, err := c.visitInline(b, fmt.Sprintf("%s-%d", name, i))
		if errors.As(err, new(noValue)) {
			continue
		}
		if err != nil {
			return "", err
		}
		alts = append(alts, e)
		kept = append(kept, b)
	}
	if len(alts) == 0 {
		return "", nothing(kw + " whose every branch the keywords beside it exclude")
	}
	if kw == "oneOf" {
		if err := c.disjoint(kept); err != nil {
			return "", err
		}
	}
	return "(" + strings.Join(alts, " | ") + ")", nil
}

// without is schema s less keyword k.
func without(s map[string]any, k string) map[string]any {
	out := make(map[string]any, len(s))
	var order []string
	for _, x := range keysOf(s) {
		if x != k {
			out[x] = s[x]
			order = append(order, x)
		}
	}
	out[orderKey] = order
	return out
}

func onlyAnnotations(s map[string]any, except string) bool {
	for _, k := range keysOf(s) {
		if k != except && !annotation(k) {
			return false
		}
	}
	return true
}

func (c *converter) typed(s map[string]any, t, name string) (string, error) {
	switch t {
	case "boolean", "null":
		return c.prim(t), nil
	case "integer":
		if !hasBounds(s) {
			return c.prim(t), nil
		}
		lo, hi, some, err := intBounds(s)
		if err != nil {
			return "", err
		}
		if !some {
			return "", nothing(fmt.Sprintf("an integer range from %v to %v", lo, hi))
		}
		return c.add(name, intRange(lo, hi)), nil
	case "number":
		if hasBounds(s) {
			return "", ErrUnsupported{"minimum, maximum and their exclusive forms on a number (not built yet)"}
		}
		return c.prim(t), nil
	case "string":
		lo, hi, err := bounds(s, "minLength", "maxLength")
		if err != nil {
			return "", err
		}
		if pt, ok := s["pattern"]; ok {
			ps, ok := pt.(string)
			if !ok {
				return "", fmt.Errorf("grammar: pattern must be a string")
			}
			if lo != 0 || hi >= 0 || s["format"] != nil {
				return "", ErrUnsupported{"pattern beside minLength, maxLength or format"}
			}
			body, err := c.pattern(ps)
			if err != nil {
				return "", err
			}
			return c.add(name, `"\"" ( `+body+` ) "\""`), nil
		}
		if f, ok := s["format"]; ok {
			fs, ok := f.(string)
			if !ok {
				return "", fmt.Errorf("grammar: format must be a string")
			}
			if body, built := c.formatString(fs); built {
				if lo != 0 || hi >= 0 {
					return "", ErrUnsupported{fmt.Sprintf("format %q beside minLength or maxLength", fs)}
				}
				return c.add(name, `"\"" `+body+` "\""`), nil
			}
		}
		if lo == 0 && hi < 0 {
			return c.prim("string"), nil
		}
		return c.add(name, fmt.Sprintf(`"\"" %s%s "\""`, c.prim("char"), rep(lo, hi))), nil
	case "array":
		return c.array(s, name)
	case "object":
		return c.object(s, name)
	}
	return "", ErrUnsupported{fmt.Sprintf("type %q", t)}
}

// rep is the repetition {lo,hi}, hi < 0 unbounded.
func rep(lo, hi int) string {
	switch {
	case hi < 0 && lo == 0:
		return "*"
	case hi < 0 && lo == 1:
		return "+"
	case hi < 0:
		return fmt.Sprintf("{%d,}", lo)
	case lo == hi:
		return fmt.Sprintf("{%d}", lo)
	}
	return fmt.Sprintf("{%d,%d}", lo, hi)
}

// bounds reads two non-negative integer keywords, 2020-12's integers (a
// zero fraction among them); hi is -1 when absent. A count past maxPositions
// is refused by name: the grammar writes a position for each.
func bounds(s map[string]any, kl, kh string) (lo, hi int, err error) {
	get := func(k string, def int) (int, error) {
		v, ok := s[k]
		if !ok {
			return def, nil
		}
		n, ok := v.(json.Number)
		if !ok {
			return 0, fmt.Errorf("grammar: %s must be a number", k)
		}
		r, ok := new(big.Rat).SetString(string(n))
		if !ok || !r.IsInt() || r.Sign() < 0 {
			return 0, fmt.Errorf("grammar: %s must be a non-negative integer", k)
		}
		if r.Num().Cmp(big.NewInt(maxPositions)) > 0 {
			return 0, ErrUnsupported{fmt.Sprintf("%s %s (past the %d positions a grammar may hold)", k, n, maxPositions)}
		}
		return int(r.Num().Int64()), nil
	}
	if lo, err = get(kl, 0); err != nil {
		return
	}
	if hi, err = get(kh, -1); err != nil {
		return
	}
	if hi >= 0 && hi < lo {
		err = fmt.Errorf("grammar: %s is below %s", kh, kl)
	}
	return
}

// array is an array schema's rule. prefixItems hold the positions an array
// has, each present or not in turn, and items (absent: any value; false:
// none) the positions after them; minItems and maxItems count both.
func (c *converter) array(s map[string]any, name string) (string, error) {
	comma := c.sep(",")
	lo, hi, err := bounds(s, "minItems", "maxItems")
	if err != nil {
		return "", err
	}
	it, hasItems := s["items"]
	var parts []string
	if p, ok := s["prefixItems"]; ok {
		ps, ok := p.([]any)
		if !ok {
			return "", fmt.Errorf("grammar: prefixItems must be an array")
		}
		for i, x := range ps {
			e, err := c.visit(x, fmt.Sprintf("%s-%d", name, i))
			if err != nil {
				return "", err
			}
			parts = append(parts, e)
		}
	}
	item := ""
	if !hasItems || it != false {
		item = c.prim("value")
		if hasItems {
			if item, err = c.visit(it, name+"-item"); err != nil {
				return "", err
			}
		}
	}
	if item == "" && lo > len(parts) {
		return "", nothing(fmt.Sprintf("minItems %d beyond the %d prefixItems items false leaves", lo, len(parts)))
	}
	if hi == 0 {
		return c.add(name, c.list("[", "]", "", false)), nil
	}
	if len(parts) == 0 {
		more := -1
		if hi > 0 {
			more = hi - 1
		}
		body := item
		if more != 0 {
			body += fmt.Sprintf(" ( %s %s )%s", comma, item, rep(max(lo-1, 0), more))
		}
		return c.add(name, c.list("[", "]", body, lo == 0)), nil
	}
	n := len(parts)
	if hi >= 0 && hi < n {
		n = hi
	}
	// from_i is the array from position i on, position i written: the
	// next one required while minItems counts it.
	tail := ""
	if n == len(parts) && item != "" {
		more := -1
		if hi >= 0 {
			more = hi - n
		}
		if more != 0 {
			tail = fmt.Sprintf(" ( %s %s )%s", comma, item, rep(max(lo-n, 0), more))
		}
	}
	from := ""
	for i := n - 1; i >= 0; i-- {
		body := parts[i] + tail
		if i < n-1 {
			if i+1 < lo {
				body = parts[i] + " " + comma + " " + from
			} else {
				body = fmt.Sprintf("%s ( %s %s )?", parts[i], comma, from)
			}
		}
		from = c.add(fmt.Sprintf("%s-from-%d", name, i), body)
	}
	return c.add(name, c.list("[", "]", from, lo == 0)), nil
}

func (c *converter) object(s map[string]any, name string) (string, error) {
	colon, comma := c.sep(":"), c.sep(",")
	props, _ := s["properties"].(map[string]any)
	if s["properties"] != nil && props == nil {
		return "", fmt.Errorf("grammar: properties must be an object")
	}
	addl, hasAddl := s["additionalProperties"]
	if !hasAddl {
		addl = true
	}
	req := map[string]bool{}
	// unlisted are the keys required and given no schema in properties:
	// written with additionalProperties' schema, any value where it gives
	// none.
	var unlisted []string
	if r, ok := s["required"]; ok {
		rs, ok := r.([]any)
		if !ok {
			return "", fmt.Errorf("grammar: required must be an array")
		}
		for _, x := range rs {
			n, ok := x.(string)
			if !ok {
				return "", fmt.Errorf("grammar: required lists names")
			}
			if _, ok := props[n]; !ok && !req[n] {
				if addl == false {
					return "", nothing(fmt.Sprintf("required %q beside additionalProperties false", n))
				}
				unlisted = append(unlisted, n)
			}
			req[n] = true
		}
	}
	if len(keysOf(props)) == 0 && len(unlisted) == 0 {
		switch {
		case addl == true:
			return c.prim("object"), nil
		case addl == false:
			return c.add(name, c.list("{", "}", "", false)), nil
		}
		val, err := c.visit(addl, name+"-value")
		if err != nil {
			return "", err
		}
		kv := c.add(name+"-kv", c.prim("string")+" "+colon+" "+val)
		return c.add(name, c.list("{", "}", fmt.Sprintf("%s ( %s %s )*", kv, comma, kv), true)), nil
	}
	// The properties in the schema's order: a map loses it, so read the
	// keys again from the source where it has one.
	order := append(keysOf(props), unlisted...)
	// more is a key and value beside the properties, a key none of them
	// names: written after them, as many as the output writes.
	more := ""
	if addl != false {
		val, err := c.visit(addl, name+"-more")
		if err != nil {
			return "", err
		}
		key, err := c.keyNot(order, name)
		if err != nil {
			return "", err
		}
		more = c.add(name+"-more-kv", key+" "+colon+" "+val)
	}
	var reqKV, optKV []string
	for _, k := range order {
		ks, ok := props[k]
		if !ok {
			ks = addl
		}
		v, err := c.visit(ks, name+"-"+k)
		if !req[k] && errors.As(err, new(noValue)) {
			continue
		}
		if err != nil {
			return "", err
		}
		kj, err := jsonText(k)
		if err != nil {
			return "", err
		}
		kv := c.add(name+"-"+k+"-kv", literal(kj)+" "+colon+" "+v)
		if req[k] {
			reqKV = append(reqKV, kv)
		} else {
			optKV = append(optKV, kv)
		}
	}
	// The optional ones as a chain: rest_i ::= kv_i ( "," rest_{i+1} )? |
	// rest_{i+1}, any subset in order, commas between.
	rest := ""
	for i := len(optKV) - 1; i >= 0; i-- {
		body := optKV[i]
		if rest != "" {
			body = fmt.Sprintf(`%s ( %s %s )? | %s`, optKV[i], comma, rest, rest)
		}
		rest = c.add(fmt.Sprintf("%s-rest-%d", name, i), body)
	}
	tail := ""
	if more != "" {
		tail = fmt.Sprintf(" ( %s %s )*", comma, more)
	}
	switch {
	case len(reqKV) > 0:
		body := strings.Join(reqKV, " "+comma+" ")
		if rest != "" {
			body += fmt.Sprintf(" ( %s %s )?", comma, rest)
		}
		return c.add(name, c.list("{", "}", body+tail, false)), nil
	case rest != "" && more != "":
		return c.add(name, c.list("{", "}", fmt.Sprintf("( %s | %s )%s", rest, more, tail), true)), nil
	case rest != "":
		return c.add(name, c.list("{", "}", rest, true)), nil
	}
	return c.add(name, c.list("{", "}", more+tail, true)), nil
}

// orderKey holds, in every decoded object, its keys in the order the text
// wrote them: a property order is the order a constrained output writes its
// fields in.
const orderKey = "\x00order"

// keysOf is an object's keys in the text's order.
func keysOf(m map[string]any) []string {
	if ks, ok := m[orderKey].([]string); ok {
		return ks
	}
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	slices.Sort(ks)
	return ks
}

// decodeOrdered decodes one JSON value, numbers as json.Number and every
// object's key order kept under orderKey. A key written twice is refused.
func decodeOrdered(d *json.Decoder) (any, error) {
	t, err := d.Token()
	if err != nil {
		return nil, err
	}
	switch t {
	case json.Delim('{'):
		m := map[string]any{}
		var order []string
		for d.More() {
			kt, err := d.Token()
			if err != nil {
				return nil, err
			}
			k, ok := kt.(string)
			if !ok {
				return nil, fmt.Errorf("an object key that is not a string")
			}
			if _, dup := m[k]; dup || k == orderKey {
				return nil, fmt.Errorf("key %q written twice", k)
			}
			if m[k], err = decodeOrdered(d); err != nil {
				return nil, err
			}
			order = append(order, k)
		}
		if _, err := d.Token(); err != nil {
			return nil, err
		}
		m[orderKey] = order
		return m, nil
	case json.Delim('['):
		a := []any{}
		for d.More() {
			v, err := decodeOrdered(d)
			if err != nil {
				return nil, err
			}
			a = append(a, v)
		}
		if _, err := d.Token(); err != nil {
			return nil, err
		}
		return a, nil
	}
	return t, nil
}

// refTarget is the schema a $ref into the root's $defs or definitions, or
// the root itself, names.
func (c *converter) refTarget(ref string) (any, error) {
	var path string
	switch {
	case strings.HasPrefix(ref, "#/$defs/"):
		path = "$defs"
	case strings.HasPrefix(ref, "#/definitions/"):
		path = "definitions"
	case ref == "#":
		return c.root, nil
	default:
		return nil, ErrUnsupported{fmt.Sprintf("$ref %q (only #, #/$defs/... and #/definitions/...)", ref)}
	}
	defs, _ := c.root.(map[string]any)[path].(map[string]any)
	t, ok := defs[strings.TrimPrefix(ref, "#/"+path+"/")]
	if !ok {
		return nil, fmt.Errorf("grammar: $ref %q names nothing", ref)
	}
	return t, nil
}

// ref is the rule for a $ref into the root's $defs or definitions, made
// once, so a recursive schema refers to itself.
func (c *converter) ref(ref string) (string, error) {
	if n, ok := c.refs[ref]; ok {
		return n, nil
	}
	target, err := c.refTarget(ref)
	if err != nil {
		return "", err
	}
	key := "root-ref"
	if ref != "#" {
		key = ref[strings.LastIndex(ref, "/")+1:]
	}
	// Reserve the name before visiting, for recursion.
	n := c.add("ref-"+key, "")
	c.refs[ref] = n
	body, err := c.visit(target, n)
	if err != nil {
		return "", err
	}
	if body == n {
		return "", fmt.Errorf("grammar: $ref %q refers only to itself", ref)
	}
	c.rules[n] = body
	return n, nil
}

package grammar

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// Builder composes one grammar from literal text and several JSON Schemas:
// a tool call is the model's own wrapper syntax around arguments that match
// the declared tool's schema, and the wrapper differs per model family. The
// schemas are converted exactly as FromJSONSchema converts one; each keeps
// its own $defs.
type Builder struct {
	c *converter
}

// NewBuilder is an empty grammar.
func NewBuilder() *Builder { return &Builder{c: newConverter(nil)} }

// Schema is a rule matching a JSON value valid under schema, named after
// name. An empty or null schema is any JSON object.
func (b *Builder) Schema(schema []byte, name string) (string, error) {
	if s := bytes.TrimSpace(schema); len(s) == 0 || string(s) == "null" {
		return b.c.prim("object"), nil
	}
	d := json.NewDecoder(bytes.NewReader(schema))
	d.UseNumber()
	root, err := decodeOrdered(d)
	if err != nil {
		return "", fmt.Errorf("grammar: the schema is not JSON: %v", err)
	}
	if _, err := d.Token(); err == nil {
		return "", fmt.Errorf("grammar: the schema has text after its value")
	}
	// A $ref resolves against the schema it was written in.
	b.c.root, b.c.refs = root, map[string]string{}
	body, err := b.c.visit(root, name)
	if err != nil {
		return "", err
	}
	return b.c.add(name, body), nil
}

// Prop is one property of an object schema: its key, the rule for its
// value, whether the schema requires it, and its type when the schema names
// one ("" otherwise, or for a list of types).
type Prop struct {
	Key, Rule, Type string
	Required        bool
}

// Properties is an object schema's properties in the order the schema
// writes them, each converted to a value rule named after name: the
// arguments of a family that writes each argument on its own (an XML
// parameter, a Python keyword) rather than as one JSON object. A schema
// with no properties has none.
func (b *Builder) Properties(schema []byte, name string) ([]Prop, error) {
	if s := bytes.TrimSpace(schema); len(s) == 0 || string(s) == "null" {
		return nil, nil
	}
	d := json.NewDecoder(bytes.NewReader(schema))
	d.UseNumber()
	root, err := decodeOrdered(d)
	if err != nil {
		return nil, fmt.Errorf("grammar: the schema is not JSON: %v", err)
	}
	s, ok := root.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("grammar: a parameters schema must be an object")
	}
	props, _ := s["properties"].(map[string]any)
	req := map[string]bool{}
	if rs, ok := s["required"].([]any); ok {
		for _, x := range rs {
			if n, ok := x.(string); ok {
				req[n] = true
			}
		}
	}
	b.c.root, b.c.refs = root, map[string]string{}
	var out []Prop
	for _, k := range keysOf(props) {
		v, err := b.c.visit(props[k], name+"-"+k)
		if err != nil {
			return nil, err
		}
		p := Prop{Key: k, Rule: b.c.add(name+"-"+k, v), Required: req[k]}
		if ps, ok := props[k].(map[string]any); ok {
			p.Type, _ = ps["type"].(string)
		}
		out = append(out, p)
	}
	return out, nil
}

// Value is the rule for any JSON value; Object for any JSON object.
func (b *Builder) Value() string  { return b.c.prim("value") }
func (b *Builder) Object() string { return b.c.prim("object") }

// String is the rule for a JSON string literal; Space the bounded whitespace
// the JSON rules end in.
func (b *Builder) String() string { return b.c.prim("string") }
func (b *Builder) Space() string  { return b.c.prim("space") }

// Rule adds body under name (numbered when taken) and returns the name.
func (b *Builder) Rule(name, body string) string { return b.c.add(name, body) }

// Until is a rule for any text that does not contain lit, named after name.
//
// It is the text-before-a-close-marker of a tool call's string argument or a
// reasoning block. The rule is a regular language over characters: a run of
// characters other than lit's first, or a proper prefix of lit followed by a
// character that neither continues it nor starts it again. So the text may
// not hold lit's first character followed by a proper prefix of lit restarting
// at that point ("<<" before "</x>"): a narrower language than "not
// containing lit", never a wider one, so whatever it allows ends where the
// grammar expects.
func (b *Builder) Until(name, lit string) string {
	rs := []rune(lit)
	if len(rs) == 0 {
		return b.c.add(name, `""`)
	}
	first := rs[0]
	alts := []string{"[^" + classRune(first) + "]"}
	for k := 1; k < len(rs); k++ {
		ex := classRune(first)
		if rs[k] != first {
			ex += classRune(rs[k])
		}
		alts = append(alts, Literal(string(rs[:k]))+" [^"+ex+"]")
	}
	return b.c.add(name, "("+strings.Join(alts, " | ")+")*")
}

// classRune is r as a character-class member.
func classRune(r rune) string {
	switch r {
	case ']', '[', '\\', '^', '-':
		return `\` + string(r)
	case '\n':
		return `\n`
	case '\r':
		return `\r`
	case '\t':
		return `\t`
	}
	if r < 0x20 || r == 0x7F {
		return fmt.Sprintf(`\x%02X`, r)
	}
	return string(r)
}

// Literal is GBNF for the exact text s.
func Literal(s string) string { return literal(s) }

// Text is the grammar whose root is the expression root.
func (b *Builder) Text(root string) string {
	b.c.rules["root"] = root
	b.c.order = append(b.c.order, "root")
	return b.c.text()
}

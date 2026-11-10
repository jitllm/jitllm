package grammar

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// JSON Schema to GBNF: the subset below, each keyword as llama.cpp's
// json-schema-to-grammar writes it, and every other validation keyword
// refused by name (ErrUnsupported) rather than ignored, since ignoring one
// would let through output the schema forbids.
//
// Supported: type (one or a list), properties, required, additionalProperties
// (false, true or a schema: beside properties, the keys none of them name),
// allOf over object schemas (schemaobj.go), items,
// prefixItems (followed by items or not), minItems, maxItems, minLength,
// maxLength, an integer's minimum, maximum, exclusiveMinimum and
// exclusiveMaximum, format (date, time, date-time and uuid; any other is an
// annotation, as JSON Schema 2020-12 reads it), pattern (pattern.go), enum,
// const, anyOf, oneOf (as
// anyOf: the grammar cannot exclude a value two branches both match), $ref
// into #/$defs or #/definitions, and the annotations. An object with
// properties takes no other keys: absent additionalProperties is read as
// false, as OpenAI's strict mode requires, and every output stays valid under
// the schema either way. A key required and given no schema is written with
// any value.

// ErrUnsupported is a schema keyword or form the converter does not build.
type ErrUnsupported struct{ What string }

func (e ErrUnsupported) Error() string {
	return fmt.Sprintf("grammar: JSON Schema %s is not supported", e.What)
}

// The primitives, as llama.cpp's json-schema-to-grammar writes them, the
// whitespace bounded so a constrained model cannot pad forever.
func primitive(n string) (string, bool) {
	switch n {
	case "space":
		return `| " " | "\n" [ \t]{0,20}`, true
	case "boolean":
		return `("true" | "false") space`, true
	case "null":
		return `"null" space`, true
	case "integral-part":
		return `[0] | [1-9] [0-9]{0,15}`, true
	case "decimal-part":
		return `[0-9]{1,16}`, true
	case "integer":
		return `("-"? integral-part) space`, true
	case "number":
		return `("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space`, true
	case "char":
		return `[^"\\\x7F\x00-\x1F] | [\\] (["\\/bfnrt] | "u" [0-9a-fA-F]{4})`, true
	case "string":
		return `"\"" char* "\"" space`, true
	case "value":
		return `object | array | string | number | boolean | null`, true
	case "object":
		return `"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space`, true
	case "array":
		return `"[" space ( value ("," space value)* )? "]" space`, true
	}
	return "", false
}

// primDeps is what each primitive refers to.
func primDeps(n string) ([]string, bool) {
	switch n {
	case "boolean":
		return []string{"space"}, true
	case "null":
		return []string{"space"}, true
	case "integer":
		return []string{"integral-part", "space"}, true
	case "number":
		return []string{"integral-part", "decimal-part", "space"}, true
	case "string":
		return []string{"char", "space"}, true
	case "value":
		return []string{"object", "array", "string", "number", "boolean", "null"}, true
	case "object":
		return []string{"string", "value", "space"}, true
	case "array":
		return []string{"value", "space"}, true
	case "char":
		return nil, true
	case "space":
		return nil, true
	case "integral-part":
		return nil, true
	case "decimal-part":
		return nil, true
	}
	return []string{}, false
}

// JSONObject is the grammar of any JSON object: OpenAI's json_object.
func JSONObject() string {
	c := newConverter(nil)
	c.prim("object")
	c.rules["root"] = "object"
	c.order = append(c.order, "root")
	return c.text()
}

// FromJSONSchema converts a JSON Schema to GBNF.
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
	c.add("root", body)
	return c.text(), nil
}

type converter struct {
	root  any
	rules map[string]string
	order []string
	refs  map[string]string // $ref -> rule name
}

func newConverter(root any) *converter {
	return &converter{root: root, rules: map[string]string{}, refs: map[string]string{}}
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
	c.rules[n], _ = primitive(n)
	c.order = append(c.order, n)
	deps, _ := primDeps(n)
	for _, d := range deps {
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
		if _, prim := primitive(n); !prim && c.rules[n] == body {
			return n
		}
		n = fmt.Sprintf("%s-%d", name, i)
	}
	c.rules[n] = body
	c.order = append(c.order, n)
	return n
}

// literal is GBNF for the exact text s.
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

// jsonLiteral is GBNF for the value v written as compact JSON, then space.
func (c *converter) jsonLiteral(v any) (string, error) {
	j, err := json.Marshal(plain(v))
	if err != nil {
		return "", err
	}
	return literal(string(j)) + " " + c.prim("space"), nil
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
	case "title", "description", "default", "examples", "$schema", "$id", "$comment", "deprecated", "readOnly", "writeOnly", "$defs", "definitions":
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

// visit is a GBNF expression for schema s; name is what a rule made for it
// is called.
func (c *converter) visit(s any, name string) (string, error) {
	switch v := s.(type) {
	case bool:
		if !v {
			return "", ErrUnsupported{"false (a schema nothing satisfies)"}
		}
		return c.prim("value"), nil
	case map[string]any:
		return c.visitObject(v, name)
	}
	return "", fmt.Errorf("grammar: a schema must be an object or a boolean, not %T", s)
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
		merged, err := c.mergeAll(s)
		if err != nil {
			return "", err
		}
		return c.visitObject(merged, name)
	}
	if v, ok := s["const"]; ok {
		return c.jsonLiteral(v)
	}
	if v, ok := s["enum"]; ok {
		vs, ok := v.([]any)
		if !ok || len(vs) == 0 {
			return "", fmt.Errorf("grammar: enum must be a non-empty array")
		}
		alts := make([]string, len(vs))
		for i, x := range vs {
			lit, err := c.jsonLiteral(x)
			if err != nil {
				return "", err
			}
			alts[i] = lit
		}
		return "(" + strings.Join(alts, " | ") + ")", nil
	}
	for _, k := range []string{"anyOf", "oneOf"} {
		if v, ok := s[k]; ok {
			vs, ok := v.([]any)
			if !ok || len(vs) == 0 {
				return "", fmt.Errorf("grammar: %s must be a non-empty array", k)
			}
			alts := make([]string, len(vs))
			for i, x := range vs {
				e, err := c.visit(x, fmt.Sprintf("%s-%d", name, i))
				if err != nil {
					return "", err
				}
				alts[i] = e
			}
			return "(" + strings.Join(alts, " | ") + ")", nil
		}
	}
	var types []string
	switch t := s["type"].(type) {
	case nil:
		switch {
		case s["properties"] != nil || s["additionalProperties"] != nil:
			types = []string{"object"}
		case s["items"] != nil || s["prefixItems"] != nil:
			types = []string{"array"}
		case s["minLength"] != nil || s["maxLength"] != nil:
			types = []string{"string"}
		case hasBounds(s):
			// Any value, a number among them within the bounds: the integers
			// there, the bounds built for them alone.
			types = []string{"object", "array", "string", "integer", "boolean", "null"}
		case s["format"] != nil || s["pattern"] != nil:
			types = []string{"object", "array", "string", "number", "boolean", "null"}
		default:
			return c.prim("value"), nil
		}
	case string:
		types = []string{t}
	case []any:
		for _, x := range t {
			n, ok := x.(string)
			if !ok {
				return "", fmt.Errorf("grammar: type lists names")
			}
			types = append(types, n)
		}
	default:
		return "", fmt.Errorf("grammar: type is a name or a list of names")
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
			return "", ErrUnsupported{fmt.Sprintf("an integer range nothing satisfies (%v to %v)", lo, hi)}
		}
		return c.add(name, "( "+intRange(lo, hi)+" ) "+c.prim("space")), nil
	case "number":
		if hasBounds(s) {
			return "", ErrUnsupported{"minimum, maximum and their exclusive forms on a number"}
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
			return c.add(name, `"\"" ( `+body+` ) "\"" `+c.prim("space")), nil
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
				return c.add(name, `"\"" `+body+` "\"" `+c.prim("space")), nil
			}
		}
		if lo == 0 && hi < 0 {
			return c.prim("string"), nil
		}
		c.prim("char")
		return c.add(name, fmt.Sprintf(`"\"" char%s "\"" %s`, rep(lo, hi), c.prim("space"))), nil
	case "array":
		return c.array(s, name)
	case "object":
		return c.object(s, name)
	}
	return "", ErrUnsupported{fmt.Sprintf("type %q", t)}
}

// rep is the GBNF repetition {lo,hi}, hi < 0 unbounded.
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

// bounds reads two non-negative integer keywords; hi is -1 when absent.
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
		i, err := n.Int64()
		if err != nil || i < 0 || i > maxRepeat {
			return 0, fmt.Errorf("grammar: %s must be an integer from 0 to %d", k, maxRepeat)
		}
		return int(i), nil
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

func (c *converter) array(s map[string]any, name string) (string, error) {
	sp := c.prim("space")
	if p, ok := s["prefixItems"]; ok {
		ps, ok := p.([]any)
		if !ok {
			return "", fmt.Errorf("grammar: prefixItems must be an array")
		}
		parts := make([]string, len(ps))
		for i, x := range ps {
			e, err := c.visit(x, fmt.Sprintf("%s-%d", name, i))
			if err != nil {
				return "", err
			}
			parts[i] = e
		}
		body := `"[" ` + sp + " "
		if len(parts) > 0 {
			body += strings.Join(parts, ` "," `+sp+" ") + " "
		}
		it, more := s["items"]
		if !more || it == false {
			return c.add(name, body+`"]" `+sp), nil
		}
		// Every prefix item, then the items after them: the array's bounds
		// count both.
		if len(parts) > 0 {
			item, err := c.visit(it, name+"-item")
			if err != nil {
				return "", err
			}
			lo, hi, err := bounds(s, "minItems", "maxItems")
			if err != nil {
				return "", err
			}
			if hi >= 0 && hi < len(parts) {
				return "", ErrUnsupported{fmt.Sprintf("maxItems %d below the %d prefixItems a written array holds", hi, len(parts))}
			}
			if hi >= 0 {
				hi -= len(parts)
			}
			return c.add(name, fmt.Sprintf(`%s( "," %s %s )%s "]" %s`, body, sp, item, rep(max(lo-len(parts), 0), hi), sp)), nil
		}
	}
	item := c.prim("value")
	if it, ok := s["items"]; ok {
		var err error
		if item, err = c.visit(it, name+"-item"); err != nil {
			return "", err
		}
	}
	lo, hi, err := bounds(s, "minItems", "maxItems")
	if err != nil {
		return "", err
	}
	next := `"," ` + sp + " " + item
	var body string
	switch {
	case hi == 0:
		body = ""
	case lo == 0:
		more := -1
		if hi > 0 {
			more = hi - 1
		}
		body = fmt.Sprintf("( %s ( %s )%s )?", item, next, rep(0, more))
	default:
		more := -1
		if hi > 0 {
			more = hi - 1
		}
		body = fmt.Sprintf("%s ( %s )%s", item, next, rep(lo-1, more))
	}
	return c.add(name, fmt.Sprintf(`"[" %s %s "]" %s`, sp, body, sp)), nil
}

func (c *converter) object(s map[string]any, name string) (string, error) {
	sp := c.prim("space")
	props, _ := s["properties"].(map[string]any)
	if s["properties"] != nil && props == nil {
		return "", fmt.Errorf("grammar: properties must be an object")
	}
	addl, hasAddl := s["additionalProperties"]
	req := map[string]bool{}
	// unlisted are the keys required and given no schema in properties:
	// written with any value, or with additionalProperties' schema.
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
					return "", ErrUnsupported{fmt.Sprintf("required %q beside additionalProperties false, which nothing satisfies", n)}
				}
				unlisted = append(unlisted, n)
			}
			req[n] = true
		}
	}
	if len(keysOf(props)) == 0 && len(unlisted) == 0 {
		switch {
		case !hasAddl || addl == true:
			return c.prim("object"), nil
		case addl == false:
			return c.add(name, `"{" `+sp+` "}" `+sp), nil
		}
		val, err := c.visit(addl, name+"-value")
		if err != nil {
			return "", err
		}
		kv := fmt.Sprintf(`%s ":" %s %s`, c.prim("string"), sp, val)
		return c.add(name, fmt.Sprintf(`"{" %s ( %s ( "," %s %s )* )? "}" %s`, sp, kv, sp, kv, sp)), nil
	}
	// The properties in the schema's order: a map loses it, so read the
	// keys again from the source where it has one.
	order := append(keysOf(props), unlisted...)
	// more is a key and value beside the properties, a key none of them
	// names: written after them, as many as the output writes.
	more := ""
	if hasAddl && addl != false && len(keysOf(props)) > 0 {
		val, err := c.visit(addl, name+"-more")
		if err != nil {
			return "", err
		}
		key, err := c.keyNot(order, name)
		if err != nil {
			return "", err
		}
		more = c.add(name+"-more-kv", fmt.Sprintf(`%s ":" %s %s`, key, sp, val))
	}
	var reqKV, optKV []string
	for _, k := range order {
		ks, ok := props[k]
		if !ok {
			ks = true
			if hasAddl && addl != true {
				ks = addl
			}
		}
		v, err := c.visit(ks, name+"-"+k)
		if err != nil {
			return "", err
		}
		kj, err := json.Marshal(k)
		if err != nil {
			return "", err
		}
		kv := c.add(name+"-"+k+"-kv", fmt.Sprintf(`%s %s ":" %s %s`, literal(string(kj)), sp, sp, v))
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
			body = fmt.Sprintf(`%s ( "," %s %s )? | %s`, optKV[i], sp, rest, rest)
		}
		rest = c.add(fmt.Sprintf("%s-rest-%d", name, i), body)
	}
	tail := ""
	if more != "" {
		tail = fmt.Sprintf(` ( "," %s %s )*`, sp, more)
	}
	body := `"{" ` + sp + " "
	switch {
	case len(reqKV) > 0:
		body += strings.Join(reqKV, ` "," `+sp+" ")
		if rest != "" {
			body += fmt.Sprintf(` ( "," %s %s )?`, sp, rest)
		}
		body += tail
	case rest != "" && more != "":
		body += fmt.Sprintf(`( ( %s | %s )%s )?`, rest, more, tail)
	case rest != "":
		body += rest + "?"
	case more != "":
		body += fmt.Sprintf(`( %s%s )?`, more, tail)
	}
	return c.add(name, body+` "}" `+sp), nil
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

// Package schemacheck is the structured-output gates' JSON Schema
// validator, for the subset grammar compiles: what an output is held
// to, written independently of the grammar that produced it.
package schemacheck

import (
	"encoding/json"
	"fmt"
	"math/big"
	"regexp"
	"slices"
	"strings"
	"time"
	"unicode/utf8"
)

// Validate holds doc (parsed by Parse) to schema s, a JSON Schema in the
// subset grammar builds; root is the document $refs resolve in.
func Validate(root, s, doc any) error {
	if b, ok := s.(bool); ok {
		if !b {
			return fmt.Errorf("false schema")
		}
		return nil
	}
	m := s.(map[string]any)
	if r, ok := m["$ref"].(string); ok {
		name := strings.TrimPrefix(r, "#/$defs/")
		return Validate(root, root.(map[string]any)["$defs"].(map[string]any)[name], doc)
	}
	if c, ok := m["const"]; ok && !jsonEqual(c, doc) {
		return fmt.Errorf("%v is not the const %v", doc, c)
	}
	if e, ok := m["enum"].([]any); ok && !slices.ContainsFunc(e, func(x any) bool { return jsonEqual(x, doc) }) {
		return fmt.Errorf("%v is not in the enum %v", doc, e)
	}
	if a, ok := m["anyOf"].([]any); ok {
		for _, x := range a {
			if Validate(root, x, doc) == nil {
				return nil
			}
		}
		return fmt.Errorf("%v matches no branch", doc)
	}
	var types []string
	switch t := m["type"].(type) {
	case string:
		types = []string{t}
	case []any:
		for _, x := range t {
			types = append(types, x.(string))
		}
	}
	if len(types) > 0 && !slices.ContainsFunc(types, func(t string) bool { return isType(doc, t) }) {
		return fmt.Errorf("%v is not of type %v", doc, types)
	}
	switch d := doc.(type) {
	case json.Number:
		if err := inBounds(m, d); err != nil {
			return err
		}
	case string:
		if pt, ok := m["pattern"].(string); ok {
			re, err := regexp.Compile(pt)
			if err != nil {
				return fmt.Errorf("pattern %q is not one Go's regexp reads: %v", pt, err)
			}
			if !re.MatchString(d) {
				return fmt.Errorf("%q does not match %s", d, pt)
			}
		}
		if f, ok := m["format"].(string); ok {
			if err := hasFormat(f, d); err != nil {
				return fmt.Errorf("%q is not a %s: %v", d, f, err)
			}
		}
		n := utf8.RuneCountInString(d)
		if x, ok := m["maxLength"].(float64); ok && float64(n) > x {
			return fmt.Errorf("%q is longer than %v", d, x)
		}
		if x, ok := m["minLength"].(float64); ok && float64(n) < x {
			return fmt.Errorf("%q is shorter than %v", d, x)
		}
	case []any:
		if x, ok := m["maxItems"].(float64); ok && float64(len(d)) > x {
			return fmt.Errorf("%d items, past %v", len(d), x)
		}
		if x, ok := m["minItems"].(float64); ok && float64(len(d)) < x {
			return fmt.Errorf("%d items, under %v", len(d), x)
		}
		// items holds the items after the prefix; a prefix with no items
		// after it is held to its own length.
		after := 0
		it, more := m["items"]
		more = more && it != false
		if p, ok := m["prefixItems"].([]any); ok {
			if len(d) < len(p) || !more && len(d) != len(p) {
				return fmt.Errorf("%d items for %d prefix items", len(d), len(p))
			}
			for i := range p {
				if err := Validate(root, p[i], d[i]); err != nil {
					return err
				}
			}
			after = len(p)
		}
		if more {
			for _, x := range d[after:] {
				if err := Validate(root, it, x); err != nil {
					return err
				}
			}
		}
	case map[string]any:
		props, _ := m["properties"].(map[string]any)
		req, _ := m["required"].([]any)
		for _, r := range req {
			if _, ok := d[r.(string)]; !ok {
				return fmt.Errorf("%v lacks %v", d, r)
			}
		}
		addl, hasAddl := m["additionalProperties"]
		for k, x := range d {
			ps, ok := props[k]
			if !ok {
				// A key no property names: refused where the schema forbids
				// others, held to additionalProperties' schema where it gives
				// one, and refused where it lists properties and does not
				// require this key (the converter's strict reading).
				switch {
				case hasAddl && addl == false:
					return fmt.Errorf("%v has %q, which additionalProperties false forbids", d, k)
				case hasAddl && addl != true:
					ps = addl
				case props != nil && !slices.Contains(req, any(k)):
					return fmt.Errorf("%v has %q, not a property", d, k)
				default:
					continue
				}
			}
			if err := Validate(root, ps, x); err != nil {
				return err
			}
		}
	}
	return nil
}

// inBounds holds number n to schema m's minimum, maximum and their
// exclusive forms, compared exactly.
func inBounds(m map[string]any, n json.Number) error {
	x, ok := new(big.Rat).SetString(string(n))
	if !ok {
		return fmt.Errorf("%s is not a number", n)
	}
	for _, k := range []string{"minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"} {
		v, ok := m[k]
		if !ok {
			continue
		}
		b, ok := new(big.Rat).SetString(fmt.Sprint(v))
		if !ok {
			return fmt.Errorf("%s %v is not a number", k, v)
		}
		c := x.Cmp(b)
		if k == "minimum" && c < 0 || k == "exclusiveMinimum" && c <= 0 || k == "maximum" && c > 0 || k == "exclusiveMaximum" && c >= 0 {
			return fmt.Errorf("%s is outside %s %v", n, k, v)
		}
	}
	return nil
}

// hasFormat holds s to format f, RFC 3339's date, time and date-time and
// RFC 4122's uuid, read by the time package and a pattern; any other format
// asserts nothing.
func hasFormat(f, s string) error {
	var err error
	switch f {
	case "date":
		_, err = time.Parse("2006-01-02", s)
	case "time":
		_, err = time.Parse("15:04:05Z07:00", s)
	case "date-time":
		_, err = time.Parse(time.RFC3339, s)
	case "uuid":
		if !regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`).MatchString(s) {
			err = fmt.Errorf("not five groups of hex digits")
		}
	}
	return err
}

func isType(doc any, t string) bool {
	switch t {
	case "string":
		_, ok := doc.(string)
		return ok
	case "number":
		_, ok := doc.(json.Number)
		return ok
	case "integer":
		n, ok := doc.(json.Number)
		return ok && !strings.ContainsAny(string(n), ".eE")
	case "boolean":
		_, ok := doc.(bool)
		return ok
	case "null":
		return doc == nil
	case "array":
		_, ok := doc.([]any)
		return ok
	case "object":
		_, ok := doc.(map[string]any)
		return ok
	}
	return false
}

// Parse parses one JSON document, numbers kept as written: a grammar
// allows sixteen digits and an exponent, past what a float64 holds.
func Parse(s string, doc *any) error {
	d := json.NewDecoder(strings.NewReader(s))
	d.UseNumber()
	if err := d.Decode(doc); err != nil {
		return err
	}
	if d.More() {
		return fmt.Errorf("text after the document")
	}
	return nil
}

func jsonEqual(a, b any) bool {
	x, err1 := json.Marshal(a)
	y, err2 := json.Marshal(b)
	return err1 == nil && err2 == nil && string(x) == string(y)
}

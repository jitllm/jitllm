// Package schemacheck is the structured-output gates' JSON Schema
// validator, for the subset grammar compiles: what an output is held
// to, written independently of the grammar that produced it.
package schemacheck

import (
	"encoding/json"
	"fmt"
	"maps"
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
	if a, ok := m["allOf"].([]any); ok {
		for _, x := range a {
			// A part reads absent additionalProperties as JSON Schema does,
			// true: its properties are not the whole object.
			if xm, ok := x.(map[string]any); ok {
				if _, has := xm["additionalProperties"]; !has {
					xm = maps.Clone(xm)
					xm["additionalProperties"] = true
					x = xm
				}
			}
			if err := Validate(root, x, doc); err != nil {
				return err
			}
		}
	}
	// Every keyword of a schema holds at once: a branch matched leaves the
	// keywords beside it to hold too.
	if a, ok := m["anyOf"].([]any); ok && !slices.ContainsFunc(a, func(x any) bool { return Validate(root, x, doc) == nil }) {
		return fmt.Errorf("%v matches no branch of anyOf", doc)
	}
	if a, ok := m["oneOf"].([]any); ok {
		n := 0
		for _, x := range a {
			if Validate(root, x, doc) == nil {
				n++
			}
		}
		if n != 1 {
			return fmt.Errorf("%v matches %d branches of oneOf, not one", doc, n)
		}
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
		// prefixItems holds the positions the array has, items those after
		// them.
		after := 0
		if p, ok := m["prefixItems"].([]any); ok {
			after = min(len(p), len(d))
			for i := range after {
				if err := Validate(root, p[i], d[i]); err != nil {
					return err
				}
			}
		}
		if it, ok := m["items"]; ok {
			for _, x := range d[after:] {
				if err := Validate(root, it, x); err != nil {
					return err
				}
			}
		}
	case map[string]any:
		props, _ := m["properties"].(map[string]any)
		req, _ := m["required"].([]any)
		if _, merged := m["allOf"]; merged && props == nil {
			// The parts hold their own properties: each part's object is
			// held to that part above.
			break
		}
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
				// one, and taken with any value where it is absent.
				switch {
				case hasAddl && addl == false:
					return fmt.Errorf("%v has %q, which additionalProperties false forbids", d, k)
				case hasAddl && addl == true:
					continue
				case hasAddl:
					ps = addl
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
	for _, k := range []string{"minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"} {
		v, ok := m[k]
		if !ok {
			continue
		}
		// Read only against a bound: an unbounded number may carry an
		// exponent past what a big.Rat is let to expand.
		x, ok := new(big.Rat).SetString(string(n))
		if !ok {
			return fmt.Errorf("%s is not a number", n)
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
// asserts nothing. RFC 3339's T and Z are case-insensitive (5.6), and the
// time package reads them upper case.
func hasFormat(f, s string) error {
	var err error
	switch f {
	case "date":
		_, err = time.Parse("2006-01-02", s)
	case "time":
		_, err = time.Parse("15:04:05Z07:00", strings.ToUpper(s))
	case "date-time":
		_, err = time.Parse(time.RFC3339, strings.ToUpper(s))
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
		// 2020-12 counts any number with a zero fraction as an integer.
		n, ok := doc.(json.Number)
		if !ok {
			return false
		}
		if !strings.ContainsAny(string(n), ".eE") {
			return true
		}
		r, ok := new(big.Rat).SetString(string(n))
		return ok && r.IsInt()
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
// writes numerals of any length, past what a float64 holds.
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

// jsonEqual is JSON Schema's equality: numbers by their value, 1 and 1.0
// alike, and objects whatever their keys' order.
func jsonEqual(a, b any) bool {
	switch x := a.(type) {
	case map[string]any:
		y, ok := b.(map[string]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for k, v := range x {
			if w, ok := y[k]; !ok || !jsonEqual(v, w) {
				return false
			}
		}
		return true
	case []any:
		y, ok := b.([]any)
		return ok && slices.EqualFunc(x, y, jsonEqual)
	}
	if p, ok := ratOf(a); ok {
		q, ok := ratOf(b)
		return ok && p.Cmp(q) == 0
	}
	return a == b
}

func ratOf(v any) (*big.Rat, bool) {
	switch x := v.(type) {
	case json.Number:
		return new(big.Rat).SetString(string(x))
	case float64:
		r := new(big.Rat).SetFloat64(x)
		return r, r != nil
	}
	return nil, false
}

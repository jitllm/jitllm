// Package schemacheck is the structured-output gates' JSON Schema
// validator, for the subset grammar compiles: what an output is held
// to, written independently of the grammar that produced it.
package schemacheck

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
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
	case string:
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
		if p, ok := m["prefixItems"].([]any); ok {
			if len(d) != len(p) {
				return fmt.Errorf("%d items for %d prefix items", len(d), len(p))
			}
			for i := range p {
				if err := Validate(root, p[i], d[i]); err != nil {
					return err
				}
			}
		}
		if it, ok := m["items"]; ok && it != false {
			for _, x := range d {
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
		for k, x := range d {
			ps, ok := props[k]
			if !ok {
				if props != nil {
					return fmt.Errorf("%v has %q, not a property", d, k)
				}
				continue
			}
			if err := Validate(root, ps, x); err != nil {
				return err
			}
		}
	}
	return nil
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

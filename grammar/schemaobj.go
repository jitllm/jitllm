package grammar

import (
	"encoding/json"
	"fmt"
	"strings"
)

// keyNot is the rule of an object's key, then space, whose decoded value is
// none of names: a trie over the names' characters, each step a character
// no name goes on with (and then any rest), one a name does (and then that
// name's subtree), or the closing quote where no name ends. A key the
// object writes beside its properties can repeat none of them.
func (c *converter) keyNot(names []string, name string) (string, error) {
	type node struct {
		kids  map[rune]*node
		order []rune
		end   bool
	}
	root := &node{kids: map[rune]*node{}}
	for _, n := range names {
		cur := root
		for _, r := range n {
			k, ok := cur.kids[r]
			if !ok {
				k = &node{kids: map[rune]*node{}}
				cur.kids[r] = k
				cur.order = append(cur.order, r)
			}
			cur = k
		}
		cur.end = true
	}
	quote, rest := `"\""`, c.prim("char")+"*"
	var walk func(n *node, id string) (string, error)
	walk = func(n *node, id string) (string, error) {
		var alts []string
		var kids []span
		for _, r := range n.order {
			kids = append(kids, span{r, r})
		}
		if other := setOf(kids...).complement(); len(other) > 0 {
			ch, err := c.jsonChar(other)
			if err != nil {
				return "", err
			}
			alts = append(alts, ch+" "+rest+" "+quote)
		}
		for i, r := range n.order {
			ch, err := c.jsonChar(setOf(span{r, r}))
			if err != nil {
				return "", err
			}
			sub, err := walk(n.kids[r], fmt.Sprintf("%s-%d", id, i))
			if err != nil {
				return "", err
			}
			alts = append(alts, ch+" "+sub)
		}
		if !n.end {
			alts = append(alts, quote)
		}
		return c.add(id, strings.Join(alts, " | ")), nil
	}
	body, err := walk(root, name+"-key")
	if err != nil {
		return "", err
	}
	return quote + " " + body + " " + c.prim("space"), nil
}

// mergeAll is schema s with its allOf folded in: every part an object
// schema, or a $ref to one, whose properties, required names and other
// keywords join s's. All of them hold at once, so the object a merged
// schema writes is valid under each; a keyword or a property two of them
// state differently is refused by name.
func (c *converter) mergeAll(s map[string]any) (map[string]any, error) {
	out := map[string]any{}
	var order, porder []string
	var props map[string]any
	var req []any
	var add func(m map[string]any) error
	add = func(m map[string]any) error {
		for _, k := range keysOf(m) {
			switch v := m[k]; k {
			case "allOf":
				parts, ok := v.([]any)
				if !ok || len(parts) == 0 {
					return fmt.Errorf("grammar: allOf must be a non-empty array")
				}
				for _, p := range parts {
					pm, err := c.part(p)
					if err != nil {
						return err
					}
					if pm != nil {
						if err := add(pm); err != nil {
							return err
						}
					}
				}
			case "properties":
				pm, ok := v.(map[string]any)
				if !ok {
					return fmt.Errorf("grammar: properties must be an object")
				}
				if props == nil {
					props = map[string]any{}
				}
				for _, pk := range keysOf(pm) {
					if old, held := props[pk]; held {
						if !sameJSON(old, pm[pk]) {
							return ErrUnsupported{fmt.Sprintf("allOf whose schemas state property %q differently", pk)}
						}
						continue
					}
					props[pk] = pm[pk]
					porder = append(porder, pk)
				}
			case "required":
				rs, ok := v.([]any)
				if !ok {
					return fmt.Errorf("grammar: required must be an array")
				}
				for _, x := range rs {
					if !containsJSON(req, x) {
						req = append(req, x)
					}
				}
			default:
				if annotation(k) {
					continue
				}
				if old, held := out[k]; held {
					if !sameJSON(old, v) {
						return ErrUnsupported{fmt.Sprintf("allOf whose schemas state %s differently", k)}
					}
					continue
				}
				out[k] = v
				order = append(order, k)
			}
		}
		return nil
	}
	if err := add(s); err != nil {
		return nil, err
	}
	if props != nil {
		props[orderKey] = porder
		out["properties"] = props
		order = append(order, "properties")
	}
	if req != nil {
		out["required"] = req
		order = append(order, "required")
	}
	out[orderKey] = order
	return out, nil
}

// part is an allOf part as an object schema: a $ref resolved, true as
// nothing to add (nil), false refused.
func (c *converter) part(p any) (map[string]any, error) {
	switch v := p.(type) {
	case bool:
		if !v {
			return nil, ErrUnsupported{"false (a schema nothing satisfies), in allOf"}
		}
		return nil, nil
	case map[string]any:
		if r, ok := v["$ref"].(string); ok && onlyAnnotations(v, "$ref") {
			t, err := c.refTarget(r)
			if err != nil {
				return nil, err
			}
			return c.part(t)
		}
		return v, nil
	}
	return nil, fmt.Errorf("grammar: an allOf part must be an object or a boolean, not %T", p)
}

func sameJSON(a, b any) bool {
	x, err1 := json.Marshal(plain(a))
	y, err2 := json.Marshal(plain(b))
	return err1 == nil && err2 == nil && string(x) == string(y)
}

func containsJSON(xs []any, x any) bool {
	for _, y := range xs {
		if sameJSON(x, y) {
			return true
		}
	}
	return false
}

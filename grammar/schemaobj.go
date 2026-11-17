package grammar

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// keyNot is the rule of an object's key whose decoded value is
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
	return quote + " " + body, nil
}

// mergeAll is schema s with its allOf folded in: every part an object
// schema, or a $ref to one, whose properties, required names and other
// keywords join s's. All of them hold at once, so the object a merged
// schema writes is valid under each: the types are the ones every part
// names, a part's additionalProperties holds the properties the other parts
// name and it does not (false leaves them out, a required one leaving
// nothing; a schema joins theirs), and a keyword or a property two of them
// state differently is refused by name. refs are the $refs folded in, which
// the merged schema's visit holds (c.merging) so that one reaching itself
// again is refused rather than unfolded without end.
func (c *converter) mergeAll(s map[string]any) (merged map[string]any, refs []string, err error) {
	out := map[string]any{}
	var order, porder []string
	var props map[string]any
	var req []any
	// closers are the parts whose additionalProperties is not true, each
	// with its own properties: 2020-12 holds a part's additionalProperties
	// to every key its own properties do not name.
	type closer struct {
		own map[string]any
		ap  any
	}
	var closers []closer
	var add func(m map[string]any) error
	add = func(m map[string]any) error {
		if ap, ok := m["additionalProperties"]; ok && ap != true {
			own, _ := m["properties"].(map[string]any)
			closers = append(closers, closer{own, ap})
		}
		for _, k := range keysOf(m) {
			switch v := m[k]; k {
			case "allOf":
				parts, ok := v.([]any)
				if !ok || len(parts) == 0 {
					return fmt.Errorf("grammar: allOf must be a non-empty array")
				}
				for _, p := range parts {
					held := len(c.merging)
					pm, err := c.part(p)
					refs = append(refs, c.merging[held:]...)
					if err == nil && pm != nil {
						err = add(pm)
					}
					c.merging = c.merging[:held]
					if err != nil {
						return err
					}
				}
			case "type":
				ts, err := typeNames(v)
				if err != nil {
					return err
				}
				if old, held := out[k]; held {
					ts = typeMeet(old.([]string), ts)
					if len(ts) == 0 {
						return nothing("allOf whose parts share no type")
					}
				} else {
					order = append(order, k)
				}
				out[k] = ts
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
		return nil, nil, err
	}
	for _, cl := range closers {
		var kept []string
		for _, pk := range porder {
			if _, mine := cl.own[pk]; mine {
				kept = append(kept, pk)
				continue
			}
			if cl.ap == false {
				if containsJSON(req, pk) {
					return nil, nil, nothing(fmt.Sprintf("allOf requiring %q, which a part's additionalProperties false forbids", pk))
				}
				delete(props, pk)
				continue
			}
			props[pk] = map[string]any{"allOf": []any{props[pk], cl.ap}, orderKey: []string{"allOf"}}
			kept = append(kept, pk)
		}
		porder = kept
	}
	if ts, ok := out["type"].([]string); ok {
		list := make([]any, len(ts))
		for i, t := range ts {
			list[i] = t
		}
		out["type"] = list
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
	return out, refs, nil
}

// typeMeet is the types both a and b admit: an integer is a number.
func typeMeet(a, b []string) []string {
	var out []string
	for _, x := range a {
		for _, y := range b {
			t := ""
			switch {
			case x == y:
				t = x
			case x == "integer" && y == "number", x == "number" && y == "integer":
				t = "integer"
			}
			if t != "" && !slices.Contains(out, t) {
				out = append(out, t)
			}
		}
	}
	return out
}

// part is an allOf part as an object schema: a $ref resolved and held on
// c.merging, true as nothing to add (nil), false refused.
func (c *converter) part(p any) (map[string]any, error) {
	switch v := p.(type) {
	case bool:
		if !v {
			return nil, nothing("false, in allOf")
		}
		return nil, nil
	case map[string]any:
		if r, ok := v["$ref"].(string); ok && onlyAnnotations(v, "$ref") {
			if slices.Contains(c.merging, r) {
				return nil, ErrUnsupported{fmt.Sprintf("$ref %q folded into itself", r)}
			}
			t, err := c.refTarget(r)
			if err != nil {
				return nil, err
			}
			c.merging = append(c.merging, r)
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

// disjoint refuses a oneOf two of whose kept branches (each folded with the
// keywords beside it) may hold one value. Two hold none in common where their
// types do not meet (an integer is a number), where both list their values
// (const, enum) and share none, or where they meet only as objects and each
// requires a property whose listed values the other's do not share.
func (c *converter) disjoint(branches []any) error {
	rs := make([]reach, len(branches))
	for i, b := range branches {
		c.spent += visitCost
		rs[i] = c.reachOf(b)
	}
	for i := range rs {
		for j := i + 1; j < len(rs); j++ {
			c.spent += 1 + min(len(rs[i].vals), len(rs[j].vals)) + len(rs[i].keyed)
			if err := c.within(); err != nil {
				return err
			}
			if !rs[i].apart(rs[j]) {
				return ErrUnsupported{"oneOf whose branches may overlap"}
			}
		}
	}
	return nil
}

// reach is what a folded schema says of the values it holds: their types,
// the values it lists (canon), nil where it lists none, and each required
// property's listed values.
type reach struct {
	types []string
	vals  map[string]bool
	keyed map[string]map[string]bool
}

func (c *converter) reachOf(b any) reach {
	m := c.folded(b)
	r := reach{types: typesOf(m), vals: listedOf(m)}
	props, _ := m["properties"].(map[string]any)
	rs, _ := m["required"].([]any)
	for _, x := range rs {
		k, ok := x.(string)
		if !ok {
			continue
		}
		if vs := listedOf(props[k]); vs != nil {
			if r.keyed == nil {
				r.keyed = map[string]map[string]bool{}
			}
			r.keyed[k] = vs
		}
	}
	return r
}

func (a reach) apart(b reach) bool {
	meet := typeMeet(a.types, b.types)
	if len(meet) == 0 {
		return true
	}
	if a.vals != nil && b.vals != nil && !meets(a.vals, b.vals) {
		return true
	}
	if len(meet) == 1 && meet[0] == "object" {
		for k, va := range a.keyed {
			if vb, ok := b.keyed[k]; ok && !meets(va, vb) {
				return true
			}
		}
	}
	return false
}

func meets(a, b map[string]bool) bool {
	if len(b) < len(a) {
		a, b = b, a
	}
	for k := range a {
		if b[k] {
			return true
		}
	}
	return false
}

// folded is schema b as one object schema, its $refs resolved and its allOf
// merged, for reading which values it holds; nil where it reads as any.
func (c *converter) folded(b any) map[string]any {
	held := len(c.merging)
	m, err := c.part(b)
	if err == nil && m != nil {
		if _, ok := m["allOf"]; ok {
			m, _, err = c.mergeAll(m)
		}
	}
	c.merging = c.merging[:held]
	if err != nil {
		return nil
	}
	return m
}

// typesOf is the types the values of folded schema m have: its type
// keyword's, narrowed to its listed values' where it lists them.
func typesOf(m map[string]any) []string {
	ts := []string{"object", "array", "string", "number", "boolean", "null"}
	if t, ok := m["type"]; ok {
		if n, err := typeNames(t); err == nil {
			ts = n
		}
	}
	vs := listed(m)
	if vs == nil {
		return ts
	}
	var out []string
	for _, v := range vs {
		for _, t := range ts {
			if isType(v, t) && !slices.Contains(out, t) {
				out = append(out, t)
			}
		}
	}
	return out
}

// listed is the values schema x lists: its const, its enum, or those of a
// part of its allOf that lists them; nil where it lists none.
func listed(x any) []any {
	m, _ := x.(map[string]any)
	if v, ok := m["const"]; ok {
		return []any{v}
	}
	if v, ok := m["enum"].([]any); ok {
		return v
	}
	if parts, ok := m["allOf"].([]any); ok {
		for _, p := range parts {
			if vs := listed(p); vs != nil {
				return vs
			}
		}
	}
	return nil
}

// listedOf is listed(x) as a set of canon texts, nil where x lists none or
// one of its values has no canon text.
func listedOf(x any) map[string]bool {
	vs := listed(x)
	if vs == nil {
		return nil
	}
	set := make(map[string]bool, len(vs))
	for _, v := range vs {
		var b strings.Builder
		if !canon(&b, v) {
			return nil
		}
		set[b.String()] = true
	}
	return set
}

// canon writes v as 2020-12 compares values: a number by its value (1, 1.0
// and 10e-1 are one), an object's keys sorted. ok false for a numeral whose
// exponent is past what an int64 holds.
func canon(b *strings.Builder, v any) bool {
	switch x := v.(type) {
	case map[string]any:
		ks := make([]string, 0, len(x))
		for k := range x {
			if k != orderKey {
				ks = append(ks, k)
			}
		}
		slices.Sort(ks)
		b.WriteByte('{')
		for i, k := range ks {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(strconv.Quote(k))
			b.WriteByte(':')
			if !canon(b, x[k]) {
				return false
			}
		}
		b.WriteByte('}')
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteByte(',')
			}
			if !canon(b, e) {
				return false
			}
		}
		b.WriteByte(']')
	case json.Number:
		d, ok := decimal(string(x))
		if !ok {
			return false
		}
		b.WriteString(d)
	case string:
		b.WriteString(strconv.Quote(x))
	case bool:
		b.WriteString(strconv.FormatBool(x))
	case nil:
		b.WriteString("null")
	default:
		return false
	}
	return true
}

// decimal is RFC 8259 numeral n as its significant digits and a power of
// ten, one text for every numeral of one value, without computing the power.
func decimal(n string) (string, bool) {
	sign := ""
	if strings.HasPrefix(n, "-") {
		sign, n = "-", n[1:]
	}
	mant, exp := n, int64(0)
	if i := strings.IndexAny(n, "eE"); i >= 0 {
		e, err := strconv.ParseInt(n[i+1:], 10, 64)
		if err != nil || e > 1<<62 || e < -(1<<62) {
			return "", false
		}
		mant, exp = n[:i], e
	}
	whole, frac, _ := strings.Cut(mant, ".")
	digits := strings.TrimLeft(whole+frac, "0")
	if digits == "" {
		return "0", true
	}
	sig := strings.TrimRight(digits, "0")
	exp += int64(len(digits)-len(sig)) - int64(len(frac))
	return sign + sig + "e" + strconv.FormatInt(exp, 10), true
}

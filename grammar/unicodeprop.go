package grammar

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"strings"
)

// property is the set a pattern's Unicode property escape names, body the text
// between its braces: a lone general category or binary property, or
// General_Category, Script or Script_Extensions = a value, each by any alias
// the Unicode Character Database gives it (ECMA-262's
// UnicodePropertyValueExpression), in the Unicode propsVersion states. The
// surrogates are left out, as complement leaves them: a string holds none.
func property(body string) (runeSet, error) {
	kind, name := "", body
	if k, v, ok := strings.Cut(body, "="); ok {
		if kind = propKind(k); kind == "" {
			return nil, fmt.Errorf("grammar: a pattern's \\p{%s}: %q is not General_Category, Script or Script_Extensions", body, k)
		}
		name = v
	}
	key := propKey(kind, name)
	if key == "" {
		return nil, fmt.Errorf("grammar: a pattern's \\p{%s} names no Unicode %s property or value", body, propsVersion)
	}
	raw, err := base64.StdEncoding.DecodeString(propRanges(key))
	if err != nil {
		return nil, fmt.Errorf("grammar: the %s property table: %w", key, err)
	}
	var out runeSet
	next := uint64(0)
	for len(raw) > 0 {
		gap, n := binary.Uvarint(raw)
		if n <= 0 {
			return nil, fmt.Errorf("grammar: the %s property table is cut short", key)
		}
		size, m := binary.Uvarint(raw[n:])
		if m <= 0 {
			return nil, fmt.Errorf("grammar: the %s property table is cut short", key)
		}
		raw = raw[n+m:]
		lo := next + gap
		out = append(out, span{rune(lo), rune(lo + size)})
		next = lo + size + 1
	}
	return out.minus(setOf(span{0xD800, 0xDFFF})), nil
}

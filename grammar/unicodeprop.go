package grammar

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/internal/uniprop"
)

// property is the set a pattern's Unicode property escape names, body the text
// between its braces: a lone general category or binary property, or
// General_Category, Script or Script_Extensions = a value, each by any alias
// the Unicode Character Database gives it (ECMA-262's
// UnicodePropertyValueExpression), in the Unicode uniprop.Version states. The
// surrogates are left out, as complement leaves them: a string holds none.
func property(body string) (runeSet, error) {
	kind, name := "", body
	if k, v, ok := strings.Cut(body, "="); ok {
		if kind = uniprop.KindOf(k); kind == "" {
			return nil, fmt.Errorf("grammar: a pattern's \\p{%s}: %q is not General_Category, Script or Script_Extensions", body, k)
		}
		name = v
	}
	key := uniprop.Key(kind, name)
	if key == "" {
		return nil, fmt.Errorf("grammar: a pattern's \\p{%s} names no Unicode %s property or value", body, uniprop.Version)
	}
	spans, err := uniprop.Spans(key)
	if err != nil {
		return nil, fmt.Errorf("grammar: %w", err)
	}
	out := make(runeSet, len(spans))
	for i, s := range spans {
		out[i] = span{s.Lo, s.Hi}
	}
	return out.minus(setOf(span{0xD800, 0xDFFF})), nil
}

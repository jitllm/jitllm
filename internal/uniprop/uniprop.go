// Package uniprop is the Unicode Character Database's properties -- general
// categories, scripts, script extensions and binary properties -- as the code
// points each holds, generated from the pinned version (scripts/genprops)
// rather than read from Go's unicode package, so what a pattern matches does
// not move under a toolchain upgrade. A grammar's pattern and a tokenizer's
// pre-split pattern read the same tables.
package uniprop

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
)

// Span is the code points Lo through Hi.
type Span struct{ Lo, Hi rune }

// Spans is the code points key (Key's) holds, as sorted, disjoint spans.
func Spans(key string) ([]Span, error) {
	raw, err := base64.StdEncoding.DecodeString(Ranges(key))
	if err != nil {
		return nil, fmt.Errorf("uniprop: the %s table: %w", key, err)
	}
	var out []Span
	next := uint64(0)
	for len(raw) > 0 {
		gap, n := binary.Uvarint(raw)
		if n <= 0 {
			return nil, fmt.Errorf("uniprop: the %s table is cut short", key)
		}
		size, m := binary.Uvarint(raw[n:])
		if m <= 0 {
			return nil, fmt.Errorf("uniprop: the %s table is cut short", key)
		}
		raw = raw[n+m:]
		lo := next + gap
		out = append(out, Span{rune(lo), rune(lo + size)})
		next = lo + size + 1
	}
	return out, nil
}

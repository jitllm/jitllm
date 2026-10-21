package jlm

import "strings"

// Strings is a table of strings held as one string and the offsets that cut
// it. A vocabulary is hundreds of thousands of strings; as a []string each is
// a pointer the collector scans on every cycle for the life of the model, and
// in this form the table is two objects, neither of them scanned.
type Strings struct {
	all string
	// end[i] is where string i ends in all; it starts where i-1 ends.
	end []uint32
}

// NewStrings is a table of s.
func NewStrings(s []string) Strings {
	n := 0
	for _, x := range s {
		n += len(x)
	}
	var b strings.Builder
	b.Grow(n)
	end := make([]uint32, len(s))
	for i, x := range s {
		b.WriteString(x)
		end[i] = uint32(b.Len())
	}
	return Strings{all: b.String(), end: end}
}

// Len is the number of strings.
func (t Strings) Len() int { return len(t.end) }

// At is string i.
func (t Strings) At(i int) string {
	lo, hi := t.Span(i)
	return t.all[lo:hi]
}

// Span is where string i lies in All.
func (t Strings) Span(i int) (lo, hi uint32) {
	if i > 0 {
		lo = t.end[i-1]
	}
	return lo, t.end[i]
}

// All is every string, joined: string i is All()[Span(i)].
func (t Strings) All() string { return t.all }

// readStrings is n length-prefixed strings of r, copied into one table.
func readStrings(r *rbuf, n int) (Strings, error) {
	t := Strings{end: make([]uint32, n)}
	var b strings.Builder
	for i := range n {
		m := int(r.u32())
		if !r.want(m) {
			return Strings{}, r.err
		}
		b.Write(r.b[:m])
		r.b = r.b[m:]
		t.end[i] = uint32(b.Len())
	}
	t.all = b.String()
	return t, nil
}

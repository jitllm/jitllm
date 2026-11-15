package graph

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestNGramRowsRefusesRowsPastAWord holds the hash's remainders to the
// bounds a word holds: a head of no rows, of more than 2^31, or ending past
// 2^32 is refused rather than written (a remainder by none never ends), and
// a head inside them is written.
func TestNGramRowsRefusesRowsPastAWord(t *testing.T) {
	for _, tc := range []struct {
		name     string
		voc, off uint64
		ok       bool
	}{
		{"a head of 97 rows", 97, 100, true},
		{"a head of 2^31 rows ending at 2^32", 1 << 31, 1 << 31, true},
		{"a head of no rows", 0, 0, false},
		{"a head past 2^31 rows", 1<<31 + 1, 0, false},
		{"a head ending past 2^32", 79, 1<<32 - 78, false},
	} {
		b := New()
		win := b.In("win", 2)
		b.G.Values[win].Elem = ElemU32
		h := &jlm.NGramHash{EOS: 2, HeadsPer: 1, Mul: []uint64{3, 5}, Off: []uint64{tc.off}, Voc: []uint64{tc.voc}}
		_, err := b.NGramRows(win, 2, h)
		if (err == nil) != tc.ok {
			t.Errorf("%s: %v", tc.name, err)
		}
	}
}

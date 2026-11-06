package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// TestNGramHashRoundTrips holds an n-gram op's hash to the program's
// contract: it round-trips with its op, and an op with none, one whose
// ranges do not cover its heads, or a hash on another op is refused.
func TestNGramHashRoundTrips(t *testing.T) {
	op := func(h *NGramHash) *Program {
		return &Program{NIn: 1, Out: 1, Ops: []TowerOp{{Code: TOpNGramRows, Bind: []Role{RoleNGramTable}, Span: 3, Width: 64, Hash: h}}}
	}
	hash := func() *NGramHash {
		return &NGramHash{EOS: 2, Image: 7, HeadsPer: 2, Mul: []uint64{3, 5, 11},
			Off: []uint64{0, 100, 200, 300}, Voc: []uint64{97, 89, 83, 79}}
	}
	p := op(hash())
	w := &wbuf{}
	encodeProgram(w, p)
	got := &Program{}
	decodeProgram(&rbuf{b: w.b}, got)
	if !reflect.DeepEqual(got, p) {
		t.Fatalf("round trip: %+v, want %+v", got, p)
	}
	if err := p.valid("n-gram"); err != nil {
		t.Fatalf("a hash that fits its op: %v", err)
	}
	short := hash()
	short.Voc = short.Voc[:3]
	other := &Program{NIn: 1, Out: 1, Ops: []TowerOp{{Code: TOpSlice, In: []uint16{0}, Width: 4, Hash: hash()}}}
	for _, tc := range []struct {
		name string
		p    *Program
		want string
	}{
		{"no hash", op(nil), "by a hash of"},
		{"a range short", op(short), "by a hash of"},
		{"a hash on a slice", other, "states a hash"},
	} {
		if err := tc.p.valid("n-gram"); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", tc.name, err, tc.want)
		}
	}
}

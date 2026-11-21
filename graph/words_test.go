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

// TestTableRowsFollowsEachGather names a row table once, by the row its
// gathers' indices are sliced from, however many gathers read it; names each
// of several tables by its own row; leaves out a table whose gathers read
// rows from two rows; and names no gather of a weight the caller does not
// hold as a row table.
func TestTableRowsFollowsEachGather(t *testing.T) {
	gather := func(b *Builder, r jlm.Role, from ValueID, col int) {
		c := b.Emit(OpSlice, []ValueID{from}, []Result{{Name: "col", Width: 1, Elem: ElemU32}}, nil, Attrs{Offset: col, Width: 1}, nil, nil)[0]
		b.Emit(OpEmbed, []ValueID{c}, []Result{{Name: "emb", Width: 4}},
			[]Binding{{Role: r, K: 4, Rows: 8, From: 8 * col}}, Attrs{}, nil, nil)
	}
	one, other := jlm.RoleNGramTable, jlm.RoleTokenEmbd
	both := []jlm.Role{one, other}
	b := New()
	rows := b.In("rows", 2)
	gather(b, one, rows, 0)
	gather(b, one, rows, 1)
	if ts := b.G.TableRows(both); len(ts) != 1 || ts[0].Role != one || ts[0].Rows != rows {
		t.Fatalf("one table gathered twice by the row %d: %+v", rows, ts)
	}
	if ts := b.G.TableRows([]jlm.Role{other}); len(ts) != 0 {
		t.Fatalf("a weight not held as a row table is named: %+v", ts)
	}
	b1 := New()
	r0, r1 := b1.In("r0", 2), b1.In("r1", 1)
	gather(b1, other, r1, 0)
	gather(b1, one, r0, 0)
	gather(b1, one, r0, 1)
	if ts := b1.G.TableRows(both); len(ts) != 2 || ts[0] != (TableRow{Role: other, Rows: r1}) || ts[1] != (TableRow{Role: one, Rows: r0}) {
		t.Fatalf("two tables, each gathered by its own row: %+v", ts)
	}
	b2 := New()
	gather(b2, one, b2.In("a", 1), 0)
	gather(b2, one, b2.In("b", 1), 0)
	if ts := b2.G.TableRows(both); len(ts) != 0 {
		t.Fatalf("a table gathered by two rows is named: %+v", ts)
	}
	if ts := New().G.TableRows(both); len(ts) != 0 {
		t.Fatalf("a graph gathering nothing names %+v", ts)
	}
}

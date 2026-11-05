package jlm_test

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestRowTableIsReadARowAtATime writes a model whose n-gram table (an IQ4_NL
// row table, RoleNGramTable) is far larger than its blocks, and holds the
// container to what a row table is: no page grows to hold it, the dense
// region does not carry it, and Open reads none of it (its view reads the
// rows: TestRowViewHoldsTheRowsItReads).
func TestRowTableIsReadARowAtATime(t *testing.T) {
	dst, h, table, rows := rowTableModel(t)
	without, _ := expertSource(t)
	hw, err := jlm.Write(filepath.Join(t.TempDir(), "plain.jlm"), without, jlm.Fingerprint{})
	if err != nil {
		t.Fatal(err)
	}
	if h.PageSize != hw.PageSize || h.DenseLen != hw.DenseLen {
		t.Fatalf("the table moved the pages (%d, want %d) or the dense region (%d, want %d)",
			h.PageSize, hw.PageSize, h.DenseLen, hw.DenseLen)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	cf := framesOf(t, c)
	defer c.Close()
	te, ok := c.Find(jlm.RoleNGramTable, jlm.DenseBlock, -1)
	if !ok || !jlm.RowTable(te) || c.InDense(te) {
		t.Fatalf("the table is not a row table outside the dense region: %+v", te)
	}
	rb, err := jlm.RowBytes(te)
	if err != nil || rb != 90 {
		t.Fatalf("rows of %d bytes (%v), want 90", rb, err)
	}
	if n := cf.RowReads(); n != 0 {
		t.Fatalf("opening the model read %d spans of its row table", n)
	}
	if uint64(len(table)) != rows*rb {
		t.Fatalf("a table of %d bytes for %d rows of %d", len(table), rows, rb)
	}
}

// rowTableModel writes a model whose n-gram table is an IQ4_NL row table of
// 3000 rows of 160 (five blocks a row, 90 bytes: rows straddle the 4 KiB
// spans), each block's scale 1: the container, its header, the table's
// bytes and its rows.
func rowTableModel(t *testing.T) (string, *jlm.Header, []byte, uint64) {
	src, _ := expertSource(t)
	const k, rows = 160, 3000
	rng := rand.New(rand.NewPCG(5, 11))
	table := make([]byte, rows*k/32*18)
	for i := range table {
		table[i] = byte(rng.Uint32())
	}
	for i := 0; i < len(table); i += 18 {
		binary.LittleEndian.PutUint16(table[i:], 0x3C00)
	}
	e := jlm.Tensor{Role: jlm.RoleNGramTable, Block: jlm.DenseBlock, Index: -1, Type: jlm.TypeIQ4NL,
		NDim: 2, Data: table, Name: "ngram"}
	e.Dims[0], e.Dims[1] = k, rows
	src.Tensors = append(src.Tensors, e)
	dst := filepath.Join(t.TempDir(), "m.jlm")
	h, err := jlm.Write(dst, src, jlm.Fingerprint{})
	if err != nil {
		t.Fatal(err)
	}
	return dst, h, table, rows
}

// TestRowViewHoldsTheRowsItReads holds a row table's view to the table: the
// rows a step names read in at their place, each span once however many of
// them it holds, read again once dropped, a row past the table refused, and
// a warm step's reads and drop allocating nothing.
func TestRowViewHoldsTheRowsItReads(t *testing.T) {
	dst, _, table, rows := rowTableModel(t)
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	cf := framesOf(t, c)
	defer c.Close()
	te, ok := c.Find(jlm.RoleNGramTable, jlm.DenseBlock, -1)
	if !ok {
		t.Fatal("no table")
	}
	v, err := cf.NewRowView(te)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	rb, _ := jlm.RowBytes(te)
	sel := []uint32{0, 1, 45, 46, 2999, 1234, 455, 46}
	spans := map[uint64]bool{}
	at := te.QSOff % jlm.Align
	for _, r := range sel {
		for p := (at + uint64(r)*rb) / jlm.Align; p <= (at+uint64(r+1)*rb-1)/jlm.Align; p++ {
			spans[p] = true
		}
	}
	check := func(when string) {
		t.Helper()
		for _, r := range sel {
			got := unsafe.Slice((*byte)(unsafe.Add(v.Base(), uint64(r)*rb)), rb)
			if !bytes.Equal(got, table[uint64(r)*rb:uint64(r+1)*rb]) {
				t.Fatalf("%s: row %d is not its source bytes", when, r)
			}
		}
	}
	before := cf.RowReads()
	if err := v.Ensure(sel); err != nil {
		t.Fatal(err)
	}
	if n := cf.RowReads() - before; n != int64(len(spans)) {
		t.Fatalf("%d reads for rows in %d spans", n, len(spans))
	}
	check("read")
	before = cf.RowReads()
	if err := v.Ensure(sel); err != nil || cf.RowReads() != before {
		t.Fatalf("rows read in were read again (%d reads, %v)", cf.RowReads()-before, err)
	}
	v.Drop()
	if v.Held() != 0 {
		t.Fatalf("%d bytes held after a drop", v.Held())
	}
	before = cf.RowReads()
	if err := v.Ensure(sel); err != nil || cf.RowReads() == before {
		t.Fatalf("dropped rows were not read again (%v)", err)
	}
	check("read again")
	if err := v.Ensure([]uint32{uint32(rows)}); err == nil {
		t.Fatal("a row past the table was read")
	}
	if n := testing.AllocsPerRun(10, func() {
		v.Drop()
		if err := v.Ensure(sel); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Fatalf("a warm step's rows allocate %.0f times", n)
	}
}

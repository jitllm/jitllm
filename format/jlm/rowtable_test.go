package jlm_test

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestRowTableIsReadARowAtATime writes a model whose n-gram table (an IQ4_NL
// row table, RoleNGramTable) is far larger than its blocks, and holds the
// container to what a row table is: no page grows to hold it, the dense
// region does not carry it, Open reads none of it, and ReadRow returns each
// row's source bytes exactly, wherever a row falls against the alignment.
func TestRowTableIsReadARowAtATime(t *testing.T) {
	src, _ := expertSource(t)
	const k, rows = 160, 3000 // five IQ4_NL blocks a row, 90 bytes: rows straddle the 4 KiB blocks
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
	without, _ := expertSource(t)
	dir := t.TempDir()
	hw, err := jlm.Write(filepath.Join(dir, "plain.jlm"), without, jlm.Fingerprint{})
	if err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "m.jlm")
	h, err := jlm.Write(dst, src, jlm.Fingerprint{})
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
	defer c.Close()
	te, ok := c.Find(jlm.RoleNGramTable, jlm.DenseBlock, -1)
	if !ok || !jlm.RowTable(te) || c.InDense(te) {
		t.Fatalf("the table is not a row table outside the dense region: %+v", te)
	}
	rb, err := jlm.RowBytes(te)
	if err != nil || rb != 90 {
		t.Fatalf("rows of %d bytes (%v), want 90", rb, err)
	}
	dstRow, scr := make([]byte, rb), jlm.RowScratch()
	for _, r := range []uint64{0, 1, 45, 46, 2999, 1234, 455} {
		before := c.RowReads()
		if err := c.ReadRow(te, r, dstRow, scr); err != nil {
			t.Fatal(err)
		}
		if c.RowReads() != before+1 {
			t.Fatalf("row %d took %d reads", r, c.RowReads()-before)
		}
		if !bytes.Equal(dstRow, table[r*rb:(r+1)*rb]) {
			t.Fatalf("row %d is not its source bytes", r)
		}
	}
	if err := c.ReadRow(te, rows, dstRow, scr); err == nil {
		t.Fatal("a row past the table was read")
	}
}

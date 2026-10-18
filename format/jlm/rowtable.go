package jlm

import (
	"fmt"
)

// RowScratch is a buffer ReadRow reads through: two Align'd blocks, enough for
// any row of up to Align bytes wherever it falls. The caller owns it and
// reuses it, so a warm read allocates nothing.
func RowScratch() []byte { return alignedBuf(2 * Align) }

// ReadRow reads row r of the row table e (RowTable) into dst, which is
// RowBytes(e) long, through scratch (RowScratch). The read is the Align'd
// span holding the row, through the direct handle: one pread of a block or
// two, never the table.
func (f *File) ReadRow(e *Entry, r uint64, dst, scratch []byte) error {
	if !RowTable(e) {
		return fmt.Errorf("jlm: %s is not a row table", e.Name)
	}
	rb, err := RowBytes(e)
	if err != nil {
		return err
	}
	if r >= e.Dims[1] || uint64(len(dst)) != rb || rb > Align {
		return fmt.Errorf("jlm: %s: row %d of %d, %d bytes into %d", e.Name, r, e.Dims[1], rb, len(dst))
	}
	off := e.QSOff + r*rb
	lo := off &^ (Align - 1)
	hi := (off + rb + Align - 1) &^ (Align - 1)
	if hi-lo > uint64(len(scratch)) {
		return fmt.Errorf("jlm: %s: a %d-byte scratch for a %d-byte span", e.Name, len(scratch), hi-lo)
	}
	if err := f.readRowSpan(scratch[:hi-lo], int64(lo)); err != nil {
		return fmt.Errorf("jlm: %s row %d: %w", e.Name, r, err)
	}
	copy(dst, scratch[off-lo:off-lo+rb])
	return nil
}

// RowReads is how many row-table reads this File has issued, apart from the
// pager's reads: a row is not a page, and a decode that reads rows by design is
// still a resident model's decode.
func (f *File) RowReads() int64 { return f.rowReads.Load() }

// readRowSpan is one pread of a row's Align'd span: no fan-out, so nothing
// escapes and a warm read allocates nothing.
func (f *File) readRowSpan(dst []byte, off int64) error {
	f.rowReads.Add(1)
	h := f.f
	if f.direct != nil && directOK(off, len(dst)) && aligned(dst) {
		h = f.direct
	}
	_, err := h.ReadAt(dst, off)
	return err
}

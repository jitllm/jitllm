package jlm

import (
	"errors"
	"fmt"
	"io"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/internal/offheap"
)

// RowView is a row table where the host reserves address space for all of
// it: row r at Base plus r times RowBytes, so a device reading the host's
// memory binds the table whole and gathers the rows a step names, as a bank
// is bound whole in the expert reservation (BankView). A row is readable once
// Ensure has read its Align'd span in from the file, and until Drop hands
// every span read back. A view is its owner's alone: one device's calls read
// it, so it takes no lock.
type RowView struct {
	f   *File
	mem []byte
	// lo is the file offset mem[0] holds, end the table's last byte's past
	// it; row0 is row 0's place in mem, rb a row's bytes.
	lo, end, row0, rb uint64
	rows              uint64
	// held is a bit an Align span of mem, set while it is read in, and read
	// the spans read in since the last Drop.
	held []uint64
	read []uint32
}

// NewRowView reserves row table e's view. It fails where e is not a row
// table or the host reserves no address space.
func (f *File) NewRowView(e *Entry) (*RowView, error) {
	if !RowTable(e) {
		return nil, fmt.Errorf("jlm: %s is not a row table", e.Name)
	}
	rb, err := RowBytes(e)
	if err != nil {
		return nil, err
	}
	lo := e.QSOff &^ (Align - 1)
	end := e.QSOff + e.Dims[1]*rb
	n := (end - lo + Align - 1) &^ (Align - 1)
	mem, mapped, err := offheap.ReserveOffHeap(int(n))
	if err != nil || !mapped {
		return nil, fmt.Errorf("jlm: %s: no reservation of %d bytes for its rows (%v)", e.Name, n, err)
	}
	return &RowView{f: f, mem: mem, lo: lo, end: end, row0: e.QSOff - lo, rb: rb, rows: e.Dims[1],
		held: make([]uint64, (n/Align+63)/64)}, nil
}

// Base is row 0's address; the rows follow at RowBytes apart.
func (v *RowView) Base() unsafe.Pointer { return unsafe.Pointer(&v.mem[v.row0]) }

// Rows is the table's rows.
func (v *RowView) Rows() uint64 { return v.rows }

// Ensure reads in each of rows' spans not read in already: one read a span,
// nothing allocated once the spans of a step's most rows have been read.
func (v *RowView) Ensure(rows []uint32) error {
	for _, r := range rows {
		if uint64(r) >= v.rows {
			return fmt.Errorf("jlm: row %d of a table of %d", r, v.rows)
		}
		at := v.row0 + uint64(r)*v.rb
		for p := at / Align; p <= (at+v.rb-1)/Align; p++ {
			if v.held[p/64]&(1<<(p%64)) != 0 {
				continue
			}
			if err := v.readSpan(p); err != nil {
				return err
			}
			v.held[p/64] |= 1 << (p % 64)
			v.read = append(v.read, uint32(p))
		}
	}
	return nil
}

// readSpan reads span p of the view from the file, the last only to the
// table's end.
func (v *RowView) readSpan(p uint64) error {
	off := v.lo + p*Align
	n := min(uint64(Align), v.end-off)
	dst := v.mem[p*Align : p*Align+n]
	if err := offheap.CommitOffHeap(dst); err != nil {
		return fmt.Errorf("jlm: row span at %d: %w", off, err)
	}
	v.f.rowReads.Add(1)
	h := v.f.f
	if v.f.direct != nil && directOK(int64(off), int(n)) && aligned(dst) {
		h = v.f.direct
	}
	if _, err := h.ReadAt(dst, int64(off)); err != nil && !errors.Is(err, io.EOF) {
		return fmt.Errorf("jlm: row span at %d: %w", off, err)
	}
	return nil
}

// RowReads is how many row-table reads this File has issued, counted apart
// from the pager's reads: a row is not a page, and a decode that reads rows by
// design is still a resident model's decode.
func (f *File) RowReads() int64 { return f.rowReads.Load() }

// Held is the bytes of the spans read in.
func (v *RowView) Held() uint64 { return uint64(len(v.read)) * Align }

// Drop hands every span read in back to the system, a run of neighbouring
// spans at a time.
func (v *RowView) Drop() {
	slices.Sort(v.read)
	for i := 0; i < len(v.read); {
		j := i + 1
		for j < len(v.read) && v.read[j] == v.read[j-1]+1 {
			j++
		}
		a, b := uint64(v.read[i])*Align, uint64(v.read[j-1]+1)*Align
		offheap.DropOffHeap(v.mem[a:min(b, uint64(len(v.mem)))])
		i = j
	}
	for _, p := range v.read {
		v.held[p/64] &^= 1 << (p % 64)
	}
	v.read = v.read[:0]
}

// Close gives the reservation back; the view's rows are gone with it.
func (v *RowView) Close() {
	if v.mem != nil {
		offheap.UnmapOffHeap(v.mem)
		v.mem, v.read = nil, nil
	}
}

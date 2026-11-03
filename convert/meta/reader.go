package meta

import (
	"encoding/binary"
	"fmt"
)

// errTruncated is the one error every bounds failure funnels into. A GGUF is
// untrusted input from the internet: the reader must return errors, never panic
// and never read out of bounds, for any byte sequence.
const errTruncated constError = "meta: truncated or malformed"

type constError string

func (e constError) Error() string { return string(e) }

// reader is a bounds-checked cursor over the mapping. It is sticky: once err is
// set every subsequent read is a no-op, so callers can read a whole record and
// check once. Every accessor must go through want().
//
// It is exported because a jlm container also stores its metadata values in
// this encoding (so Value can stay lazy), and one cursor serves both.
type Reader struct {
	B   []byte
	O   uint64
	Err error
}

func (r *Reader) Fail(format string, a ...any) {
	if r.Err == nil {
		r.Err = fmt.Errorf("meta: at offset %d: "+format, append([]any{r.O}, a...)...)
	}
}

// want reports whether n more bytes are available, setting err if not. The
// addition is overflow-checked because n comes from the file.
func (r *Reader) Want(n uint64) bool {
	if r.Err != nil {
		return false
	}
	if end := r.O + n; end < r.O || end > uint64(len(r.B)) {
		r.Err = fmt.Errorf("%w: wanted %d bytes at offset %d, have %d", errTruncated, n, r.O, uint64(len(r.B))-min(r.O, uint64(len(r.B))))
		return false
	}
	return true
}

func (r *Reader) U32() uint32 {
	if !r.Want(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.B[r.O:])
	r.O += 4
	return v
}

func (r *Reader) U64() uint64 {
	if !r.Want(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(r.B[r.O:])
	r.O += 8
	return v
}

// str reads a length-prefixed string as a subslice of the mapping. No copy: a
// GGUF string is only turned into a Go string when someone asks for it, which
// keeps a large vocabulary allocation-free to walk.
func (r *Reader) Str() []byte {
	n := r.U64()
	if !r.Want(n) {
		return nil
	}
	s := r.B[r.O : r.O+n]
	r.O += n
	return s
}

// skipValue advances past one KV value of type t without decoding it. Arrays
// must be walked element by element because element widths are variable
// (strings), but nothing is allocated.
func (r *Reader) SkipValue(t ValueType, depth int) {
	if r.Err != nil {
		return
	}
	if n, fixed := t.width(); fixed {
		if r.Want(n) {
			r.O += n
		}
		return
	}
	switch t {
	case String:
		r.Str()
	case Array:
		if depth > 0 { // ggml has never nested arrays; refuse rather than recurse on hostile input.
			r.Fail("nested array")
			return
		}
		et := ValueType(r.U32())
		n := r.U64()
		if r.Err != nil {
			return
		}
		if w, fixed := et.width(); fixed {
			// Fixed-width: one bounds check for the whole array, not n of them.
			total := n * w
			if w != 0 && total/w != n { // overflow
				r.Fail("array of %d x %d bytes overflows", n, w)
				return
			}
			if r.Want(total) {
				r.O += total
			}
			return
		}
		if et != String {
			r.Fail("array of unknown element type %d", et)
			return
		}
		for i := uint64(0); i < n; i++ {
			r.Str()
			if r.Err != nil {
				return
			}
		}
	default:
		r.Fail("unknown value type %d", t)
	}
}

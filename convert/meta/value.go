package meta

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// ValueType tags a metadata value. ABI: these appear in a .jlm file.
//
// The numbers and framing match GGUF's, which lets the GGUF parser hand out
// subslices of its own mapping with no copy; the spec is still meta's own.
type ValueType uint32

const (
	Uint8   ValueType = 0
	Int8    ValueType = 1
	Uint16  ValueType = 2
	Int16   ValueType = 3
	Uint32  ValueType = 4
	Int32   ValueType = 5
	Float32 ValueType = 6
	Bool    ValueType = 7
	String  ValueType = 8
	Array   ValueType = 9
	Uint64  ValueType = 10
	Int64   ValueType = 11
	Float64 ValueType = 12
)

// width returns the fixed byte width of a scalar type. String and Array are
// variable, and anything else is unknown — both report fixed=false, so callers
// must handle them explicitly rather than defaulting to a size.
func (t ValueType) width() (n uint64, fixed bool) {
	switch t {
	case Uint8, Int8, Bool:
		return 1, true
	case Uint16, Int16:
		return 2, true
	case Uint32, Int32, Float32:
		return 4, true
	case Uint64, Int64, Float64:
		return 8, true
	}
	return 0, false
}

// Value is one metadata entry, left undecoded. Raw points into the mapping.
//
// Lazy on purpose: a vocabulary is several parallel arrays of a quarter-million
// entries, and materializing them at Open would cost that many allocations.
// Scalars decode on demand; the tokenizer decodes the arrays once.
type Value struct {
	Type ValueType
	Raw  []byte // the value's bytes, exactly

	// Array only:
	Elem ValueType
	N    uint64
}

// Uint decodes any unsigned or signed integer value as uint64. Signed negatives
// are returned as their two's-complement bits; callers wanting sign use Int.
func (v Value) Uint() (uint64, bool) {
	i, ok := v.Int()
	return uint64(i), ok
}

// Int decodes any integer or bool value, sign-extended.
//
// The zero Value (a missing key, `f.KV["absent"]`) must not panic: ValueType's
// zero is Uint8, so the length checks are what make a short or empty Raw
// report false rather than index a nil slice.
func (v Value) Int() (int64, bool) {
	switch v.Type {
	case Uint8:
		if len(v.Raw) < 1 {
			return 0, false
		}
		return int64(v.Raw[0]), true
	case Int8:
		if len(v.Raw) < 1 {
			return 0, false
		}
		return int64(int8(v.Raw[0])), true
	case Bool:
		if len(v.Raw) < 1 {
			return 0, false
		}
		if v.Raw[0] != 0 {
			return 1, true
		}
		return 0, true
	case Uint16:
		if len(v.Raw) < 2 {
			return 0, false
		}
		return int64(binary.LittleEndian.Uint16(v.Raw)), true
	case Int16:
		if len(v.Raw) < 2 {
			return 0, false
		}
		return int64(int16(binary.LittleEndian.Uint16(v.Raw))), true
	case Uint32:
		if len(v.Raw) < 4 {
			return 0, false
		}
		return int64(binary.LittleEndian.Uint32(v.Raw)), true
	case Int32:
		if len(v.Raw) < 4 {
			return 0, false
		}
		return int64(int32(binary.LittleEndian.Uint32(v.Raw))), true
	case Uint64, Int64:
		if len(v.Raw) < 8 {
			return 0, false
		}
		return int64(binary.LittleEndian.Uint64(v.Raw)), true
	}
	return 0, false
}

// Float decodes a float value, and also integers — GGUF writers are
// inconsistent about whether e.g. rope.freq_base is f32 or u32.
func (v Value) Float() (float64, bool) {
	switch v.Type {
	case Float32:
		if len(v.Raw) < 4 {
			return 0, false
		}
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(v.Raw))), true
	case Float64:
		if len(v.Raw) < 8 {
			return 0, false
		}
		return math.Float64frombits(binary.LittleEndian.Uint64(v.Raw)), true
	}
	if i, ok := v.Int(); ok {
		return float64(i), true
	}
	return 0, false
}

// String returns a string value. Not valid for arrays of strings.
func (v Value) String() (string, bool) {
	if v.Type != String {
		return "", false
	}
	return string(v.Raw), true
}

// Len is the element count of an array value, or 0 for a scalar.
func (v Value) Len() int {
	if v.Type != Array {
		return 0
	}
	return int(v.N)
}

// EachString walks a string array, handing fn a subslice of the mapping for each
// element. No allocation: the caller copies only what it keeps. Return false to
// stop early.
//
// This is why Value is lazy.
func (v Value) EachString(fn func(i int, s []byte) bool) error {
	if v.Type != Array || v.Elem != String {
		return fmt.Errorf("meta: value is %v, not an array of strings", v.Type)
	}
	r := &Reader{B: v.Raw}
	for i := 0; i < int(v.N); i++ {
		s := r.Str()
		if r.Err != nil {
			return fmt.Errorf("meta: string array element %d: %w", i, r.Err)
		}
		if !fn(i, s) {
			return nil
		}
	}
	return nil
}

// Float32s decodes a float32 array.
func (v Value) Float32s() ([]float32, error) {
	if v.Type != Array || v.Elem != Float32 {
		return nil, fmt.Errorf("meta: value is %v, not an array of float32", v.Type)
	}
	if uint64(len(v.Raw)) < v.N*4 {
		return nil, fmt.Errorf("meta: float32 array of %d needs %d bytes, have %d", v.N, v.N*4, len(v.Raw))
	}
	out := make([]float32, v.N)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(v.Raw[i*4:]))
	}
	return out, nil
}

// Int32s decodes an int32 or uint32 array.
func (v Value) Int32s() ([]int32, error) {
	if v.Type != Array || (v.Elem != Int32 && v.Elem != Uint32) {
		return nil, fmt.Errorf("meta: value is %v of %v, not an array of int32", v.Type, v.Elem)
	}
	if uint64(len(v.Raw)) < v.N*4 {
		return nil, fmt.Errorf("meta: int32 array of %d needs %d bytes, have %d", v.N, v.N*4, len(v.Raw))
	}
	out := make([]int32, v.N)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(v.Raw[i*4:]))
	}
	return out, nil
}

// Uint64s decodes an array of any integer width as uint64, refusing a
// negative element: the hash constants a converter carries are unsigned.
func (v Value) Uint64s() ([]uint64, error) {
	if v.Type != Array {
		return nil, fmt.Errorf("meta: value is %v, not an array", v.Type)
	}
	w, fixed := v.Elem.width()
	if !fixed || v.Elem == Float32 || v.Elem == Float64 || v.Elem == Bool {
		return nil, fmt.Errorf("meta: array of %v, not of integers", v.Elem)
	}
	if uint64(len(v.Raw)) < v.N*w {
		return nil, fmt.Errorf("meta: array of %d needs %d bytes, have %d", v.N, v.N*w, len(v.Raw))
	}
	out := make([]uint64, v.N)
	for i := range out {
		x, _ := Value{Type: v.Elem, Raw: v.Raw[uint64(i)*w : uint64(i+1)*w]}.Int()
		if x < 0 && v.Elem != Uint64 {
			return nil, fmt.Errorf("meta: element %d is %d", i, x)
		}
		out[i] = uint64(x)
	}
	return out, nil
}

// Short renders a value for human inspection: scalars in full, arrays as their
// type and length plus the first few elements.
//
// It truncates: printing a quarter-million-entry token list is never what the
// reader wanted.
func (v Value) Short() string {
	if s, ok := v.String(); ok {
		if len(s) > 120 {
			return fmt.Sprintf("%q... (%d bytes)", s[:120], len(s))
		}
		return fmt.Sprintf("%q", s)
	}
	if v.Type == Bool {
		if b, ok := v.Int(); ok {
			return fmt.Sprintf("%v", b != 0)
		}
	}
	if v.Type != Array {
		if f, ok := v.Float(); ok {
			if i, ok := v.Int(); ok {
				return fmt.Sprintf("%d", i)
			}
			return fmt.Sprintf("%g", f)
		}
		return fmt.Sprintf("<%v>", v.Type)
	}
	out := fmt.Sprintf("arr[%v,%d]", v.Elem, v.N)
	if v.Elem == String {
		var parts []string
		v.EachString(func(i int, s []byte) bool {
			parts = append(parts, fmt.Sprintf("%q", s))
			return i < 2
		})
		if len(parts) > 0 {
			return out + " = [" + strings.Join(parts, ", ") + ", ...]"
		}
		return out
	}
	if n := v.Len(); n > 0 && n <= 8 {
		if xs, err := v.Int32s(); err == nil {
			var parts []string
			for _, x := range xs {
				parts = append(parts, fmt.Sprintf("%d", x))
			}
			return out + " = [" + strings.Join(parts, ", ") + "]"
		}
		if xs, err := v.Float32s(); err == nil {
			var parts []string
			for _, x := range xs {
				parts = append(parts, fmt.Sprintf("%g", x))
			}
			return out + " = [" + strings.Join(parts, ", ") + "]"
		}
	}
	return out
}

// The Make* constructors build every value type from ordinary Go values, for
// a caller that wants to produce a meta.File without learning the encoding.
// The converters build a jlm.Source directly; the one converter caller is
// Kimi-K3's safetensors config (convert/hfkimik3.go), which states text_config
// as the GGUF keys llama.cpp writes so configOf reads it. Arrays are built
// eagerly, since the caller already has the slice; the reader stays lazy.

// MakeUint tags an unsigned integer, narrowing to the smallest type that holds
// it so a round-trip through a file does not widen every count to 64 bits.
func MakeUint(n uint64) Value {
	switch {
	case n <= math.MaxUint8:
		return Value{Type: Uint8, Raw: []byte{byte(n)}}
	case n <= math.MaxUint16:
		return Value{Type: Uint16, Raw: binary.LittleEndian.AppendUint16(nil, uint16(n))}
	case n <= math.MaxUint32:
		return Value{Type: Uint32, Raw: binary.LittleEndian.AppendUint32(nil, uint32(n))}
	}
	return Value{Type: Uint64, Raw: binary.LittleEndian.AppendUint64(nil, n)}
}

// MakeInt tags a signed integer. It does not narrow: a negative that fits a
// smaller type still round-trips, but the width is part of what a caller means
// by a signed key and guessing it back is how a -1 sentinel becomes 255.
func MakeInt(n int64) Value {
	return Value{Type: Int64, Raw: binary.LittleEndian.AppendUint64(nil, uint64(n))}
}

// MakeFloat tags a 32-bit float, which is every float key any model writes.
func MakeFloat(f float32) Value {
	return Value{Type: Float32, Raw: binary.LittleEndian.AppendUint32(nil, math.Float32bits(f))}
}

// MakeBool tags a boolean.
func MakeBool(v bool) Value {
	b := byte(0)
	if v {
		b = 1
	}
	return Value{Type: Bool, Raw: []byte{b}}
}

// MakeString tags a string.
//
// Raw is the string's bytes and not its framing, which a constructor can get
// wrong silently: the parser consumes the u64 length before it stores Raw, so a
// value carrying its own prefix reads back with the prefix in it. Likewise an
// array's Raw is the elements, with the element type and count in Elem and N.
func MakeString(s string) Value {
	return Value{Type: String, Raw: []byte(s)}
}

// MakeStrings tags an array of strings -- the vocabulary, the merge table, the
// token-type names.
func MakeStrings(ss []string) Value {
	b := make([]byte, 0, 8*len(ss)+16)
	for _, s := range ss {
		b = appendStr(b, s)
	}
	return Value{Type: Array, Elem: String, N: uint64(len(ss)), Raw: b}
}

// MakeFloat32s tags an array of float32 -- token scores.
func MakeFloat32s(fs []float32) Value {
	b := make([]byte, 0, 4*len(fs))
	for _, f := range fs {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(f))
	}
	return Value{Type: Array, Elem: Float32, N: uint64(len(fs)), Raw: b}
}

// MakeInt32s tags an array of int32 -- token types.
func MakeInt32s(is []int32) Value {
	b := make([]byte, 0, 4*len(is))
	for _, i := range is {
		b = binary.LittleEndian.AppendUint32(b, uint32(i))
	}
	return Value{Type: Array, Elem: Int32, N: uint64(len(is)), Raw: b}
}

func appendStr(b []byte, s string) []byte {
	b = binary.LittleEndian.AppendUint64(b, uint64(len(s)))
	return append(b, s...)
}

package meta

import (
	"sort"

	"github.com/jitllm/jitllm/format/quant"
)

// Package meta is the model metadata and tensor directory, and it belongs to
// neither the GGUF parser nor the engine. It exists so the inference path does
// not import gguf: a jlm container answers "what architecture, what vocabulary,
// what shape" through this, and gguf is a parser that produces one. The value
// encoding is GGUF's, byte for byte, which the container reuses deliberately
// (see Reader).

// Tensor is one entry of the tensor table.
type Tensor struct {
	Name   string
	Dims   []uint64 // ne, fastest-varying first, exactly as stored
	Type   quant.Type
	Off    uint64 // relative to File.DataStart
	NBytes uint64 // computed from Dims and quant.Type
}

// Elems is the logical element count.
func (t *Tensor) Elems() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// File is an open GGUF mapping.
type File struct {
	Path      string
	Version   uint32
	Alignment uint64
	DataStart uint64 // absolute file offset of the tensor data section
	KV        map[string]Value
	Tensors   []Tensor

	// Fetch, when non-nil, supplies a tensor's bytes instead of the mapping.
	//
	// It is how jlm serves a file whose layout is not GGUF's: quantised
	// tensors are in the device layout and the rest sit at container offsets,
	// so t.Off means nothing. Every weight read goes through Bytes, so one
	// hook covers them all.
	Fetch func(*Tensor) []byte

	// Data is the whole-file mapping, and CloseFn releases it. Both are set by
	// the parser that produced this -- a container sets neither, because its
	// Fetch hook serves every byte and Bytes must never reach a mapping that
	// does not exist.
	Data    []byte
	CloseFn func() error

	index map[string]int
}

// Close unmaps the file. Every []byte handed out by Bytes becomes invalid.
func (f *File) Close() error {
	if f.CloseFn == nil {
		return nil
	}
	return f.CloseFn()
}

// Size is the mapped file size.
func (f *File) Size() uint64 { return uint64(len(f.Data)) }

// Bytes returns t's data as a subslice of the mapping. No copy. Valid until Close.
func (f *File) Bytes(t *Tensor) []byte {
	if f.Fetch != nil {
		return f.Fetch(t)
	}
	lo := f.DataStart + t.Off
	return f.Data[lo : lo+t.NBytes : lo+t.NBytes]
}

// New builds a File from metadata alone, with no tensor data, for jlm, whose
// weights are in the device layout (Bytes must never be called on one without
// Fetch). It builds the name index, without which Get finds nothing.
func New(path string, kv map[string]Value, tensors []Tensor) *File {
	f := &File{Path: path, KV: kv, Tensors: tensors}
	f.Reindex()
	return f
}

// Reindex rebuilds the name lookup. A parser filling Tensors field by field
// calls it once at the end; New does it for a caller that has the whole slice.
func (f *File) Reindex() {
	f.index = make(map[string]int, len(f.Tensors))
	for i := range f.Tensors {
		f.index[f.Tensors[i].Name] = i
	}
}

// Get looks a tensor up by its exact name.
func (f *File) Get(name string) (*Tensor, bool) {
	i, ok := f.index[name]
	if !ok {
		return nil, false
	}
	return &f.Tensors[i], true
}

// Layer resolves one tensor of block n by role, e.g. Layer(3, "attn_q.weight").
//
// Always resolve by name: the intra-block tensor order is converter-dependent,
// differs between models, and is not execution order.
func (f *File) Layer(n int, role string) (*Tensor, bool) {
	return f.Get("blk." + itoa(uint64(n)) + "." + role)
}

// LayerRange is the byte span covering every tensor of block n, relative to
// DataStart. Every block is contiguous in all reference models (0 internal
// gaps), which is what makes a per-layer madvise or prefetch possible later.
func (f *File) LayerRange(n int) (off, size uint64, ok bool) {
	prefix := "blk." + itoa(uint64(n)) + "."
	lo, hi := ^uint64(0), uint64(0)
	for i := range f.Tensors {
		t := &f.Tensors[i]
		if len(t.Name) < len(prefix) || t.Name[:len(prefix)] != prefix {
			continue
		}
		lo = min(lo, t.Off)
		hi = max(hi, t.Off+t.NBytes)
		ok = true
	}
	if !ok {
		return 0, 0, false
	}
	return lo, hi - lo, true
}

// LayerBytes is the mapping covering block n, or nil.
//
// It is LayerRange resolved against the mapping, so a caller that prefetches
// or evicts one block does not re-derive the bounds arithmetic over untrusted
// input.
//
// It never panics, for any n: a negative n (converted unsigned, so it matches
// no prefix), a missing block or an empty one all give nil. The bound is
// re-checked here rather than inherited from validateTensorData's proof.
//
// The result aliases the mapping and is invalid after Close, exactly like Bytes.
func (f *File) LayerBytes(n int) []byte {
	off, size, ok := f.LayerRange(n)
	if !ok || size == 0 {
		return nil
	}
	lo := f.DataStart + off
	hi := lo + size
	if lo < f.DataStart || hi < lo || hi > uint64(len(f.Data)) {
		return nil // overflow, or a span the mapping does not cover
	}
	return f.Data[lo:hi:hi]
}

// TypeHistogram counts tensors and bytes per quantization type, largest first.
func (f *File) TypeHistogram() []TypeCount {
	m := map[quant.Type]*TypeCount{}
	for i := range f.Tensors {
		t := &f.Tensors[i]
		c := m[t.Type]
		if c == nil {
			c = &TypeCount{Type: t.Type}
			m[t.Type] = c
		}
		c.Count++
		c.Bytes += t.NBytes
	}
	out := make([]TypeCount, 0, len(m))
	for _, c := range m {
		out = append(out, *c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Bytes > out[j].Bytes })
	return out
}

// TypeCount is one row of a TypeHistogram.
type TypeCount struct {
	Type  quant.Type
	Count int
	Bytes uint64
}

// TensorBytes is the total size of all tensor data.
func (f *File) TensorBytes() uint64 {
	var n uint64
	for i := range f.Tensors {
		n += f.Tensors[i].NBytes
	}
	return n
}

// Arch is general.architecture: "llama", "gemma", "bert", ...
// Every hyperparameter key is namespaced under it.
func (f *File) Arch() string {
	s, _ := f.KV["general.architecture"].String()
	return s
}

// Key looks up an architecture-namespaced hyperparameter, e.g.
// Key("block_count") reads "llama.block_count" or "gemma.block_count".
func (f *File) Key(suffix string) (Value, bool) {
	v, ok := f.KV[f.Arch()+"."+suffix]
	return v, ok
}

// UintKey reads an architecture-namespaced integer, falling back to def when the
// key is absent. Absent is normal (gemma has no rope.freq_base), and a wrong
// default yields a model that runs and emits confident nonsense.
func (f *File) UintKey(suffix string, def uint64) uint64 {
	v, ok := f.Key(suffix)
	if !ok {
		return def
	}
	n, ok := v.Uint()
	if !ok {
		return def
	}
	return n
}

// FloatKey is UintKey for floats.
func (f *File) FloatKey(suffix string, def float64) float64 {
	v, ok := f.Key(suffix)
	if !ok {
		return def
	}
	n, ok := v.Float()
	if !ok {
		return def
	}
	return n
}

// itoa is a local copy, to avoid exporting one from package quant for one
// error string.
func itoa(v uint64) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

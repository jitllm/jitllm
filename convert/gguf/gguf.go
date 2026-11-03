// Package gguf parses GGUF model files into a meta.File. It is the converter's
// input and nothing else reaches it.
//
// The whole file is mmap'd read-only and every tensor is handed out as a
// subslice of that mapping. Nothing is copied and no tensor is ever
// materialized as []float32. The data model lives in package meta.
package gguf

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/quant"
)

const magic = 0x46554747 // "GGUF" little-endian

// Option configures how a GGUF is opened. The package reads no environment;
// cmd/jitllm maps JITLLM_NO_WARM onto WithoutWarm.
type Option func(*openCfg)

type openCfg struct{ noWarm bool }

// WithoutWarm skips the read-ahead pass over the mapping. The warm-up is what
// makes the first token fast; skipping it is for measuring what it buys.
func WithoutWarm(on bool) Option { return func(c *openCfg) { c.noWarm = on } }

// Open maps path and parses its header, metadata and tensor table, joining a
// split model's parts. Every field that indexes into the mapping is validated
// before Open returns; see validate.go. It never panics, for any input.
func Open(path string, opts ...Option) (*meta.File, error) {
	f, err := openOne(path, opts...)
	if err != nil {
		return nil, err
	}
	g, err := joinSplit(f, opts...)
	if err != nil {
		f.Close()
		return nil, err
	}
	return g, nil
}

// openOne is Open for a single file, without joining a split model's parts.
func openOne(path string, opts ...Option) (*meta.File, error) {
	var c openCfg
	for _, o := range opts {
		o(&c)
	}
	b, closeFn, err := mmapFile(path, c.noWarm)
	if err != nil {
		return nil, err
	}
	f, err := parse(b, b, path, closeFn)
	if err != nil {
		closeFn()
		return nil, err
	}
	return f, nil
}

// parse is Open without the mapping, so the fuzzer can hammer it directly on a
// []byte. It must return an error — never panic — for every possible input.
//
// The header is read from b, a prefix of whole, which is the file: the two are
// the same mapping for a local file, and for a streamed one (OpenAt) b is the
// part of the file read so far.
func parse(b, whole []byte, path string, closeFn func() error) (f *meta.File, err error) {
	r := &meta.Reader{B: b}
	if got := r.U32(); got != magic {
		return nil, fmt.Errorf("gguf: %s: not a GGUF file (magic %#08x)", path, got)
	}
	f = &meta.File{Path: path, Version: r.U32(), Data: whole, CloseFn: closeFn}
	nTensors := r.U64()
	nKV := r.U64()
	if r.Err != nil {
		return nil, r.Err
	}
	if f.Version != 2 && f.Version != 3 {
		return nil, fmt.Errorf("gguf: %s: unsupported version %d (want 2 or 3)", path, f.Version)
	}
	// Refuse counts the file cannot possibly hold, before allocating anything
	// sized by them: a tiny input claiming a million tensors must not become a
	// multi-gigabyte allocation.
	if err := checkCounts(nTensors, nKV, uint64(len(b))-r.O); err != nil {
		return nil, fmt.Errorf("gguf: %s: %w", path, err)
	}

	f.KV = make(map[string]meta.Value, nKV)
	for i := uint64(0); i < nKV; i++ {
		key := string(r.Str())
		vt := meta.ValueType(r.U32())
		if r.Err != nil {
			return nil, r.Err
		}
		v := meta.Value{Type: vt}
		switch {
		case vt == meta.Array:
			v.Elem = meta.ValueType(r.U32())
			v.N = r.U64()
			start := r.O
			r.O -= 12 // skipValue re-reads the elem type and count
			r.SkipValue(vt, 0)
			if r.Err == nil {
				v.Raw = b[start:r.O]
			}
		case vt == meta.String:
			v.Raw = r.Str()
		default:
			start := r.O
			r.SkipValue(vt, 0)
			if r.Err == nil {
				v.Raw = b[start:r.O]
			}
		}
		if r.Err != nil {
			return nil, fmt.Errorf("gguf: %s: kv[%d] %q: %w", path, i, key, r.Err)
		}
		f.KV[key] = v
	}

	// general.alignment must be read before it is used, and defaults to 32.
	// None of the three reference models actually sets it.
	f.Alignment = 32
	if v, ok := f.KV["general.alignment"]; ok {
		a, ok := v.Uint()
		if !ok {
			return nil, fmt.Errorf("gguf: %s: general.alignment is %v, not an integer", path, v.Type)
		}
		f.Alignment = a
	}
	if err := checkAlignment(f.Alignment); err != nil {
		return nil, fmt.Errorf("gguf: %s: %w", path, err)
	}

	f.Tensors = make([]meta.Tensor, 0, nTensors)
	index := make(map[string]int, nTensors)
	for i := uint64(0); i < nTensors; i++ {
		t := meta.Tensor{Name: string(r.Str())}
		nd := r.U32()
		if r.Err != nil {
			return nil, r.Err
		}
		if nd > maxDims {
			return nil, fmt.Errorf("gguf: %s: tensor %q has %d dimensions (max %d)", path, t.Name, nd, maxDims)
		}
		t.Dims = make([]uint64, nd)
		for j := range t.Dims {
			t.Dims[j] = r.U64()
		}
		t.Type = quant.Type(r.U32())
		t.Off = r.U64()
		if r.Err != nil {
			return nil, fmt.Errorf("gguf: %s: tensor[%d] %q: %w", path, i, t.Name, r.Err)
		}
		if err := sizeTensor(&t); err != nil {
			return nil, fmt.Errorf("gguf: %s: tensor %q: %w", path, t.Name, err)
		}
		if _, dup := index[t.Name]; dup {
			return nil, fmt.Errorf("gguf: %s: duplicate tensor name %q", path, t.Name)
		}
		index[t.Name] = len(f.Tensors)
		f.Tensors = append(f.Tensors, t)
	}

	f.Reindex()
	f.DataStart = alignUp(r.O, f.Alignment)
	if f.DataStart > uint64(len(whole)) {
		return nil, fmt.Errorf("gguf: %s: data section starts at %d, past end of file %d", path, f.DataStart, len(whole))
	}
	if err := validateTensorData(f); err != nil {
		return nil, fmt.Errorf("gguf: %s: %w", path, err)
	}
	return f, nil
}

// alignUp is the GGUF data section's padding rule, applied to the tensor table's
// end to find where the data starts.
func alignUp(v, a uint64) uint64 { return (v + a - 1) &^ (a - 1) }

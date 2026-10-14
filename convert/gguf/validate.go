package gguf

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/quant"
)

// A GGUF is untrusted input downloaded from the internet, and every field below
// indexes into the mapping. Each check here is one out-of-bounds read that
// does not happen. They are cheap: all of it is O(tensors), once, at Open.
//
// The counts are generous against real models (the largest has about 1,200
// tensors) and still tight enough that a hostile header cannot make Open
// allocate much: a tensor costs ~100 bytes of Go memory per 24 bytes of file.
const (
	maxTensors = 1 << 16
	maxKV      = 1 << 12
	maxDims    = 4 // ggml's GGML_MAX_DIMS
)

// The smallest possible on-disk encoding of one entry, used to reject header
// counts that the remaining bytes cannot hold:
//
//	kv:     u64 key length + u32 type + >=1 byte of value
//	tensor: u64 name length + u32 n_dims + u32 type + u64 offset
const (
	minKVBytes     = 8 + 4 + 1
	minTensorBytes = 8 + 4 + 4 + 8
)

func checkCounts(nTensors, nKV, remaining uint64) error {
	if nTensors > maxTensors || nKV > maxKV {
		return fmt.Errorf("implausible header: %d tensors, %d kv", nTensors, nKV)
	}
	if nKV > remaining/minKVBytes {
		return fmt.Errorf("header claims %d metadata entries but only %d bytes remain", nKV, remaining)
	}
	if nTensors > remaining/minTensorBytes {
		return fmt.Errorf("header claims %d tensors but only %d bytes remain", nTensors, remaining)
	}
	// Both sections must fit together, not just each alone.
	if nKV*minKVBytes+nTensors*minTensorBytes > remaining {
		return fmt.Errorf("header claims %d kv + %d tensors, which cannot fit in %d bytes", nKV, nTensors, remaining)
	}
	return nil
}

func checkAlignment(a uint64) error {
	if a == 0 || a&(a-1) != 0 {
		return fmt.Errorf("general.alignment %d is not a nonzero power of two", a)
	}
	if a > 1<<20 {
		return fmt.Errorf("general.alignment %d is implausible", a)
	}
	return nil
}

// sizeTensor validates a tensor's shape and type and fills in NBytes.
func sizeTensor(t *meta.Tensor) error {
	if t.Type == quant.I2_S {
		return sizeI2S(t)
	}
	if t.Type.Engine() {
		return fmt.Errorf("type %d is one of this engine's own formats, which no GGUF carries", uint32(t.Type))
	}
	if !t.Type.Known() {
		return fmt.Errorf("unknown ggml type %d", uint32(t.Type))
	}
	if len(t.Dims) == 0 {
		return fmt.Errorf("tensor has no dimensions")
	}
	be, bb := t.Type.BlockElems(), t.Type.BlockBytes()

	elems := uint64(1)
	for i, d := range t.Dims {
		if d == 0 {
			return fmt.Errorf("dimension %d is zero", i)
		}
		if elems > ^uint64(0)/d {
			return fmt.Errorf("shape %v overflows", t.Dims)
		}
		elems *= d
	}
	// ne[0] is the fastest-varying dimension and is the one that must tile the
	// block: a quantized row is a whole number of blocks, never a partial one.
	if t.Dims[0]%be != 0 {
		return fmt.Errorf("ne[0]=%d is not a multiple of %s's %d elements per block", t.Dims[0], t.Type, be)
	}
	blocks := elems / be
	if blocks > ^uint64(0)/bb {
		return fmt.Errorf("size overflows")
	}
	t.NBytes = blocks * bb
	return nil
}

// validateTensorData checks every tensor lies inside the mapping, and that the
// data section is exactly covered: offsets strictly increasing, with only
// alignment padding between neighbours.
//
// That last property holds on all three reference models, so it doubles as a
// free whole-file integrity check — a truncated or spliced download fails here
// rather than producing fluent nonsense 20 minutes later.
func validateTensorData(f *meta.File) error {
	avail := uint64(len(f.Data)) - f.DataStart
	var prevEnd uint64
	var prevName string
	for i := range f.Tensors {
		t := &f.Tensors[i]
		if t.Off > avail || t.NBytes > avail-t.Off {
			return fmt.Errorf("tensor %q spans [%d, %d) of a %d byte data section",
				t.Name, t.Off, t.Off+t.NBytes, avail)
		}
		if i > 0 {
			if t.Off < prevEnd {
				return fmt.Errorf("tensor %q at offset %d overlaps %q which ends at %d",
					t.Name, t.Off, prevName, prevEnd)
			}
			if gap := t.Off - prevEnd; gap >= f.Alignment {
				return fmt.Errorf("tensor %q at offset %d leaves a %d byte hole after %q (alignment is %d)",
					t.Name, t.Off, gap, prevName, f.Alignment)
			}
		} else if t.Off != 0 {
			return fmt.Errorf("first tensor %q starts at %d, not 0", t.Name, t.Off)
		}
		prevEnd, prevName = t.Off+t.NBytes, t.Name
	}
	return nil
}

// sizeI2S sizes bitnet.cpp's I2_S, which has no block: the codes of the whole
// tensor, four to a byte in 128-element groups, then a 32-byte tail holding
// its f32 scale (quant.I2SBytes).
func sizeI2S(t *meta.Tensor) error {
	if len(t.Dims) == 0 {
		return fmt.Errorf("tensor has no dimensions")
	}
	elems := uint64(1)
	for i, d := range t.Dims {
		if d == 0 {
			return fmt.Errorf("dimension %d is zero", i)
		}
		if elems > ^uint64(0)/d {
			return fmt.Errorf("shape %v overflows", t.Dims)
		}
		elems *= d
	}
	if t.Dims[0]%128 != 0 {
		return fmt.Errorf("ne[0]=%d is not a multiple of I2_S's 128-element groups", t.Dims[0])
	}
	t.NBytes = quant.I2SBytes(elems)
	return nil
}

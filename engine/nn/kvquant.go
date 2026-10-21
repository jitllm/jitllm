//go:build amd64 || arm64

package nn

import (
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// QuantizeKVRows is the q8 cache's append: n head rows of hd float32 (row r at
// src[r*hd:]) written to dst as n consecutive q8_0 rows of
// cpu.KVRowBytes(cpu.KVQ8, hd) bytes, through the generated activation
// quantizer at a 32-element window (cpu.KVFmt has the row's layout).
//
// A head width that is not a whole number of blocks is copied into a padded
// run first: the padding is zeros, which no amax counts and which quantize to
// zero, so the last block is exactly the reference's over the row.
//
// It runs on the caller's goroutine, outside any pool region, with spill space
// of its own, and allocates nothing once ReserveKVQuant has sized the padding.
func (f *JIT) QuantizeKVRows(dst, src []float32, hd, n int) {
	nb := (hd + cpu.KVQ8Block - 1) / cpu.KVQ8Block
	k := nb * cpu.KVQ8Block
	rs := cpu.KVRowBytes(cpu.KVQ8, hd) / 4
	if len(dst) < n*rs || len(src) < n*hd {
		panic("nn: QuantizeKVRows: a run shorter than its rows")
	}
	f.ReserveKVQuant(hd)
	qk := f.quantKernel(quant.Q8_0)
	for r := 0; r < n; r++ {
		x := src[r*hd : (r+1)*hd]
		if hd != k {
			copy(f.kvPad, x)
			clear(f.kvPad[hd:k])
			x = f.kvPad[:k]
		}
		row := dst[r*rs : (r+1)*rs]
		q := unsafe.Slice((*int8)(unsafe.Pointer(&row[0])), k)
		qk.Run(quant.Q8_0, q, row[k/4:k/4+2*nb], nil, f.kvScr, x, k, 0, nb, cpu.KVQ8Block)
	}
}

// WidenKVRows is QuantizeKVRows' inverse: n q8_0 head rows of hd elements
// widened to n float32 rows of hd, each element d*q -- the exact value the q8
// attention kernels read -- through the generated widening (cpu.EmitKVWiden).
// The kernel is emitted on the first call at a head width.
func (f *JIT) WidenKVRows(dst, src []float32, hd, n int) {
	if n <= 0 {
		return
	}
	if len(dst) < n*hd || len(src) < n*cpu.KVRowBytes(cpu.KVQ8, hd)/4 {
		panic("nn: WidenKVRows: a run shorter than its rows")
	}
	c := f.kvWiden[hd]
	if c == nil {
		b, err := f.em.KVWiden(hd)
		if err != nil {
			panic(err)
		}
		c = mustMap("kv_widen_hd"+itoaN(hd), b)
		if f.kvWiden == nil {
			f.kvWiden = map[int]*cpu.Code{}
		}
		f.kvWiden[hd] = c
	}
	c.Call(&cpu.Args{Out: &dst[0], W: (*byte)(unsafe.Pointer(&src[0])), Rows: int64(n)})
}

// ReserveKVQuant sizes QuantizeKVRows' padding and spill space for head width
// hd, so a model that provisions its cache at load runs the append without
// allocating.
func (f *JIT) ReserveKVQuant(hd int) {
	k := (hd + cpu.KVQ8Block - 1) / cpu.KVQ8Block * cpu.KVQ8Block
	if len(f.kvPad) < k {
		f.kvPad = make([]float32, k)
	}
	if len(f.kvScr) < cpu.QuantActNarrowScratch {
		f.kvScr = make([]float32, cpu.QuantActNarrowScratch)
	}
	f.quantKernel(quant.Q8_0)
}

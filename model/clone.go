package model

import (
	"slices"
	"unsafe"
)

// CloneTo makes dst an instance's own copy of block l: every binding a page-in
// writes is dst's -- each matrix and its packed planes, a bank's expert views,
// the attention heads' views, a prediction block's own weights -- so binding dst (BindBlock) leaves l and every other copy as they
// were. The vectors are read once at open and never bound, and stay shared.
func (l *Layer) CloneTo(dst *Layer) {
	*dst = *l
	var w *mtpWeights
	if l.Mtp != nil {
		c := *l.Mtp
		w = &c
		dst.Mtp = w
	}
	dst.Experts = slices.Clone(l.Experts)
	dst.kbHead = slices.Clone(l.kbHead)
	dst.vbHead = slices.Clone(l.vbHead)
	// A matrix is recorded by the field that holds it, and the field moves
	// with the copy; one held by no field (a tower matrix with no slot of its
	// own) is the copy's own tensor.
	moved := func(t *Tensor) *Tensor {
		at := func(base unsafe.Pointer, size uintptr, to unsafe.Pointer) (*Tensor, bool) {
			p, b := uintptr(unsafe.Pointer(t)), uintptr(base)
			if p < b || p >= b+size {
				return nil, false
			}
			return (*Tensor)(unsafe.Add(to, p-b)), true
		}
		if n, ok := at(unsafe.Pointer(l), unsafe.Sizeof(*l), unsafe.Pointer(dst)); ok {
			return n
		}
		if w != nil {
			if n, ok := at(unsafe.Pointer(l.Mtp), unsafe.Sizeof(*l.Mtp), unsafe.Pointer(w)); ok {
				return n
			}
		}
		if len(l.Experts) > 0 {
			size := uintptr(len(l.Experts)) * unsafe.Sizeof(l.Experts[0])
			if n, ok := at(unsafe.Pointer(&l.Experts[0]), size, unsafe.Pointer(&dst.Experts[0])); ok {
				return n
			}
		}
		c := *t
		return &c
	}
	// Each tensor the copy binds holds its own packed planes: bindOne writes
	// them in place.
	own := func(t *Tensor) {
		if t.Packed != nil {
			p := *t.Packed
			t.Packed = &p
		}
	}
	dst.Mats = make([]roleTensor, len(l.Mats))
	for i, m := range l.Mats {
		dst.Mats[i] = roleTensor{m.Role, moved(m.T)}
		own(dst.Mats[i].T)
	}
	for i := range dst.Experts {
		own(&dst.Experts[i].Gate)
		own(&dst.Experts[i].Up)
		own(&dst.Experts[i].Down)
	}
	for i := range dst.kbHead {
		own(&dst.kbHead[i])
	}
	for i := range dst.vbHead {
		own(&dst.vbHead[i])
	}
	if w != nil {
		own(&w.EhProj)
		own(&w.Head)
		own(&w.Embd)
	}
}

package cpudev

import (
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/internal/offheap"
)

// ownOffHeap is the words past which an own buffer is anonymous memory
// rather than the heap's: an attention's scores are rows by heads by the
// context's keys, which the kernel commits as a step's rows reach them, and
// which on the heap would be the collector's to account and mark (RULE 2f).
const ownOffHeap = 1 << 14

// Own is form f's own buffers on d, made at the form's first run on it:
// each zeroed but for its constant words.
func (d *Device) Own(f *Form) [][]uint32 {
	if b, ok := d.own[f]; ok {
		return b
	}
	if d.own == nil {
		d.own = map[*Form][][]uint32{}
	}
	b := make([][]uint32, len(f.Own))
	for i, n := range f.Own {
		b[i] = d.ownWords(max(n, 1))
		copy(b[i], f.Consts[i])
	}
	d.own[f] = b
	return b
}

// ownWords is n words of an own buffer, zeroed.
func (d *Device) ownWords(n int) []uint32 {
	if n >= ownOffHeap {
		if m, mapped, err := offheap.MapOffHeap(4 * n); err == nil && mapped {
			d.mapped = append(d.mapped, m)
			return unsafe.Slice((*uint32)(unsafe.Pointer(&m[0])), n)
		}
	}
	return make([]uint32, n)
}

// dropWords gives back w, words ownWords made, before d closes: a buffer a
// larger one replaced.
func (d *Device) dropWords(w []uint32) {
	if len(w) == 0 {
		return
	}
	p := unsafe.Pointer(&w[0])
	for i, m := range d.mapped {
		if unsafe.Pointer(&m[0]) == p {
			offheap.UnmapOffHeap(m)
			d.mapped = slices.Delete(d.mapped, i, i+1)
			return
		}
	}
}

// Close returns the own buffers d holds off the heap.
func (d *Device) Close() {
	for _, m := range d.mapped {
		offheap.UnmapOffHeap(m)
	}
	d.mapped, d.own = nil, nil
}

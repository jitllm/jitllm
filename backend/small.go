package backend

import (
	"slices"
	"sync"
)

// smallMax is the largest buffer a device carves from a shared piece. CUDA
// hands an allocation of 1 MiB or less out in multiples of 64 KiB
// (cudaFootprint) and Vulkan every allocation (vulkanFootprint), so a
// block's norms, biases and scale planes, a few KiB each, cost a piece
// apiece. Carved from one, sixteen 4 KiB buffers cost what one did.
const smallMax = 16 << 10

// smallMin is the smallest slot: a multiple of the alignment a launch reads a
// buffer's words at, and of 256, the most a Vulkan device's
// minStorageBufferOffsetAlignment may be, so every slot binds where a view
// must.
const smallMin = 256

// smallClasses is the slot sizes, smallMin doubling to smallMax.
const smallClasses = 7

// viewer is a device buffer a slot is a view of.
type viewer[B any] interface {
	View(off, n int) (B, error)
}

// smallPool is a device's pieces for small buffers. Each piece is one
// smallPage allocation cut into equal slots of one class; open holds the
// pieces of each class with a slot free.
type smallPool[B viewer[B]] struct {
	mu   sync.Mutex
	open [smallClasses][]*smallPiece[B]
}

// smallPiece is one piece: its buffer and count, its class, the slots it
// has handed out and the offsets of those it has not.
type smallPiece[B any] struct {
	b     B
	acc   *counted
	class int
	used  int
	free  []int
}

// smallClass is the class of an n-byte buffer: the smallest slot that holds it.
func smallClass(n int) int {
	k := 0
	for smallMin<<k < n {
		k++
	}
	return k
}

// smallFootprint is what an n-byte small buffer costs the card: its slot.
// The piece a slot is the first of costs more, by at most a piece a class
// over any run of buffers (AllocCounter.Slack).
func smallFootprint(n int) int { return smallMin << smallClass(n) }

// take is a slot for an n-byte buffer and its view, from a piece of its
// class with one free, else from a piece newPiece makes, whose count cnt
// takes as the card's spending.
func (p *smallPool[B]) take(n int, cnt *allocCount, newPiece func() (B, error)) (*smallPiece[B], B, int, error) {
	k := smallClass(n)
	p.mu.Lock()
	defer p.mu.Unlock()
	var none B
	if len(p.open[k]) == 0 {
		b, err := newPiece()
		if err != nil {
			return nil, none, 0, err
		}
		pc := &smallPiece[B]{b: b, acc: cnt.take(0, smallPage), class: k}
		for off := smallPage - smallMin<<k; off >= 0; off -= smallMin << k {
			pc.free = append(pc.free, off)
		}
		p.open[k] = append(p.open[k], pc)
	}
	open := p.open[k]
	pc := open[len(open)-1]
	off := pc.free[len(pc.free)-1]
	v, err := pc.b.View(off, n)
	if err != nil {
		return nil, none, 0, err
	}
	pc.free = pc.free[:len(pc.free)-1]
	pc.used++
	if len(pc.free) == 0 {
		p.open[k] = open[:len(open)-1]
	}
	return pc, v, off, nil
}

// give hands slot off of piece pc back, reporting whether it was the
// piece's last: the caller then frees the piece's buffer and gives its count.
func (p *smallPool[B]) give(pc *smallPiece[B], off int) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	pc.free = append(pc.free, off)
	pc.used--
	open := p.open[pc.class]
	if len(pc.free) == 1 {
		open = append(open, pc)
	}
	last := pc.used == 0
	if last {
		open = slices.DeleteFunc(open, func(q *smallPiece[B]) bool { return q == pc })
	}
	p.open[pc.class] = open
	return last
}

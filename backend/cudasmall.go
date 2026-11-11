package backend

import (
	"errors"
	"slices"
	"sync"

	"github.com/jitllm/jitllm/backend/cuda"
)

// smallMax is the largest buffer CUDA's Alloc carves from a shared piece: an
// allocation of 1 MiB or less costs the card a multiple of 64 KiB
// (cudaFootprint), so a block's norms, biases and scale planes, a few KiB
// each, cost a piece apiece. Carved from one, sixteen 4 KiB buffers cost
// what one did.
const smallMax = 16 << 10

// smallMin is the smallest slot: a multiple of the alignment a launch reads a
// buffer's words at.
const smallMin = 256

// smallClasses is the slot sizes, smallMin doubling to smallMax.
const smallClasses = 7

// smallPool is a device's pieces for small buffers. Each piece is one
// smallPage cuMemAlloc cut into equal slots of one class; open holds the
// pieces of each class with a slot free.
type smallPool struct {
	mu   sync.Mutex
	open [smallClasses][]*smallPiece
}

// smallPiece is one piece: its buffer and count, its class, the slots it
// has handed out and the offsets of those it has not.
type smallPiece struct {
	b     *cuda.Buffer
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
// over any run of buffers (Slack).
func smallFootprint(n int) int { return smallMin << smallClass(n) }

// allocSmall is an n-byte buffer in a slot of a piece of its class, a new
// piece made where none has one free. The piece's count holds the card's
// spending, the slot's the bytes asked for.
func (c *cudaDev) allocSmall(n int) (Buf, error) {
	if c.closed() {
		return nil, errCUDAClosed
	}
	k := smallClass(n)
	c.small.mu.Lock()
	defer c.small.mu.Unlock()
	if len(c.small.open[k]) == 0 {
		var b *cuda.Buffer
		var err error = errCUDAClosed
		c.do(func() { b, err = c.d.Alloc(smallPage) })
		if err != nil {
			if errors.Is(err, cuda.ErrOutOfMemory) {
				c.ctr.allocRefused.Add(1)
			}
			return nil, err
		}
		p := &smallPiece{b: b, acc: c.take(0, smallPage), class: k}
		for off := smallPage - smallMin<<k; off >= 0; off -= smallMin << k {
			p.free = append(p.free, off)
		}
		c.small.open[k] = append(c.small.open[k], p)
	}
	open := c.small.open[k]
	p := open[len(open)-1]
	off := p.free[len(p.free)-1]
	v, err := p.b.View(off, n)
	if err != nil {
		return nil, err
	}
	p.free = p.free[:len(p.free)-1]
	p.used++
	if len(p.free) == 0 {
		c.small.open[k] = open[:len(open)-1]
	}
	return &cudaBuf{b: v, dev: c, acc: c.take(n, 0), piece: p, off: off}, nil
}

// closed reports that Close has run: a piece made before it has no context
// behind it, and a slot of one is no buffer.
func (c *cudaDev) closed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.reqs == nil
}

// freeSmall hands slot off of piece p back once the context has finished
// with it, as cuMemFree waits for the work before it, and frees the piece
// with its last slot.
func (c *cudaDev) freeSmall(p *smallPiece, off int) {
	c.do(func() { c.drv.Sync() })
	c.small.mu.Lock()
	defer c.small.mu.Unlock()
	p.free = append(p.free, off)
	p.used--
	open := c.small.open[p.class]
	if len(p.free) == 1 {
		open = append(open, p)
	}
	if p.used == 0 {
		open = slices.DeleteFunc(open, func(q *smallPiece) bool { return q == p })
		c.do(func() { p.b.Free() })
		p.acc.give()
	}
	c.small.open[p.class] = open
}

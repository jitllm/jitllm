package backend

import (
	"errors"

	"github.com/jitllm/jitllm/backend/cuda"
)

// allocSmall is an n-byte buffer in a slot of a shared piece (smallPool),
// the piece one cuMemAlloc. The piece's count holds the card's spending, the
// slot's the bytes asked for.
func (c *cudaDev) allocSmall(n int) (Buf, error) {
	if c.closed() {
		return nil, errCUDAClosed
	}
	p, v, off, err := c.small.take(n, &c.allocCount, func() (*cuda.Buffer, error) {
		var b *cuda.Buffer
		var err error = errCUDAClosed
		c.do(func() { b, err = c.d.Alloc(smallPage) })
		if errors.Is(err, cuda.ErrOutOfMemory) {
			c.ctr.allocRefused.Add(1)
		}
		return b, err
	})
	if err != nil {
		return nil, err
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
func (c *cudaDev) freeSmall(p *smallPiece[*cuda.Buffer], off int) {
	c.do(func() { c.drv.Sync() })
	if c.small.give(p, off) {
		c.do(func() { p.b.Free() })
		p.acc.give()
	}
}

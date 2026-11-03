package gpu

import (
	"encoding/binary"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/graph"
)

// markArena is the rotary marks of every sequence the device's sessions hold
// (graph.OpRopePos, nn.RopeMarksDevice): one buffer, a block a session's
// slot -- its count, then its marks -- where the session's devSess places it
// (markAt), a block outgrowing its room moved to a larger one at the arena's
// end. Word 0 is an empty block, the marks of a sequence told none. It is
// written where a sequence's marks change, never by a call.
type markArena struct {
	buf        backend.Buf
	size, used int
}

// freeMarks gives the arena back, refunded. Callers hold g.mu, with no
// session's marks in it or none to keep.
func (g *devTier) freeMarks() {
	a := &g.marks
	if a.buf != nil {
		a.buf.Free()
		g.refund(uint64(4 * a.size))
	}
	*a = markArena{}
}

// marksPlaced says a session's marks are in the arena: past word 0, the
// empty block every sequence told none reads.
func (a *markArena) marksPlaced() bool { return a.used > 1 }

// SetSeqMarks places slot's rotary marks on every device (nn.RopeMarksDevice).
func (s *gpuSession) SetSeqMarks(slot int, block []uint32) error {
	s.g.mu.Lock()
	ds := s.g.devs
	s.g.mu.Unlock()
	for _, d := range ds {
		d.mu.Lock()
		err := d.placeMarks(s.sid, slot, block)
		d.mu.Unlock()
		if err != nil {
			return err
		}
	}
	return nil
}

// placeMarks writes session sid's slot's marks block into the arena: in its
// block, or one twice as large at the arena's end where it has outgrown it.
// Callers hold g.mu.
func (g *devTier) placeMarks(sid uint64, slot int, block []uint32) error {
	ds := g.sessOf(sid)
	for len(ds.markAt) <= slot {
		ds.markAt, ds.markRoom = append(ds.markAt, 0), append(ds.markRoom, 0)
	}
	a := &g.marks
	if err := g.growMarks(1); err != nil {
		return err
	}
	if ds.markAt[slot] == 0 || ds.markRoom[slot] < len(block) {
		room := max(1+4*graph.MarkWords, 2*len(block))
		if err := g.growMarks(a.used + room); err != nil {
			return err
		}
		ds.markAt[slot], ds.markRoom[slot] = a.used, room
		a.used += room
	}
	return a.buf.WriteAt(4*ds.markAt[slot], u32view(block))
}

// growMarks makes the arena hold words words, word 0 its empty block: a
// larger buffer with the old one's blocks copied in. Callers hold g.mu.
func (g *devTier) growMarks(words int) error {
	a := &g.marks
	if a.buf != nil && words <= a.size {
		return nil
	}
	size := max(words, 2*a.size, 1024)
	b, err := g.dev.Alloc(4 * size)
	if err != nil {
		return err
	}
	if err := b.Write(make([]byte, 4*size)); err != nil {
		b.Free()
		return err
	}
	if a.buf != nil {
		if err := g.dev.Copy(b, 0, a.buf, 0, 4*a.used); err != nil {
			b.Free()
			return err
		}
		a.buf.Free()
		g.refund(uint64(4 * a.size))
	}
	// The device's, charged as itself, never a scratch build's.
	g.charge(uint64(4 * size))
	a.buf, a.size, a.used = b, size, max(a.used, 1)
	// A recording names the old buffer by address.
	g.dropGraph()
	return nil
}

// ropeRowsWords is a call's R rows as the rotary tables read them
// (composer.TableRopeRows): the live count nrow, then each row's cache position
// and where its sequence's marks start, a row past the live ones the last
// live one's. Session sid's one sequence at pos.. where rag is nil, else each
// row's session and sequence (ragStep.seqLen).
func (g *devTier) ropeRowsWords(sid uint64, pos, nrow, R int, rag *ragStep) []byte {
	b := binary.LittleEndian.AppendUint32(g.subBytes.ropeRows[:0], uint32(nrow))
	for r := range R {
		i := min(r, max(nrow, 1)-1)
		p, s, slot := pos+i, sid, 0
		if rag != nil && i < len(rag.pos) {
			p = rag.pos[i]
			if rag.sid != nil {
				s = rag.sid[i]
			} else if rag.self != 0 {
				s = rag.self
			}
			if rag.seqLen > 0 {
				slot = (rag.slot[i] - rag.pos[i]) / rag.seqLen
			}
		}
		off := 0
		if ds := g.sess[s]; ds != nil && slot < len(ds.markAt) {
			off = ds.markAt[slot]
		}
		if f := g.kb.ropeFault; f[0] != 0 {
			p = f[0]*p + f[1]
		}
		b = binary.LittleEndian.AppendUint32(b, uint32(p))
		b = binary.LittleEndian.AppendUint32(b, uint32(off))
	}
	g.subBytes.ropeRows = b
	return b
}

// rowPosWords is a call's R rows' cache positions as a block reading them
// takes them (composer.TablePos): the live count nrow, then a position a row, a
// row past the live ones the last live one's.
func (g *devTier) rowPosWords(pos, nrow, R int, rag *ragStep) []byte {
	b := binary.LittleEndian.AppendUint32(g.subBytes.rowPos[:0], uint32(nrow))
	for r := range R {
		i := min(r, max(nrow, 1)-1)
		p := pos + i
		if rag != nil && i < len(rag.pos) {
			p = rag.pos[i]
		}
		b = binary.LittleEndian.AppendUint32(b, uint32(p))
	}
	g.subBytes.rowPos = b
	return b
}

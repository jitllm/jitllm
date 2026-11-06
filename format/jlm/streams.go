package jlm

import "fmt"

// Streams is a residual of several streams the converter states: their
// count, what each block's sublayers read from them and how each output
// joins the running residual, the stream a block banks its input into, and
// the row the head reads. Stream 0 is the running residual; the engine knows
// nothing of why a model has the others.
type Streams struct {
	// N is the streams, the running residual among them; 0 or 1 is none.
	N uint8
	// Expand is each stream past the first as the input rows make it from
	// the embedding, a program over it (nil a copy of it); empty where the
	// input rows write stream 0 alone.
	Expand []*Program
	// Blocks is each block's reads and writes, one a block, the prediction
	// blocks after the model's.
	Blocks []BlockStreams
	// Head is the row the head reads, its Join unused.
	Head SubStreams
}

// BlockStreams is one block's use of the streams.
type BlockStreams struct {
	// Bank is the stream the block's input is written into before any
	// read; 0 is none.
	Bank uint8
	// Attn and FFN are each sublayer's input and join.
	Attn, FFN SubStreams
}

// SubStreams is a sublayer's read of the streams and the join of its output.
type SubStreams struct {
	// From are Read's inputs, in order: a stream (its value now), StreamIn
	// or StreamRun.
	From []int8
	// Read is the sublayer's input over From; nil is the running residual.
	Read *Program
	// Exports are values of Read the writes read after the sublayer's output.
	Exports []uint16
	// Join is the running residual after the sublayer, a program over the
	// running residual and the sublayer's output; nil is their sum.
	Join *Program
	// Writes is, where the sublayer writes every stream, each stream after
	// it: a program over the stream, the sublayer's output and Exports (nil
	// leaves the stream as it is). Empty where Join is the write.
	Writes []*Program
}

const (
	// StreamRun is the running residual where the read is made.
	StreamRun int8 = -1
	// StreamIn is stream 0 as the block received it.
	StreamIn int8 = -2
)

func (s *Streams) valid(nLayer int) error {
	if s.N <= 1 {
		if s.N == 1 || len(s.Blocks) != 0 || len(s.Head.From) != 0 || s.Head.Read != nil {
			return fmt.Errorf("jlm: %d streams state %d blocks", s.N, len(s.Blocks))
		}
		return nil
	}
	if len(s.Blocks) != nLayer {
		return fmt.Errorf("jlm: the streams state %d blocks of %d", len(s.Blocks), nLayer)
	}
	if n := len(s.Expand); n != 0 && n != int(s.N)-1 {
		return fmt.Errorf("jlm: the streams expand %d of %d", n, s.N-1)
	}
	for _, p := range s.Expand {
		if p == nil {
			continue
		}
		if p.NIn != 1 {
			return fmt.Errorf("jlm: a stream's expansion of %d inputs", p.NIn)
		}
		if err := p.valid("expansion"); err != nil {
			return err
		}
	}
	for i := range s.Blocks {
		b := &s.Blocks[i]
		if int(b.Bank) >= int(s.N) {
			return fmt.Errorf("jlm: block %d banks into stream %d of %d", i, b.Bank, s.N)
		}
		for _, u := range []*SubStreams{&b.Attn, &b.FFN} {
			if err := u.valid(s.N, fmt.Sprintf("block %d", i)); err != nil {
				return err
			}
		}
	}
	return s.Head.valid(s.N, "the head")
}

func (u *SubStreams) valid(n uint8, where string) error {
	for _, k := range u.From {
		if k < StreamIn || int(k) >= int(n) {
			return fmt.Errorf("jlm: %s reads stream %d of %d", where, k, n)
		}
	}
	for _, e := range u.Exports {
		if u.Read == nil || int(e) >= int(u.Read.NIn)+len(u.Read.Ops) {
			return fmt.Errorf("jlm: %s exports value %d of its read", where, e)
		}
	}
	if w := len(u.Writes); w != 0 && (w != int(n) || u.Join != nil) {
		return fmt.Errorf("jlm: %s writes %d of %d streams (join %v)", where, w, n, u.Join != nil)
	}
	for _, p := range u.Writes {
		if p == nil {
			continue
		}
		if int(p.NIn) != 2+len(u.Exports) {
			return fmt.Errorf("jlm: %s writes a stream through a program of %d inputs, want %d", where, p.NIn, 2+len(u.Exports))
		}
		if err := p.valid(where); err != nil {
			return err
		}
	}
	switch {
	case u.Read == nil && len(u.From) != 0:
		return fmt.Errorf("jlm: %s reads %d streams through no program", where, len(u.From))
	case u.Read != nil && int(u.Read.NIn) != len(u.From):
		return fmt.Errorf("jlm: %s reads %d streams through a program of %d inputs", where, len(u.From), u.Read.NIn)
	case u.Join != nil && u.Join.NIn != 2:
		return fmt.Errorf("jlm: %s joins through a program of %d inputs", where, u.Join.NIn)
	}
	for _, p := range []*Program{u.Read, u.Join} {
		if p == nil {
			continue
		}
		// A program may return one of its inputs as it is, with no op.
		if len(p.Ops) == 0 {
			if p.Out >= p.NIn {
				return fmt.Errorf("jlm: %s returns value %d of %d", where, p.Out, p.NIn)
			}
			continue
		}
		if err := p.valid(where); err != nil {
			return err
		}
	}
	return nil
}

func encodeStreams(w *wbuf, s *Streams) {
	w.u8(s.N)
	w.u32(uint32(len(s.Blocks)))
	for i := range s.Blocks {
		b := &s.Blocks[i]
		w.u8(b.Bank)
		encodeSub(w, &b.Attn)
		encodeSub(w, &b.FFN)
	}
	encodeSub(w, &s.Head)
}

// encodeSub writes a read and its join, the sixteenth tail's part of it.
func encodeSub(w *wbuf, u *SubStreams) {
	w.u32(uint32(len(u.From)))
	for _, k := range u.From {
		w.u8(uint8(k))
	}
	for _, p := range []*Program{u.Read, u.Join} {
		w.bool(p != nil)
		if p != nil {
			encodeProgram(w, p)
		}
	}
}

// writesStreams reports that the streams expand or a read exports or
// writes: the nineteenth tail.
func (s *Streams) writesStreams() bool {
	if len(s.Expand) != 0 {
		return true
	}
	subs := []*SubStreams{&s.Head}
	for i := range s.Blocks {
		subs = append(subs, &s.Blocks[i].Attn, &s.Blocks[i].FFN)
	}
	for _, u := range subs {
		if len(u.Exports) != 0 || len(u.Writes) != 0 {
			return true
		}
	}
	return false
}

// encodeStreamWrites is the nineteenth tail: the expansions, then each
// block's sublayers' exports and writes, then the head's.
func encodeStreamWrites(w *wbuf, s *Streams) {
	encodePrograms(w, s.Expand)
	w.u32(uint32(len(s.Blocks)))
	for i := range s.Blocks {
		encodeWrites(w, &s.Blocks[i].Attn)
		encodeWrites(w, &s.Blocks[i].FFN)
	}
	encodeWrites(w, &s.Head)
}

func encodeWrites(w *wbuf, u *SubStreams) {
	w.u32(uint32(len(u.Exports)))
	for _, e := range u.Exports {
		w.u16(e)
	}
	encodePrograms(w, u.Writes)
}

// encodePrograms writes a list of programs, each present or nil.
func encodePrograms(w *wbuf, ps []*Program) {
	w.u32(uint32(len(ps)))
	for _, p := range ps {
		w.bool(p != nil)
		if p != nil {
			encodeProgram(w, p)
		}
	}
}

func decodePrograms(r *rbuf) []*Program {
	var ps []*Program
	for range r.count(1) {
		var p *Program
		if r.bool() {
			p = &Program{}
			decodeProgram(r, p)
		}
		ps = append(ps, p)
	}
	return ps
}

func decodeStreams(r *rbuf, s *Streams) {
	s.N = r.u8()
	for range r.count(3) {
		var b BlockStreams
		b.Bank = r.u8()
		decodeSub(r, &b.Attn)
		decodeSub(r, &b.FFN)
		s.Blocks = append(s.Blocks, b)
	}
	decodeSub(r, &s.Head)
}

func decodeSub(r *rbuf, u *SubStreams) {
	for range r.count(1) {
		u.From = append(u.From, int8(r.u8()))
	}
	for _, p := range []**Program{&u.Read, &u.Join} {
		if r.bool() {
			*p = &Program{}
			decodeProgram(r, *p)
		}
	}
}

// decodeStreamWrites reads the nineteenth tail into s, whose blocks the
// sixteenth stated.
func decodeStreamWrites(r *rbuf, s *Streams) {
	s.Expand = decodePrograms(r)
	if n := int(r.u32()); r.err == nil && n != len(s.Blocks) {
		r.err = fmt.Errorf("jlm: the streams' writes state %d blocks of %d", n, len(s.Blocks))
		return
	}
	for i := range s.Blocks {
		decodeWrites(r, &s.Blocks[i].Attn)
		decodeWrites(r, &s.Blocks[i].FFN)
	}
	decodeWrites(r, &s.Head)
}

func decodeWrites(r *rbuf, u *SubStreams) {
	for range r.count(2) {
		u.Exports = append(u.Exports, r.u16())
	}
	u.Writes = decodePrograms(r)
}

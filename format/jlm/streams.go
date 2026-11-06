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
	// Blocks is each block's reads and writes, one a block.
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
	// From are Read's inputs, in order: StreamIn, StreamRun, or a stream
	// past the first.
	From []int8
	// Read is the sublayer's input over From; nil is the running residual.
	Read *Program
	// Join is the running residual after the sublayer, a program over the
	// running residual and the sublayer's output; nil is their sum.
	Join *Program
}

const (
	// StreamIn is stream 0 as the block received it.
	StreamIn int8 = 0
	// StreamRun is the running residual where the read is made.
	StreamRun int8 = -1
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
		if k < StreamRun || int(k) >= int(n) {
			return fmt.Errorf("jlm: %s reads stream %d of %d", where, k, n)
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

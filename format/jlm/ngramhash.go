package jlm

// heads is how many heads a hash over n-grams up to span reads.
func (h *NGramHash) heads(span int32) int { return int(span-1) * int(h.HeadsPer) }

// fits reports a hash an n-gram op of span and width can read: a head count
// that divides the width, a range a head, a multiplier a token.
func (h *NGramHash) fits(span, width int32) bool {
	if h == nil || h.HeadsPer == 0 {
		return false
	}
	n := h.heads(span)
	return n > 0 && int(width)%n == 0 && len(h.Off) == n && len(h.Voc) == n && len(h.Mul) >= int(span)
}

func encodeNGramHash(w *wbuf, h *NGramHash) {
	w.bool(h != nil)
	if h == nil {
		return
	}
	w.u32(h.EOS)
	w.u32(h.Image)
	w.u32(h.HeadsPer)
	for _, s := range [][]uint64{h.Mul, h.Off, h.Voc} {
		w.u32(uint32(len(s)))
		for _, v := range s {
			w.u64(v)
		}
	}
}

func decodeNGramHash(r *rbuf) *NGramHash {
	if !r.bool() {
		return nil
	}
	h := &NGramHash{EOS: r.u32(), Image: r.u32(), HeadsPer: r.u32()}
	for _, s := range []*[]uint64{&h.Mul, &h.Off, &h.Voc} {
		for range r.count(8) {
			*s = append(*s, r.u64())
		}
	}
	return h
}

// states reports that a block's statement holds an op of code c.
func (s *Streams) states(c OpCode) bool {
	for i := range s.Blocks {
		b := &s.Blocks[i]
		for _, sub := range []*SubStreams{&b.Pre, &b.Attn, &b.FFN} {
			for _, p := range append([]*Program{sub.Read, sub.Join, sub.Post}, sub.Writes...) {
				if p == nil {
					continue
				}
				for _, op := range p.Ops {
					if op.Code == c {
						return true
					}
				}
			}
		}
	}
	return false
}

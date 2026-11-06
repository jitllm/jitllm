package jlm

import "fmt"

// Selection is how blocks' attention selects the history it reads: an
// indexer scores the cached positions and attention reads the best. Its key
// rides each cached row after the attention's own, and the converter states
// what the reference computes; the engine knows no indexer by name.
type Selection struct {
	// Blocks is each block's: true where it selects, the prediction blocks
	// after the model's.
	Blocks []bool
	// Key is a position's key, a program over the block's normed input,
	// cached KeyWidth wide (zeros past the program's result). Query is the
	// query, a program over the normed input and the attention's query
	// latent (the normed input again where the attention has none). Weight,
	// where stated, is a weight a query head, a program over the normed
	// input.
	Key, Query, Weight *Program
	KeyWidth           uint32
	// Heads query heads of HeadDim score Groups keys: one per kv group
	// (Groups == the attention's kv heads) or one every head reads.
	Heads, HeadDim, Groups uint32
	// Rank is how a block of Block positions is ranked, Keep of them read,
	// the Local ending at the query's own forced in.
	Rank               Rank
	Block, Keep, Local uint32
	// Relu passes each head's score through relu before the heads' sum.
	Relu bool
	// PoolNorm is the pooled key's RMS norm (RankPooled), at Eps.
	PoolNorm Role
	Eps      float32
}

// Rank is how a selection ranks a block of positions.
type Rank uint8

const (
	// RankBest ranks a block by its best position's score.
	RankBest Rank = 1
	// RankPooled ranks a block by its keys pooled by their max, RMS-normed
	// and turned by the block's first position's rotary before scoring; the
	// block holding the query is always read.
	RankPooled Rank = 2
)

// At reports that block li selects.
func (s *Selection) At(li int) bool {
	return s != nil && li >= 0 && li < len(s.Blocks) && s.Blocks[li]
}

func (s *Selection) valid(nLayer int) error {
	if s == nil {
		return nil
	}
	switch {
	case len(s.Blocks) != nLayer:
		return fmt.Errorf("jlm: the selection states %d blocks of %d", len(s.Blocks), nLayer)
	case s.Key == nil || s.Query == nil:
		return fmt.Errorf("jlm: a selection with no key or no query")
	case s.Key.NIn != 1 || s.Query.NIn != 2 || s.Weight != nil && s.Weight.NIn != 1:
		return fmt.Errorf("jlm: a selection's programs read %d, %d inputs", s.Key.NIn, s.Query.NIn)
	case s.Rank != RankBest && s.Rank != RankPooled:
		return fmt.Errorf("jlm: selection rank %d is not one this format defines", s.Rank)
	case s.Heads == 0 || s.HeadDim == 0 || s.Groups == 0 || s.KeyWidth < s.HeadDim:
		return fmt.Errorf("jlm: a selection of %d heads of %d over %d keys %d wide", s.Heads, s.HeadDim, s.Groups,
			s.KeyWidth)
	case s.Block == 0 || s.Keep == 0 || s.Local > s.Keep:
		return fmt.Errorf("jlm: a selection keeping %d blocks of %d, %d local", s.Keep, s.Block, s.Local)
	}
	for _, p := range []*Program{s.Key, s.Query, s.Weight} {
		if p == nil {
			continue
		}
		if err := p.valid("selection"); err != nil {
			return err
		}
	}
	return nil
}

func encodeSelection(w *wbuf, s *Selection) {
	w.bool(s != nil)
	if s == nil {
		return
	}
	w.u32(uint32(len(s.Blocks)))
	for _, b := range s.Blocks {
		w.bool(b)
	}
	encodePrograms(w, []*Program{s.Key, s.Query, s.Weight})
	for _, v := range []uint32{s.KeyWidth, s.Heads, s.HeadDim, s.Groups, s.Block, s.Keep, s.Local} {
		w.u32(v)
	}
	w.u8(uint8(s.Rank))
	w.bool(s.Relu)
	w.u32(uint32(s.PoolNorm))
	w.f32(s.Eps)
}

func decodeSelection(r *rbuf) *Selection {
	if !r.bool() {
		return nil
	}
	s := &Selection{}
	for range r.count(1) {
		s.Blocks = append(s.Blocks, r.bool())
	}
	ps := decodePrograms(r)
	if len(ps) != 3 {
		if r.err == nil {
			r.err = fmt.Errorf("jlm: a selection of %d programs", len(ps))
		}
		return nil
	}
	s.Key, s.Query, s.Weight = ps[0], ps[1], ps[2]
	for _, v := range []*uint32{&s.KeyWidth, &s.Heads, &s.HeadDim, &s.Groups, &s.Block, &s.Keep, &s.Local} {
		*v = r.u32()
	}
	s.Rank = Rank(r.u8())
	s.Relu = r.bool()
	s.PoolNorm = Role(r.u32())
	s.Eps = r.f32()
	return s
}

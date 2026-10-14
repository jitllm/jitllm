package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// QSA: Qwen3.8-Flash-Next's block selection (transformers'
// Qwen4ExpTextQSAIndexer, llama.cpp's build_qsa_top_k).
//
// The indexer projects the block's input to IdxHeads queries and one key of
// IdxHeadDim. The key is cached raw -- no norm, no rotary -- as one more kv
// head of the attention row (Config.KVRowAt; its value zero, the head padded
// to HeadDim), so it pages, relocates and is prefix-cached with k. A query
// at p sees its visible positions in blocks of QSARatio: every complete
// block's keys are averaged, RMSNormed and rotated at the block's first
// position, and scored sum_h relu(q_h . key_b), each query head RMSNormed and
// rotated at p; attention reads the IdxTopK/QSARatio best complete blocks and
// the incomplete tail, through a -inf mask added before the softmax. Under
// IdxTopK + QSARatio visible positions every key is read and the mask is
// nil. transformers scales the score by 1/sqrt(IdxHeadDim) and llama.cpp
// does not; a positive scale on a sum of ReLUs selects the same blocks, and
// the engine takes llama.cpp's.

// qsaScratch is a State's QSA buffers.
type qsaScratch struct {
	q, key           []float32 // the decode row's query heads and raw key
	pooled           []float32 // every complete block's key
	score, tmp, bias []float32
	cs               []float32 // one block's rotary table
	kept             []bool
	ord              nn.SampleOrder
	dot              *nn.AttnSet
	// per row of a chunk: queries, keys and masks.
	bq, bk, bbias []float32
	masks         [][]float32
	rows          int
}

// loadQSA reads block bi's indexer tensors and checks their shapes.
func loadQSA(l *layer, bi int32, cfg *Config,
	get func(jlm.Role, int32, int32) (tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if l.idxQ, err = get(jlm.RoleIdxQ, bi, -1); err != nil {
		return err
	}
	if l.idxK, err = get(jlm.RoleIdxK, bi, -1); err != nil {
		return err
	}
	if l.idxQNorm, err = vec(jlm.RoleIdxQNorm, bi); err != nil {
		return err
	}
	if l.idxKNorm, err = vec(jlm.RoleIdxKNorm, bi); err != nil {
		return err
	}
	h, d := cfg.IdxHeads, cfg.IdxHeadDim
	if l.idxQ.k != cfg.NEmbd || l.idxQ.rows != h*d || l.idxK.k != cfg.NEmbd || l.idxK.rows != d ||
		len(l.idxQNorm) != d || len(l.idxKNorm) != d {
		return fmt.Errorf("model: block %d's QSA indexer is q %dx%d, k %dx%d, norms %d/%d; "+
			"want %d heads of %d over a %d-wide input", bi, l.idxQ.k, l.idxQ.rows, l.idxK.k,
			l.idxK.rows, len(l.idxQNorm), len(l.idxKNorm), h, d, cfg.NEmbd)
	}
	return nil
}

// allocQSA sizes the decode token's selection buffers.
func (s *State) allocQSA() {
	c := s.m.Cfg
	if c.QSARatio == 0 {
		return
	}
	g := &qsaScratch{}
	s.q4.qsa = g
	nb := s.maxSeq/c.QSARatio + 1
	g.q = make([]float32, c.IdxHeads*c.IdxHeadDim)
	g.key = make([]float32, c.HeadDim)
	g.pooled = make([]float32, nb*c.IdxHeadDim)
	g.score, g.tmp = make([]float32, nb), make([]float32, nb)
	g.bias = make([]float32, s.attStride)
	g.cs = make([]float32, c.NRot)
	g.kept = make([]bool, nb)
	g.ord.Reserve(nb)
	g.dot = s.attnSetF32(c.IdxHeadDim)
}

// growQSARows sizes a chunk's: each row's query, key and mask.
func (s *State) growQSARows(n int) {
	c := s.m.Cfg
	g := s.qsa()
	if g == nil || g.rows >= n {
		return
	}
	g.rows = n
	g.bq = make([]float32, n*c.IdxHeads*c.IdxHeadDim)
	g.bk = make([]float32, n*c.HeadDim)
	g.bbias = make([]float32, n*s.attStride)
	g.masks = make([][]float32, n)
}

func (s *State) qsa() *qsaScratch {
	if s.q4 == nil {
		return nil
	}
	return s.q4.qsa
}

// qsaProject writes one position's raw key (padded to HeadDim, the cached
// head) into key and its query heads, each normed and rotated at cs, into q.
func (s *State) qsaProject(key, q []float32, l *layer, h, cs []float32) error {
	c := s.m.Cfg
	clear(key)
	if err := s.mv(key[:c.IdxHeadDim], l.idxK, h); err != nil {
		return err
	}
	if err := s.mv(q, l.idxQ, h); err != nil {
		return err
	}
	s.qsaQueryRest(q, l, cs)
	return nil
}

// qsaQueryRest is the query's norm and rotary, after its projection.
func (s *State) qsaQueryRest(q []float32, l *layer, cs []float32) {
	c := s.m.Cfg
	d := c.IdxHeadDim
	for h := 0; h < c.IdxHeads; h++ {
		t := q[h*d : (h+1)*d]
		s.rmsnorm(t, t, l.idxQNorm, c.RMSEps)
	}
	if c.NRot > 0 {
		nn.RoPE32JIT(q, d, cs, s.m.rope.Neox)
	}
	s.jit.NewInput()
}

// qsaMask fills bias (n long) with the selection of a query whose window is
// the first n cached positions of sequence slot in layer li's history, and
// returns it, or nil when every position is read. The window's pages must be
// resident (kvEnsureWindow).
//
// The history read is o's, the scratch s's: a step's rows may be other
// sessions' sequences.
func (s *State) qsaMask(o *State, li, slot, n int, q, bias []float32, l *layer) []float32 {
	c, g := s.m.Cfg, s.qsa()
	R, d := c.QSARatio, c.IdxHeadDim
	nb := n / R
	keep := c.IdxTopK / R
	if c.q4eFault == q4eFaultDenseQSA || nb <= keep {
		return nil
	}
	if c.q4eFault == q4eFaultNoTail && n%R == 0 {
		// The violation's selection still has a tail to drop: below.
	}
	// Each complete block's key: the mean of its raw keys, normed, rotated at
	// its first position.
	for b := 0; b < nb; b++ {
		p := g.pooled[b*d : (b+1)*d]
		o.qsaKeyAt(li, slot, b*R, p)
		if c.q4eFault != q4eFaultQSAFirstKey {
			for t := 1; t < R; t++ {
				o.qsaAddKey(li, slot, b*R+t, p)
			}
			s.scale(p, 1/float32(R))
		}
		s.rmsnorm(p, p, l.idxKNorm, c.RMSEps)
		if c.NRot > 0 {
			s.jit.RopeTable(s.m.rope, g.cs, o.ropeOf(slot, b*R))
			nn.RoPE32JIT(p, d, g.cs, s.m.rope.Neox)
		}
	}
	s.jit.NewInput()
	score := g.score[:nb]
	clear(score)
	for h := 0; h < c.IdxHeads; h++ {
		g.dot.AttnScores(g.tmp[:nb], g.pooled[:nb*d], q[h*d:(h+1)*d], nb)
		nn.Act32JIT(g.tmp[:nb], nn.ActReLU)
		s.axpy(score, g.tmp[:nb], 1)
	}
	s.jit.NewInput()
	kept := g.kept[:nb]
	clear(kept)
	g.ord.Begin(score)
	for have := 0; have < keep; {
		_, b, ok := g.ord.Next()
		if !ok {
			break
		}
		if !kept[b] {
			kept[b] = true
			have++
		}
	}
	ninf := float32(math.Inf(-1))
	row := bias[:n]
	for b := 0; b < nb; b++ {
		v := ninf
		if kept[b] {
			v = 0
		}
		for t := b * R; t < (b+1)*R; t++ {
			row[t] = v
		}
	}
	tail := float32(0)
	if c.q4eFault == q4eFaultNoTail {
		tail = ninf
	}
	for t := nb * R; t < n; t++ {
		row[t] = tail
	}
	return row
}

// qsaKeyAt copies position pos's cached raw key (layer li, slot) into dst.
func (s *State) qsaKeyAt(li, slot, pos int, dst []float32) {
	c := s.m.Cfg
	k, _ := s.kv.layers[li].at(s.kvlAt(li), slot, pos, c.NKVHead)
	copy(dst, k[:len(dst)])
}

// qsaAddKey adds position pos's cached raw key into dst.
func (s *State) qsaAddKey(li, slot, pos int, dst []float32) {
	c := s.m.Cfg
	k, _ := s.kv.layers[li].at(s.kvlAt(li), slot, pos, c.NKVHead)
	s.axpy(dst, k[:len(dst)], 1)
}

// qsaProjectRows fills every row's raw key (bk, padded to HeadDim) and query
// heads (bq, normed and rotated at the row's table) from a chunk's n block
// inputs in bh.
func (s *State) qsaProjectRows(l *layer, n int, cs, csSWA []float32, li int) error {
	c, g := s.m.Cfg, s.qsa()
	d, hq := c.IdxHeadDim, c.IdxHeads*c.IdxHeadDim
	s.growQSARows(n)
	clear(g.bk[:n*c.HeadDim])
	for i := 0; i < n; i++ {
		if err := s.mv(g.bk[i*c.HeadDim:i*c.HeadDim+d], l.idxK, s.bh[i*c.NEmbd:(i+1)*c.NEmbd]); err != nil {
			return err
		}
	}
	if err := s.mm(g.bq[:n*hq], l.idxQ, s.bh, n); err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		s.qsaQueryRest(g.bq[i*hq:(i+1)*hq], l, s.ropeTable(li, cs, csSWA, i))
	}
	return nil
}

// qsaRowOf is row i's cache vectors for a QSA block, assembled in kc and vc
// from its k and v and its raw indexer key in bk.
func (s *State) qsaRowOf(li, i int, k, v []float32) ([]float32, []float32) {
	c, g := s.m.Cfg, s.qsa()
	kvr := c.KVRowAt(li)
	msaRow(s.kc[:kvr], s.vc[:kvr], k, v, g.bk[i*c.HeadDim:(i+1)*c.HeadDim])
	return s.kc[:kvr], s.vc[:kvr]
}

// qsaRows computes the selection for each of n rows before a batched
// attention: row i's query heads are in bq, its history is slot(i)'s first
// win(i) positions of layer li, and its mask lands in bbias at i*attStride
// (masks[i], nil for a dense row).
func (s *State) qsaRows(li int, l *layer, n int, owner func(i int) (*State, int), win func(i int) int,
	masks [][]float32) {
	c, g := s.m.Cfg, s.qsa()
	hq := c.IdxHeads * c.IdxHeadDim
	for i := 0; i < n; i++ {
		o, slot := owner(i)
		masks[i] = s.qsaMask(o, li, slot, win(i), g.bq[i*hq:(i+1)*hq], g.bbias[i*s.attStride:(i+1)*s.attStride], l)
	}
}

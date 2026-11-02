package model

import "github.com/jitllm/jitllm/jit"

// A compressing block's entries (ds4kv.go) as the composed CPU binds them:
// an arena like a layer's history, reserved one entry past those the
// instance's positions fold. That last entry is the sink: a row completing
// no entry writes its pool there, and no row's entry range reaches it.

// useEntArenas gives every compressing block's entries the model's arena,
// for an instance whose CPU blocks run composed.
func (kc *kvCache) useEntArenas(maxSeq int, arenaFor func(li, pp int, latent bool, minSlots int) (*kvArena, error)) error {
	for li := kc.lo; li < len(kc.ent); li++ {
		if r := kc.entRate(li); r > 0 && maxSeq > 0 {
			if err := kc.ent[li].useArena(li, maxSeq/r+1, arenaFor); err != nil {
				return err
			}
		}
	}
	return nil
}

// entWrite is the entry a row at pos of block li writes: the one it
// completes, else the sink.
func (kc *kvCache) entWrite(li, pos int) int {
	e, r := &kc.ent[li], kc.entRate(li)
	if (pos+1)%r == 0 {
		return pos / r
	}
	return e.reserved*e.p - 1
}

// cpuEntReady makes the entry pages block li's rows at pos write and read
// resident in the entries' arena, or says why they are not: each row's
// entry (or the sink) written, every entry before the last row's read.
func (s *Instance) cpuEntReady(li int, pos ...int) jit.Decline {
	r := s.kv.entRate(li)
	if r == 0 {
		return jit.DeclineServes
	}
	e := &s.kv.ent[li]
	if e.arena == nil {
		return jit.DeclinePageOut
	}
	end := 0
	for _, p := range pos {
		n := s.kv.entWrite(li, p) / e.p
		e.grow(n)
		e.cow(n)
		end = max(end, (p+1)/r)
	}
	for n := 0; n*e.p < end; n++ {
		if !e.arenaPage(n) {
			return jit.DeclinePageOut
		}
	}
	return jit.DeclineServes
}

// cpuEntBind binds block li's entries to b for its rows of sequences seq at
// positions pos, the descriptors in t (4 words a row): the sequence's
// table, the entries it reads, the one it writes.
func (s *Instance) cpuEntBind(b *cpuBound, li int, t []uint32, seq, pos []int) {
	r := s.kv.entRate(li)
	if r == 0 {
		return
	}
	e := &s.kv.ent[li]
	slots := max(e.arenaSlots(), 1)
	for i, p := range pos {
		q := 0
		if seq != nil {
			q = seq[i]
		}
		w := t[4*i : 4*i+4]
		w[0], w[1], w[2], w[3] = uint32(q*slots), 0, uint32((p+1)/r), uint32(s.kv.entWrite(li, p))
	}
	b.ent, b.entPages, b.entRows = e, e.table(max(s.nseq, 1)), t
}

// cpuTokens keeps the step's n rows' tokens, row i's id(i), for the blocks
// that bind them (a mixture routed by token id), padded for a call's form
// past the last row.
func (s *Instance) cpuTokens(n int, id func(i int) int32) {
	need := n + cpuRowsCap
	if cap(s.cpu.tok) < need {
		s.cpu.tok = make([]uint32, need)
	}
	s.cpu.tok = s.cpu.tok[:need]
	for i := range n {
		s.cpu.tok[i] = uint32(id(i))
	}
	clear(s.cpu.tok[n:])
}

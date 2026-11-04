package model

import (
	"context"
	"fmt"
)

// sessBatch is nseq sessions leasing one instance, driven as the gates on
// batching drive a batch: a token a row a step (StepRuns), a row's prompt its
// session's prefill, a retired row's session started over. The instance is
// embedded for placement (SetDevice, GPULayers, ...); a row's history is
// sess[i]'s.
type sessBatch struct {
	*Instance
	pool   *Pool
	sess   []*Session
	runs   []Run
	out    []float32
	nseq   int
	maxSeq int
}

// newSessBatch is nseq sessions of up to maxSeq positions on a fresh instance
// of m.
func newSessBatch(m *Model, nseq, maxSeq int) *sessBatch {
	in := m.NewInstance(maxSeq)
	pool := NewPool()
	if err := pool.Serve(in, nseq); err != nil {
		panic(err)
	}
	return &sessBatch{Instance: in, pool: pool, sess: make([]*Session, nseq), runs: make([]Run, nseq), nseq: nseq, maxSeq: maxSeq}
}

// row is row i's session, opened at its first use: after the gate placed
// the instance, whose placement may have narrowed its history's format.
func (b *sessBatch) row(i int) *Session {
	if b.sess[i] == nil {
		s, err := b.pool.NewSession(b.maxSeq)
		if err != nil {
			panic(err)
		}
		b.sess[i] = s
	}
	return b.sess[i]
}

// ForwardBatch steps token i of every row at the row's next position and
// returns the rows' logits back to back.
func (b *sessBatch) ForwardBatch(tokens []int32) ([]float32, error) {
	if len(tokens) != b.nseq {
		return nil, fmt.Errorf("model: ForwardBatch got %d tokens for %d sequences", len(tokens), b.nseq)
	}
	for i := range b.sess {
		s := b.row(i)
		if err := s.Acquire(context.Background(), PhaseGenerate); err != nil {
			return nil, err
		}
		b.runs[i] = Run{Session: s, Tokens: tokens[i : i+1], Logits: true}
	}
	lg, err := StepRuns(b.runs)
	if err != nil {
		return nil, err
	}
	b.out = b.out[:0]
	for _, l := range lg {
		b.out = append(b.out, l...)
	}
	return b.out, nil
}

// ForwardBatchGreedy is ForwardBatch with each row's greedy token.
func (b *sessBatch) ForwardBatchGreedy(tokens []int32) ([]int32, error) {
	if _, err := b.ForwardBatch(tokens); err != nil {
		return nil, err
	}
	out := make([]int32, b.nseq)
	for i := range out {
		out[i] = Greedy(b.BatchLogits(i))
	}
	return out, nil
}

// BatchLogits is row i of the last ForwardBatch.
func (b *sessBatch) BatchLogits(i int) []float32 {
	nv := b.c.NVocab
	return b.out[i*nv : (i+1)*nv]
}

// SeqPos is row i's position.
func (b *sessBatch) SeqPos(i int) int { return b.row(i).Pos() }

// PrefillSeq runs tokens into row i's session and returns its logits.
func (b *sessBatch) PrefillSeq(i int, tokens []int32) ([]float32, error) {
	if i < 0 || i >= b.nseq {
		return nil, fmt.Errorf("model: sequence slot %d out of range for a batch of %d", i, b.nseq)
	}
	var lg []float32
	err := b.row(i).Run(context.Background(), PhasePrefill, func(in *Instance) error {
		var err error
		lg, err = in.Prefill(tokens)
		return err
	})
	return lg, err
}

// PrefillSeqMixed is PrefillSeq for a mixed prompt.
func (b *sessBatch) PrefillSeqMixed(i int, spans ...Span) ([]float32, error) {
	var lg []float32
	err := b.row(i).Run(context.Background(), PhasePrefill, func(in *Instance) error {
		var err error
		lg, err = in.PrefillMixed(spans...)
		return err
	})
	return lg, err
}

// Retire starts row i's sequence over.
func (b *sessBatch) Retire(i int) {
	b.row(i).Run(context.Background(), PhasePrefill, func(in *Instance) error {
		in.Reset()
		return nil
	})
}

// Close closes the sessions and the instance.
func (b *sessBatch) Close() error {
	for _, s := range b.sess {
		if s != nil {
			s.Close()
		}
	}
	b.pool.Unserve(b.Instance)
	return b.Instance.Close()
}

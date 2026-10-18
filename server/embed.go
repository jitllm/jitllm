package server

import (
	"context"
	"fmt"
	"time"

	"github.com/jitllm/jitllm/engine/model"
)

// Embeddings: one vector per input from an embedding model's pooled readout.
//
// It is the engine's Embedder and nothing else: the Connect InferenceService
// and /v1/embeddings both call Engine.Embed, which tokenizes as `jitllm embed`
// does and hands each sequence to model.Embedder.Embed, so a vector served here
// is the vector the CLI prints for the same model and text.
//
// An embedder runs on the host -- an encoder has no device tier, and a decoder
// embedding model's attention may be bidirectional, which the device does not
// implement -- so a request queues on the host gate like any host session, and
// all of its inputs run in that one turn through one embedder.

// EmbedOptions is Embed's input: the Connect RPC and the OpenAI shim both build
// one.
type EmbedOptions struct {
	// Model is a loaded model's id, or the file name it was loaded from.
	Model string
	// Inputs are PromptText or PromptIDs, embedded in order.
	Inputs []Prompt
	// Dimensions is the vector width asked for. Zero means the model's own.
	Dimensions int
}

// EmbedResult is one vector per input, in input order.
type EmbedResult struct {
	ModelID    string
	Vectors    [][]float32
	Tokens     []int
	Pooling    string
	QueuedFor  time.Duration
	QueueDepth int32
	Took       time.Duration
}

// TotalTokens is what usage reports: every input's tokens, specials included.
func (r *EmbedResult) TotalTokens() int {
	n := 0
	for _, t := range r.Tokens {
		n += t
	}
	return n
}

// Embed runs every input through the model's embedder.
func (e *Engine) Embed(ctx context.Context, o EmbedOptions) (*EmbedResult, error) {
	if len(o.Inputs) == 0 {
		return nil, fmt.Errorf("%w: embed needs at least one input", ErrInvalid)
	}
	lm, err := e.resolveModel(o.Model)
	if err != nil {
		return nil, err
	}
	m := lm.m
	if !m.IsEmbedding() {
		return nil, fmt.Errorf("%w: model %q (%s) is not an embedding model: its container states no "+
			"pooling, so its last residual has no readout the weights were trained for; generate from it instead",
			ErrInvalid, lm.id, m.Cfg.Arch)
	}
	dim := m.Cfg.NEmbd
	if o.Dimensions != 0 && o.Dimensions != dim {
		return nil, fmt.Errorf("%w: model %q embeds in %d dimensions and its container records no "+
			"Matryoshka training, so a %d-dimension prefix would not be an embedding in the same space; "+
			"omit dimensions or pass %d", ErrInvalid, lm.id, dim, o.Dimensions, dim)
	}

	seqs := make([][]int32, len(o.Inputs))
	for i, p := range o.Inputs {
		switch p.Kind {
		case PromptText:
			if m.Vocab == nil {
				return nil, fmt.Errorf("%w: model %q has no tokenizer (%v); send token ids", ErrInvalid, lm.id, m.TokErr)
			}
			seqs[i] = m.EmbedIDs(p.Text)
		case PromptIDs:
			for _, id := range p.IDs {
				if id < 0 || int(id) >= m.Cfg.NVocab {
					return nil, fmt.Errorf("%w: input %d carries token id %d, outside the vocabulary of %d",
						ErrInvalid, i, id, m.Cfg.NVocab)
				}
			}
			seqs[i] = p.IDs
		default:
			return nil, fmt.Errorf("%w: input %d is neither text nor token ids", ErrInvalid, i)
		}
		if len(seqs[i]) == 0 {
			return nil, fmt.Errorf("%w: input %d is empty", ErrInvalid, i)
		}
	}

	gs := e.gatesFor(lm.gateIDs())
	var waited time.Duration
	depth := gs.acquire(e.nextID("emb"))
	defer gs.release()

	emb, err := lm.takeEmbedder()
	if err != nil {
		return nil, err
	}
	defer lm.giveEmbedder(emb)
	if limit := emb.MaxTokens(); limit > 0 {
		for i, ids := range seqs {
			if len(ids) > limit {
				return nil, fmt.Errorf("%w: input %d is %d tokens and model %q embeds at most %d; "+
					"a truncated input would be an embedding of a different text, so shorten it",
					ErrInvalid, i, len(ids), lm.id, limit)
			}
		}
	}

	res := &EmbedResult{
		ModelID:    lm.id,
		Vectors:    make([][]float32, len(seqs)),
		Tokens:     make([]int, len(seqs)),
		Pooling:    m.Pooling().String(),
		QueuedFor:  waited,
		QueueDepth: depth,
	}
	t0 := time.Now()
	for i, ids := range seqs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		v, err := emb.Embed(ids)
		if err != nil {
			return nil, err
		}
		// The embedder's slice is overwritten by its next call.
		res.Vectors[i] = append([]float32(nil), v...)
		res.Tokens[i] = len(ids)
	}
	res.Took = time.Since(t0)
	lm.tokensPrefilled.Add(int64(res.TotalTokens()))
	return res, nil
}

// embedders is a model's idle embedders. An Embedder is one sequence at a time
// and owns a worker pool (an encoder) or a session (a decoder), so it is built
// once and reused. Nothing bounds how many run at once: each concurrent request
// takes one (building it when the list is empty), so the list grows to the
// highest concurrency the model has seen and keeps that many idle.
type embedders struct {
	free   []*model.Embedder
	busy   int
	closed bool
	idle   chan struct{} // closed when busy reaches zero after close
}

func (lm *LoadedModel) takeEmbedder() (*model.Embedder, error) {
	lm.embMu.Lock()
	if lm.emb.closed {
		lm.embMu.Unlock()
		return nil, fmt.Errorf("%w: model %q was unloaded", ErrNotFound, lm.id)
	}
	lm.emb.busy++
	if n := len(lm.emb.free); n > 0 {
		x := lm.emb.free[n-1]
		lm.emb.free = lm.emb.free[:n-1]
		lm.embMu.Unlock()
		return x, nil
	}
	lm.embMu.Unlock()
	x, err := lm.m.NewEmbedder()
	if err != nil {
		lm.giveEmbedder(nil)
		return nil, err
	}
	// Its blocks go where a session's would: an encoder's through its
	// segment, a decoder's through its State.
	if lm.device() != nil {
		if err := x.SetDeviceLayers(lm.device(), lm.deviceBlocks()); err != nil {
			x.Close()
			lm.giveEmbedder(nil)
			return nil, err
		}
	}
	return x, nil
}

// giveEmbedder returns an embedder to the free list, or closes it when the
// model is being unloaded -- before the model, which is the order the engine
// requires of everything holding a State.
func (lm *LoadedModel) giveEmbedder(x *model.Embedder) {
	lm.embMu.Lock()
	defer lm.embMu.Unlock()
	lm.emb.busy--
	if lm.emb.closed {
		if x != nil {
			x.Close()
		}
		if lm.emb.busy == 0 {
			close(lm.emb.idle)
		}
		return
	}
	if x != nil {
		lm.emb.free = append(lm.emb.free, x)
	}
}

// closeEmbedders closes the idle embedders and waits for the busy ones to come
// back, so none outlives the model. An embed in flight runs to its end.
func (lm *LoadedModel) closeEmbedders() {
	lm.embMu.Lock()
	if lm.emb.closed {
		lm.embMu.Unlock()
		return
	}
	lm.emb.closed = true
	free := lm.emb.free
	lm.emb.free = nil
	lm.emb.idle = make(chan struct{})
	busy := lm.emb.busy
	idle := lm.emb.idle
	lm.embMu.Unlock()
	for _, x := range free {
		x.Close()
	}
	if busy > 0 {
		<-idle
	}
}

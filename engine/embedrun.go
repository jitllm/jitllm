package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/nn"
)

// Embedder is the working memory for embedding one sequence at a time. It is
// not safe for concurrent use; make one per goroutine.
type Embedder struct {
	m      *model.Model
	st     *Instance
	pooled []float32
	// skipDense reads the vector out before any projection head, matching
	// llama.cpp on a GGUF converted without the dense modules. Tests only.
	skipDense bool
}

// NewEmbedder builds the working memory Embed needs, an instance of m with
// opts. A model whose container carries no pooling is refused: its last
// residual has no readout the weights were trained for.
func NewEmbedder(m *model.Model, opts ...Option) (*Embedder, error) {
	if !m.IsEmbedding() {
		return nil, fmt.Errorf("model: %s is not an embedding model (its container sets no pooling)", m.Cfg.Arch)
	}
	c := m.Cfg
	st, err := NewInstance(m, min(c.NCtx, model.EmbedMaxSeq), opts...)
	if err != nil {
		return nil, err
	}
	return &Embedder{m: m, st: st, pooled: make([]float32, st.poolWidth())}, nil
}

// Close releases the embedder's session.
func (e *Embedder) Close() { e.st.Close() }

// MaxTokens is the longest sequence Embed takes: the session's context.
func (e *Embedder) MaxTokens() int { return e.st.maxSeq }

// EmbedText tokenizes text the way the model's own tokenizer does -- the
// special tokens it adds, and for a last-token pooled model the EOS it pools
// -- and embeds it.
func (e *Embedder) EmbedText(text string) ([]float32, error) {
	if e.m.Vocab == nil {
		return nil, fmt.Errorf("model: no tokenizer: %v", e.m.TokErr)
	}
	return e.Embed(e.m.EmbedIDs(text))
}

// Embed runs one token sequence and returns its pooled, L2-normalised
// embedding. The slice is the embedder's own and is overwritten by the next
// call; copy it to keep it.
func (e *Embedder) Embed(ids []int32) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("model: Embed called with no tokens")
	}
	if n := e.MaxTokens(); n > 0 && len(ids) > n {
		return nil, fmt.Errorf("model: %d tokens, and this model embeds at most %d", len(ids), n)
	}
	if err := e.st.embedPrefill(ids, e.pooled, e.skipDense); err != nil {
		return nil, err
	}
	if e.skipDense {
		return e.pooled[:e.m.Cfg.NEmbd], nil
	}
	return e.pooled, nil
}

// SetDeviceLayers places the session's blocks on d (Instance.SetDeviceLayers):
// max caps how many are offered, -1 offers every one. A bidirectional
// model's (EmbeddingGemma's) blocks are planned non-causal and placed the
// same way.
func (e *Embedder) SetDeviceLayers(d nn.LayerDevice, max int) error {
	return e.st.SetDeviceLayers(d, max)
}

// DeviceBlocks is how many of the session's blocks run on a device.
func (e *Embedder) DeviceBlocks() int { return e.st.GPULayers() }

// DeviceDeclines is why the device refused the blocks it did not take.
func (e *Embedder) DeviceDeclines() []DeviceDecline { return e.st.DeviceDeclines() }

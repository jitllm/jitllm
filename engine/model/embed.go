package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// Embeddings: a sequence in, one L2-normalised vector out. An embedding
// model (qwen3-embedding, EmbeddingGemma) is a language model whose container
// says how to pool its last residual and whether attention is bidirectional,
// and runs through Instance.embedPrefill:
//
//	e, _ := m.NewEmbedder()
//	defer e.Close()
//	v, _ := e.EmbedText("some text")

// embedMaxSeq caps an embedding session's context: the KV cache is committed
// as it grows, so this bounds the worst case and costs nothing below it.
const embedMaxSeq = 8192

// Embedder is the working memory for embedding one sequence at a time. It is
// not safe for concurrent use; make one per goroutine.
type Embedder struct {
	m      *Model
	st     *Instance
	pooled []float32
	// skipDense reads the vector out before any projection head, matching
	// llama.cpp on a GGUF converted without the dense modules. Tests only.
	skipDense bool
}

// Pooling is how this model reads an embedding out of its last residual, or
// jlm.PoolNone for a model that is not an embedding model.
func (m *Model) Pooling() jlm.Pooling { return m.Cfg.Pooling }

// IsEmbedding reports whether the container describes an embedding model.
func (m *Model) IsEmbedding() bool { return m.Cfg.Pooling != jlm.PoolNone }

// NewEmbedder builds the working memory Embed needs. A model whose container
// carries no pooling is refused: its last residual has no readout the weights
// were trained for.
func (m *Model) NewEmbedder() (*Embedder, error) {
	if !m.IsEmbedding() {
		return nil, fmt.Errorf("model: %s is not an embedding model (its container sets no pooling)", m.Cfg.Arch)
	}
	c := m.Cfg
	st := m.NewInstance(min(c.NCtx, embedMaxSeq))
	return &Embedder{m: m, st: st, pooled: make([]float32, st.cpuPoolWidth())}, nil
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

// EmbedIDs is the token sequence EmbedText embeds.
//
// For a last-token pooled model the EOS is part of the input: qwen3-embedding
// pools at its <|endoftext|> (add_eos_token). tok.Encode does not add it,
// because a generation prompt must never carry one, so this is the one caller
// that acts on the flag.
func (m *Model) EmbedIDs(text string) []int32 {
	v := m.Vocab
	ids := v.Encode(text, true)
	if v.AddEOS && v.EOS >= 0 && (len(ids) == 0 || ids[len(ids)-1] != v.EOS) {
		ids = append(ids, v.EOS)
	}
	return ids
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

// loadDenseHeads binds an embedding model's projection heads, when the
// container carries them. Both or neither: one alone is a model this graph
// cannot express.
func loadDenseHeads(c *jlm.File, m *Model) error {
	has1 := c.Has(jlm.RoleEmbdDense1, jlm.DenseBlock, -1)
	has2 := c.Has(jlm.RoleEmbdDense2, jlm.DenseBlock, -1)
	if !has1 && !has2 {
		return nil
	}
	if has1 != has2 {
		return fmt.Errorf("model: the container carries one of two dense heads")
	}
	for _, h := range []struct {
		role jlm.Role
		dst  *tensor
	}{{jlm.RoleEmbdDense1, &m.dense1}, {jlm.RoleEmbdDense2, &m.dense2}} {
		e, _ := c.Find(h.role, jlm.DenseBlock, -1)
		typ, ok := jlm.SourceType(e.Type)
		if !ok || e.NDim != 2 {
			return fmt.Errorf("model: %v is %v in %d dimensions", h.role, e.Type, e.NDim)
		}
		qs, _, _ := c.Span(e)
		*h.dst = tensor{e: e, typ: typ, data: qs, rows: int(e.Dims[1]), k: int(e.Dims[0])}
		if jlm.Packed(e.Type) {
			h.dst.packed = &nn.Packed{}
		}
	}
	if m.dense1.k != m.Cfg.NEmbd || m.dense2.k != m.dense1.rows || m.dense2.rows != m.Cfg.NEmbd {
		return fmt.Errorf("model: dense heads %dx%d then %dx%d do not map %d back to itself",
			m.dense1.rows, m.dense1.k, m.dense2.rows, m.dense2.k, m.Cfg.NEmbd)
	}
	return nil
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

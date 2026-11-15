package model

import (
	"fmt"

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
const EmbedMaxSeq = 8192

// Pooling is how this model reads an embedding out of its last residual, or
// jlm.PoolNone for a model that is not an embedding model.
func (m *Model) Pooling() jlm.Pooling { return m.Cfg.Pooling }

// IsEmbedding reports whether the container describes an embedding model.
func (m *Model) IsEmbedding() bool { return m.Cfg.Pooling != jlm.PoolNone }

// EmbedIDs is the token sequence EmbedText embeds.
//
// For a last-token pooled model the EOS is part of the input: qwen3-embedding
// pools at its <|endoftext|> (add_eos_token). tok.Encode does not add it,
// because a generation prompt must never carry one, so this is the one caller
// that acts on the flag.
func (m *Model) EmbedIDs(text string) []int32 {
	v := m.Vocab
	ids := v.EncodeSpecial(text, true)
	if v.AddEOS && v.EOS >= 0 && (len(ids) == 0 || ids[len(ids)-1] != v.EOS) {
		ids = append(ids, v.EOS)
	}
	return ids
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
	var d [2]*Tensor
	for i, r := range [...]jlm.Role{jlm.RoleEmbdDense1, jlm.RoleEmbdDense2} {
		e, _ := c.Find(r, jlm.DenseBlock, -1)
		if _, ok := jlm.SourceType(e.Type); !ok || e.NDim != 2 {
			return fmt.Errorf("model: %v is %v in %d dimensions", r, e.Type, e.NDim)
		}
		if d[i] = m.DenseMatrix(r); d[i] == nil {
			return fmt.Errorf("model: %v is not among the model's dense matrices", r)
		}
	}
	if d[0].K != m.Cfg.NEmbd || d[1].K != d[0].Rows || d[1].Rows != m.Cfg.NEmbd {
		return fmt.Errorf("model: dense heads %dx%d then %dx%d do not map %d back to itself",
			d[0].Rows, d[0].K, d[1].Rows, d[1].K, m.Cfg.NEmbd)
	}
	return nil
}

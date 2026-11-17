package convert

import (
	"bytes"
	"fmt"
	"strconv"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// A prediction block is a block like any other after the trunk: its graph
// is stated with the trunk's, and what makes it one is what it stores -- a
// projection of the next token's normed embedding beside the trunk's normed
// hidden state (RolePredictProj), the two norms, and optionally its own head
// norm, head and embedding. The container keeps no count of them; a GGUF
// counts them in nextn_predict_layers and names their tensors blk.N.nextn.*,
// and both end here.

// predictOf splits a GGUF's block_count into the trunk and the prediction
// blocks after it: a GGUF writes both into block_count and the second into
// nextn_predict_layers. From here on NLayer is the trunk alone, and the
// blocks after it are the container's prediction blocks.
//
// Each prediction block must carry its projection: a count with no block
// behind it would be a model that says it can draft and cannot.
func predictOf(f *meta.File, c *Config) error {
	n := uint32(f.UintKey("nextn_predict_layers", 0))
	if n == 0 {
		return nil
	}
	if n >= c.NLayer {
		return fmt.Errorf("nextn_predict_layers %d in a %d-block file", n, c.NLayer)
	}
	c.NLayer -= n
	c.NPredict = n
	for i := uint32(0); i < n; i++ {
		name := "blk." + strconv.Itoa(int(c.NLayer+i)) + ".nextn.eh_proj.weight"
		if _, ok := f.Get(name); !ok {
			return fmt.Errorf("nextn_predict_layers is %d and the file has no %s", n, name)
		}
	}
	return nil
}

// predictionBlocks is how many prediction blocks follow c's trunk in ts: the
// blocks from NLayer on, each storing its projection.
func predictionBlocks(c *jlm.Config, ts []jlm.Tensor) uint32 {
	held := map[int32]bool{}
	for i := range ts {
		if ts[i].Role == jlm.RolePredictProj {
			held[ts[i].Block] = true
		}
	}
	n := uint32(0)
	for held[int32(c.NLayer+n)] {
		n++
	}
	return n
}

// predictRole is a prediction block's entry or head tensor, which its
// entry's and head's graphs bind, not its body's.
func predictRole(r jlm.Role) bool {
	switch r {
	case jlm.RolePredictProj, jlm.RolePredictTokenNorm, jlm.RolePredictHiddenNorm, jlm.RolePredictHeadNorm,
		jlm.RolePredictHead, jlm.RolePredictEmbd:
		return true
	}
	return false
}

// predictTensors checks the prediction blocks and drops the copies they
// carry.
//
// A prediction block needs its projection and both input norms, and the
// roles only a prediction block has may not appear outside one. Its own head
// and embedding (DeepSeek-V3 and GLM ship both, as copies of the trunk's) are
// dropped when they are byte for byte the trunk's tensor, which is every
// published checkpoint here: the model then reads the trunk's, and the
// container does not carry a vocabulary-sized matrix twice. A copy that
// differs is kept, and the engine reads it.
func predictTensors(ts []jlm.Tensor, c *Config) ([]jlm.Tensor, error) {
	var head, embd *jlm.Tensor
	for i := range ts {
		t := &ts[i]
		if t.Block != jlm.DenseBlock {
			continue
		}
		switch t.Role {
		case jlm.RoleOutput:
			head = t
		case jlm.RoleTokenEmbd:
			embd = t
		}
	}
	// A tied model's head is its embedding.
	if head == nil {
		head = embd
	}
	same := func(a, b *jlm.Tensor) bool {
		return a != nil && b != nil && a.Type == b.Type && a.NDim == b.NDim && a.Dims == b.Dims &&
			bytes.Equal(a.Data, b.Data)
	}
	type need struct{ proj, tn, hn bool }
	have := make([]need, c.NPredict)
	out := make([]jlm.Tensor, 0, len(ts))
	for i := range ts {
		t := &ts[i]
		pb := t.Block >= int32(c.NLayer) && t.Block < int32(c.NLayer+c.NPredict)
		if predictRole(t.Role) && !pb {
			if c.NPredict == 0 {
				return nil, fmt.Errorf("convert: %s is a prediction block's tensor in block %d, "+
					"and the model declares no prediction block", t.Name, t.Block)
			}
			return nil, fmt.Errorf("convert: %s is a prediction block's tensor in block %d, "+
				"and the prediction blocks are %d..%d", t.Name, t.Block, c.NLayer, c.NLayer+c.NPredict-1)
		}
		if pb {
			h := &have[t.Block-int32(c.NLayer)]
			switch t.Role {
			case jlm.RolePredictProj:
				if t.NDim != 2 || t.Dims[0] != 2*uint64(c.NEmbd) || t.Dims[1] != uint64(c.NEmbd) {
					return nil, fmt.Errorf("convert: %s is %v, want [%d %d]", t.Name, t.Dims[:t.NDim],
						2*c.NEmbd, c.NEmbd)
				}
				h.proj = true
			case jlm.RolePredictTokenNorm:
				h.tn = true
			case jlm.RolePredictHiddenNorm:
				h.hn = true
			case jlm.RolePredictHead:
				if same(t, head) {
					continue
				}
			case jlm.RolePredictEmbd:
				if same(t, embd) {
					continue
				}
			}
		}
		out = append(out, *t)
	}
	for i, h := range have {
		if !h.proj || !h.tn || !h.hn {
			return nil, fmt.Errorf("convert: prediction block %d lacks its projection, token norm or hidden norm "+
				"(have %v, %v, %v)", int(c.NLayer)+i, h.proj, h.tn, h.hn)
		}
	}
	return out, nil
}

package convert

import (
	"bytes"
	"fmt"
	"strconv"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// nextnOf splits a GGUF's block_count into the trunk and the
// multi-token-prediction blocks after it: llama.cpp writes both into
// block_count and the second into nextn_predict_layers. They stay, as blocks
// NLayer.. of the container (jlm.Config.NMTP), and from here on NLayer is the
// trunk alone.
//
// Each prediction block must carry its eh_proj: a count with no block behind
// it would be a model that says it can draft and cannot.
func nextnOf(f *meta.File, c *Config) error {
	nextn := uint32(f.UintKey("nextn_predict_layers", 0))
	if nextn == 0 {
		return nil
	}
	if nextn >= c.NLayer {
		return fmt.Errorf("nextn_predict_layers %d in a %d-block file", nextn, c.NLayer)
	}
	c.NLayer -= nextn
	c.NMTP = nextn
	for i := uint32(0); i < nextn; i++ {
		name := "blk." + strconv.Itoa(int(c.NLayer+i)) + ".nextn.eh_proj.weight"
		if _, ok := f.Get(name); !ok {
			return fmt.Errorf("nextn_predict_layers is %d and the file has no %s", nextn, name)
		}
	}
	return nil
}

// nextnTensors checks the prediction blocks and drops the copies they carry.
//
// A prediction block needs its eh_proj and both input norms, and the roles
// only a prediction block has may not appear outside one. Its own head and
// embedding (DeepSeek-V3 and GLM ship both, as copies of the trunk's) are
// dropped when they are byte for byte the trunk's tensor, which is every
// published checkpoint here: the model then reads the trunk's, and the
// container does not carry a vocabulary-sized matrix twice. A copy that
// differs is kept, and the engine reads it.
func nextnTensors(ts []jlm.Tensor, c *Config) ([]jlm.Tensor, error) {
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
	type need struct{ eh, en, hn bool }
	have := make([]need, c.NMTP)
	out := make([]jlm.Tensor, 0, len(ts))
	for i := range ts {
		t := &ts[i]
		mtp := t.Block >= int32(c.NLayer) && t.Block < int32(c.NLayer+c.NMTP)
		switch t.Role {
		case jlm.RoleNextnEHProj, jlm.RoleNextnENorm, jlm.RoleNextnHNorm,
			jlm.RoleNextnHeadNorm, jlm.RoleNextnHead, jlm.RoleNextnEmbd:
			if !mtp {
				return nil, fmt.Errorf("convert: %s is a prediction block's tensor in block %d, "+
					"and the prediction blocks are %d..%d", t.Name, t.Block, c.NLayer, c.NLayer+c.NMTP)
			}
		}
		if mtp {
			h := &have[t.Block-int32(c.NLayer)]
			switch t.Role {
			case jlm.RoleNextnEHProj:
				if t.NDim != 2 || t.Dims[0] != 2*uint64(c.NEmbd) || t.Dims[1] != uint64(c.NEmbd) {
					return nil, fmt.Errorf("convert: %s is %v, want [%d %d]", t.Name, t.Dims[:t.NDim],
						2*c.NEmbd, c.NEmbd)
				}
				h.eh = true
			case jlm.RoleNextnENorm:
				h.en = true
			case jlm.RoleNextnHNorm:
				h.hn = true
			case jlm.RoleNextnHead:
				if same(t, head) {
					continue
				}
			case jlm.RoleNextnEmbd:
				if same(t, embd) {
					continue
				}
			}
		}
		out = append(out, *t)
	}
	for i, h := range have {
		if !h.eh || !h.en || !h.hn {
			return nil, fmt.Errorf("convert: prediction block %d lacks its eh_proj, enorm or hnorm "+
				"(have %v, %v, %v)", int(c.NLayer)+i, h.eh, h.en, h.hn)
		}
	}
	return out, nil
}

package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
)

// sharedExpertWidth takes the shared expert's width off its tensor wherever
// one ships, for every graph: a key or a count can disagree with the
// weights (llama.cpp's ERNIE converter writes intermediate_size //
// num_key_value_heads where the class builds moe_intermediate_size *
// moe_num_shared_experts; Hunyuan builds it at intermediate_size), and the
// tensor is the fact. A file shipping none keeps the key's or the count's.
func sharedExpertWidth(f *meta.File, c *Config) error {
	for i := 0; i < int(c.NLayer); i++ {
		t, ok := f.Get("blk." + itoa(i) + ".ffn_gate_shexp.weight")
		if !ok {
			continue
		}
		if len(t.Dims) < 2 {
			return fmt.Errorf("blk.%d.ffn_gate_shexp.weight is %v", i, t.Dims)
		}
		c.NFFNShExp = uint32(t.Dims[1])
		return nil
	}
	return nil
}

package convert

import (
	"errors"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/quant"
)

// TestPhimoeRefusesAHeadWidthTheFileDoesNotState is Phi-tiny-MoE's GGUF:
// sixteen heads of 128 on a 4096-wide model, and llama.cpp's phimoe converter
// writes no attention.key_length, so the file implies heads of 256 while its q
// projection has 2048 rows. Converting it would run every head at the wrong
// width (it panicked at the first token); the file is refused, naming the
// safetensors input that reads head_dim. The control is the same file with q
// at the width the keys imply.
func TestPhimoeRefusesAHeadWidthTheFileDoesNotState(t *testing.T) {
	file := func(qRows uint64) *meta.File {
		k := baseKeys(2)
		k["expert_count"] = meta.MakeUint(4)
		k["expert_used_count"] = meta.MakeUint(2)
		f := headerOnly("phimoe", k)
		ts := append(f.Tensors, meta.Tensor{Name: "blk.0.attn_q.weight", Dims: []uint64{64, qRows},
			Type: quant.F32, NBytes: 4 * 64 * qRows})
		return meta.New(f.Path, f.KV, ts)
	}
	if _, err := configOf(file(64)); err != nil {
		t.Fatalf("q at the keys' width: %v", err)
	}
	_, err := configOf(file(32))
	if err == nil || !strings.Contains(err.Error(), "key_length") || !errors.Is(err, ErrBadInput) {
		t.Fatalf("q at half the keys' width converted, or not as a bad input: %v", err)
	}
}

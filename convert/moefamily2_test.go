package convert

import (
	"errors"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/quant"
)

// TestPhimoeReadsItsHeadWidthOffItsProjections is Phi-tiny-MoE's GGUF:
// sixteen heads of 128 on a 4096-wide model, and its GGUF holds no
// attention.key_length, so its keys imply heads of 256 (and state a rotary of
// 256, the same quotient) while its q projection has 2048 rows. The head width
// is q's rows over the heads, held to k's and v's over the kv heads, and the
// rotary and value widths follow it: the class turns the whole head. Here four
// heads of 8 on a 64-wide model, the keys implying 16. A k or v projection
// that disagrees, or a q that is not whole heads, is refused naming the
// tensors; so is a rotary at embedding/heads under a class that does not turn
// the whole head, which could mean either width.
func TestPhimoeReadsItsHeadWidthOffItsProjections(t *testing.T) {
	file := func(arch string, q, k, v uint64, keys map[string]meta.Value) *meta.File {
		kv := baseKeys(2)
		kv["expert_count"] = meta.MakeUint(4)
		kv["expert_used_count"] = meta.MakeUint(2)
		for n, x := range keys {
			kv[n] = x
		}
		f := headerOnly(arch, kv)
		ts := f.Tensors
		for _, p := range []struct {
			name string
			rows uint64
		}{{"attn_q", q}, {"attn_k", k}, {"attn_v", v}} {
			ts = append(ts, meta.Tensor{Name: "blk.0." + p.name + ".weight", Dims: []uint64{64, p.rows},
				Type: quant.F32, NBytes: 4 * 64 * p.rows})
		}
		return meta.New(f.Path, f.KV, ts)
	}
	rot16 := map[string]meta.Value{"rope.dimension_count": meta.MakeUint(16)}
	for _, tc := range []struct {
		name, arch    string
		q, k, v       uint64
		keys          map[string]meta.Value
		head, val, nr uint32
	}{
		{"the keys' width", "phimoe", 64, 64, 64, nil, 16, 0, 16},
		{"half the keys' width", "phimoe", 32, 32, 32, nil, 8, 0, 8},
		{"half, the rotary at embedding/heads", "phimoe", 32, 32, 32, rot16, 8, 0, 8},
		{"half, grouped", "phimoe", 32, 16, 16, map[string]meta.Value{"attention.head_count_kv": meta.MakeUint(2)}, 8, 0, 8},
		{"half, values of 16", "phimoe", 32, 32, 64, map[string]meta.Value{"attention.value_length": meta.MakeUint(16)}, 8, 16, 8},
		{"half, a partial rotary", "llama", 32, 32, 32, map[string]meta.Value{"rope.dimension_count": meta.MakeUint(4)}, 8, 0, 4},
	} {
		c, err := configOf(file(tc.arch, tc.q, tc.k, tc.v, tc.keys))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if c.HeadDim != tc.head || c.HeadDimV != tc.val || c.NRot != tc.nr {
			t.Fatalf("%s: heads of %d, values %d, rotary %d; want %d, %d, %d", tc.name, c.HeadDim, c.HeadDimV,
				c.NRot, tc.head, tc.val, tc.nr)
		}
	}
	for _, tc := range []struct {
		name, arch, names string
		q, k, v           uint64
		keys              map[string]meta.Value
		err               error
	}{
		{"k at the keys' width", "phimoe", "attn_k", 32, 64, 32, nil, ErrBadInput},
		{"v at the keys' width", "phimoe", "attn_v", 32, 32, 64, nil, ErrBadInput},
		{"q not whole heads", "phimoe", "attn_q", 30, 32, 32, nil, ErrBadInput},
		{"a rotary the class does not turn whole", "llama", "rope.dimension_count", 32, 32, 32, rot16, ErrBadInput},
		{"a rotary wider than the class's whole head", "phimoe", "whole head", 32, 32, 32,
			map[string]meta.Value{"rope.dimension_count": meta.MakeUint(12)}, ErrNotImplemented},
	} {
		_, err := configOf(file(tc.arch, tc.q, tc.k, tc.v, tc.keys))
		if err == nil || !strings.Contains(err.Error(), tc.names) || !errors.Is(err, tc.err) {
			t.Fatalf("%s converted, or not as %v naming %s: %v", tc.name, tc.err, tc.names, err)
		}
	}
}

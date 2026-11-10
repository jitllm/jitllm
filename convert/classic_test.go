package convert

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestUnfuseSplitsAFusedQKVBias holds the fused q|k|v bias (phi-2's older
// GGUFs, falcon, starcoder) to the three element ranges llama.cpp's build_qkv
// views it as, in that order, with GQA widths. Every element carries its own
// index, so a swapped or shifted range reads the wrong numbers.
func TestUnfuseSplitsAFusedQKVBias(t *testing.T) {
	c := &Config{Arch: ArchPhi2, Config: &jlm.Config{NHead: 4, NKVHead: 2, HeadDim: 8}}
	q, kv := 32, 16
	n := q + 2*kv
	data := make([]byte, 4*n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(float32(i)))
	}
	e := jlm.Tensor{Role: jlm.RoleAttnQKVBias, Block: 3, Index: -1, Type: jlm.TypeF32, NDim: 1,
		Data: data, Name: "blk.3.attn_qkv.bias"}
	e.Dims[0] = uint64(n)
	parts, split, err := unfuse(&e, c)
	if err != nil || !split {
		t.Fatalf("unfuse: split %v err %v", split, err)
	}
	want := []struct {
		role jlm.Role
		from int
		n    int
	}{{jlm.RoleAttnQBias, 0, q}, {jlm.RoleAttnKBias, q, kv}, {jlm.RoleAttnVBias, q + kv, kv}}
	if len(parts) != len(want) {
		t.Fatalf("%d parts, want %d", len(parts), len(want))
	}
	for i, w := range want {
		p := parts[i]
		if p.Role != w.role || p.Block != 3 || p.NDim != 1 || int(p.Dims[0]) != w.n || len(p.Data) != 4*w.n {
			t.Fatalf("part %d: %v block %d dims %v bytes %d, want %v of %d", i, p.Role, p.Block,
				p.Dims[:p.NDim], len(p.Data), w.role, w.n)
		}
		for j := 0; j < w.n; j++ {
			if v := math.Float32frombits(binary.LittleEndian.Uint32(p.Data[4*j:])); v != float32(w.from+j) {
				t.Fatalf("%v[%d] = %g, want %d", w.role, j, v, w.from+j)
			}
		}
	}
	// A width that does not match the heads is refused rather than guessed.
	e.Dims[0]--
	if _, _, err := unfuse(&e, c); err == nil {
		t.Error("a fused bias one element short was split anyway")
	}
}

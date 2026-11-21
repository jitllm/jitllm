package convert

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAnI32TensorIsItsValuesAsF32: a GGUF I32 tensor becomes the F32 of its
// values, exactly, and one past 2^24 is refused.
func TestAnI32TensorIsItsValuesAsF32(t *testing.T) {
	in := make([]byte, 12)
	for i, v := range []int32{0, 7, -3} {
		binary.LittleEndian.PutUint32(in[4*i:], uint32(v))
	}
	out, err := i32AsF32("x", in)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{0, 7, -3} {
		if got := math.Float32frombits(binary.LittleEndian.Uint32(out[4*i:])); got != want {
			t.Errorf("value %d is %v, want %v", i, got, want)
		}
	}
	binary.LittleEndian.PutUint32(in, 1<<24)
	if _, err := i32AsF32("x", in); err == nil {
		t.Error("2^24 was carried as F32")
	}
}

// TestHashExpertsFitTheirBanks: a token-to-expert table is two ids per token
// of three tokens here, each an expert of four and none twice for a token; an
// id past the experts, a repeated one and a table of another shape are each
// refused.
func TestHashExpertsFitTheirBanks(t *testing.T) {
	c := &Config{Config: &jlm.Config{NExpertUsed: 2, NVocab: 3, NExpert: 4}}
	table := func(ids []float32, k, n uint64) *jlm.Source {
		e := jlm.Tensor{Role: jlm.RoleHashExperts, Block: 0, Index: -1, Name: "blk.0.ffn_gate_tid2eid.weight"}
		setF32Dims(&e, ids, k, n)
		return &jlm.Source{Tensors: []jlm.Tensor{e}}
	}
	if err := hashExpertsFit(c, table([]float32{0, 1, 2, 3, 3, 0}, 2, 3)); err != nil {
		t.Fatalf("a table that fits was refused: %v", err)
	}
	for _, bad := range []struct {
		what string
		s    *jlm.Source
	}{
		{"an id past the experts", table([]float32{0, 1, 2, 4, 3, 0}, 2, 3)},
		{"an id twice for a token", table([]float32{0, 1, 2, 2, 3, 0}, 2, 3)},
		{"a fraction", table([]float32{0, 1, 2.5, 3, 3, 0}, 2, 3)},
		{"another shape", table([]float32{0, 1, 2, 3}, 2, 2)},
	} {
		if err := hashExpertsFit(c, bad.s); err == nil {
			t.Errorf("%s was not refused", bad.what)
		}
	}
}

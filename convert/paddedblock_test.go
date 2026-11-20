package convert

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAPaddedTowerBlockIsNotABlock holds dropPaddedBlock to the file's own
// bytes: a last block that repeats the one before it is the padding the MLP
// projector's statement names and goes, and a last block of its own, or one
// a byte away from the one before it, stays a block.
func TestAPaddedTowerBlockIsNotABlock(t *testing.T) {
	lib, err := projectorLibrary()
	if err != nil {
		t.Fatal(err)
	}
	p := lib["mlp"]
	if p == nil || p.Blocks != "padded" {
		t.Fatal("the MLP projector does not state its tower padded")
	}
	tower := func(last func(b []byte)) *jlm.Source {
		s := &jlm.Source{Vision: &jlm.Vision{NLayer: 3}}
		for b := range int32(3) {
			for _, r := range []jlm.Role{jlm.RoleVFC1, jlm.RoleVFC2} {
				d := make([]byte, 8)
				src := min(b, 1)
				binary.LittleEndian.PutUint32(d, math.Float32bits(float32(src)+0.5))
				binary.LittleEndian.PutUint32(d[4:], math.Float32bits(float32(r)))
				if b == 2 && last != nil {
					last(d)
				}
				s.Tensors = append(s.Tensors, jlm.Tensor{Role: r, Block: b, Index: -1, Type: jlm.TypeF32,
					NDim: 1, Dims: [4]uint64{2}, Data: d})
			}
		}
		return s
	}
	blocks := func(s *jlm.Source) int {
		n := 0
		for _, t := range s.Tensors {
			n = max(n, int(t.Block)+1)
		}
		return n
	}
	for _, c := range []struct {
		name  string
		last  func([]byte)
		p     *projStates
		layer uint32
	}{
		{"its last block a copy of the one before", nil, p, 2},
		{"its last block its own", func(d []byte) { d[0] ^= 1 }, p, 3},
		{"a statement that names no padding", nil, &projStates{}, 3},
	} {
		s := tower(c.last)
		if err := dropPaddedBlock(s, c.p); err != nil {
			t.Fatal(err)
		}
		if s.Vision.NLayer != c.layer || blocks(s) != int(c.layer) {
			t.Errorf("%s: %d blocks stated and %d carried, want %d", c.name, s.Vision.NLayer, blocks(s), c.layer)
		}
	}
}

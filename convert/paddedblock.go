package convert

import (
	"bytes"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// dropPaddedBlock reads a tower whose statement says blocks "padded". A
// llava projector's GGUF is read to the block before its block_count (the
// projector takes the second-to-last block's rows), and a converter that
// writes every block of a model the projector reads whole pads the tower with
// a copy of its last block, which that reading never runs: moondream2's
// mmproj states 28 blocks, its block 27 is block 26 byte for byte, and the
// model has 27. Where the file's last block repeats the one before it tensor
// for tensor and byte for byte it is that padding and not a block here; a
// tower whose last block is its own (llava-phi-3's) runs every block it
// states.
func dropPaddedBlock(s *jlm.Source, p *projStates) error {
	if p == nil || p.Blocks != "padded" || s.Vision == nil || s.Vision.NLayer < 2 {
		return nil
	}
	last, prev := int32(s.Vision.NLayer-1), int32(s.Vision.NLayer-2)
	of := func(b int32) map[jlm.Role]*jlm.Tensor {
		m := map[jlm.Role]*jlm.Tensor{}
		for i := range s.Tensors {
			if t := &s.Tensors[i]; t.Role.Vision() && t.Block == b {
				m[t.Role] = t
			}
		}
		return m
	}
	a, b := of(last), of(prev)
	if len(a) == 0 || len(a) != len(b) {
		return nil
	}
	for r, t := range a {
		u, ok := b[r]
		if !ok || t.Type != u.Type || t.NDim != u.NDim || t.Dims != u.Dims {
			return nil
		}
	}
	for r, t := range a {
		x, err := t.Bytes()
		if err != nil {
			return err
		}
		y, err := b[r].Bytes()
		if err != nil {
			return err
		}
		if !bytes.Equal(x, y) {
			return nil
		}
	}
	s.Tensors = slices.DeleteFunc(s.Tensors, func(t jlm.Tensor) bool { return t.Role.Vision() && t.Block == last })
	s.Vision.NLayer--
	return nil
}

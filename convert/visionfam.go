package convert

import (
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// The converter's half of the vision families whose codes sit at the top of
// the code space (format/jlm/visionfam.go): the names their towers' tensors
// carry, and the transforms they need, each exact.

// clampSlot is where a clipped linear's bounds go in a block's RoleVClamp
// vector: its matrix's place, times four.
func clampSlot(name string) (int, bool) {
	switch name {
	case "attn_q":
		return 0, true
	case "attn_k":
		return 1, true
	case "attn_v":
		return 2, true
	case "attn_out":
		return 3, true
	case "ffn_gate":
		return 4, true
	case "ffn_up":
		return 5, true
	case "ffn_down":
		return 6, true
	}
	return 0, false
}

// clampWhich is a bound's place among a matrix's four.
func clampWhich(name string) (int, bool) {
	switch name {
	case "input_min":
		return 0, true
	case "input_max":
		return 1, true
	case "output_min":
		return 2, true
	case "output_max":
		return 3, true
	}
	return 0, false
}

// clampOf parses one of Gemma 4's clipped-linear bounds,
// "v.blk.N.<matrix>.<input|output>_<min|max>": a scalar with no role of its
// own, gathered into the block's RoleVClamp vector (clampTensors).
func clampOf(name string) (block int32, at int, ok bool) {
	rest, found := strings.CutPrefix(name, "v.blk.")
	if !found {
		return 0, 0, false
	}
	f := strings.Split(rest, ".")
	if len(f) != 3 {
		return 0, 0, false
	}
	n, err := strconv.Atoi(f[0])
	s, okS := clampSlot(f[1])
	w, okW := clampWhich(f[2])
	if err != nil || n < 0 || !okS || !okW {
		return 0, 0, false
	}
	return int32(n), 4*s + w, true
}

// clampTensors writes each block's gathered bounds as its RoleVClamp vector;
// a bound the file did not carry is infinite, which clamps nothing.
func clampTensors(s *jlm.Source, clamps map[int32]*[28]float32) error {
	for _, b := range slices.Sorted(maps.Keys(clamps)) {
		v := clamps[b]
		if b < 0 || uint32(b) >= s.Vision.NLayer {
			return fmt.Errorf("convert: clipped-linear bounds for block %d of %d", b, s.Vision.NLayer)
		}
		e := jlm.Tensor{Role: jlm.RoleVClamp, Block: b, Index: -1, Name: fmt.Sprintf("v.blk.%d.clamp", b)}
		setF32Dims(&e, v[:], 28)
		s.Tensors = append(s.Tensors, e)
	}
	return nil
}

// newClamps is a block's bounds before the file has stated any: all
// infinite.
func newClamps() *[28]float32 {
	var c [28]float32
	for i := range c {
		c[i] = float32(math.Inf(1))
		if i%2 == 0 {
			c[i] = float32(math.Inf(-1))
		}
	}
	return &c
}

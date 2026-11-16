package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// rowBytes is the bytes of k elements of t, or 0 when k is not whole blocks.
func rowBytes(t quant.Type, k uint64) uint64 {
	be := t.BlockElems()
	if be == 0 || k%be != 0 {
		return 0
	}
	return k / be * t.BlockBytes()
}

// checkDeepstack holds a Qwen3-VL tower's deepstack mergers to the file's own
// list of tapped blocks (clip.vision.is_deepstack_layers): a merger the list
// does not name, or a named block with no merger, is a file that disagrees
// with itself, and either way a text block would add the wrong rows.
func checkDeepstack(f *meta.File, s *jlm.Source, p *projStates) error {
	v := s.Vision
	if v == nil {
		return nil
	}
	if p == nil || !p.Taps {
		return nil
	}
	have := map[int32]int{}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role >= jlm.RoleVDsNorm && t.Role <= jlm.RoleVDsFC2Bias {
			have[t.Index]++
		}
	}
	var want []bool
	if kv, ok := f.KV["clip.vision.is_deepstack_layers"]; ok {
		if kv.Type != meta.Array || kv.Elem != meta.Bool || uint64(len(kv.Raw)) < kv.N {
			return fmt.Errorf("convert: clip.vision.is_deepstack_layers is not an array of bools")
		}
		for i := uint64(0); i < kv.N; i++ {
			want = append(want, kv.Raw[i] != 0)
		}
	}
	for b := range want {
		if want[b] && have[int32(b)] != 6 {
			return fmt.Errorf("convert: block %d is a deepstack tap and carries %d of its merger's 6 tensors",
				b, have[int32(b)])
		}
	}
	for b := range have {
		if int(b) >= len(want) || !want[b] {
			return fmt.Errorf("convert: block %d carries a deepstack merger the file does not list", b)
		}
	}
	return nil
}

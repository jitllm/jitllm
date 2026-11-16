package convert

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The tower tensors the vision families ship that the shared name table
// does not cover: Qwen3-VL's deepstack mergers, which go to the dense region
// at the index of the block they tap (jlm.RoleVDsNorm says why not the
// block's page).

// familyTowerTensors is the container's tensors for the tower tensor t, or
// ok=false when t is not one of the names above.
func familyTowerTensors(f *meta.File, v *jlm.Vision, p *projStates, t *meta.Tensor) (out []jlm.Tensor, ok bool, err error) {
	if v == nil {
		return nil, false, nil
	}
	if b, part, found := deepstackName(t.Name); found {
		role, known := map[string]jlm.Role{
			"norm.weight": jlm.RoleVDsNorm, "norm.bias": jlm.RoleVDsNormBias,
			"fc1.weight": jlm.RoleVDsFC1, "fc1.bias": jlm.RoleVDsFC1Bias,
			"fc2.weight": jlm.RoleVDsFC2, "fc2.bias": jlm.RoleVDsFC2Bias,
		}[part]
		if !known {
			return nil, true, fmt.Errorf("convert: %s: a deepstack tensor this engine does not know", t.Name)
		}
		if p == nil || !p.Taps {
			return nil, true, fmt.Errorf("convert: %s: deepstack on a %v tower", t.Name, v.Projector)
		}
		if uint32(b) >= v.NLayer {
			return nil, true, fmt.Errorf("convert: %s taps block %d of %d", t.Name, b, v.NLayer)
		}
		e, err := tensorOf(f, t, role, jlm.DenseBlock, b)
		return []jlm.Tensor{e}, true, err
	}
	return nil, false, nil
}

// familyTowerName reports whether a tower tensor name is one
// familyTowerTensors resolves rather than the shared name table.
func familyTowerName(name string) bool {
	_, _, ok := deepstackName(name)
	return ok
}

// deepstackName parses "v.deepstack.N.<part>".
func deepstackName(name string) (block int32, part string, ok bool) {
	rest, found := strings.CutPrefix(name, "v.deepstack.")
	if !found {
		return 0, "", false
	}
	i := strings.IndexByte(rest, '.')
	if i <= 0 {
		return 0, "", false
	}
	n, err := strconv.Atoi(rest[:i])
	if err != nil || n < 0 {
		return 0, "", false
	}
	return int32(n), rest[i+1:], true
}

// blockPart parses "v.blk.N.<part>".
func blockPart(name string) (block int32, part string, ok bool) {
	rest, found := strings.CutPrefix(name, "v.blk.")
	if !found {
		return 0, "", false
	}
	i := strings.IndexByte(rest, '.')
	if i <= 0 {
		return 0, "", false
	}
	n, err := strconv.Atoi(rest[:i])
	if err != nil || n < 0 {
		return 0, "", false
	}
	return int32(n), rest[i+1:], true
}

// tensorOf is t as the container's tensor under the given triple.
func tensorOf(f *meta.File, t *meta.Tensor, role jlm.Role, block, index int32) (jlm.Tensor, error) {
	ty, err := typeOf(t.Type)
	if err != nil {
		return jlm.Tensor{}, fmt.Errorf("convert: %s: %w", t.Name, err)
	}
	e := jlm.Tensor{Role: role, Block: block, Index: index, Type: ty,
		NDim: uint8(len(t.Dims)), Data: f.Bytes(t), Name: t.Name}
	copy(e.Dims[:], t.Dims)
	return e, nil
}

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

package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
)

// layerCount reads a count a file may write per layer: a number, or one per
// layer whose zeros mark the layers it does not apply to (llama.cpp's is_recr
// reads that zero on a hybrid's recurrent layers). An array whose other
// entries agree is that number, and one of zeros is zero; one whose entries
// differ comes back whole (per), with n zero.
func layerCount(f *meta.File, key string, def uint32) (n uint32, per []int32, err error) {
	kv, ok := f.Key(key)
	if !ok {
		return def, nil, nil
	}
	if v, ok := kv.Uint(); ok {
		return uint32(v), nil, nil
	}
	vs, err := kv.Int32s()
	if err != nil {
		return 0, nil, fmt.Errorf("%s: %w", key, err)
	}
	if len(vs) == 0 {
		return def, nil, nil
	}
	for i, v := range vs {
		switch {
		case v < 0:
			return 0, nil, fmt.Errorf("%s: %d at layer %d", key, v, i)
		case v == 0:
		case n == 0:
			n = uint32(v)
		case uint32(v) != n:
			return 0, vs, nil
		}
	}
	return n, nil, nil
}

// uniformCount is layerCount refusing a count that differs by layer.
func uniformCount(f *meta.File, key string, def uint32) (uint32, error) {
	n, per, err := layerCount(f, key, def)
	if err == nil && per != nil {
		err = fmt.Errorf("%s differs by layer (%v): %w", key, per, ErrNotImplemented)
	}
	return n, err
}

// countsHeld refuses a per-layer kv head count no block's geometry holds, and
// a file stating no heads whose layers attend.
func countsHeld(c *Config, perKV []int32, headless bool) error {
	for i, v := range perKV {
		if g := c.GeomAt(i); v != 0 && i < int(c.NLayer) && g.NKVHead != uint32(v) {
			return fmt.Errorf("attention.head_count_kv differs by layer (%d at layer %d, %d in its geometry): %w",
				v, i, g.NKVHead, ErrNotImplemented)
		}
	}
	for i := 0; headless && i < int(c.NLayer); i++ {
		if i >= len(c.LayerKinds) || c.LayerKinds[i].Attends() {
			return fmt.Errorf("attention.head_count is zero and layer %d attends", i)
		}
	}
	return nil
}

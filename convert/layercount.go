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
// a zero head or kv head count on a layer that attends: a zero marks a layer a
// count does not apply to, for every model, and a head count of zero written
// as one number gives every layer zero. Where the converter merged the
// source's layers into fewer blocks, a count over the source's layers is
// matched in order: its n-th nonzero entry to the n-th block that attends.
func countsHeld(c *Config, f *meta.File) error {
	attends := func(b int) bool { return b >= len(c.LayerKinds) || c.LayerKinds[b].Attends() }
	for _, key := range []string{"attention.head_count", "attention.head_count_kv"} {
		kv, ok := f.Key(key)
		if !ok {
			continue
		}
		var vs []int32
		if v, one := kv.Uint(); one {
			if v != 0 || key != "attention.head_count" {
				continue
			}
			vs = make([]int32, c.NLayer)
		} else if p, err := kv.Int32s(); err != nil {
			return fmt.Errorf("%s: %w", key, err)
		} else {
			vs = p
		}
		var blocks []int
		if src := int(f.UintKey("block_count", 0)); src != int(c.NLayer) && len(vs) == src {
			for b := range int(c.NLayer) {
				if attends(b) {
					blocks = append(blocks, b)
				}
			}
			var held []int32
			for _, v := range vs {
				if v != 0 {
					held = append(held, v)
				}
			}
			if len(held) != len(blocks) {
				return fmt.Errorf("%s states heads on %d source layers, and %d blocks attend", key, len(held), len(blocks))
			}
			vs = held
		} else {
			vs = vs[:min(len(vs), int(c.NLayer))]
			for b := range vs {
				blocks = append(blocks, b)
			}
		}
		for i, v := range vs {
			b := blocks[i]
			if v == 0 && attends(b) {
				return fmt.Errorf("%s is zero at layer %d, which attends", key, b)
			}
			if g := c.GeomAt(b); v != 0 && key == "attention.head_count_kv" && g.NKVHead != uint32(v) {
				return fmt.Errorf("%s differs by layer (%d at layer %d, %d in its geometry): %w",
					key, v, b, g.NKVHead, ErrNotImplemented)
			}
		}
	}
	return nil
}

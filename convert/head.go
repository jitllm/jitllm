package convert

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
)

// A multi-token-prediction head published as a file of its own (unsloth's
// mtp-*.gguf for Qwen3.8-Flash-Next) is the same model's prediction block,
// cut out of the GGUF it was exported beside. The separate file is a
// property of the input, as an mmproj is: it ends here, where withHead puts
// the head's block back after the trunk's and the conversion proceeds as for
// a GGUF that carried it (nextnOf, nextnTensors).

// withHead is trunk f with head h's prediction blocks appended: block_count
// the trunk's plus the head's nextn_predict_layers, every per-block array
// the head's (which must agree with the trunk's over the trunk's blocks),
// and the head's blk.N.* tensors served from the head's file. Every other key
// the two files share must be the same bytes: a head trained against a
// different trunk drafts nothing useful, and a hyperparameter that differs
// says it was.
func withHead(f, h *meta.File, src, head string) (*meta.File, error) {
	if f.Arch() != h.Arch() {
		return nil, fmt.Errorf("convert: %s is %q and the head %s is %q", src, f.Arch(), head, h.Arch())
	}
	if f.UintKey("nextn_predict_layers", 0) != 0 {
		return nil, fmt.Errorf("convert: %s already carries a prediction block; the head %s would be a second",
			src, head)
	}
	n := f.UintKey("block_count", 0)
	k := h.UintKey("nextn_predict_layers", 0)
	if k == 0 {
		return nil, fmt.Errorf("convert: %s states no nextn_predict_layers: it is not a prediction head", head)
	}
	if hb := h.UintKey("block_count", 0); hb != n+k {
		return nil, fmt.Errorf("convert: the head %s has %d blocks with %d to predict, and %s has %d: "+
			"these two files do not belong together", head, hb, k, src, n)
	}
	pre := f.Arch() + "."
	kv := make(map[string]meta.Value, len(f.KV)+4)
	for key, v := range f.KV {
		kv[key] = v
	}
	for key, hv := range h.KV {
		if !strings.HasPrefix(key, pre) {
			continue // the tokenizer and general.*: the trunk's are the model's
		}
		tv, ok := f.KV[key]
		switch {
		case key == pre+"block_count":
			continue
		case !ok:
			kv[key] = hv // nextn_predict_layers and what only a head states
		case tv.Type == meta.Array && hv.Type == meta.Array && tv.Elem == hv.Elem && tv.N == n && hv.N == n+k &&
			bytes.HasPrefix(hv.Raw, tv.Raw):
			kv[key] = hv // a per-block array, the head's blocks after the trunk's
		case tv.Type != hv.Type || tv.Elem != hv.Elem || tv.N != hv.N || !bytes.Equal(tv.Raw, hv.Raw):
			return nil, fmt.Errorf("convert: %s states %s as %s and the head %s as %s: "+
				"these two files do not belong together", src, key, tv.Short(), head, hv.Short())
		}
	}
	bc := pre + "block_count"
	v, ok := kv[bc]
	if !ok || v.Type != meta.Uint32 {
		return nil, fmt.Errorf("convert: %s's block_count is not a uint32", src)
	}
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint32(raw, uint32(n+k))
	kv[bc] = meta.Value{Type: meta.Uint32, Raw: raw}

	ts := make([]meta.Tensor, 0, len(f.Tensors)+len(h.Tensors))
	ts = append(ts, f.Tensors...)
	fromHead := make(map[string]bool)
	for i := range h.Tensors {
		t := h.Tensors[i]
		b, ok := blockOf(t.Name)
		if !ok {
			// A self-contained head carries the token embedding and the
			// output projection; they must be the trunk's, which the model
			// reads (nextnTensors drops a block's copy the same way).
			tt, have := f.Get(t.Name)
			if !have || tt.Type != t.Type || !slices.Equal(tt.Dims, t.Dims) || !bytes.Equal(f.Bytes(tt), h.Bytes(&t)) {
				return nil, fmt.Errorf("convert: the head %s carries %s, which is not %s's", head, t.Name, src)
			}
			continue
		}
		if b < n || b >= n+k {
			return nil, fmt.Errorf("convert: the head %s carries %s, outside its prediction blocks %d..%d",
				head, t.Name, n, n+k-1)
		}
		if _, dup := f.Get(t.Name); dup {
			return nil, fmt.Errorf("convert: %s and the head %s both carry %s", src, head, t.Name)
		}
		fromHead[t.Name] = true
		ts = append(ts, t)
	}
	m := meta.New(f.Path, kv, ts)
	m.Version, m.Alignment = f.Version, f.Alignment
	m.Fetch = func(t *meta.Tensor) []byte {
		if fromHead[t.Name] {
			return h.Bytes(t)
		}
		return f.Bytes(t)
	}
	return m, nil
}

// blockOf is the block a blk.N.* tensor name belongs to.
func blockOf(name string) (uint64, bool) {
	rest, ok := strings.CutPrefix(name, "blk.")
	if !ok {
		return 0, false
	}
	num, _, ok := strings.Cut(rest, ".")
	if !ok {
		return 0, false
	}
	b, err := strconv.ParseUint(num, 10, 32)
	return b, err == nil
}

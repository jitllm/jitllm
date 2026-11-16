package convert

import (
	"encoding/binary"
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// readoutTrunk is a readout whose head's blocks block_count counts after the
// trunk's (decision.block_count of them), every per-layer key written for
// all of them: the trunk is read through a view of the file cut to its
// blocks (trunkView), as the backbone its architecture names, then stated as
// the graph Arch names, NonCausal where its attention is not causal, Context
// its context; the head's blocks convert after it. Require is what the file
// must state for the trunk to be that graph, Context an expression, both
// over the file's keys (src).
type readoutTrunk struct {
	Require   []mjRequire `json:"require"`
	Context   string      `json:"context"`
	Arch      string      `json:"arch"`
	NonCausal bool        `json:"non_causal"`
}

// trunkOf is the readout f states and its trunk, nil where its statement
// has none.
func trunkOf(f *meta.File) (string, *readoutTrunk, error) {
	v, ok := f.Key("decision.type")
	if !ok {
		return "", nil, nil
	}
	name, _ := v.String()
	lib, err := readoutLibrary()
	if err != nil {
		return "", nil, err
	}
	if r := lib.Readouts[name]; r != nil && r.Trunk != nil {
		return name, r.Trunk, nil
	}
	return "", nil, nil
}

// config is configOf for a file of readout name's whose trunk t states: the
// trunk's config as its backbone reads it, stated as t's graph, with the
// head's blocks after it.
func (t *readoutTrunk) config(name string, f *meta.File) (*Config, error) {
	head, total := uint32(f.UintKey("decision.block_count", 0)), uint32(f.UintKey("block_count", 0))
	if head == 0 || head >= total {
		return nil, fmt.Errorf("convert: a %s readout with %d head blocks in %d", name, head, total)
	}
	by := "readouts.json: " + name
	vars := map[string]jinja.Value{"src": ggufKeys(f)}
	for _, r := range t.Require {
		v, err := libraryEval(by, "require", r.That, vars)
		if err != nil {
			return nil, err
		}
		if !v.IsTruthy() {
			return nil, fmt.Errorf("convert: %s", r.Else)
		}
	}
	view, err := trunkView(f, total-head)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	c, err := configOfFile(view, false)
	if err != nil {
		return nil, err
	}
	// The FFN width is per block in the file; the trunk's are one width.
	if ffn, ok := view.Key("feed_forward_length"); ok && c.NFFN == 0 {
		vs, err := ffn.Int32s()
		if err != nil || len(vs) == 0 {
			return nil, fmt.Errorf("convert: %s: feed_forward_length: %v", name, err)
		}
		for _, v := range vs {
			if v != vs[0] || v <= 0 {
				return nil, fmt.Errorf("convert: %s: feed_forward_length differs between trunk blocks: %w",
					name, ErrNotImplemented)
			}
		}
		c.NFFN = uint32(vs[0])
	}
	ctx, err := libraryEval(by, "context", t.Context, vars)
	if err != nil {
		return nil, err
	}
	if !ctx.IsInt() || ctx.AsInt() <= 0 {
		return nil, fmt.Errorf("convert: %s: the trunk's context is not a count", by)
	}
	c.NCtx = uint32(ctx.AsInt())
	arch, ok := archNamed(t.Arch)
	if !ok {
		return nil, fmt.Errorf("convert: %s: the trunk's graph %q is no graph the container names", by, t.Arch)
	}
	c.Arch = arch
	if t.NonCausal {
		c.Flags |= jlm.FlagNonCausal
	}
	return c, nil
}

// archNamed is the graph named name.
func archNamed(name string) (Arch, bool) {
	for a := Arch(1); a != 0; a++ {
		if a.name() == name {
			return a, true
		}
	}
	return 0, false
}

// trunkView is f with block_count and every per-layer array cut to the first
// n blocks.
func trunkView(f *meta.File, n uint32) (*meta.File, error) {
	total := f.UintKey("block_count", 0)
	kv := make(map[string]meta.Value, len(f.KV))
	for k, v := range f.KV {
		if v.Type == meta.Array && v.N == total && (v.Elem == meta.Int32 || v.Elem == meta.Uint32 ||
			v.Elem == meta.Float32) {
			v.N, v.Raw = uint64(n), v.Raw[:4*n]
		}
		kv[k] = v
	}
	bc := f.Arch() + ".block_count"
	if v, ok := kv[bc]; !ok || v.Type != meta.Uint32 {
		return nil, fmt.Errorf("block_count is not a uint32")
	}
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint32(raw, n)
	kv[bc] = meta.Value{Type: meta.Uint32, Raw: raw}
	view := meta.New(f.Path, kv, f.Tensors)
	return view, nil
}

package gguf

import (
	"fmt"
	"regexp"
	"strconv"

	"github.com/jitllm/jitllm/convert/meta"
)

// splitName matches llama.cpp's gguf-split naming: NAME-00001-of-00003.gguf.
func splitName() *regexp.Regexp {
	return regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)
}

// joinSplit opens the remaining parts of a split GGUF and serves every tensor
// through f, so a caller sees one file. f is part 1, which carries all the
// metadata; the others carry only tensors.
//
// Any model over ~50 GB ships split, since Hugging Face caps a file there. A
// part's tensor offsets are relative to its own data section, so each part
// keeps its mapping and Fetch routes by tensor.
func joinSplit(f *meta.File, opts ...Option) (*meta.File, error) {
	v, ok := f.KV["split.count"]
	if !ok {
		return f, nil
	}
	n, ok := v.Uint()
	if !ok || n <= 1 {
		return f, nil
	}
	m := splitName().FindStringSubmatch(f.Path)
	if m == nil {
		return nil, fmt.Errorf("gguf: %s: split.count is %d and the name is not NAME-00001-of-%05d.gguf", f.Path, n, n)
	}
	if i, _ := strconv.Atoi(m[2]); i != 1 {
		return nil, fmt.Errorf("gguf: %s: part %d of a split model; open part 1", f.Path, i)
	}
	if total, _ := strconv.Atoi(m[3]); uint64(total) != n {
		return nil, fmt.Errorf("gguf: %s: the name says %d parts and split.count says %d", f.Path, total, n)
	}
	owner := make(map[string]*meta.File, len(f.Tensors))
	for i := range f.Tensors {
		owner[f.Tensors[i].Name] = f
	}
	parts := []*meta.File{f}
	closeAll := func() {
		for _, p := range parts[1:] {
			p.Close()
		}
	}
	for i := uint64(2); i <= n; i++ {
		p, err := openOne(fmt.Sprintf("%s-%05d-of-%05d.gguf", m[1], i, n), opts...)
		if err != nil {
			closeAll()
			return nil, err
		}
		parts = append(parts, p)
		for j := range p.Tensors {
			t := p.Tensors[j]
			if _, dup := owner[t.Name]; dup {
				closeAll()
				return nil, fmt.Errorf("gguf: %s: tensor %q is in two parts", p.Path, t.Name)
			}
			owner[t.Name] = p
			f.Tensors = append(f.Tensors, t)
		}
	}
	f.Reindex()
	first := f.CloseFn
	f.CloseFn = func() error {
		closeAll()
		return first()
	}
	f.Fetch = func(t *meta.Tensor) []byte {
		p := owner[t.Name]
		lo := p.DataStart + t.Off
		return p.Data[lo : lo+t.NBytes : lo+t.NBytes]
	}
	return f, nil
}

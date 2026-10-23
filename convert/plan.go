package convert

import (
	"fmt"
	"io/fs"
	"os"
	"strings"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
)

// ShardPlan is what converting a safetensors model would write, decided from
// the shards' headers alone (PlanShards).
type ShardPlan struct {
	Header *jlm.Header
	// Size is the container's length in bytes.
	Size uint64
	// ByType is the container's tensor bytes by type, planes aligned.
	ByType map[jlm.Type]uint64
	// Ignored is how many source tensors are named and not carried (a vision
	// tower the converter does not build).
	Ignored int
	// BF16 is the bytes of BF16 source tensors the container carries, at
	// their BF16 size: the default path stores each as F32, twice this.
	BF16 uint64
}

// PlanSafetensors is PlanShards for a model on the disk, src as
// FromSafetensors takes it.
func PlanSafetensors(src string, opts ...Option) (*ShardPlan, error) {
	dir, paths, err := hfShards(src)
	if err != nil {
		return nil, err
	}
	var open []*safetensors.File
	defer func() {
		for _, f := range open {
			f.Close()
		}
	}()
	for _, p := range paths {
		f, err := safetensors.Open(p)
		if err != nil {
			return nil, err
		}
		open = append(open, f)
	}
	return PlanShards(os.DirFS(dir), dir, open, opts...)
}

// PlanShards runs the whole conversion of shards up to the first byte of a
// tensor -- the config, the vocabulary, every name identified or refused,
// every shape checked, the layout decided -- reading only the headers: each
// tensor reads as zeros of its length (safetensors.File.Headers), so a remote
// repository transfers nothing past them. A conversion that plans cleanly
// cannot be refused later on anything a header decides.
func PlanShards(meta fs.FS, name string, shards []*safetensors.File, opts ...Option) (*ShardPlan, error) {
	hs := make([]*safetensors.File, len(shards))
	for i, f := range shards {
		hs[i] = f.Headers()
	}
	dir := hfDir{meta, name}
	s, err := hfSourceOf(dir, hs)
	if err != nil {
		return nil, err
	}
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	if o.head != "" {
		return nil, fmt.Errorf("convert: WithHead joins a GGUF head to a GGUF; %s is safetensors", name)
	}
	if o.q8 {
		quantizeMatrices(s.Tensors)
	}
	if err := stateTower(s); err != nil {
		return nil, err
	}
	h, size, err := jlm.Plan(s, jlm.Fingerprint{Writer: jlm.WriterID()})
	if err != nil {
		return nil, err
	}
	p := &ShardPlan{Header: h, Size: size, ByType: map[jlm.Type]uint64{}}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		n, err := jlm.StoredBytes(t.Type, t.Dims, t.NDim)
		if err != nil {
			return nil, err
		}
		p.ByType[t.Type] += n
	}
	// What the source carries, by dtype: hfSourceOf has refused every name it
	// could not place, so a name identify cannot place here is one a gather
	// or the MXFP4 merge consumed.
	hc, err := readHFConfig(dir)
	if err != nil {
		return nil, err
	}
	ha, err := hfArchOf(hc)
	if err != nil {
		return nil, err
	}
	for _, f := range shards {
		for i := range f.Tensors {
			t := &f.Tensors[i]
			if strings.HasSuffix(t.Name, ctPacked) || strings.HasSuffix(t.Name, ctScale) {
				continue
			}
			_, _, _, ignored, err := ha.identify(t.Name, nil)
			switch {
			case err == nil && ignored:
				p.Ignored++
			case t.DType == safetensors.BF16:
				p.BF16 += t.NBytes()
			}
		}
	}
	return p, nil
}

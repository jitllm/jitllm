package convert

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// layerTensors is every name a layer's tensor takes past its "blk.N." prefix,
// as a set an expression asks with `in`.
func layerTensors(f *meta.File) jinja.Value {
	d := jinja.NewDict()
	for i := range f.Tensors {
		rest, ok := strings.CutPrefix(f.Tensors[i].Name, "blk.")
		if j := strings.IndexByte(rest, '.'); ok && j > 0 {
			d.AsDict().Set(rest[j+1:], jinja.NewBool(true))
		}
	}
	return d
}

// recurrentFits refuses a recurrent geometry whose parts do not divide: the
// value heads split the inner width, the key heads the value heads and the
// inner width.
func recurrentFits(s jlm.SSMConfig) error {
	switch {
	case s.ConvKernel < 2:
		return fmt.Errorf("a convolution of %d taps: a recurrent layer convolves two or more", s.ConvKernel)
	case s.NHeadV > 0 && s.Inner%s.NHeadV != 0:
		return fmt.Errorf("recurrent inner size %d is not %d whole heads", s.Inner, s.NHeadV)
	case s.Groups > 0 && s.NHeadV > 0 && s.NHeadV%s.Groups != 0:
		return fmt.Errorf("%d value heads over %d key heads is not a whole number per key head", s.NHeadV, s.Groups)
	case s.Groups > 0 && s.Inner%s.Groups != 0:
		return fmt.Errorf("recurrent inner size %d does not divide into %d groups", s.Inner, s.Groups)
	}
	return nil
}

// stateLayers sets each layer's kind from what the library states for f's
// GGUF architecture, where no reader has: a layer takes the last kind whose
// tensors it all carries, attention where none. A statement no layer meets
// is refused.
func stateLayers(f *meta.File, c *Config) error {
	st, by, err := ggufStates(f.Arch())
	if err != nil || st == nil || len(st.Layers) == 0 || c.LayerKinds != nil {
		return err
	}
	kinds := make([]jlm.LayerKind, c.NLayer)
	stated := 0
	for i := range kinds {
		kinds[i] = jlm.LayerFullAttn
		for _, l := range st.Layers {
			carries := true
			for _, t := range l.Tensors {
				_, ok := f.Get("blk." + itoa(i) + "." + t)
				carries = carries && ok
			}
			if carries {
				kinds[i] = mjLayerKinds()[l.Kind]
			}
		}
		if kinds[i] != jlm.LayerFullAttn {
			stated++
		}
	}
	if stated == 0 {
		return fmt.Errorf("%s states the layers' kinds, and no layer carries the tensors of one", by)
	}
	c.LayerKinds = kinds
	return nil
}

// selectiveNormsFit refuses ssm.dt_b_c_rms where a layer's mixer has no dt, B
// and C norms to apply it to (an SSD mixer's step takes dt in the gate's
// place and B and C off the convolution).
func selectiveNormsFit(f *meta.File, c *Config) error {
	kv, ok := f.Key("ssm.dt_b_c_rms")
	if v, _ := kv.Uint(); !ok || v == 0 {
		return nil
	}
	for i, k := range c.LayerKinds {
		if k == jlm.LayerSSD || k == jlm.LayerSSDAttn {
			return fmt.Errorf("ssm.dt_b_c_rms on layer %d, an SSD mixer: %w", i, ErrNotImplemented)
		}
	}
	return nil
}

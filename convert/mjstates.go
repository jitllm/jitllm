package convert

import (
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// mjStates is what the container states for one GGUF architecture a file
// names, as expressions over the GGUF's keys (src, as a recipe reads them:
// src['attention.layer_norm_epsilon']). A GGUF and a config.json the file
// maps onto those keys (config.keys) reach it alike, through configOfFile.
type mjStates struct {
	// Flags are graph features by name (mjFlags), each true or false.
	Flags map[string]string `json:"flags"`
	// Activation is the feed-forward's activation by its transformers name
	// (mjActivations).
	Activation string `json:"activation"`
	// NormEps is the blocks' norm epsilon and EmbdScale the factor the input
	// rows are scaled by, each a number.
	NormEps   string `json:"norm_eps"`
	EmbdScale string `json:"embd_scale"`
	// Window is the sliding window, and AttnSoftcap and FinalSoftcap the
	// caps on the scores and on the logits, each a number (0 for none).
	Window       *mjStatesWindow `json:"window"`
	AttnSoftcap  string          `json:"attn_softcap"`
	FinalSoftcap string          `json:"final_softcap"`
	// LogitDiv divides the logits (0 for none).
	LogitDiv string `json:"logit_div"`
	// Recurrent is the recurrent layers' geometry (jlm.SSMConfig), each a
	// count.
	Recurrent *mjStatesRecurrent `json:"recurrent"`
	// RopeSections is the M-RoPE split of a head's rotary pairs between a
	// position's axes, a list of counts (empty for none).
	RopeSections string `json:"rope_sections"`
	// Layers is each layer's kind, read off the tensors it carries
	// (stateLayers).
	Layers []mjStatesLayer `json:"layers"`
	exprs  map[string]*jinja.Expr
}

// mjStatesRecurrent is a recurrent geometry: the convolution's taps, the key
// heads (groups), the inner width, the state's size and the value heads. A
// field left out is zero; one stated is at least one.
type mjStatesRecurrent struct {
	ConvKernel string `json:"conv_kernel"`
	Groups     string `json:"groups"`
	Inner      string `json:"inner"`
	StateSize  string `json:"state_size"`
	HeadsV     string `json:"heads_v"`
}

// mjStatesLayer is a layer kind (mjLayerKinds) for a layer carrying every
// one of Tensors, named past its "blk.N." prefix.
type mjStatesLayer struct {
	Tensors []string `json:"tensors"`
	Kind    string   `json:"kind"`
}

// mjLayerKinds is each layer kind a file may state by name.
func mjLayerKinds() map[string]jlm.LayerKind {
	return map[string]jlm.LayerKind{
		"attention":        jlm.LayerFullAttn,
		"linear_attention": jlm.LayerLinearAttn,
		"ssd":              jlm.LayerSSD,
		"ssd_attention":    jlm.LayerSSDAttn,
		"short_conv":       jlm.LayerShortConv,
		"scan":             jlm.LayerScan,
	}
}

// mjStatesWindow is a sliding window: Size positions (0 for none, and then
// nothing else of it applies); every Period'th layer attends globally, or
// Pattern's where the file's attention.sliding_window_pattern does not say
// (swaPatternOf); RopeBase the windowed layers' rotary base.
type mjStatesWindow struct {
	Size     string `json:"size"`
	Period   string `json:"period"`
	Pattern  string `json:"pattern"`
	RopeBase string `json:"rope_base"`
}

// checkStates refuses a statement for a GGUF architecture the file does not
// name, a name no reader states, and an expression reading anything but the
// keys.
func (mj *modelJSON) checkStates() error {
	for _, arch := range slices.Sorted(maps.Keys(mj.Container)) {
		if !slices.Contains(mj.GGUF, arch) {
			return fmt.Errorf("convert: %s states the container of %s, which it names no GGUF architecture", mj.name, arch)
		}
		st := mj.Container[arch]
		srcs := []string{st.NormEps, st.EmbdScale, st.AttnSoftcap, st.FinalSoftcap, st.LogitDiv, st.RopeSections}
		if w := st.Window; w != nil {
			if w.Size == "" || w.Period != "" && w.Pattern != "" {
				return fmt.Errorf("convert: %s: %s: a window states its size, and a period or a pattern, not both", mj.name, arch)
			}
			srcs = append(srcs, w.Size, w.Period, w.Pattern, w.RopeBase)
		}
		if r := st.Recurrent; r != nil {
			if r.ConvKernel == "" {
				return fmt.Errorf("convert: %s: %s: a recurrent geometry states its convolution's taps", mj.name, arch)
			}
			srcs = append(srcs, r.ConvKernel, r.Groups, r.Inner, r.StateSize, r.HeadsV)
		}
		for _, l := range st.Layers {
			if _, ok := mjLayerKinds()[l.Kind]; !ok || len(l.Tensors) == 0 {
				return fmt.Errorf("convert: %s: %s: layer kind %q over tensors %v", mj.name, arch, l.Kind, l.Tensors)
			}
		}
		for _, k := range slices.Sorted(maps.Keys(st.Flags)) {
			_, ok := mjFlags()[k]
			if _, no := mjFlagsNotImplemented()[k]; !ok && !no {
				return fmt.Errorf("convert: %s: %s: flag %q is no graph feature", mj.name, arch, k)
			}
			srcs = append(srcs, st.Flags[k])
		}
		if _, ok := mjActivations()[st.Activation]; st.Activation != "" && !ok {
			return fmt.Errorf("convert: %s: %s: activation %q is not one the container states", mj.name, arch, st.Activation)
		}
		if _, carried := mjActivationParams()[st.Activation]; carried {
			return fmt.Errorf("convert: %s: %s: activation %q carries its parameters in the blocks: %w",
				mj.name, arch, st.Activation, ErrNotImplemented)
		}
		if err := st.compile(srcs); err != nil {
			return fmt.Errorf("convert: %s: %s: %w", mj.name, arch, err)
		}
	}
	return nil
}

// compile compiles each of srcs, refusing one that reads anything but the
// keys and the layers' tensors (layer_tensors: every name a layer's tensor
// takes past its "blk.N." prefix).
func (st *mjStates) compile(srcs []string) error {
	st.exprs = map[string]*jinja.Expr{}
	for _, src := range srcs {
		if src == "" || st.exprs[src] != nil {
			continue
		}
		x, err := jinja.CompileExpr(src)
		if err != nil {
			return err
		}
		for _, n := range x.Names() {
			if n != "src" && n != "layer_tensors" {
				return fmt.Errorf("%q reads %q, which is not the keys (src) or the layers' tensors (layer_tensors)", src, n)
			}
		}
		st.exprs[src] = x
	}
	return nil
}

// ggufStates is what the library states for the container of a GGUF of
// architecture arch, nil where no file states it.
func ggufStates(arch string) (*mjStates, string, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, "", err
	}
	for _, mj := range files {
		if st, ok := mj.Container[arch]; ok {
			return st, mj.name, nil
		}
	}
	return nil, "", nil
}

// stateContainer applies what the library states for the container of f's
// GGUF architecture to c.
func stateContainer(f *meta.File, c *Config) error {
	st, by, err := ggufStates(f.Arch())
	if err != nil || st == nil {
		return err
	}
	return st.apply(by, f, c)
}

// apply sets what st states, read off f's keys, in c; by names the file.
func (st *mjStates) apply(by string, f *meta.File, c *Config) error {
	arch := f.Arch()
	var err error
	vars := map[string]jinja.Value{"src": ggufKeys(f), "layer_tensors": layerTensors(f)}
	value := func(what, src string) (jinja.Value, error) {
		v, err := st.exprs[src].Eval(vars, blockExprSteps)
		if err == nil && v.IsUndefined() {
			err = fmt.Errorf("%q is undefined", src)
		}
		if err != nil {
			return jinja.Undefined(), fmt.Errorf("convert: %s: %s: %s: %w", by, arch, what, err)
		}
		return v, nil
	}
	number := func(what, src string) (float32, error) {
		v, err := value(what, src)
		switch {
		case err != nil:
			return 0, err
		case v.IsInt():
			return float32(v.AsInt()), nil
		case v.IsFloat():
			return float32(v.AsFloat()), nil
		}
		return 0, fmt.Errorf("convert: %s: %s: %s is %v, not a number", by, arch, what, v)
	}
	for _, k := range slices.Sorted(maps.Keys(st.Flags)) {
		v, err := value("flag "+k, st.Flags[k])
		if err != nil {
			return err
		}
		if !v.IsBool() {
			return fmt.Errorf("convert: %s: %s: flag %s is %v, not true or false", by, arch, k, v)
		}
		switch why, no := mjFlagsNotImplemented()[k]; {
		case no && v.AsBool():
			return fmt.Errorf("convert: %s: %s (%s): %w", arch, k, why, ErrNotImplemented)
		case v.AsBool():
			c.Flags |= mjFlags()[k]
		default:
			c.Flags &^= mjFlags()[k]
		}
	}
	if st.Activation != "" {
		c.Flags = c.Flags&^mjActivationFlags | mjActivations()[st.Activation]
	}
	if st.NormEps != "" {
		if c.RMSEps, err = number("norm_eps", st.NormEps); err != nil {
			return err
		}
	}
	if st.EmbdScale != "" {
		if c.EmbdScale, err = number("embd_scale", st.EmbdScale); err != nil {
			return err
		}
	}
	for _, s := range []struct {
		what, src string
		to        *float32
	}{{"attn_softcap", st.AttnSoftcap, &c.AttnSoftcap}, {"final_softcap", st.FinalSoftcap, &c.FinalSoftcap},
		{"logit_div", st.LogitDiv, &c.LogitScale}} {
		if s.src == "" {
			continue
		}
		if *s.to, err = number(s.what, s.src); err != nil {
			return err
		}
		if x := float64(*s.to); x < 0 || math.IsInf(x, 0) || math.IsNaN(x) {
			return fmt.Errorf("convert: %s: %s: %s is %g", by, arch, s.what, x)
		}
	}
	whole := func(what, src string) (uint32, error) {
		x, err := number(what, src)
		if err == nil && (x < 0 || x != float32(uint32(x))) {
			err = fmt.Errorf("convert: %s: %s: %s is %g, not a count", by, arch, what, x)
		}
		return uint32(x), err
	}
	if st.RopeSections != "" {
		v, err := value("rope_sections", st.RopeSections)
		if err != nil {
			return err
		}
		if !v.IsList() {
			return fmt.Errorf("convert: %s: %s: rope_sections is %v, not a list", by, arch, v)
		}
		secs := make([]int32, v.AsList().Len())
		for i := range secs {
			x := v.AsList().Get(i)
			if !x.IsInt() || x.AsInt() < 0 || x.AsInt() > math.MaxInt32 {
				return fmt.Errorf("convert: %s: %s: rope section %d is %v, not a count", by, arch, i, x)
			}
			secs[i] = int32(x.AsInt())
		}
		if len(secs) > 0 {
			if err := ropeSectionsFit(arch, secs, c); err != nil {
				return err
			}
		}
	}
	if r := st.Recurrent; r != nil {
		c.SSM = jlm.SSMConfig{}
		for _, s := range []struct {
			what, src string
			to        *uint32
		}{{"conv_kernel", r.ConvKernel, &c.SSM.ConvKernel}, {"groups", r.Groups, &c.SSM.Groups},
			{"inner", r.Inner, &c.SSM.Inner}, {"state_size", r.StateSize, &c.SSM.StateSize},
			{"heads_v", r.HeadsV, &c.SSM.NHeadV}} {
			if s.src == "" {
				continue
			}
			if *s.to, err = whole("recurrent."+s.what, s.src); err != nil {
				return err
			}
			if *s.to == 0 {
				return fmt.Errorf("convert: %s: incomplete recurrent geometry: %s is zero", arch, s.what)
			}
		}
		if err := recurrentFits(c.SSM); err != nil {
			return fmt.Errorf("convert: %s: %w", arch, err)
		}
	}
	w := st.Window
	if w == nil {
		return nil
	}
	size, err := whole("window.size", w.Size)
	if err != nil {
		return err
	}
	if size == 0 {
		if w.Period != "" {
			return fmt.Errorf("convert: %s: a window period over no attention.sliding_window: %w", arch, ErrNotImplemented)
		}
		return nil
	}
	c.SWAWindow = size
	switch {
	case w.Period != "":
		if c.SWAPeriod, err = whole("window.period", w.Period); err != nil {
			return err
		}
	case w.Pattern != "":
		def, err := whole("window.pattern", w.Pattern)
		if err != nil {
			return err
		}
		if err := swaPatternOf(f, c, def); err != nil {
			return fmt.Errorf("convert: %s: %w", arch, err)
		}
	}
	if w.RopeBase != "" {
		if c.RopeBaseSWA, err = number("window.rope_base", w.RopeBase); err != nil {
			return err
		}
	}
	return nil
}

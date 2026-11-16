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
	// AttnTemp scales each layer's q by its position.
	AttnTemp *mjStatesAttnTemp `json:"attn_temp"`
	// Require holds the file to what the reference's graph reads, as the
	// config's do (mjRequire).
	Require []mjRequire `json:"require"`
	// NoRopeEvery is n where every n-th layer, (il+1) % n == 0, skips the
	// rotary, a count (0 for none).
	NoRopeEvery string `json:"no_rope_every"`
	// ActQuant is how every projection's input is quantized, by name
	// (mjActQuants): row_int8 is a ternary projection's int8 rows.
	ActQuant string `json:"act_quant"`
	exprs    map[string]*jinja.Expr
}

// mjActQuants are the activation quantizations a statement names.
func mjActQuants() map[string]jlm.ActQuant {
	return map[string]jlm.ActQuant{"none": jlm.ActQuantNone, "row_int8": jlm.ActQuantRowInt8}
}

// mjStatesAttnTemp scales q by 1 + Scale*ln(1 + floor((pos+Offset)/Floor)),
// each a number; a Scale of 0 states none.
type mjStatesAttnTemp struct {
	Scale  string `json:"scale"`
	Floor  string `json:"floor"`
	Offset string `json:"offset"`
}

// mjStatesWindow is a sliding window: Size positions; every Period'th layer
// attends globally, or Pattern's where the file's
// attention.sliding_window_pattern does not say (swaPatternOf); RopeBase the
// windowed layers' rotary base and RopeScale its magnitude (0 for the global
// layers'). A Size of 0 slides nothing.
type mjStatesWindow struct {
	Size      string `json:"size"`
	Period    string `json:"period"`
	Pattern   string `json:"pattern"`
	RopeBase  string `json:"rope_base"`
	RopeScale string `json:"rope_scale"`
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
		srcs := []string{st.NormEps, st.EmbdScale, st.AttnSoftcap, st.FinalSoftcap, st.LogitDiv, st.NoRopeEvery}
		for i, r := range st.Require {
			if r.That == "" || r.Else == "" {
				return fmt.Errorf("convert: %s: %s: require %d states what it holds and why", mj.name, arch, i)
			}
			srcs = append(srcs, r.That)
		}
		if w := st.Window; w != nil {
			if w.Size == "" || w.Period != "" && w.Pattern != "" {
				return fmt.Errorf("convert: %s: %s: a window states its size, and a period or a pattern, not both", mj.name, arch)
			}
			srcs = append(srcs, w.Size, w.Period, w.Pattern, w.RopeBase, w.RopeScale)
		}
		if t := st.AttnTemp; t != nil {
			if t.Scale == "" || t.Floor == "" {
				return fmt.Errorf("convert: %s: %s: a q temperature states its scale and its floor", mj.name, arch)
			}
			srcs = append(srcs, t.Scale, t.Floor, t.Offset)
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
		if _, ok := mjActQuants()[st.ActQuant]; st.ActQuant != "" && !ok {
			return fmt.Errorf("convert: %s: %s: act_quant %q is not one the container states", mj.name, arch, st.ActQuant)
		}
		if err := st.compile(srcs); err != nil {
			return fmt.Errorf("convert: %s: %s: %w", mj.name, arch, err)
		}
	}
	return nil
}

// compile compiles each of srcs, refusing one that reads anything but the
// keys.
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
			if n != "src" && n != "tensors" {
				return fmt.Errorf("%q reads %q, which is not the keys (src) or the tensors the file stores (tensors)", src, n)
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
	tensors := jinja.NewDict()
	for i := range f.Tensors {
		tensors.AsDict().Set(f.Tensors[i].Name, jinja.NewBool(true))
	}
	vars := map[string]jinja.Value{"src": ggufKeys(f), "tensors": tensors}
	value := func(what, src string) (jinja.Value, error) {
		v, err := st.exprs[src].Eval(vars, blockExprSteps)
		if err == nil && v.IsUndefined() {
			err = fmt.Errorf("%q is undefined: the file does not state it, and the publisher's config.json "+
				"does (jitllm convert -config, WithModelConfig)", src)
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
	for i, r := range st.Require {
		v, err := value(fmt.Sprintf("require %d", i), r.That)
		if err != nil {
			return err
		}
		if !v.IsBool() {
			return fmt.Errorf("convert: %s: %s: require %d is %v, not true or false", by, arch, i, v)
		}
		if !v.AsBool() {
			return fmt.Errorf("convert: %s: %s: %w", arch, r.Else, ErrNotImplemented)
		}
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
	if st.ActQuant != "" {
		c.ActQuant = mjActQuants()[st.ActQuant]
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
	if t := st.AttnTemp; t != nil {
		if err := stateAttnTemp(by, arch, t, number, whole, c); err != nil {
			return err
		}
	}
	if w := st.Window; w != nil {
		if err := stateWindow(f, arch, w, number, whole, c); err != nil {
			return err
		}
	}
	if st.NoRopeEvery == "" {
		return nil
	}
	n, err := whole("no_rope_every", st.NoRopeEvery)
	switch {
	case err != nil:
		return err
	case n == 0:
		return nil
	case n == 1:
		return fmt.Errorf("convert: %s: every layer skips the rotary: %w", arch, ErrNotImplemented)
	case c.SWAWindow != 0:
		return fmt.Errorf("convert: %s: layers skipping the rotary beside a sliding window of %d: %w",
			arch, c.SWAWindow, ErrNotImplemented)
	}
	c.SWAPeriod = n
	c.Flags |= jlm.FlagNoPEGlobal
	return nil
}

// stateAttnTemp states the q temperature t in c.
func stateAttnTemp(by, arch string, t *mjStatesAttnTemp, number func(what, src string) (float32, error),
	whole func(what, src string) (uint32, error), c *Config) error {
	s, err := number("attn_temp.scale", t.Scale)
	if err != nil || s == 0 {
		return err
	}
	floor, err := whole("attn_temp.floor", t.Floor)
	if err != nil {
		return err
	}
	if floor == 0 {
		return fmt.Errorf("convert: %s: %s: a q temperature of %g with no floor to divide positions by", by, arch, s)
	}
	off := float32(0)
	if t.Offset != "" {
		if off, err = number("attn_temp.offset", t.Offset); err != nil {
			return err
		}
	}
	c.AttnTempScale, c.AttnTempFloor, c.AttnTempOffset = s, floor, off
	return nil
}

// stateWindow states the sliding window w of f in c.
func stateWindow(f *meta.File, arch string, w *mjStatesWindow, number func(what, src string) (float32, error),
	whole func(what, src string) (uint32, error), c *Config) error {
	size, err := whole("window.size", w.Size)
	if err != nil {
		return err
	}
	if size == 0 {
		if w.Period == "" {
			return nil
		}
		p, err := whole("window.period", w.Period)
		if err == nil && p != 0 {
			err = fmt.Errorf("convert: %s: a window period over no attention.sliding_window: %w", arch, ErrNotImplemented)
		}
		return err
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
	if w.RopeScale != "" {
		if c.RopeScaleSWA, err = number("window.rope_scale", w.RopeScale); err != nil {
			return err
		}
	}
	return nil
}

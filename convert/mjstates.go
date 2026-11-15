package convert

import (
	"fmt"
	"maps"
	"slices"

	"github.com/jitllm/jitllm/convert/meta"
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
	exprs     map[string]*jinja.Expr
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
		srcs := []string{st.NormEps, st.EmbdScale}
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
			if n != "src" {
				return fmt.Errorf("%q reads %q, which is not the keys (src)", src, n)
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
	vars := map[string]jinja.Value{"src": ggufKeys(f)}
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
	return nil
}

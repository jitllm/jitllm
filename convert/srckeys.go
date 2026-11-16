package convert

import (
	"encoding/json"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/tok/jinja"
)

// maxSrcArray is the longest array of a source's keys a recipe reads; a
// vocabulary's arrays are not hyperparameters.
const maxSrcArray = 1 << 16

// ggufKeys is f's architecture keys, the namespace stripped, as a recipe
// reads them (the fact src): scalars, and arrays of numbers.
func ggufKeys(f *meta.File) jinja.Value {
	d := jinja.NewDict()
	pre := f.Arch() + "."
	for k, v := range f.KV {
		name, ok := strings.CutPrefix(k, pre)
		if !ok {
			continue
		}
		if x, ok := ggufValue(v); ok {
			d.AsDict().Set(name, x)
		}
	}
	return d
}

// ggufEnv is f's architecture keys as a model.json expression reads them: a
// path is the key past the architecture's prefix, a number a float64.
func ggufEnv(f *meta.File) exprEnv {
	pre := f.Arch() + "."
	return func(path string) any {
		v, ok := f.KV[pre+path]
		if !ok {
			return nil
		}
		x, ok := ggufValue(v)
		if !ok {
			return nil
		}
		switch {
		case x.IsInt():
			return float64(x.AsInt())
		case x.IsFloat():
			return x.AsFloat()
		case x.IsString():
			return x.AsString()
		case x.IsBool():
			return x.AsBool()
		}
		return nil
	}
}

// ggufValue is one key's value, false for one a recipe does not read.
func ggufValue(v meta.Value) (jinja.Value, bool) {
	switch v.Type {
	case meta.String:
		s, ok := v.String()
		return jinja.NewString(s), ok
	case meta.Bool:
		i, ok := v.Int()
		return jinja.NewBool(i != 0), ok
	case meta.Float32, meta.Float64:
		f, ok := v.Float()
		return jinja.NewFloat(f), ok
	case meta.Array:
		if v.N > maxSrcArray {
			return jinja.Value{}, false
		}
		var items []jinja.Value
		switch v.Elem {
		case meta.Float32:
			fs, err := v.Float32s()
			if err != nil {
				return jinja.Value{}, false
			}
			for _, f := range fs {
				items = append(items, jinja.NewFloat(float64(f)))
			}
		default:
			is, err := v.Uint64s()
			if err != nil {
				return jinja.Value{}, false
			}
			for _, i := range is {
				items = append(items, jinja.NewInt(int64(i)))
			}
		}
		return jinja.NewList(items), true
	}
	i, ok := v.Int()
	return jinja.NewInt(i), ok
}

// hfKeys is a config.json's top-level keys as a recipe reads them (the fact
// src).
func hfKeys(c *hfConfig) jinja.Value {
	d := jinja.NewDict()
	for k, raw := range c.keys {
		var x any
		if json.Unmarshal(raw, &x) == nil {
			d.AsDict().Set(k, jinja.FromGoValue(x))
		}
	}
	return d
}

package convert

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"maps"
	"path"
	"slices"
	"strings"
	"unicode"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// A feature is one thing a statement of the library states: an op of a
// block's graph by what it computes -- its step, kind, weights, attributes
// and expressions, not the values it names or the condition it runs under,
// so one op two files state is one feature -- a graph's residual streams,
// and a field a container statement sets (each of an object's, and a field
// naming a word by the word). engine.TestTheCensusCoversEveryFeature holds
// every feature to a model the census runs that takes it.

// streamsFeature is a graph's residual streams.
const streamsFeature = "graph streams"

// opFeature is o as a feature: its fields as the file writes them, but the
// values it names and its condition, the unset ones left out.
func opFeature(o mjBlockOp) (string, error) {
	o.If, o.In, o.Out, o.Name, o.Hist, o.Streams, o.Ex, o.From, o.Steps, o.runs = "", nil, "", "", nil, "", "", "", nil, nil
	b, err := json.Marshal(o)
	if err != nil {
		return "", fmt.Errorf("convert: an op as a feature: %w", err)
	}
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return "", fmt.Errorf("convert: an op as a feature: %w", err)
	}
	if b, err = json.Marshal(setFields(v)); err != nil {
		return "", fmt.Errorf("convert: an op as a feature: %w", err)
	}
	return "op " + string(b), nil
}

// setFields is v, a decoded JSON value, with every unset field left out: an
// empty string, a null, an empty list or object, false and zero.
func setFields(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := map[string]any{}
		for k, e := range x {
			if e = setFields(e); e != nil {
				out[k] = e
			}
		}
		if len(out) == 0 {
			return nil
		}
		return out
	case []any:
		var out []any
		for _, e := range x {
			out = append(out, setFields(e))
		}
		if len(out) == 0 {
			return nil
		}
		return out
	case string:
		if x == "" {
			return nil
		}
	case bool:
		if !x {
			return nil
		}
	case float64:
		if x == 0 {
			return nil
		}
	}
	return v
}

// StatedBy is the library files that state feature f, by their names in
// convert/models.
func StatedBy(f string) ([]string, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	var out []string
	for _, mj := range files {
		in := map[string]bool{}
		if g := mj.Graph; g != nil {
			if g.Streams != nil {
				in[streamsFeature] = true
			}
			if err := opFeatures(g.Block, in); err != nil {
				return nil, err
			}
		}
		if in[f] {
			out = append(out, strings.TrimSuffix(path.Base(mj.name), ".json"))
		}
	}
	return out, nil
}

// Features is every feature the library states, sorted.
func Features() ([]string, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	set := map[string]bool{}
	for _, mj := range files {
		if g := mj.Graph; g != nil {
			if g.Streams != nil {
				set[streamsFeature] = true
			}
			if err := opFeatures(g.Block, set); err != nil {
				return nil, err
			}
		}
	}
	raw, err := rawContainers()
	if err != nil {
		return nil, err
	}
	for _, c := range raw {
		for _, f := range fieldFeatures(c) {
			set[f] = true
		}
	}
	return slices.Sorted(maps.Keys(set)), nil
}

// opFeatures adds each of ops' features to set: an each's steps for it, and
// the concat that joins its runs.
func opFeatures(ops []mjBlockOp, set map[string]bool) error {
	for _, o := range ops {
		if o.Op == "each" {
			if err := opFeatures(o.Steps, set); err != nil {
				return err
			}
			o = mjBlockOp{Op: "concat"}
		}
		f, err := opFeature(o)
		if err != nil {
			return err
		}
		set[f] = true
	}
	return nil
}

// FeaturesOf is the features converting src by opts takes, sorted: a GGUF,
// with its tower where mmproj is not "", or a safetensors directory (read
// from its headers, PlanShards), read as its conversion reads it and
// written nowhere.
func FeaturesOf(src, mmproj string, opts ...Option) ([]string, error) {
	taken := map[string]bool{}
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	o.taken = taken
	var arch string
	if IsSafetensors(src) {
		if mmproj != "" {
			return nil, fmt.Errorf("convert: %s is safetensors, and a tower beside it is a GGUF's", src)
		}
		if _, err := PlanSafetensors(src, append(opts, func(o *options) { o.taken = taken })...); err != nil {
			return nil, err
		}
		dir, _, err := hfShards(src)
		if err != nil {
			return nil, err
		}
		hc, err := readHFConfig(dirFiles(dir))
		if err != nil {
			return nil, err
		}
		ha, err := hfArchFor(dirFiles(dir), hc, o.modelJSON)
		if err != nil {
			return nil, err
		}
		arch = ha.name
	} else {
		if err := withGGUF(src, mmproj, o, func(f *meta.File, _ *jlm.Source) error {
			arch = f.Arch()
			return nil
		}); err != nil {
			return nil, err
		}
	}
	raw, err := rawContainers()
	if err != nil {
		return nil, err
	}
	for _, f := range fieldFeatures(raw[arch]) {
		taken[f] = true
	}
	return slices.Sorted(maps.Keys(taken)), nil
}

// rawContainers is every container statement of the library as its file
// writes it, by architecture.
func rawContainers() (map[string]map[string]json.RawMessage, error) {
	ents, err := fs.ReadDir(modelLibraryFS, "models")
	if err != nil {
		return nil, err
	}
	out := map[string]map[string]json.RawMessage{}
	for _, e := range ents {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		p := path.Join("models", e.Name())
		b, err := modelLibraryFS.ReadFile(p)
		if err != nil {
			return nil, err
		}
		var f struct {
			Container map[string]map[string]json.RawMessage `json:"container"`
		}
		if err := json.Unmarshal(b, &f); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", p, err)
		}
		maps.Copy(out, f.Container)
	}
	return out, nil
}

// fieldFeatures is a container statement's fields as features: each field it
// sets but its tensor table, each of an object's, and a field naming a word
// (a gate, a layout) by the word.
func fieldFeatures(c map[string]json.RawMessage) []string {
	var out []string
	for _, k := range slices.Sorted(maps.Keys(c)) {
		if k == "tensors" {
			continue
		}
		var sub map[string]json.RawMessage
		if json.Unmarshal(c[k], &sub) == nil && sub != nil {
			for _, s := range slices.Sorted(maps.Keys(sub)) {
				out = append(out, "container "+k+"."+s+word(sub[s]))
			}
			continue
		}
		out = append(out, "container "+k+word(c[k]))
	}
	return out
}

// word is " = " and v where v is a word (letters and underscores), "" for
// any other value.
func word(v json.RawMessage) string {
	var s string
	if json.Unmarshal(v, &s) != nil || s == "" ||
		strings.TrimFunc(s, func(r rune) bool { return unicode.IsLetter(r) || r == '_' }) != "" {
		return ""
	}
	return " = " + s
}

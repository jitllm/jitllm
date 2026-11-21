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
// naming a word by the word) -- and each choice a statement table beside the
// library states that changes what runs (tableFeatures).
// engine.TestTheCensusCoversEveryFeature holds every feature to a model the
// census runs that takes it.

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
	tables, err := tableFeatures()
	if err != nil {
		return nil, err
	}
	if by, ok := tables[f]; ok {
		return []string{by}, nil
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
	tables, err := tableFeatures()
	if err != nil {
		return nil, err
	}
	for f := range tables {
		set[f] = true
	}
	return slices.Sorted(maps.Keys(set)), nil
}

// take adds feature f to taken where that is not nil.
func take(taken map[string]bool, f string) {
	if taken != nil {
		taken[f] = true
	}
}

// tableFeatures is each choice the statement tables beside the library
// state that changes what runs, and the table stating it: each op of a
// part front.json states, by the part; a readout and a temperature band of
// readouts.json; each source's reading of a rotary method and each
// convention it is read under (rotary.json); a sampling key of
// generation.json; a tool-call form of toolforms.json; a projector's own
// tensor name, indexed part and rename of towertensors.json (a name it
// leaves unread runs nothing); and each op of a projector's tower and each
// picture rule of projectors.json.
func tableFeatures() (map[string]string, error) {
	out := map[string]string{}
	add := func(by string, fs ...string) {
		for _, f := range fs {
			out[f] = by
		}
	}
	var parts map[string][]mjBlockOp
	if err := json.Unmarshal(frontJSON, &parts); err != nil {
		return nil, fmt.Errorf("convert: front.json: %w", err)
	}
	for part, steps := range parts {
		set := map[string]bool{}
		if err := opFeatures(steps, set); err != nil {
			return nil, err
		}
		for f := range set {
			add("convert/front.json", frontFeature(part, f))
		}
	}
	ro, err := readoutLibrary()
	if err != nil {
		return nil, err
	}
	for name := range ro.Readouts {
		add("convert/readouts.json", "readout "+name)
	}
	for band := range ro.Bands {
		add("convert/readouts.json", "readout band "+band)
	}
	rot, err := readRotary()
	if err != nil {
		return nil, err
	}
	for name, m := range rot.Methods {
		if m.HF != nil {
			add("convert/rotary.json", rotaryFeature(name, "hf", ""))
		}
		if m.GGUF != nil {
			add("convert/rotary.json", rotaryFeature(name, "gguf", ""))
			for c := range m.GGUF.Conventions {
				add("convert/rotary.json", rotaryFeature(name, "gguf", c))
			}
		}
	}
	gen, err := generationKeys()
	if err != nil {
		return nil, err
	}
	for field := range gen {
		add("convert/generation.json", "generation "+field)
	}
	tf, err := readToolForms()
	if err != nil {
		return nil, err
	}
	for _, f := range tf.Forms {
		add("convert/toolforms.json", "tool form "+f.Form.Name)
	}
	tt, err := readTowerTensors()
	if err != nil {
		return nil, err
	}
	for p, tp := range tt.Projectors {
		for name := range tp.Roles {
			add("convert/towertensors.json", towerTensorFeature(p, "role", name))
		}
		for _, ix := range tp.Indexed {
			for part := range ix.Roles {
				add("convert/towertensors.json", towerTensorFeature(p, "indexed", ix.Prefix+part))
			}
		}
		for _, rn := range tp.Renames {
			add("convert/towertensors.json", towerTensorFeature(p, "rename", rn.Prefix+rn.Infix))
		}
	}
	lib, err := projectorLibrary()
	if err != nil {
		return nil, err
	}
	var pf projectorFile
	if err := json.Unmarshal(projectorsJSON, &pf); err != nil {
		return nil, fmt.Errorf("convert: projectors.json: %w", err)
	}
	// An entry that is another's (as) is converted by that one's tower.
	towers := map[string]*mjTower{}
	for name, ps := range lib {
		if pf.Projectors[name].As == "" {
			towers[name] = ps.Tower
		}
	}
	for name, a := range pf.Audio {
		towers[name] = a.Tower
	}
	for name, t := range towers {
		if t == nil {
			continue
		}
		for _, ops := range t.programs() {
			if err := towerOpFeatures(ops, func(f string) { add("convert/projectors.json", towerFeature(name, f)) }); err != nil {
				return nil, err
			}
		}
	}
	pics, err := pictureLibrary()
	if err != nil {
		return nil, err
	}
	for rn := range pics.Rules {
		add("convert/projectors.json", "picture rule "+rn)
	}
	return out, nil
}

// frontFeature is op feature f of front.json's part.
func frontFeature(part, f string) string { return "front " + part + " " + f }

// rotaryFeature is a method's reading by source (hf or gguf), under
// convention where that is not "".
func rotaryFeature(method, source, convention string) string {
	f := "rotary " + method + " by " + source
	if convention != "" {
		f += ", convention " + convention
	}
	return f
}

// towerTensorFeature is projector p's towertensors.json entry of kind
// (role, indexed, rename) for name.
func towerTensorFeature(p, kind, name string) string {
	return "tower tensors " + p + " " + kind + " " + name
}

// towerFeature is op feature f of projector p's tower.
func towerFeature(p, f string) string { return "tower " + p + " " + f }

// towerOpFeature is o as a feature: its fields as the file writes them but
// its condition, the values it names and the ops it runs, the unset ones
// left out.
func towerOpFeature(o mjTowerOp) (string, error) {
	o.If, o.In, o.Out, o.Var, o.Ops = "", nil, "", "", nil
	b, err := json.Marshal(o)
	if err != nil {
		return "", fmt.Errorf("convert: a tower op as a feature: %w", err)
	}
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return "", fmt.Errorf("convert: a tower op as a feature: %w", err)
	}
	if b, err = json.Marshal(setFields(v)); err != nil {
		return "", fmt.Errorf("convert: a tower op as a feature: %w", err)
	}
	return "op " + string(b), nil
}

// towerOpFeatures calls add with each of ops' features, an each's ops among
// them.
func towerOpFeatures(ops []mjTowerOp, add func(string)) error {
	for _, o := range ops {
		f, err := towerOpFeature(o)
		if err != nil {
			return err
		}
		add(f)
		if err := towerOpFeatures(o.Ops, add); err != nil {
			return err
		}
	}
	return nil
}

// towerTook records each op projector p's tower runs in taken; nil where
// taken is.
func towerTook(taken map[string]bool, p string) func(mjTowerOp) error {
	if taken == nil {
		return nil
	}
	return func(o mjTowerOp) error {
		f, err := towerOpFeature(o)
		if err != nil {
			return err
		}
		taken[towerFeature(p, f)] = true
		return nil
	}
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

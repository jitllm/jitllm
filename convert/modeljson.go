package convert

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"slices"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// A model.json states what the converter reads from a model's files: which
// config keys mean what, which tensor names hold which roles, and the
// transforms from the converter's closed vocabulary that reshape a tensor on
// the way in. It is data; the transforms are Go. docs/design/model-json.md
// is the design.
//
// One is found, in order of authority: the caller's (WithModelJSON), the
// model directory's own model.json, the library embedded here
// (convert/models), and last the Go table (hfArchTable) for the classes not
// yet moved to the library.

//go:embed models/*.json
var modelLibraryFS embed.FS

type modelJSON struct {
	Classes  []string  `json:"classes"`
	Template string    `json:"template"`
	Config   mjConfig  `json:"config"`
	Tensors  mjTensors `json:"tensors"`
	name     string    // for messages
}

type mjConfig struct {
	// Root is the object the paths are read from (text_config for a model
	// with a tower); $.key reads the file's top.
	Root string `json:"root"`
	// Arch is the graph configOf builds, by its GGUF name, and Keys the GGUF
	// keys (without the architecture's prefix) it reads.
	Arch   string           `json:"arch"`
	Keys   map[string]mjKey `json:"keys"`
	Vocab  string           `json:"vocab"`
	Refuse []mjRefusal      `json:"refuse"`
	Layers *mjLayers        `json:"layers"`
	Head   mjHead           `json:"head"`
}

// mjKey is one GGUF key's value, an expression with the key's type.
type mjKey struct {
	Uint   string `json:"uint"`
	Float  string `json:"float"`
	Int32s string `json:"int32s"`
}

type mjRefusal struct {
	If  string `json:"if"`
	Why string `json:"why"`
}

// mjLayers reads each block's kind from a list in the config.
type mjLayers struct {
	Key   string            `json:"key"`
	Kinds map[string]string `json:"kinds"` // the list's value -> "recurrent" | "attention"
}

// mjHead is the output head: its source name, and the config's statement
// that the embedding is the head instead.
type mjHead struct {
	Tensor string `json:"tensor"`
	Tied   string `json:"tied"`
}

type mjTensors struct {
	BlockPrefix  string              `json:"block_prefix"`
	Model        map[string]mjTensor `json:"model"`
	Block        map[string]mjTensor `json:"block"`
	Gather       []mjGather          `json:"gather"`
	Ignore       []string            `json:"ignore"`
	IgnorePrefix []string            `json:"ignore_prefix"`
}

// mjTensor is a source tensor's role, by the role's GGUF tensor name (roleOf),
// and the transforms it takes in order. A bare string is the role alone.
type mjTensor struct {
	Role string   `json:"role"`
	Do   []string `json:"do"`
}

func (t *mjTensor) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		t.Role = s
		return nil
	}
	type plain mjTensor
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	return d.Decode((*plain)(t))
}

// mjGather builds one tensor a block from several sources by a transform.
type mjGather struct {
	Role string   `json:"role"`
	Do   string   `json:"do"`
	From []string `json:"from"`
}

func parseModelJSON(b []byte, name string) (*modelJSON, error) {
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	mj := &modelJSON{name: name}
	if err := d.Decode(mj); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if len(mj.Classes) == 0 {
		return nil, fmt.Errorf("convert: %s names no class", name)
	}
	if _, ok := archOf()[mj.Config.Arch]; !ok {
		return nil, fmt.Errorf("convert: %s: graph %q is not one configOf builds", name, mj.Config.Arch)
	}
	if mj.Config.Vocab == "" || mj.Config.Head.Tensor == "" {
		return nil, fmt.Errorf("convert: %s states no vocab or no head tensor", name)
	}
	return mj, nil
}

// modelLibrary is the embedded library by class.
func modelLibrary() (map[string]*modelJSON, error) {
	ents, err := fs.ReadDir(modelLibraryFS, "models")
	if err != nil {
		return nil, err
	}
	lib := map[string]*modelJSON{}
	for _, e := range ents {
		p := path.Join("models", e.Name())
		b, err := modelLibraryFS.ReadFile(p)
		if err != nil {
			return nil, err
		}
		mj, err := parseModelJSON(b, "convert/"+p)
		if err != nil {
			return nil, err
		}
		for _, c := range mj.Classes {
			if prev, dup := lib[c]; dup {
				return nil, fmt.Errorf("convert: class %s is in both %s and %s", c, prev.name, mj.name)
			}
			if _, dup := hfArchTable()[c]; dup {
				return nil, fmt.Errorf("convert: class %s is in %s and in the Go table", c, mj.name)
			}
			lib[c] = mj
		}
	}
	return lib, nil
}

// WithModelJSON converts a safetensors model by the model.json at path,
// ahead of the model's own and the library's.
func WithModelJSON(path string) Option {
	return func(o *options) { o.modelJSON = path }
}

// hfArchFor is the entry a model converts by: the caller's model.json, the
// directory's, the library's, or the Go table's.
func hfArchFor(dir hfDir, c *hfConfig, caller string) (*hfArch, error) {
	var mj *modelJSON
	switch {
	case caller != "":
		b, err := os.ReadFile(caller)
		if err != nil {
			return nil, fmt.Errorf("convert: the model.json asked for: %w", err)
		}
		if mj, err = parseModelJSON(b, caller); err != nil {
			return nil, err
		}
	case dir.has("model.json"):
		b, err := dir.read("model.json")
		if err != nil {
			return nil, err
		}
		if mj, err = parseModelJSON(b, dir.path("model.json")); err != nil {
			return nil, err
		}
	default:
		lib, err := modelLibrary()
		if err != nil {
			return nil, err
		}
		for _, n := range c.Architectures {
			if mj = lib[n]; mj != nil {
				break
			}
		}
		if mj == nil {
			return hfArchOf(c)
		}
	}
	if !slices.ContainsFunc(c.Architectures, func(a string) bool { return slices.Contains(mj.Classes, a) }) {
		return nil, fmt.Errorf("convert: %s states %v, and %s's architectures are %v",
			mj.name, mj.Classes, c.path, c.Architectures)
	}
	return mj.hfArch()
}

// hfArch lowers a model.json onto the entry the safetensors path reads.
func (mj *modelJSON) hfArch() (*hfArch, error) {
	a := &hfArch{
		arch:         archOf()[mj.Config.Arch],
		template:     mj.Template,
		blockPrefix:  mj.Tensors.BlockPrefix,
		model:        map[string]jlm.Role{},
		block:        map[string]jlm.Role{},
		ignore:       map[string]bool{},
		ignorePrefix: mj.Tensors.IgnorePrefix,
		split:        map[string]hfSplit{},
	}
	for _, n := range mj.Tensors.Ignore {
		a.ignore[n] = true
	}
	elem := map[string][]mjElem{}
	add := func(dst map[string]jlm.Role, name string, t mjTensor) error {
		r, ok := roleOf(t.Role)
		if !ok {
			return fmt.Errorf("convert: %s: %s's role %q is not a tensor name the container knows", mj.name, name, t.Role)
		}
		dst[name] = retarget(a.arch, r)
		var shape []hfSplit
		for _, d := range t.Do {
			if f, ok := mjElems()[d]; ok {
				elem[name] = append(elem[name], f)
				continue
			}
			if f, ok := mjShapes()[d]; ok {
				shape = append(shape, f)
				continue
			}
			return fmt.Errorf("convert: %s: %s: transform %q is not in the vocabulary", mj.name, name, d)
		}
		switch len(shape) {
		case 0:
		case 1:
			a.split[name] = shape[0]
		default:
			return fmt.Errorf("convert: %s: %s takes %d shape transforms; one is the limit", mj.name, name, len(shape))
		}
		return nil
	}
	for n, t := range mj.Tensors.Model {
		if err := add(a.model, n, t); err != nil {
			return nil, err
		}
	}
	for n, t := range mj.Tensors.Block {
		if err := add(a.block, n, t); err != nil {
			return nil, err
		}
	}
	if len(elem) > 0 {
		a.fixup = func(e *jlm.Tensor, data []byte, c *Config) error {
			for _, f := range elem[a.blockRest(e.Name)] {
				var err error
				if data, err = f(e, data); err != nil {
					return err
				}
			}
			e.Data = data
			return nil
		}
	}
	if len(mj.Tensors.Gather) > 0 {
		gs := make([]hfGatherStep, len(mj.Tensors.Gather))
		for i, g := range mj.Tensors.Gather {
			r, ok := roleOf(g.Role)
			if !ok {
				return nil, fmt.Errorf("convert: %s: gathered role %q is not a tensor name the container knows", mj.name, g.Role)
			}
			f, ok := mjGathers()[g.Do]
			if !ok {
				return nil, fmt.Errorf("convert: %s: gather %q is not in the vocabulary", mj.name, g.Do)
			}
			gs[i] = hfGatherStep{role: retarget(a.arch, r), from: g.From, f: f}
		}
		a.gather = func(c *Config, names []string, read func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
			return gatherBlocks(c, a.blockPrefix, gs, names, read)
		}
	}
	a.config = mj.config
	return a, nil
}

// config reads config.json through the model.json into the GGUF keys and
// hands them to configOf, so one reader decides what each key means.
func (mj *modelJSON) config(c *hfConfig, names []string) (*Config, error) {
	top := map[string]any{}
	for k, raw := range c.keys {
		var v any
		if err := json.Unmarshal(raw, &v); err != nil {
			return nil, fmt.Errorf("convert: %s: %s: %w", c.path, k, err)
		}
		top[k] = v
	}
	root := top
	if mj.Config.Root != "" {
		r, ok := top[mj.Config.Root].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("convert: %s: %s reads %s, which this config has not as an object",
				c.path, mj.name, mj.Config.Root)
		}
		root = r
	}
	env := func(p string) any {
		at, rest := root, p
		if r, ok := strings.CutPrefix(p, "$."); ok {
			at, rest = top, r
		}
		var cur any = at
		for _, k := range strings.Split(rest, ".") {
			m, ok := cur.(map[string]any)
			if !ok {
				return nil
			}
			cur = m[k]
		}
		return cur
	}
	eval := func(what, src string) (any, error) {
		n, err := parseExpr(src)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %s: %w", mj.name, what, err)
		}
		v, err := evalMJ(n, env)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %s: %s: %w", c.path, mj.name, what, err)
		}
		return v, nil
	}
	for _, r := range mj.Config.Refuse {
		v, err := eval("refuse "+r.If, r.If)
		if err != nil {
			return nil, err
		}
		if b, ok := v.(bool); !ok {
			return nil, fmt.Errorf("convert: %s: refuse %q is %s, not a boolean", mj.name, r.If, exprKind(v))
		} else if b {
			return nil, fmt.Errorf("convert: %s: %s: %w", c.path, r.Why, ErrNotImplemented)
		}
	}

	ns := mj.Config.Arch + "."
	kvs := map[string]meta.Value{"general.architecture": meta.MakeString(mj.Config.Arch)}
	keys := make([]string, 0, len(mj.Config.Keys))
	for k := range mj.Config.Keys {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		spec := mj.Config.Keys[k]
		var src string
		n := 0
		for _, s := range []string{spec.Uint, spec.Float, spec.Int32s} {
			if s != "" {
				src, n = s, n+1
			}
		}
		if n != 1 {
			return nil, fmt.Errorf("convert: %s: key %s states %d types; one is the rule", mj.name, k, n)
		}
		v, err := eval(k, src)
		if err != nil {
			return nil, err
		}
		if v == nil {
			// An absent key takes the CLASS's default, which the model.json
			// writes with ??; the reader's own default is another class's.
			return nil, fmt.Errorf("convert: %s: %s reads %q, which is absent, and %s states no default",
				c.path, k, src, mj.name)
		}
		switch {
		case spec.Uint != "":
			u, err := exprUint(k, v)
			if err != nil {
				return nil, err
			}
			kvs[ns+k] = meta.MakeUint(u)
		case spec.Float != "":
			f, ok := v.(float64)
			if !ok {
				return nil, fmt.Errorf("convert: %s: %s is %s, not a number", c.path, k, exprKind(v))
			}
			kvs[ns+k] = meta.MakeFloat(float32(f))
		default:
			list, ok := v.([]any)
			if !ok {
				return nil, fmt.Errorf("convert: %s: %s is %s, not a list", c.path, k, exprKind(v))
			}
			out := make([]int32, len(list))
			for i, x := range list {
				u, err := exprUint(fmt.Sprintf("%s[%d]", k, i), x)
				if err != nil || u > 1<<31-1 {
					return nil, fmt.Errorf("convert: %s: %s[%d] is not a 32-bit count", c.path, k, i)
				}
				out[i] = int32(u)
			}
			kvs[ns+k] = meta.MakeInt32s(out)
		}
	}

	// configOf reads the vocabulary's size, whether the head is tied and
	// which blocks are recurrent off the tensor table; these stand in for the
	// checkpoint's.
	vv, err := eval("vocab", mj.Config.Vocab)
	if err != nil {
		return nil, err
	}
	vocab, err := exprUint("vocab", vv)
	if err != nil {
		return nil, err
	}
	embd := uint64(0)
	if v, ok := kvs[ns+"embedding_length"]; ok {
		embd, _ = v.Uint()
	}
	ts := []meta.Tensor{{Name: "token_embd.weight", Dims: []uint64{embd, vocab}}}
	tied := false
	if mj.Config.Head.Tied != "" {
		v, err := eval("tied", mj.Config.Head.Tied)
		if err != nil {
			return nil, err
		}
		tied = v == true
	}
	switch hasHead := slices.Contains(names, mj.Config.Head.Tensor); {
	case hasHead && tied:
		return nil, fmt.Errorf("convert: %s: the head is tied and %s is present; which one is the head "+
			"is not a question this converter may guess at", c.path, mj.Config.Head.Tensor)
	case hasHead:
		ts = append(ts, meta.Tensor{Name: "output.weight", Dims: ts[0].Dims})
	case !tied:
		return nil, fmt.Errorf("convert: %s: no %s and the head is not tied: this model's output head "+
			"is missing, not tied", c.path, mj.Config.Head.Tensor)
	}
	if l := mj.Config.Layers; l != nil {
		v, err := eval("layers", l.Key)
		if err != nil {
			return nil, err
		}
		list, ok := v.([]any)
		if !ok {
			return nil, fmt.Errorf("convert: %s: %s is %s, not a list", c.path, l.Key, exprKind(v))
		}
		// A list shorter than the model leaves blocks of no stated kind.
		if bc, ok := kvs[ns+"block_count"]; ok {
			if n, _ := bc.Uint(); uint64(len(list)) != n {
				return nil, fmt.Errorf("convert: %s: %s has %d entries in a %d-block model", c.path, l.Key, len(list), n)
			}
		}
		for i, x := range list {
			s, _ := x.(string)
			switch l.Kinds[s] {
			case "recurrent":
				ts = append(ts, meta.Tensor{Name: "blk." + itoa(i) + ".ssm_conv1d.weight", Dims: []uint64{1, embd}})
			case "attention":
			default:
				return nil, fmt.Errorf("convert: %s: %s[%d] is %v, which %s does not name", c.path, l.Key, i, x, mj.name)
			}
		}
	}
	return configOf(meta.New(c.path, kvs, ts))
}

func exprUint(what string, v any) (uint64, error) {
	f, ok := v.(float64)
	if !ok || f < 0 || f != float64(uint64(f)) {
		return 0, fmt.Errorf("convert: %s is %v, not a count", what, v)
	}
	return uint64(f), nil
}

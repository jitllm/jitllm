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
// config keys mean what, which tensor names hold which roles, and how a tensor
// is reshaped on the way in, as a composition of the tensor primitives
// (mjtransforms.go). It is data: the Go is the primitives and the generic
// readers of its statements, and knows no model. What the converter cannot
// express it refuses in code, by name, one rule for every file.
// docs/design/model-json.md is the design.
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
	Layers *mjLayers        `json:"layers"`
	Head   mjHead           `json:"head"`
	// Activation is the feed-forward's activation by its transformers name
	// (mjActivations).
	Activation string `json:"activation"`
	// RopeScaling is the rope scaling object (rope_parameters or
	// rope_scaling), null for none.
	RopeScaling string `json:"rope_scaling"`
	// Window is a sliding window: its size, and whether it is in use.
	Window *mjWindow `json:"window"`
	// Flags states named graph features (mjFlags), each a boolean.
	Flags map[string]string `json:"flags"`
}

type mjWindow struct {
	Size string `json:"size"`
	Used string `json:"used"`
}

// mjActivations is each feed-forward activation the container states, by its
// transformers name, as the activation flags it sets; another name is refused.
func mjActivations() map[string]jlm.Flags {
	return map[string]jlm.Flags{
		"silu":              0,
		"swish":             0,
		"gelu_pytorch_tanh": jlm.FlagGELU,
		"relu2":             jlm.FlagReLU2,
	}
}

// mjActivationFlags are the flags mjActivations decides between.
const mjActivationFlags = jlm.FlagGELU | jlm.FlagReLU2

// mjFlags is each graph feature a file may state by name, as the container
// flag the engine runs it by.
func mjFlags() map[string]jlm.Flags {
	return map[string]jlm.Flags{
		"attention_output_gate": jlm.FlagAttnOutGate,
		"non_causal":            jlm.FlagNonCausal,
	}
}

// mjKey is one GGUF key's value, an expression with the key's type.
type mjKey struct {
	Uint   string `json:"uint"`
	Float  string `json:"float"`
	Int32s string `json:"int32s"`
	Bool   string `json:"bool"`
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
	Experts      *mjExperts          `json:"experts"`
}

// mjExperts names a mixture's per-expert matrices: the block-relative prefix
// before an expert's index, and each matrix's bank by the bank's GGUF tensor
// name. The converter stacks one bank per role.
type mjExperts struct {
	Prefix string            `json:"prefix"`
	Names  map[string]string `json:"names"`
}

// mjTensor is a source tensor's role, by the role's GGUF tensor name (roleOf),
// and the primitives it takes in order. A bare string is the role alone.
type mjTensor struct {
	Role string `json:"role"`
	Do   mjPipe `json:"do"`
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

// mjGather builds one tensor a block from several sources, each entering as
// the vector of its values, by primitives that leave one tensor.
type mjGather struct {
	Role string   `json:"role"`
	From []string `json:"from"`
	Do   mjPipe   `json:"do"`
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
	if mj.Config.Activation == "" {
		return nil, fmt.Errorf("convert: %s states no activation", name)
	}
	for f := range mj.Config.Flags {
		if _, ok := mjFlags()[f]; !ok {
			return nil, fmt.Errorf("convert: %s: flag %q is not a graph feature the container states", name, f)
		}
	}
	if w := mj.Config.Window; w != nil && (w.Size == "" || w.Used == "") {
		return nil, fmt.Errorf("convert: %s: a window states its size and whether it is used", name)
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
	if e := mj.Tensors.Experts; e != nil {
		a.expertPrefix, a.expert = e.Prefix, map[string]jlm.Role{}
		for n, r := range e.Names {
			role, ok := roleOf(r)
			if !ok {
				return nil, fmt.Errorf("convert: %s: expert matrix %s's bank %q is not a tensor name the container knows", mj.name, n, r)
			}
			a.expert[n] = role
		}
	}
	st := &mjState{src: map[string]mjVal{}}
	pipes := map[string]mjPipe{}
	add := func(dst map[string]jlm.Role, name string, t mjTensor) error {
		r, ok := roleOf(t.Role)
		if !ok {
			return fmt.Errorf("convert: %s: %s's role %q is not a tensor name the container knows", mj.name, name, t.Role)
		}
		dst[name] = retarget(a.arch, r)
		if len(t.Do) > 0 {
			if err := t.Do.check(); err != nil {
				return fmt.Errorf("convert: %s: %s: %w", mj.name, name, err)
			}
			pipes[name] = t.Do
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
	if len(pipes) > 0 {
		// The result's shape and type are stated before the tensor is laid out,
		// from its source's; the bytes follow when it is read.
		a.shape = func(e *jlm.Tensor) error {
			p, ok := pipes[a.blockRest(e.Name)]
			if !ok {
				return nil
			}
			in := mjVal{shape: mjTorch(e), typ: e.Type}
			st.mu.Lock()
			env := st.env
			st.src[e.Name] = in
			st.mu.Unlock()
			v, err := p.run(env, []mjVal{in})
			if err != nil {
				return fmt.Errorf("convert: %s: %w", e.Name, err)
			}
			return mjSetShape(e, v)
		}
		a.fixup = func(e *jlm.Tensor, data []byte, c *Config) error {
			p, ok := pipes[a.blockRest(e.Name)]
			if !ok {
				e.Data = data
				return nil
			}
			st.mu.Lock()
			env, in, ok := st.env, st.src[e.Name], len(st.src[e.Name].shape) > 0
			st.mu.Unlock()
			if !ok {
				return fmt.Errorf("convert: %s: transformed without its shape stated", e.Name)
			}
			if len(data) == 0 {
				return fmt.Errorf("convert: %s: no bytes to transform", e.Name)
			}
			in.data = data
			v, err := p.run(env, []mjVal{in})
			if err != nil {
				return fmt.Errorf("convert: %s: %w", e.Name, err)
			}
			if err := mjSetShape(e, v); err != nil {
				return err
			}
			e.Data = v.data
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
			if len(g.From) == 0 {
				return nil, fmt.Errorf("convert: %s: the gather of %s has no sources", mj.name, g.Role)
			}
			if err := g.Do.check(); err != nil {
				return nil, fmt.Errorf("convert: %s: the gather of %s: %w", mj.name, g.Role, err)
			}
			gs[i] = hfGatherStep{role: retarget(a.arch, r), from: g.From, do: g.Do}
		}
		a.gather = func(c *Config, names []string, read func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
			st.mu.Lock()
			env := st.env
			st.mu.Unlock()
			return gatherBlocks(env, a.blockPrefix, gs, names, read)
		}
	}
	a.config = func(c *hfConfig, names []string) (*Config, error) {
		env, err := mj.env(c)
		if err != nil {
			return nil, err
		}
		st.mu.Lock()
		st.env = env
		st.mu.Unlock()
		return mj.configIn(env, c, names)
	}
	return a, nil
}

// config reads config.json through the model.json into the GGUF keys and
// hands them to configOf, so one reader decides what each key means.
func (mj *modelJSON) config(c *hfConfig, names []string) (*Config, error) {
	env, err := mj.env(c)
	if err != nil {
		return nil, err
	}
	return mj.configIn(env, c, names)
}

// env resolves the file's paths in config.json: from the root it names, or
// the file's top for $.key.
func (mj *modelJSON) env(c *hfConfig) (exprEnv, error) {
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
	return env, nil
}

// mjGraphKeys are config keys that change the graph, each with the value that
// leaves it unchanged. One a file does not read, present with another value,
// is refused: the file does not state that graph.
func mjGraphKeys() map[string]any {
	return map[string]any{
		"partial_rotary_factor":                 1.0,
		"rope_parameters.partial_rotary_factor": 1.0,
		"sliding_window":                        nil,
		// clip_qkv clamps q, k and v; where the clamp falls against a q/k norm
		// is the graph's, and no file states one yet.
		"clip_qkv": nil,
	}
}

func (mj *modelJSON) configIn(env exprEnv, c *hfConfig, names []string) (*Config, error) {
	read := map[string]bool{}
	renv := func(p string) any {
		read[p] = true
		return env(p)
	}
	eval := func(what, src string) (any, error) {
		n, err := parseExpr(src)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %s: %w", mj.name, what, err)
		}
		v, err := evalMJ(n, renv)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %s: %s: %w", c.path, mj.name, what, err)
		}
		return v, nil
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
		for _, s := range []string{spec.Uint, spec.Float, spec.Int32s, spec.Bool} {
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
		case spec.Bool != "":
			b, ok := v.(bool)
			if !ok {
				return nil, fmt.Errorf("convert: %s: %s is %s, not a boolean", c.path, k, exprKind(v))
			}
			kvs[ns+k] = meta.MakeBool(b)
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
	window := uint64(0)
	if w := mj.Config.Window; w != nil {
		used, err := eval("window used", w.Used)
		if err != nil {
			return nil, err
		}
		if used != true && used != false {
			return nil, fmt.Errorf("convert: %s: window used is %s, not a boolean", c.path, exprKind(used))
		}
		size, err := eval("window size", w.Size)
		if err != nil {
			return nil, err
		}
		if used == true && size != nil {
			if window, err = exprUint("window size", size); err != nil {
				return nil, err
			}
			kvs[ns+"attention.sliding_window"] = meta.MakeUint(window)
		}
	}
	cfg, err := configOf(meta.New(c.path, kvs, ts))
	if err != nil {
		return nil, err
	}
	// The statements configOf does not read decide after it.
	if window != 0 && uint64(cfg.SWAWindow) != window {
		return nil, fmt.Errorf("convert: %s: a sliding window of %d is not one graph %s runs: %w",
			c.path, window, mj.Config.Arch, ErrNotImplemented)
	}
	av, err := eval("activation", mj.Config.Activation)
	if err != nil {
		return nil, err
	}
	act, _ := av.(string)
	af, ok := mjActivations()[act]
	if !ok {
		return nil, fmt.Errorf("convert: %s: activation %v is not implemented: %w", c.path, av, ErrNotImplemented)
	}
	cfg.Flags = cfg.Flags&^mjActivationFlags | af
	fnames := make([]string, 0, len(mj.Config.Flags))
	for f := range mj.Config.Flags {
		fnames = append(fnames, f)
	}
	sort.Strings(fnames)
	for _, f := range fnames {
		v, err := eval("flag "+f, mj.Config.Flags[f])
		if err != nil {
			return nil, err
		}
		b, ok := v.(bool)
		if !ok {
			return nil, fmt.Errorf("convert: %s: flag %s is %s, not a boolean", c.path, f, exprKind(v))
		}
		if b {
			cfg.Flags |= mjFlags()[f]
		} else {
			cfg.Flags &^= mjFlags()[f]
		}
	}
	if mj.Config.RopeScaling != "" {
		v, err := eval("rope_scaling", mj.Config.RopeScaling)
		if err != nil {
			return nil, err
		}
		if err := mjRopeScaling(cfg, v); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", c.path, err)
		}
	}
	if err := mj.mixtureCovers(cfg, names); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", c.path, err)
	}
	gk := make([]string, 0, len(mjGraphKeys()))
	for k := range mjGraphKeys() {
		gk = append(gk, k)
	}
	sort.Strings(gk)
	for _, k := range gk {
		if v := env(k); !read[k] && v != nil {
			if eq, _ := exprEqual(v, mjGraphKeys()[k]); !eq {
				return nil, fmt.Errorf("convert: %s: %s is %v, and %s does not read it: %w", c.path, k, v, mj.name, ErrNotImplemented)
			}
		}
	}
	return cfg, nil
}

// mixtureCovers refuses a mixture whose experts some block does not carry:
// dense and mixture blocks in one model are not a pattern a file states yet.
func (mj *modelJSON) mixtureCovers(cfg *Config, names []string) error {
	e := mj.Tensors.Experts
	if e == nil {
		return nil
	}
	has := map[uint32]bool{}
	for _, n := range names {
		rest, ok := strings.CutPrefix(n, mj.Tensors.BlockPrefix)
		if !ok {
			continue
		}
		if b, tail, err := leadingIndex(n, rest); err == nil && strings.HasPrefix(tail, e.Prefix) {
			has[uint32(b)] = true
		}
	}
	if len(has) == 0 {
		return nil
	}
	for b := uint32(0); b < cfg.NLayer; b++ {
		if !has[b] {
			return fmt.Errorf("block %d carries no experts in a mixture of %d blocks: dense and mixture blocks "+
				"in one model are not implemented: %w", b, cfg.NLayer, ErrNotImplemented)
		}
	}
	return nil
}

// mjRopeScaling states a rope scaling object, transformers' rope_parameters
// or rope_scaling, by its published method: none and default state nothing,
// and a method not implemented is refused by name.
func mjRopeScaling(cfg *Config, v any) error {
	if v == nil {
		return nil
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return fmt.Errorf("the rope scaling is %s, not an object", exprKind(v))
	}
	t := obj["rope_type"]
	if t == nil {
		t = obj["type"]
	}
	switch t {
	case nil, "default":
		return nil
	}
	return fmt.Errorf("rope scaling %v is not implemented: %w", t, ErrNotImplemented)
}

func exprUint(what string, v any) (uint64, error) {
	f, ok := v.(float64)
	if !ok || f < 0 || f != float64(uint64(f)) {
		return 0, fmt.Errorf("convert: %s is %v, not a count", what, v)
	}
	return uint64(f), nil
}

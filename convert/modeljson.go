package convert

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"maps"
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
	// Graph states the blocks' graph (blockgraph.go); absent, the engine
	// builds it from the config. GGUF names the GGUF architectures
	// (general.architecture) whose files the graph states too.
	Graph *mjGraph `json:"graph"`
	GGUF  []string `json:"gguf"`
	name  string   // for messages
}

type mjConfig struct {
	// Root is the object the paths are read from (text_config for a model
	// with a tower); $.key reads the file's top.
	Root string `json:"root"`
	// Defaults are the class's config defaults, each a key of the root a
	// config.json that lacks it takes, as the class's dataclass does; a key
	// present as null keeps its null, which an expression maps where the
	// class does (a ?? b).
	Defaults map[string]json.RawMessage `json:"defaults"`
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
	// Gating is the class's router by name (mjGatings): the gate it computes
	// whatever config.json says.
	Gating string `json:"gating"`
	// RotaryLayers states which layers rotate.
	RotaryLayers *mjRotaryLayers `json:"rotary_layers"`
}

// mjRotaryLayers is which layers rotate, as transformers states it for the
// classes with layers that skip the rotary (no_rope_layers): Each is a list
// a layer, 1 rotating and 0 not, and where it is absent every Every-th layer
// skips the rotary.
type mjRotaryLayers struct {
	Each  string `json:"each"`
	Every string `json:"every"`
}

// mjGatings is each router gate a file may state, as llama.cpp's
// expert_gating_func, which configOf reads.
func mjGatings() map[string]uint64 {
	return map[string]uint64{"softmax": 1, "sigmoid": 2}
}

type mjWindow struct {
	Size string `json:"size"`
	Used string `json:"used"`
	// Layers states which layers slide, deciding after configOf; absent, the
	// graph configOf builds decides.
	Layers *mjWindowLayers `json:"layers"`
}

// mjWindowLayers is which layers slide, as a class states it: List, a list
// of each layer's type, a layer sliding where its type is Slides; where the
// list is null, Period (every layer but each Period'th, (i+1) % Period != 0)
// or From (each layer from From on); with none of the three, every layer.
type mjWindowLayers struct {
	List   string `json:"list"`
	Slides string `json:"slides"`
	Period string `json:"period"`
	From   string `json:"from"`
}

// mjActivations is each feed-forward activation the container states, by its
// transformers name, as the activation flags it sets; another name is refused.
func mjActivations() map[string]jlm.Flags {
	return map[string]jlm.Flags{
		"silu":              0,
		"swish":             0,
		"gelu_pytorch_tanh": jlm.FlagGELU,
		"relu2":             jlm.FlagReLU2,
		"xielu":             0,
	}
}

// mjActivationParams is each activation whose parameters a block carries, by
// the role they are carried in: the activation is the graph's when every
// block has them (mjActivationCover).
func mjActivationParams() map[string]jlm.Role {
	return map[string]jlm.Role{"xielu": jlm.RoleXIELU}
}

// mjActivationFlags are the flags mjActivations decides between.
const mjActivationFlags = jlm.FlagGELU | jlm.FlagReLU2

// mjFlags is each graph feature a file may state by name, as the container
// flag the engine runs it by.
func mjFlags() map[string]jlm.Flags {
	return map[string]jlm.Flags{
		"attention_output_gate": jlm.FlagAttnOutGate,
		"non_causal":            jlm.FlagNonCausal,
		"qk_norm":               jlm.FlagQKNorm,
	}
}

// mjFlagsNotImplemented is each graph feature a file may state by name that
// the engine does not run, with what it is: stated true, it is refused.
func mjFlagsNotImplemented() map[string]string {
	return map[string]string{"post_norm": "each sublayer normed after it in place of before"}
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
	BlockPrefix  string               `json:"block_prefix"`
	Model        map[string]mjTargets `json:"model"`
	Block        map[string]mjTargets `json:"block"`
	Gather       []mjGather           `json:"gather"`
	Ignore       []string             `json:"ignore"`
	IgnorePrefix []string             `json:"ignore_prefix"`
	Experts      *mjExperts           `json:"experts"`
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

// mjTargets is what one source tensor becomes: a role, or a list of two or
// more, each cut from the source by its own primitives (a fused projection's
// row ranges), in the order they are laid out.
type mjTargets []mjTensor

func (ts *mjTargets) UnmarshalJSON(b []byte) error {
	if b = bytes.TrimSpace(b); len(b) == 0 || b[0] != '[' {
		var t mjTensor
		if err := json.Unmarshal(b, &t); err != nil {
			return err
		}
		*ts = mjTargets{t}
		return nil
	}
	var list []mjTensor
	if err := json.Unmarshal(b, &list); err != nil {
		return err
	}
	if len(list) < 2 {
		return fmt.Errorf("a list of targets names two or more, not %d", len(list))
	}
	*ts = list
	return nil
}

// mjGather builds one tensor a block from several sources, each entering as
// the vector of its values, by primitives that leave one tensor.
type mjGather struct {
	Role string      `json:"role"`
	From []mjOperand `json:"from"`
	Do   mjPipe      `json:"do"`
}

func parseModelJSON(b []byte, name string) (*modelJSON, error) {
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	mj := &modelJSON{name: name}
	if err := d.Decode(mj); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	// A file of no class states a GGUF architecture's graph alone.
	if len(mj.Classes) == 0 && len(mj.GGUF) != 0 && mj.Graph != nil {
		if mj.Config.Arch != "" || len(mj.Tensors.Block) != 0 || len(mj.Tensors.Model) != 0 {
			return nil, fmt.Errorf("convert: %s names no class and states one's config or tensors", name)
		}
		if err := mj.Graph.check(); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		return mj, nil
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
		if _, ok := mjFlags()[f]; !ok && mjFlagsNotImplemented()[f] == "" {
			return nil, fmt.Errorf("convert: %s: flag %q is not a graph feature the container states", name, f)
		}
	}
	for k, raw := range mj.Config.Defaults {
		var v any
		if strings.ContainsAny(k, ".$") || json.Unmarshal(raw, &v) != nil {
			return nil, fmt.Errorf("convert: %s: default %q is not a key of the root and a value", name, k)
		}
	}
	if r := mj.Config.RotaryLayers; r != nil && (r.Each == "" || r.Every == "") {
		return nil, fmt.Errorf("convert: %s: rotary_layers states each layer and the period where the list is absent", name)
	}
	if w := mj.Config.Window; w != nil && (w.Size == "" || w.Used == "") {
		return nil, fmt.Errorf("convert: %s: a window states its size and whether it is used", name)
	}
	if w := mj.Config.Window; w != nil && w.Layers != nil && w.Layers.List != "" && w.Layers.Slides == "" {
		return nil, fmt.Errorf("convert: %s: a window's layer list states the type that slides", name)
	}
	if g := mj.Config.Gating; g != "" {
		if _, ok := mjGatings()[g]; !ok {
			return nil, fmt.Errorf("convert: %s: gating %q is not a router gate the container states", name, g)
		}
	}
	if g := mj.Graph; g != nil {
		if err := g.check(); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if len(mj.GGUF) != 0 && mj.Graph == nil {
		return nil, fmt.Errorf("convert: %s names GGUF architectures and states no graph for them", name)
	}
	return mj, nil
}

// libraryFiles is every file of the embedded library.
func libraryFiles() ([]*modelJSON, error) {
	ents, err := fs.ReadDir(modelLibraryFS, "models")
	if err != nil {
		return nil, err
	}
	var out []*modelJSON
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
		out = append(out, mj)
	}
	return out, nil
}

// modelLibrary is the embedded library by class.
func modelLibrary() (map[string]*modelJSON, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	lib := map[string]*modelJSON{}
	for _, mj := range files {
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

// ggufGraph is the graph the library states for a GGUF of architecture
// arch, nil where no file states one.
func ggufGraph(arch string) (*mjGraph, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	var g *mjGraph
	var by string
	for _, mj := range files {
		if !slices.Contains(mj.GGUF, arch) {
			continue
		}
		if g != nil {
			return nil, fmt.Errorf("convert: GGUF architecture %s's graph is stated by both %s and %s", arch, by, mj.name)
		}
		g, by = mj.Graph, mj.name
	}
	return g, nil
}

// archGraph is the graph the library states for the GGUF architectures that
// convert as a, nil where no file claims one: a safetensors class of the Go
// table states what a GGUF of its architecture states, its container being
// that architecture's.
func archGraph(a Arch) (*mjGraph, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	archs := archOf()
	var g *mjGraph
	var by string
	for _, mj := range files {
		if !slices.ContainsFunc(mj.GGUF, func(n string) bool { return archs[n] == a }) {
			continue
		}
		if g != nil {
			return nil, fmt.Errorf("convert: architecture %v's graph is stated by both %s and %s", a, by, mj.name)
		}
		g, by = mj.Graph, mj.name
	}
	return g, nil
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
		graph:        mj.Graph,
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
	// A source of several targets is cut into each, read whole when it is
	// reached, the parts laid out in the file's order in its place.
	addAll := func(dst map[string]jlm.Role, name string, ts mjTargets) error {
		if len(ts) == 1 {
			return add(dst, name, ts[0])
		}
		roles := make([]jlm.Role, len(ts))
		for i, t := range ts {
			r, ok := roleOf(t.Role)
			if !ok {
				return fmt.Errorf("convert: %s: %s's role %q is not a tensor name the container knows", mj.name, name, t.Role)
			}
			if len(t.Do) == 0 {
				return fmt.Errorf("convert: %s: %s feeds %d roles, and %s takes no primitives to cut it by", mj.name, name, len(ts), t.Role)
			}
			if err := t.Do.check(); err != nil {
				return fmt.Errorf("convert: %s: %s: %s: %w", mj.name, name, t.Role, err)
			}
			roles[i] = retarget(a.arch, r)
		}
		dst[name] = roles[0]
		a.split[name] = func(e jlm.Tensor, data []byte, c *Config) ([]jlm.Tensor, error) {
			st.mu.Lock()
			env := st.env
			st.mu.Unlock()
			in := mjVal{shape: mjTorch(&e), typ: e.Type, data: data}
			out := make([]jlm.Tensor, len(ts))
			for i, t := range ts {
				v, err := t.Do.run(env, []mjVal{in})
				if err != nil {
					return nil, fmt.Errorf("convert: %s: %v: %w", e.Name, roles[i], err)
				}
				p := e
				p.Role, p.Name = roles[i], fmt.Sprintf("%s[%v]", e.Name, roles[i])
				if err := mjSetShape(&p, v); err != nil {
					return nil, err
				}
				p.Data = v.data
				out[i] = p
			}
			return out, nil
		}
		return nil
	}
	for n, ts := range mj.Tensors.Model {
		if err := addAll(a.model, n, ts); err != nil {
			return nil, err
		}
	}
	for n, ts := range mj.Tensors.Block {
		if err := addAll(a.block, n, ts); err != nil {
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
	var steps []hfGatherStep
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
			for _, o := range g.From {
				if err := o.Do.check(); err != nil {
					return nil, fmt.Errorf("convert: %s: the gather of %s: %w", mj.name, g.Role, err)
				}
			}
			gs[i] = hfGatherStep{role: retarget(a.arch, r), from: g.From, do: g.Do}
		}
		steps = gs
	}
	a.gather = func(c *Config, names []string, read func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
		st.mu.Lock()
		env, act := st.env, st.act
		st.mu.Unlock()
		out, used, err := gatherBlocks(env, a.blockPrefix, steps, names, read)
		if err != nil {
			return nil, nil, err
		}
		if err := mjActivationCover(act, c, out); err != nil {
			return nil, nil, err
		}
		return out, used, nil
	}
	a.config = func(c *hfConfig, names []string) (*Config, error) {
		env, err := mj.env(c)
		if err != nil {
			return nil, err
		}
		cfg, err := mj.configIn(env, c, names)
		if err != nil {
			return nil, err
		}
		n, err := parseExpr(mj.Config.Activation)
		if err != nil {
			return nil, err
		}
		av, err := evalMJ(n, env)
		if err != nil {
			return nil, err
		}
		act, _ := av.(string)
		st.mu.Lock()
		st.env, st.act = env, act
		st.mu.Unlock()
		return cfg, nil
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
	if len(mj.Config.Defaults) > 0 {
		root = maps.Clone(root)
		for k, raw := range mj.Config.Defaults {
			if _, ok := root[k]; ok {
				continue
			}
			var v any
			if err := json.Unmarshal(raw, &v); err != nil {
				return nil, fmt.Errorf("convert: %s: default %s: %w", mj.name, k, err)
			}
			root[k] = v
		}
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

// mjWindowSlides is which of n layers slide, as l states it.
func mjWindowSlides(l *mjWindowLayers, eval func(what, src string) (any, error), n int) ([]bool, error) {
	slides := make([]bool, n)
	if l.List != "" {
		v, err := eval("window layers", l.List)
		if err != nil {
			return nil, err
		}
		if v != nil {
			list, ok := v.([]any)
			if !ok || len(list) != n {
				return nil, fmt.Errorf("the window's layers are %v, not a list of %d", v, n)
			}
			for i, t := range list {
				s, ok := t.(string)
				if !ok {
					return nil, fmt.Errorf("layer %d's type is %v, not a name", i, t)
				}
				slides[i] = s == l.Slides
			}
			return slides, nil
		}
	}
	for i, src := range []string{l.Period, l.From} {
		if src == "" {
			continue
		}
		v, err := eval("window layers", src)
		if err != nil {
			return nil, err
		}
		if v == nil {
			continue
		}
		k, err := exprUint("window layers", v)
		if err != nil {
			return nil, err
		}
		if i == 0 && k == 0 {
			return nil, fmt.Errorf("a window period of 0")
		}
		for li := range slides {
			if i == 0 {
				slides[li] = uint64(li+1)%k != 0
			} else {
				slides[li] = uint64(li) >= k
			}
		}
		return slides, nil
	}
	for i := range slides {
		slides[i] = true
	}
	return slides, nil
}

// windowPeriod is the container's statement of which layers slide: the
// period whose last layer (or, with first, whose first) attends to every
// position, a period past the last layer where every layer slides, and 0
// where none does. A pattern no period describes is refused.
func windowPeriod(slides []bool) (uint32, bool, error) {
	all, none := true, true
	for _, s := range slides {
		all, none = all && s, none && !s
	}
	switch {
	case none:
		return 0, false, nil
	case all:
		return allLocal(uint32(len(slides))), false, nil
	}
	for _, first := range []bool{false, true} {
		for p := 2; p <= len(slides); p++ {
			ok := true
			for i, s := range slides {
				want := i%p < p-1
				if first {
					want = i%p != 0
				}
				if s != want {
					ok = false
					break
				}
			}
			if ok {
				return uint32(p), first, nil
			}
		}
	}
	return 0, false, fmt.Errorf("the sliding layers %v follow no period the container states: %w", slides, ErrNotImplemented)
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
		// use_bidirectional_attention drops the causal mask where a class
		// honours it everywhere; a class whose attention paths disagree on it
		// (Gemma: eager causal, SDPA bidirectional) leaves it unread.
		"use_bidirectional_attention": false,
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
	// configOf reads facts off a GGUF's tensor table (which blocks route, a
	// shared expert's width), so the checkpoint's block tensors stand in, each
	// under the name the file gives its role, its shape in the container's
	// order.
	have := map[string]bool{}
	for _, t := range ts {
		have[t.Name] = true
	}
	more, err := mj.blockStandIns(renv, c.shapes, names, have)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", c.path, err)
	}
	ts = append(ts, more...)
	// A prediction block the file states and the checkpoint does not ship is
	// not counted: a checkpoint saved without them converts as its trunk.
	if nv, ok := kvs[ns+"nextn_predict_layers"]; ok {
		n, _ := nv.Uint()
		b, _ := kvs[ns+"block_count"].Uint()
		if n > 0 && n <= b && !have["blk."+itoa(int(b-n))+".nextn.eh_proj.weight"] {
			kvs[ns+"nextn_predict_layers"] = meta.MakeUint(0)
			kvs[ns+"block_count"] = meta.MakeUint(b - n)
		}
	}
	if g := mj.Config.Gating; g != "" {
		kvs[ns+"expert_gating_func"] = meta.MakeUint(mjGatings()[g])
		// The class computes its gate whatever the file says, so a
		// scoring_func naming another describes weights it never ran.
		sf, err := eval("scoring_func", "scoring_func")
		if err != nil {
			return nil, err
		}
		if sf != nil && sf != g {
			return nil, fmt.Errorf("convert: %s: scoring_func %v, and %s states the class's %s router: %w",
				c.path, sf, mj.name, g, ErrNotImplemented)
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
		}
		kvs[ns+"attention.sliding_window"] = meta.MakeUint(window)
	}
	cfg, err := configOf(meta.New(c.path, kvs, ts))
	if err != nil {
		return nil, err
	}
	// A stated shared-expert width holds the checkpoint to it: a graph that
	// takes the width off the tensor would run whatever ships, where the class
	// builds the width its config states.
	if v, ok := kvs[ns+"expert_shared_feed_forward_length"]; ok {
		ships := false
		for n := range have {
			ships = ships || strings.HasSuffix(n, ".ffn_gate_shexp.weight") || strings.HasSuffix(n, ".ffn_up_shexp.weight")
		}
		got := uint64(cfg.NFFNShExp)
		if !ships {
			got = 0
		}
		if w, _ := v.Uint(); got != w {
			return nil, fmt.Errorf("convert: %s: the shared experts are %d wide in the checkpoint, and the config states %d",
				c.path, got, w)
		}
	}
	// The statements configOf does not read decide after it.
	if w := mj.Config.Window; w != nil && w.Layers != nil {
		slides, err := mjWindowSlides(w.Layers, eval, int(cfg.NLayer))
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %w", c.path, err)
		}
		cfg.SWAWindow, cfg.Flags2 = 0, cfg.Flags2&^jlm.Flag2SWAGlobalFirst
		if window != 0 {
			p, first, err := windowPeriod(slides)
			if err != nil {
				return nil, fmt.Errorf("convert: %s: %w", c.path, err)
			}
			if p == 0 && cfg.Flags.Has(jlm.FlagNoPEGlobal) {
				return nil, fmt.Errorf("convert: %s: a window of %d slides no layer, and the global layers skip "+
					"the rotary: no layer rotates: %w", c.path, window, ErrNotImplemented)
			}
			if p != 0 {
				cfg.SWAWindow, cfg.SWAPeriod = uint32(window), p
				if first {
					cfg.Flags2 |= jlm.Flag2SWAGlobalFirst
				}
			}
		}
	} else if window != 0 && uint64(cfg.SWAWindow) != window {
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
		if why := mjFlagsNotImplemented()[f]; why != "" {
			if b {
				return nil, fmt.Errorf("convert: %s: %s is true (%s), which is not implemented: %w",
					c.path, f, why, ErrNotImplemented)
			}
			continue
		}
		if b {
			cfg.Flags |= mjFlags()[f]
		} else {
			cfg.Flags &^= mjFlags()[f]
		}
	}
	if r := mj.Config.RotaryLayers; r != nil {
		if err := mjRotaryLayersOf(cfg, r, eval); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", c.path, err)
		}
	}
	if mj.Config.RopeScaling != "" {
		v, err := eval("rope_scaling", mj.Config.RopeScaling)
		if err != nil {
			return nil, err
		}
		if err := mjRopeScaling(cfg, c, v, renv); err != nil {
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

// mixtureCovers holds a mixture's blocks to its dense lead and step: a block
// carries experts, and a router where the checkpoint ships routers, exactly
// where the keys place a mixture, so a pattern the two cannot state is refused
// rather than read as one they can. The prediction blocks the checkpoint
// ships are held to it too, and a dense block carries its feed-forward and
// no shared experts.
func (mj *modelJSON) mixtureCovers(cfg *Config, names []string) error {
	if mj.Tensors.Experts == nil || cfg.NExpert == 0 {
		return nil
	}
	experts, routers, dense, shared := map[int]bool{}, map[int]bool{}, map[int]bool{}, map[int]bool{}
	for _, n := range names {
		role, blk, expert, ok := mj.blockRole(n)
		if !ok {
			continue
		}
		switch r, _ := roleOf(role); {
		case expert:
			experts[blk] = true
		case r == jlm.RoleRouter:
			routers[blk] = true
		case r == jlm.RoleFFNGate || r == jlm.RoleFFNUp || r == jlm.RoleFFNDown:
			dense[blk] = true
		case strings.HasSuffix(role, "_shexp.weight"):
			shared[blk] = true
		}
	}
	// A file that names a dense feed-forward holds every dense block to it.
	named := false
	for _, ms := range mj.Tensors.Block {
		for _, m := range ms {
			if r, _ := roleOf(m.Role); r == jlm.RoleFFNGate || r == jlm.RoleFFNUp || r == jlm.RoleFFNDown {
				named = true
			}
		}
	}
	if len(experts) == 0 {
		return nil
	}
	step := max(cfg.MoEStep, 1)
	for b := range int(cfg.NLayer + cfg.NMTP) {
		want := uint32(b) >= cfg.NDenseLead && uint32(b+1)%step == 0
		var what string
		switch {
		case !want && named && !dense[b]:
			what = "carries no feed-forward"
		case !want && shared[b]:
			what = "carries shared experts"
		case experts[b] && !want:
			what = "carries experts"
		case !experts[b] && want:
			what = "carries no experts"
		case len(routers) > 0 && routers[b] && !want:
			what = "carries a router"
		case len(routers) > 0 && !routers[b] && want:
			what = "carries no router"
		default:
			continue
		}
		place := "a dense block"
		if want {
			place = "a mixture"
		}
		return fmt.Errorf("block %d %s, where a dense lead of %d and a step of %d place %s: %w",
			b, what, cfg.NDenseLead, step, place, ErrNotImplemented)
	}
	return nil
}

// blockRole is a block tensor's role, by the GGUF name the file gives it,
// its block, and whether it is an expert's matrix, which has its bank's.
// blockStandIns is each block tensor configOf may read, under the GGUF name
// its role gives it and in the container's order: a source feeding several
// roles stands in for each, at the shape its primitives cut it to.
func (mj *modelJSON) blockStandIns(env exprEnv, shapes map[string][]uint64, names []string, have map[string]bool) ([]meta.Tensor, error) {
	var ts []meta.Tensor
	for _, n := range names {
		targets, blk, ok := mj.blockTargets(n)
		if !ok {
			continue
		}
		for _, t := range targets {
			gn := "blk." + itoa(blk) + "." + t.Role
			if have[gn] {
				continue
			}
			have[gn] = true
			shape := shapes[n]
			if len(t.Do) > 0 && shape != nil {
				v, err := t.Do.run(env, []mjVal{{shape: shape, typ: jlm.TypeF32}})
				if err != nil {
					return nil, fmt.Errorf("%s as %s: %w", n, t.Role, err)
				}
				shape = v.shape
			}
			dims := make([]uint64, len(shape))
			for i, d := range shape {
				dims[len(shape)-1-i] = d
			}
			ts = append(ts, meta.Tensor{Name: gn, Dims: dims})
		}
	}
	return ts, nil
}

// blockTargets is the roles a block's source feeds and its block: the
// file's targets, or an expert's one role.
func (mj *modelJSON) blockTargets(name string) (mjTargets, int, bool) {
	t := mj.Tensors
	rest, ok := strings.CutPrefix(name, t.BlockPrefix)
	if t.BlockPrefix == "" || !ok {
		return nil, 0, false
	}
	b, r, err := leadingIndex(name, rest)
	if err != nil {
		return nil, 0, false
	}
	if m, ok := t.Block[r]; ok && len(m) > 0 {
		return m, int(b), true
	}
	if role, blk, expert, ok := mj.blockRole(name); ok && expert {
		return mjTargets{{Role: role}}, blk, true
	}
	return nil, 0, false
}

func (mj *modelJSON) blockRole(name string) (role string, blk int, expert, ok bool) {
	t := mj.Tensors
	rest, ok := strings.CutPrefix(name, t.BlockPrefix)
	if t.BlockPrefix == "" || !ok {
		return "", 0, false, false
	}
	b, r, err := leadingIndex(name, rest)
	if err != nil {
		return "", 0, false, false
	}
	if m, ok := t.Block[r]; ok && len(m) > 0 {
		return m[0].Role, int(b), false, true
	}
	if e := t.Experts; e != nil {
		if er, ok := strings.CutPrefix(r, e.Prefix); ok {
			if _, en, err := leadingIndex(name, er); err == nil && e.Names[en] != "" {
				return e.Names[en], int(b), true, true
			}
		}
	}
	return "", 0, false, false
}

// mjRotaryLayersOf states which layers skip the rotary. The container states
// them as a period, every period-th layer skipping it (Config.SWA with no
// window, FlagNoPEGlobal), so a list no period describes is refused, as is one
// where no layer rotates or a window beside it.
func mjRotaryLayersOf(cfg *Config, r *mjRotaryLayers, eval func(what, src string) (any, error)) error {
	n := int(cfg.NLayer)
	v, err := eval("rotary_layers each", r.Each)
	if err != nil {
		return err
	}
	var rope []bool
	switch list := v.(type) {
	case nil:
		ev, err := eval("rotary_layers every", r.Every)
		if err != nil {
			return err
		}
		every, err := exprUint("rotary_layers every", ev)
		if err != nil {
			return err
		}
		if every == 0 {
			return fmt.Errorf("every 0th layer skipping the rotary")
		}
		for i := 0; i < n; i++ {
			rope = append(rope, uint64(i+1)%every != 0)
		}
	case []any:
		if len(list) < n {
			return fmt.Errorf("rotary_layers has %d entries for %d layers", len(list), n)
		}
		for i, x := range list[:n] {
			switch x {
			case 0.0, 1.0:
				rope = append(rope, x == 1.0)
			default:
				return fmt.Errorf("rotary_layers[%d] is %v, not 1 (rotates) or 0", i, x)
			}
		}
	default:
		return fmt.Errorf("rotary_layers is %s, not a list", exprKind(v))
	}
	period := 0
	for i, r := range rope {
		if !r {
			period = i + 1
			break
		}
	}
	cfg.Flags &^= jlm.FlagNoPEGlobal
	if period == 0 {
		if cfg.SWAWindow == 0 {
			cfg.SWAPeriod = 0
		}
		return nil
	}
	if period == 1 {
		return fmt.Errorf("no layer rotates (rotary layers %v): %w", rope, ErrNotImplemented)
	}
	for i, r := range rope {
		if r != ((i+1)%period != 0) {
			return fmt.Errorf("rotary layers %v are not every %d-th layer skipping the rotary (layer %d): %w",
				rope, period, i, ErrNotImplemented)
		}
	}
	if cfg.SWAWindow != 0 {
		return fmt.Errorf("layers skipping the rotary beside a sliding window of %d: %w", cfg.SWAWindow, ErrNotImplemented)
	}
	cfg.SWAPeriod = uint32(period)
	cfg.Flags |= jlm.FlagNoPEGlobal
	return nil
}

// mjRopeScaling states a rope scaling object, transformers' rope_parameters
// or rope_scaling, by its published method: none and default state nothing;
// llama3 states its per-pair factors over the rotated width as
// rope_freqs.weight, reading what transformers reads from env (llama3RopeOf);
// yarn states its factor, original context, correction range and magnitude
// as transformers computes them (hfYarnApply); a method not implemented is
// refused by name.
func mjRopeScaling(cfg *Config, c *hfConfig, v any, env exprEnv) error {
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
	case "llama3":
		raw, err := json.Marshal(obj)
		if err != nil {
			return err
		}
		num := func(k string) *float64 {
			if f, ok := env(k).(float64); ok {
				return &f
			}
			return nil
		}
		freqs, err := llama3RopeOf(raw, num("original_max_position_embeddings"), num("max_position_embeddings"))
		if err != nil {
			return fmt.Errorf("rope scaling llama3: %w", err)
		}
		rot := cfg.NRot
		if rot == 0 {
			rot = cfg.HeadDim
		}
		c.extra = append(c.extra, ropeFreqsTensor(freqs(rot, float64(cfg.RopeBase))))
		return nil
	case "yarn":
		raw, err := json.Marshal(obj)
		if err != nil {
			return err
		}
		var y hfYarn
		if err := json.Unmarshal(raw, &y); err != nil {
			return fmt.Errorf("rope scaling yarn: %w", err)
		}
		if !(y.Factor >= 1) {
			return fmt.Errorf("rope scaling yarn: a factor of %g", y.Factor)
		}
		// standardize_rope_params: a top-level original context first, then the
		// object's, then max_position_embeddings (hfYarnApply's NCtx).
		if f, ok := env("original_max_position_embeddings").(float64); ok && f > 0 {
			y.OrigCtx = uint32(f)
		}
		hfYarnApply(&y, cfg)
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

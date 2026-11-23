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
	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
)

// A model.json states what the converter reads from a model's files: which
// config keys mean what, which tensor names hold which roles, and how a tensor
// is reshaped on the way in, as a composition of the tensor primitives
// (mjtransforms.go). It is data: the Go is the primitives and the generic
// readers of its statements, and knows no model. What the converter cannot
// express it refuses in code, by name, one rule for every file.
// .notes/design/model-json.md is the design.
//
// One is found, in order of authority: the caller's (WithModelJSON), the
// model directory's own model.json, the library embedded here
// (convert/models).

//go:embed models/*.json models/*.jinja
var modelLibraryFS embed.FS

type modelJSON struct {
	// Describe is the line docs/models.md gives each class and GGUF
	// architecture the file names: which well-known models it covers.
	Describe map[string]string `json:"describe"`
	Classes  []string          `json:"classes"`
	// Template is the class's chat template, a file of the library
	// (convert/models) where the release ships none a conversion finds.
	Template string    `json:"template"`
	Config   mjConfig  `json:"config"`
	Tensors  mjTensors `json:"tensors"`
	// Graph states the blocks' graph (blockgraph.go); absent, the engine
	// builds it from the config. GGUF names the GGUF architectures
	// (general.architecture) whose files the graph states too.
	Graph *mjGraph `json:"graph"`
	GGUF  []string `json:"gguf"`
	// Container is what the container states for each GGUF architecture
	// the file names (mjStates).
	Container map[string]*mjStates `json:"container"`
	name      string               // for messages
	names     tensorNames          // the tensor names its roles are read in
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
	// YarnLogMultiplier is the coefficient of ln(factor) the attention's
	// softmax scale takes where a yarn scaling applies, as a GGUF's
	// rope.scaling.yarn_log_multiplier: absent for none.
	YarnLogMultiplier string `json:"yarn_log_multiplier"`
	// RopeMethods lists the scaling methods the class's attention is gated
	// against; absent, every method the reader implements is accepted.
	RopeMethods []string `json:"rope_methods"`
	// Window is a sliding window: its size, and whether it is in use.
	Window *mjWindow `json:"window"`
	// Flags states named graph features (mjFlags), each a boolean.
	Flags map[string]string `json:"flags"`
	// Gating is the class's router by name (mjGatings): the gate it computes
	// whatever config.json says.
	Gating string `json:"gating"`
	// RotaryLayers states which layers rotate.
	RotaryLayers *mjRotaryLayers `json:"rotary_layers"`
	// Unroll states a loop: a list of the source layer each block runs, a
	// layer named again a later pass reading the first pass's weights with
	// a history of its own (unrollBlocks); null is every layer once.
	Unroll string `json:"unroll"`
	// Require holds the config to what the class's reference computes and
	// the container does not vary.
	Require []mjRequire `json:"require"`
}

// mjRequire is a statement about the config the file depends on: That is a
// boolean expression, and a config it is false for is refused, with Else
// saying why.
type mjRequire struct {
	That string `json:"that"`
	Else string `json:"else"`
}

// mjRotaryLayers is which layers rotate, as transformers states it for the
// classes with layers that skip the rotary (no_rope_layers): Each is a list
// a layer, 1 rotating and 0 not, and where it is absent every Every-th layer
// skips the rotary.
type mjRotaryLayers struct {
	Each  string `json:"each"`
	Every string `json:"every"`
}

// mjGatings is each router gate a file may state, by the code a GGUF's
// expert_gating_func key stores, which configOf reads.
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
// of each layer's type, a layer sliding where its type is Slides; with Full
// stated, a type that is neither is refused, as no layer of the class has it;
// where the list is null, Period (every layer but each Period'th, (i+1) % Period != 0)
// or From (each layer from From on); with none of the three, every layer.
type mjWindowLayers struct {
	List   string `json:"list"`
	Slides string `json:"slides"`
	Full   string `json:"full"`
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
		// The situ bounds are keys the architecture's statement reads.
		"situ": 0,
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
		"qk_norm_wide":          jlm.FlagQKNormWide,
		"qk_norm_post_rope":     jlm.FlagQKNormPostRope,
		"rope_neox":             jlm.FlagRopeNeox,
		"no_expert_norm":        jlm.FlagNoExpertNorm,
		"layer_norm":            jlm.FlagLayerNorm,
		"parallel":              jlm.FlagParallel,
		"swiglu_oai":            jlm.FlagSwiGLUOAI,
		"yarn_exact":            jlm.FlagYarnExact,
		"no_pe_global":          jlm.FlagNoPEGlobal,
		"no_pos_enc":            jlm.FlagNoPosEnc,
		"delta_key_tiled":       jlm.FlagDeltaKeyTiled,
		"swa_chunked":           jlm.FlagSWAChunked,
		"qk_l2_norm":            jlm.FlagQKL2Norm,
		"expert_weight_in":      jlm.FlagExpertWeightIn,
	}
}

// mjFlags2 is each graph feature a file may state by name, as the container's
// second flag word states it.
func mjFlags2() map[string]jlm.Flags2 {
	return map[string]jlm.Flags2{
		"swiglu_clamp":  jlm.Flag2SwiGLUClamp,
		"rope_tail":     jlm.Flag2RopeTail,
		"key_value":     jlm.Flag2KeyValue,
		"q_unit_norm":   jlm.Flag2QUnitNorm,
		"out_unrotated": jlm.Flag2OutUnrotated,
		"v_norm":        jlm.Flag2VNorm,
		"dense_moe":     jlm.Flag2DenseMoE,
	}
}

// mjFlagsNotImplemented is each graph feature a file may state by name that
// the engine does not run, with what it is: stated true, it is refused.
func mjFlagsNotImplemented() map[string]string {
	return map[string]string{
		"post_norm": "each sublayer normed after it in place of before",
	}
}

// mjKey is one GGUF key's value, an expression with the key's type.
type mjKey struct {
	Uint   string `json:"uint"`
	Float  string `json:"float"`
	Int32s string `json:"int32s"`
	Bool   string `json:"bool"`
	// If, where set, is a boolean expression: the key is written only where
	// it holds, as a reference that writes the key only when it has one.
	If string `json:"if"`
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
	BlockPrefix string               `json:"block_prefix"`
	Model       map[string]mjTargets `json:"model"`
	Block       map[string]mjTargets `json:"block"`
	// BlockLinear is what a recurrent block's (Config.LayerKinds) names
	// are, where they are not Block's; Fuse states each role several sources
	// are concatenated into, by its name, the sources in row order.
	BlockLinear map[string]mjTargets `json:"block_linear"`
	Fuse        map[string][]string  `json:"fuse"`
	// Joined, a GGUF table's, is the primitives a joined role takes once its
	// sources are joined.
	Joined       map[string]mjPipe `json:"joined"`
	Gather       []mjGather        `json:"gather"`
	Ignore       []string          `json:"ignore"`
	IgnorePrefix []string          `json:"ignore_prefix"`
	Experts      *mjExperts        `json:"experts"`
	Ternary      *mjTernary        `json:"ternary"`
	// Present states a block tensor the config says every trunk block has or
	// lacks: Tensor, relative to the block, is shipped exactly where If holds.
	// Drop, a GGUF table's, drops a block tensor where If, over the GGUF's
	// keys and block (the block's index), holds.
	Present []mjPresent `json:"present"`
	Drop    []mjPresent `json:"drop"`
	// Keys, a GGUF table's, are the GGUF keys its gather reads as a block's
	// sources: a key's value for every block, or its block's entry of a list.
	// Sizes, a GGUF table's, are values its targets read as size.<name>, each
	// an expression over the GGUF's keys, block and packed.
	Keys  []string          `json:"keys"`
	Sizes map[string]string `json:"sizes"`
}

type mjPresent struct {
	Tensor string `json:"tensor"`
	If     string `json:"if"`
}

// mjTernary names the projections a checkpoint stores as ternary weights, by
// the roles' GGUF tensor names, and how: Scale is the suffix of the tensor
// beside a packed weight that holds its scale; Latent and Reciprocal are
// boolean expressions over the config, true for weights stored as latent
// floats the converter ternarizes, and for a stored scale that is the
// reciprocal of the weights' magnitude. They are parsed with the file.
type mjTernary struct {
	Roles      []string `json:"roles"`
	Scale      string   `json:"scale"`
	Latent     string   `json:"latent"`
	Reciprocal string   `json:"reciprocal"`

	latent, reciprocal *exprNode
}

// form evaluates the statements over the config that env reads.
func (t *mjTernary) form(file string, env exprEnv) (ternaryForm, error) {
	var f ternaryForm
	for _, s := range []struct {
		what string
		n    *exprNode
		into *bool
	}{{"latent", t.latent, &f.latent}, {"reciprocal", t.reciprocal, &f.reciprocal}} {
		v, err := evalMJ(s.n, env)
		if err != nil {
			return f, fmt.Errorf("convert: %s: ternary %s: %w", file, s.what, err)
		}
		b, ok := v.(bool)
		if !ok {
			return f, fmt.Errorf("convert: %s: ternary %s is %s, not a boolean", file, s.what, exprKind(v))
		}
		*s.into = b
	}
	return f, nil
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
	// If, a GGUF table's, is an expression over the GGUF's keys: the tensor
	// is cut where it holds and kept as it is where it does not.
	If string `json:"if"`
	// Block, Each and Where place a model-level source's target in a block,
	// a class's model table's: Block is the block it belongs to, Each a
	// count of targets cut from the source (i 0 first), Where a condition the
	// target is cut under, each an expression over the config and i, i being
	// also the index a name's {i} matches.
	Block string `json:"block"`
	Each  string `json:"each"`
	Where string `json:"where"`
}

// placed reports whether a target is placed in a block.
func (t mjTensor) placed() bool { return t.Block != "" || t.Each != "" || t.Where != "" }

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

// transforms reports whether a source takes primitives of its own.
func (ts mjTargets) transforms() bool {
	for _, t := range ts {
		if len(t.Do) > 0 {
			return true
		}
	}
	return false
}

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
// the vector of its values, by primitives that leave one tensor. If, where
// set, is a boolean expression over the config: the gather runs only where
// it holds. Every has each block carry the sources; Model gathers one tensor
// of the model from sources named whole. Name is the result's name, {block}
// its block's prefix; Count, where set, the values each source holds.
// Precision "f64" computes in float64 from sources read exactly, until a type
// step rounds the result once; its primitives read shape, the first source's
// shape slowest axis first, where the source's reader knows it.
type mjGather struct {
	Role      string      `json:"role"`
	From      []mjOperand `json:"from"`
	Do        mjPipe      `json:"do"`
	If        string      `json:"if"`
	Every     bool        `json:"every"`
	Model     bool        `json:"model"`
	Name      string      `json:"name"`
	Count     string      `json:"count"`
	Precision string      `json:"precision"`
}

// wide reports whether g computes in float64, refusing a precision that is
// neither.
func (g mjGather) wide() (bool, error) {
	if g.Precision != "" && g.Precision != "f64" {
		return false, fmt.Errorf("the gather of %s computes in %q, not f64", g.Role, g.Precision)
	}
	return g.Precision == "f64", nil
}

func parseModelJSON(b []byte, name string, names tensorNames) (*modelJSON, error) {
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	mj := &modelJSON{name: name, names: names}
	if err := d.Decode(mj); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := mj.checkStates(); err != nil {
		return nil, err
	}
	for n := range mj.Describe {
		if !slices.Contains(mj.Classes, n) && !slices.Contains(mj.GGUF, n) {
			return nil, fmt.Errorf("convert: %s describes %s, which it does not name", name, n)
		}
	}
	// A file of no class states a GGUF architecture's graph alone, or a
	// graph other files' blocks use (mjUse).
	if len(mj.Classes) == 0 && mj.Graph != nil {
		if mj.Config.Arch != "" || len(mj.Tensors.Block) != 0 || len(mj.Tensors.Model) != 0 {
			return nil, fmt.Errorf("convert: %s names no class and states one's config or tensors", name)
		}
		if err := mj.Graph.check(names); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
		return mj, nil
	}
	if len(mj.Classes) == 0 {
		return nil, fmt.Errorf("convert: %s names no class", name)
	}
	if mj.Config.Vocab == "" || mj.Config.Head.Tensor == "" {
		return nil, fmt.Errorf("convert: %s states no vocab or no head tensor", name)
	}
	if mj.Config.Activation == "" && (mj.Graph == nil || len(mj.Graph.Use) != 0 || readsActivation(mj.Graph.Block)) {
		return nil, fmt.Errorf("convert: %s states no activation, and its graph has an act op to read one", name)
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
	if t := mj.Tensors.Ternary; t != nil {
		if len(t.Roles) == 0 || t.Scale == "" || t.Latent == "" || t.Reciprocal == "" {
			return nil, fmt.Errorf("convert: %s: ternary states its roles, its scale's suffix, and when the weights are latent and the scale is a reciprocal", name)
		}
		var err error
		if t.latent, err = parseExpr(t.Latent); err != nil {
			return nil, fmt.Errorf("convert: %s: ternary latent: %w", name, err)
		}
		if t.reciprocal, err = parseExpr(t.Reciprocal); err != nil {
			return nil, fmt.Errorf("convert: %s: ternary reciprocal: %w", name, err)
		}
	}
	for i, r := range mj.Config.Require {
		if r.Else == "" {
			return nil, fmt.Errorf("convert: %s: require %d states no refusal", name, i)
		}
		if _, err := parseExpr(r.That); err != nil {
			return nil, fmt.Errorf("convert: %s: require %d: %w", name, i, err)
		}
	}
	for _, p := range mj.Tensors.Present {
		if _, err := parseExpr(p.If); err != nil || p.Tensor == "" {
			return nil, fmt.Errorf("convert: %s: a presence states a block tensor and the expression that ships it", name)
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
		if err := g.check(names); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}
	if len(mj.GGUF) != 0 && mj.Graph == nil {
		return nil, fmt.Errorf("convert: %s names architectures and states no graph for them", name)
	}
	return mj, nil
}

// libraryFiles is every file of the embedded library.
func libraryFiles() ([]*modelJSON, error) {
	ents, err := fs.ReadDir(modelLibraryFS, "models")
	if err != nil {
		return nil, err
	}
	names, err := readTensorNames()
	if err != nil {
		return nil, err
	}
	var out []*modelJSON
	for _, e := range ents {
		// A class's template is a file of the library beside the statements.
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		p := path.Join("models", e.Name())
		b, err := modelLibraryFS.ReadFile(p)
		if err != nil {
			return nil, err
		}
		mj, err := parseModelJSON(b, "convert/"+p, names)
		if err != nil {
			return nil, err
		}
		out = append(out, mj)
	}
	if err := libraryUses(out); err != nil {
		return nil, err
	}
	if err := libraryContainers(out); err != nil {
		return nil, err
	}
	return out, nil
}

// libraryContainers refuses a container of a GGUF architecture its file does
// not name, a class's own container (one of a name no file's GGUF list has)
// that no class names, and a class whose graph is neither a GGUF
// architecture nor a container the library states.
func libraryContainers(files []*modelJSON) error {
	gguf, named, stated := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, mj := range files {
		for _, n := range mj.GGUF {
			gguf[n] = true
		}
		if len(mj.Classes) > 0 {
			named[mj.Config.Arch] = true
		}
		for arch := range mj.Container {
			stated[arch] = true
		}
	}
	for _, mj := range files {
		for _, arch := range slices.Sorted(maps.Keys(mj.Container)) {
			switch {
			case gguf[arch] && !slices.Contains(mj.GGUF, arch):
				return fmt.Errorf("convert: %s states the container of %s, a GGUF architecture it does not name", mj.name, arch)
			case !gguf[arch] && !named[arch]:
				return fmt.Errorf("convert: %s states the container of %s, which no GGUF architecture and no class names", mj.name, arch)
			}
		}
		if len(mj.Classes) > 0 && !gguf[mj.Config.Arch] && !stated[mj.Config.Arch] {
			return fmt.Errorf("convert: %s: graph %q is no GGUF architecture and no container the library states", mj.name, mj.Config.Arch)
		}
	}
	return nil
}

// libraryUses refuses a use naming no graph of files, or one that uses
// others in turn, and a graph of no class and no GGUF architecture that no
// file's use names: it would state no model's blocks.
func libraryUses(files []*modelJSON) error {
	used := map[string]bool{}
	for _, mj := range files {
		if mj.Graph == nil {
			continue
		}
		for _, u := range mj.Graph.Use {
			n := "convert/models/" + u.Graph + ".json"
			i := slices.IndexFunc(files, func(f *modelJSON) bool { return f.name == n })
			switch {
			case i < 0 || files[i].Graph == nil:
				return fmt.Errorf("convert: %s uses %q, which no library file states", mj.name, u.Graph)
			case len(files[i].Graph.Use) != 0:
				return fmt.Errorf("convert: %s uses %q, which uses others in turn", mj.name, u.Graph)
			}
			used[n] = true
		}
	}
	for _, mj := range files {
		if len(mj.Classes) == 0 && len(mj.GGUF) == 0 && !used[mj.name] {
			return fmt.Errorf("convert: %s names no class and no GGUF architecture, and no file uses its graph", mj.name)
		}
	}
	return nil
}

// modelLibrary is the embedded library by class.
func modelLibrary() (map[string]*modelJSON, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	return libraryOf(files)
}

// libraryOf is modelLibrary over the library's files.
func libraryOf(files []*modelJSON) (map[string]*modelJSON, error) {
	lib := map[string]*modelJSON{}
	for _, mj := range files {
		for _, c := range mj.Classes {
			if prev, dup := lib[c]; dup {
				return nil, fmt.Errorf("convert: class %s is in both %s and %s", c, prev.name, mj.name)
			}
			lib[c] = mj
		}
	}
	return lib, nil
}

// ggufGraph is the graph the library states for a GGUF of architecture
// arch, or for a class's own container of that name, nil where no file
// states one.
func ggufGraph(arch string) (*mjGraph, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	var g *mjGraph
	var by string
	for _, mj := range files {
		if _, own := mj.Container[arch]; !own && !slices.Contains(mj.GGUF, arch) {
			continue
		}
		if g != nil {
			return nil, fmt.Errorf("convert: GGUF architecture %s's graph is stated by both %s and %s", arch, by, mj.name)
		}
		g, by = mj.Graph, mj.name
	}
	return g, nil
}

// classGraph is the graph the library states for a safetensors class: the
// one for the GGUF architecture name it converts as (hfArch.name).
func classGraph(ha *hfArch) (*mjGraph, error) {
	return ggufGraph(ha.name)
}

// WithModelJSON converts a safetensors model by the model.json at path,
// ahead of the model's own and the library's.
func WithModelJSON(path string) Option {
	return func(o *options) { o.modelJSON = path }
}

// hfArchFor is the entry a model converts by: the caller's model.json, the
// directory's, or the library's.
func hfArchFor(dir hfDir, c *hfConfig, caller string) (*hfArch, error) {
	var mj *modelJSON
	names, err := readTensorNames()
	if err != nil {
		return nil, err
	}
	switch {
	case caller != "":
		b, err := os.ReadFile(caller)
		if err != nil {
			return nil, fmt.Errorf("convert: the model.json asked for: %w", err)
		}
		if mj, err = parseModelJSON(b, caller, names); err != nil {
			return nil, err
		}
	case dir.has("model.json"):
		b, err := dir.read("model.json")
		if err != nil {
			return nil, err
		}
		if mj, err = parseModelJSON(b, dir.path("model.json"), names); err != nil {
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
			return nil, classRefusal(c)
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
	template := ""
	if mj.Template != "" {
		b, err := modelLibraryFS.ReadFile("models/" + mj.Template)
		if err != nil {
			return nil, fmt.Errorf("convert: %s: its template %s is no file of the library: %w", mj.name, mj.Template, err)
		}
		template = string(b)
	}
	a := &hfArch{
		name:         mj.Config.Arch,
		template:     template,
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
	moved, err := retargets(mj.Config.Arch, nil)
	if err != nil {
		return nil, err
	}
	retarget := func(r jlm.Role) jlm.Role {
		if t, ok := moved[r]; ok {
			return t
		}
		return r
	}
	if e := mj.Tensors.Experts; e != nil {
		a.expertPrefix, a.expert = e.Prefix, map[string]jlm.Role{}
		for n, r := range e.Names {
			role, ok := mj.names.roleOf(r)
			if !ok {
				return nil, fmt.Errorf("convert: %s: expert matrix %s's bank %q is not a tensor name the container knows", mj.name, n, r)
			}
			a.expert[n] = role
		}
	}
	if len(mj.Tensors.Drop)+len(mj.Tensors.Keys)+len(mj.Tensors.Sizes)+len(mj.Tensors.Joined) > 0 {
		return nil, fmt.Errorf("convert: %s: a class's table states no drop, keys, sizes or joined, which are a GGUF table's", mj.name)
	}
	st := &mjState{src: map[string]mjVal{}}
	pipes := map[string]mjPipe{}
	add := func(dst map[string]jlm.Role, name string, t mjTensor) error {
		if t.If != "" {
			return fmt.Errorf("convert: %s: %s: a class's target states no if, which is a GGUF table's", mj.name, name)
		}
		r, ok := mj.names.roleOf(t.Role)
		if !ok {
			return fmt.Errorf("convert: %s: %s's role %q is not a tensor name the container knows", mj.name, name, t.Role)
		}
		dst[name] = retarget(r)
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
			r, ok := mj.names.roleOf(t.Role)
			if !ok {
				return fmt.Errorf("convert: %s: %s's role %q is not a tensor name the container knows", mj.name, name, t.Role)
			}
			if len(t.Do) == 0 {
				return fmt.Errorf("convert: %s: %s feeds %d roles, and %s takes no primitives to cut it by", mj.name, name, len(ts), t.Role)
			}
			if err := t.Do.check(); err != nil {
				return fmt.Errorf("convert: %s: %s: %s: %w", mj.name, name, t.Role, err)
			}
			roles[i] = retarget(r)
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
		if strings.Contains(n, "{i}") || slices.ContainsFunc(ts, mjTensor.placed) {
			if err := mj.addPlaced(a, st, n, ts, retarget); err != nil {
				return nil, err
			}
			continue
		}
		if err := addAll(a.model, n, ts); err != nil {
			return nil, err
		}
	}
	for n, ts := range mj.Tensors.Block {
		if slices.ContainsFunc(ts, mjTensor.placed) {
			return nil, fmt.Errorf("convert: %s: block tensor %s is placed in a block, which a model tensor is", mj.name, n)
		}
		if err := addAll(a.block, n, ts); err != nil {
			return nil, err
		}
	}
	if len(mj.Tensors.BlockLinear) > 0 {
		a.blockLinear = map[string]jlm.Role{}
		// One name's primitives serve every block it is read in: a name with
		// its own in both tables would take whichever was stated last.
		blockPipes := maps.Clone(pipes)
		for n, ts := range mj.Tensors.BlockLinear {
			if _, ok := blockPipes[n]; ok && ts.transforms() {
				return nil, fmt.Errorf("convert: %s: %s takes primitives of its own in both the block and the recurrent block", mj.name, n)
			}
			if err := addAll(a.blockLinear, n, ts); err != nil {
				return nil, err
			}
		}
	}
	for r, srcs := range mj.Tensors.Fuse {
		role, ok := mj.names.roleOf(r)
		if !ok {
			return nil, fmt.Errorf("convert: %s: fused role %q is not a tensor name the container knows", mj.name, r)
		}
		if len(srcs) < 2 {
			return nil, fmt.Errorf("convert: %s: %s is fused from %d source(s), not two or more", mj.name, r, len(srcs))
		}
		if a.fuseOrder == nil {
			a.fuseOrder = map[jlm.Role][]string{}
		}
		a.fuseOrder[retarget(role)] = srcs
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
			r, ok := mj.names.roleOf(g.Role)
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
			if g.Model && g.Every {
				return nil, fmt.Errorf("convert: %s: the gather of %s is the model's and every block's", mj.name, g.Role)
			}
			if g.From[0].Fill != nil && !g.Model && !g.Every {
				return nil, fmt.Errorf("convert: %s: the gather of %s finds its blocks by its first source, and that is a "+
					"filled operand", mj.name, g.Role)
			}
			step := hfGatherStep{role: retarget(r), from: g.From, do: g.Do, every: g.Every, model: g.Model, name: g.Name}
			if step.f64, err = g.wide(); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", mj.name, err)
			}
			for _, x := range []struct {
				src string
				to  **exprNode
			}{{g.If, &step.cond}, {g.Count, &step.count}} {
				if x.src == "" {
					continue
				}
				n, err := parseExpr(x.src)
				if err != nil {
					return nil, fmt.Errorf("convert: %s: the gather of %s: %w", mj.name, g.Role, err)
				}
				*x.to = n
			}
			gs[i] = step
		}
		steps = gs
	}
	tern := mj.Tensors.Ternary
	ternaryRoles := map[jlm.Role]bool{}
	if tern != nil {
		for _, n := range tern.Roles {
			r, ok := mj.names.roleOf(n)
			if !ok {
				return nil, fmt.Errorf("convert: %s: ternary role %q is not a tensor name the container knows", mj.name, n)
			}
			ternaryRoles[retarget(r)] = true
		}
	}
	a.gather = func(c *Config, names []string, read func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
		st.mu.Lock()
		env, act := st.env, st.act
		st.mu.Unlock()
		out, used, err := gatherBlocks(env, a.blockPrefix, steps, names, read, nil, int(c.NLayer))
		if err != nil {
			return nil, nil, err
		}
		if err := mjActivationCover(act, c, out); err != nil {
			return nil, nil, err
		}
		if tern != nil {
			scales, err := ternaryGather(tern.Scale, names)
			if err != nil {
				return nil, nil, err
			}
			maps.Copy(used, scales)
		}
		return out, used, nil
	}
	if tern != nil {
		a.retype = func(_ *hfConfig, role jlm.Role, name string, t *safetensors.Tensor,
			read func(string) ([]byte, *safetensors.Tensor, error)) (*hfRetyped, error) {
			if !ternaryRoles[role] {
				return nil, nil
			}
			st.mu.Lock()
			form := st.ternary
			st.mu.Unlock()
			return ternaryRetype(form, tern.Scale, name, t, read)
		}
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
		var act string
		if mj.Config.Activation != "" {
			n, err := parseExpr(mj.Config.Activation)
			if err != nil {
				return nil, err
			}
			av, err := evalMJ(n, env)
			if err != nil {
				return nil, err
			}
			act, _ = av.(string)
		}
		var form ternaryForm
		if tern != nil {
			if form, err = tern.form(mj.name, env); err != nil {
				return nil, err
			}
		}
		st.mu.Lock()
		st.env, st.act, st.ternary = env, act, form
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

// absentKeyError is a key whose expression reads what the source does not
// state, with no default in its file.
type absentKeyError struct{ error }

// keyValue is the GGUF key k as the file states it (Config.Keys), its
// expression evaluated by eval over config.json at path.
func (mj *modelJSON) keyValue(path, k string, eval func(what, src string) (any, error)) (meta.Value, error) {
	spec := mj.Config.Keys[k]
	var src string
	n := 0
	for _, s := range []string{spec.Uint, spec.Float, spec.Int32s, spec.Bool} {
		if s != "" {
			src, n = s, n+1
		}
	}
	if n != 1 {
		return meta.Value{}, fmt.Errorf("convert: %s: key %s states %d types; one is the rule", mj.name, k, n)
	}
	v, err := eval(k, src)
	if err != nil {
		return meta.Value{}, err
	}
	if v == nil {
		// An absent key takes the CLASS's default, which the model.json
		// writes with ??; the reader's own default is another class's.
		return meta.Value{}, absentKeyError{fmt.Errorf("convert: %s: %s reads %q, which is absent, and %s states no default",
			path, k, src, mj.name)}
	}
	switch {
	case spec.Uint != "":
		u, err := exprUint(k, v)
		if err != nil {
			return meta.Value{}, err
		}
		return meta.MakeUint(u), nil
	case spec.Bool != "":
		b, ok := v.(bool)
		if !ok {
			return meta.Value{}, fmt.Errorf("convert: %s: %s is %s, not a boolean", path, k, exprKind(v))
		}
		return meta.MakeBool(b), nil
	case spec.Float != "":
		f, ok := v.(float64)
		if !ok {
			return meta.Value{}, fmt.Errorf("convert: %s: %s is %s, not a number", path, k, exprKind(v))
		}
		return meta.MakeFloat(float32(f)), nil
	}
	list, ok := v.([]any)
	if !ok {
		return meta.Value{}, fmt.Errorf("convert: %s: %s is %s, not a list", path, k, exprKind(v))
	}
	out := make([]int32, len(list))
	for i, x := range list {
		u, err := exprUint(fmt.Sprintf("%s[%d]", k, i), x)
		if err != nil || u > 1<<31-1 {
			return meta.Value{}, fmt.Errorf("convert: %s: %s[%d] is not a 32-bit count", path, k, i)
		}
		out[i] = int32(u)
	}
	return meta.MakeInt32s(out), nil
}

// windowSize is the sliding window the file states (Config.Window) over
// config.json at path, 0 where none is in use.
func (mj *modelJSON) windowSize(path string, eval func(what, src string) (any, error)) (uint64, error) {
	w := mj.Config.Window
	used, err := eval("window used", w.Used)
	if err != nil {
		return 0, err
	}
	if used != true && used != false {
		return 0, fmt.Errorf("convert: %s: window used is %s, not a boolean", path, exprKind(used))
	}
	size, err := eval("window size", w.Size)
	if err != nil || used != true || size == nil {
		return 0, err
	}
	return exprUint("window size", size)
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
				if l.Full != "" && s != l.Slides && s != l.Full {
					return nil, fmt.Errorf("layer %d's type is %q, not %q or %q: %w", i, s, l.Slides, l.Full, ErrNotImplemented)
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
		// A rope dict carries it too, and the default and YaRN rotaries read
		// it from there: one per layer type, or the legacy rope_scaling.
		"rope_scaling.partial_rotary_factor":                      1.0,
		"rope_parameters.full_attention.partial_rotary_factor":    1.0,
		"rope_parameters.sliding_attention.partial_rotary_factor": 1.0,
		"sliding_window": nil,
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
		if cond := mj.Config.Keys[k].If; cond != "" {
			v, err := eval(k+" if", cond)
			if err != nil {
				return nil, err
			}
			b, ok := v.(bool)
			if !ok {
				return nil, fmt.Errorf("convert: %s: %s: key %s's if is %s, not a boolean", c.path, mj.name, k, exprKind(v))
			}
			if !b {
				continue
			}
		}
		v, err := mj.keyValue(c.path, k, eval)
		if err != nil {
			return nil, err
		}
		kvs[ns+k] = v
	}

	for i, r := range mj.Config.Require {
		v, err := eval(fmt.Sprintf("require %d", i), r.That)
		if err != nil {
			return nil, err
		}
		ok, isBool := v.(bool)
		if !isBool {
			return nil, fmt.Errorf("convert: %s: %s: require %d is %s, not a boolean", c.path, mj.name, i, exprKind(v))
		}
		if !ok {
			return nil, fmt.Errorf("convert: %s: %s: %s: %w", c.path, mj.name, r.Else, ErrNotImplemented)
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
	if more, err = mj.modelStandIns(renv, c.shapes, names, have); err != nil {
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
	if mj.Config.Window != nil {
		var err error
		if window, err = mj.windowSize(c.path, eval); err != nil {
			return nil, err
		}
		kvs[ns+"attention.sliding_window"] = meta.MakeUint(window)
	}
	mapped := meta.New(c.path, kvs, ts)
	cfg, err := classConfigOf(mapped)
	if err != nil {
		return nil, err
	}
	cfg.keys = ggufKeys(mapped)
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
				// A pattern no period describes is each block's own window.
				p, first = 0, false
				cfg.Windows = make([]uint32, len(slides))
				for i, sl := range slides {
					if sl {
						cfg.Windows[i] = uint32(window)
					}
				}
				cfg.SWAWindow, cfg.SWAPeriod = uint32(window), 0
			}
			if p == 0 && cfg.Windows == nil && cfg.Flags.Has(jlm.FlagNoPEGlobal) {
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
	var af jlm.Flags
	if mj.Config.Activation != "" {
		av, err := eval("activation", mj.Config.Activation)
		if err != nil {
			return nil, err
		}
		act, _ := av.(string)
		var ok bool
		if af, ok = mjActivations()[act]; !ok {
			return nil, fmt.Errorf("convert: %s: activation %v is not implemented: %w", c.path, av, ErrNotImplemented)
		}
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
	if mj.Config.Unroll != "" {
		v, err := eval("unroll", mj.Config.Unroll)
		if err != nil {
			return nil, err
		}
		if cfg.Unroll, err = exprInts("unroll", v); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", c.path, err)
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
		if err := mjRopeScaling(cfg, c, v, mj.Config.RopeMethods, renv); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", c.path, err)
		}
		if mj.Config.YarnLogMultiplier != "" && cfg.YarnFactor > 1 {
			m, err := eval("yarn_log_multiplier", mj.Config.YarnLogMultiplier)
			if err != nil {
				return nil, err
			}
			f, ok := m.(float64)
			if !ok {
				return nil, fmt.Errorf("convert: %s: yarn_log_multiplier is %s, not a number", c.path, exprKind(m))
			}
			cfg.YarnLogMul = float32(f)
		}
	}
	if err := mj.mixtureCovers(cfg, names); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", c.path, err)
	}
	if err := mj.presence(cfg, names, eval); err != nil {
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

// presence holds each trunk block to the tensors the config says it has: a
// tensor a presence states is shipped by every block where its expression is
// true and by none where it is false, so a config and weights that disagree
// are refused rather than read as one the graph can state.
func (mj *modelJSON) presence(cfg *Config, names []string, eval func(what, src string) (any, error)) error {
	for _, p := range mj.Tensors.Present {
		v, err := eval("presence of "+p.Tensor, p.If)
		if err != nil {
			return err
		}
		want, ok := v.(bool)
		if !ok {
			return fmt.Errorf("presence of %s is %s, not a boolean", p.Tensor, exprKind(v))
		}
		for b := range int(cfg.NLayer) {
			has := slices.Contains(names, mj.Tensors.BlockPrefix+itoa(b)+"."+p.Tensor)
			switch {
			case has && !want:
				return fmt.Errorf("block %d carries %s, and %q is false: nothing would read it: %w", b, p.Tensor, p.If, ErrNotImplemented)
			case !has && want:
				return fmt.Errorf("block %d carries no %s, and %q is true: %w", b, p.Tensor, p.If, ErrNotImplemented)
			}
		}
	}
	return nil
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
		switch r, _ := mj.names.roleOf(role); {
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
			if r, _ := mj.names.roleOf(m.Role); r == jlm.RoleFFNGate || r == jlm.RoleFFNUp || r == jlm.RoleFFNDown {
				named = true
			}
		}
	}
	if len(experts) == 0 {
		return nil
	}
	step := max(cfg.MoEStep, 1)
	for b := range int(cfg.NLayer + cfg.NPredict) {
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
// modelStandIns are the checkpoint's model-level tensors the file maps, each
// under its role's name, its shape the one its steps leave, in the container's
// order; the embedding and the head stand in already (have).
func (mj *modelJSON) modelStandIns(env exprEnv, shapes map[string][]uint64, names []string, have map[string]bool) ([]meta.Tensor, error) {
	var ts []meta.Tensor
	for _, n := range names {
		shape := shapes[n]
		if shape == nil {
			continue
		}
		for _, t := range mj.Tensors.Model[n] {
			if have[t.Role] {
				continue
			}
			have[t.Role] = true
			sh := shape
			if len(t.Do) > 0 {
				v, err := t.Do.run(env, []mjVal{{shape: shape, typ: jlm.TypeF32}})
				if err != nil {
					return nil, fmt.Errorf("%s as %s: %w", n, t.Role, err)
				}
				sh = v.shape
			}
			dims := make([]uint64, len(sh))
			for i, d := range sh {
				dims[len(sh)-1-i] = d
			}
			ts = append(ts, meta.Tensor{Name: t.Role, Dims: dims})
		}
	}
	return ts, nil
}

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
// or rope_scaling, by its method's statement in rotary.json (rotaryScaling,
// rotaryMethod.state): none and default state nothing; a method the file's
// rope_methods does not list, or one rotary.json states no reading of, is
// refused by name.
func mjRopeScaling(cfg *Config, c *hfConfig, v any, methods []string, env exprEnv) error {
	if v == nil {
		return nil
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return fmt.Errorf("the rope scaling is %s, not an object", exprKind(v))
	}
	r, err := rotaryScaling(cfg, obj, env, methods)
	if err != nil || r == nil {
		return err
	}
	if ok, err := r.holds(cfg); err != nil || !ok {
		return err
	}
	ts, err := r.settle(cfg)
	if err != nil {
		return err
	}
	div, err := r.divisors(cfg)
	if err != nil {
		return err
	}
	if div != nil {
		ts = append(ts, ropeFreqsTensor(div))
		if err := r.clear(cfg); err != nil {
			return err
		}
	}
	c.extra = append(c.extra, ts...)
	return nil
}

func exprUint(what string, v any) (uint64, error) {
	f, ok := v.(float64)
	if !ok || f < 0 || f != float64(uint64(f)) {
		return 0, fmt.Errorf("convert: %s is %v, not a count", what, v)
	}
	return uint64(f), nil
}

// readsActivation reports whether a block's ops, an each's steps among them,
// hold an act op: the one op that reads the activation a file names.
func readsActivation(ops []mjBlockOp) bool {
	for _, o := range ops {
		if o.Op == "act" || readsActivation(o.Steps) {
			return true
		}
	}
	return false
}

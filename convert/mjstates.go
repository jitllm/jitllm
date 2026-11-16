package convert

import (
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
	"github.com/jitllm/jitllm/tok/pretok"
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
	// LogitDiv divides the logits and ClampKQV bounds q, k and v (0 for
	// none).
	LogitDiv string `json:"logit_div"`
	ClampKQV string `json:"clamp_kqv"`
	// RopeBase is the rotary base of the layers no window states its own for.
	RopeBase string `json:"rope_base"`
	// Latent is latent attention's two ranks, and Indexer the indexer that
	// selects the positions attention reads.
	Latent  *mjStatesLatent  `json:"latent"`
	Indexer *mjStatesIndexer `json:"indexer"`
	// Recurrent is the recurrent layers' geometry (jlm.SSMConfig), each a
	// count.
	Recurrent *mjStatesRecurrent `json:"recurrent"`
	// RopeSections is the M-RoPE split of a head's rotary pairs between a
	// position's axes, a list of counts (empty for none).
	RopeSections string `json:"rope_sections"`
	// Layers is each layer's kind, read off the tensors it carries
	// (stateLayers).
	Layers []mjStatesLayer `json:"layers"`
	// ResidualScale is the factor each block's attention and FFN output is
	// scaled by (0 for none), and MoEStep the mixture blocks' interleave
	// (jlm.Config.MoEStep), each a number.
	ResidualScale string `json:"residual_scale"`
	MoEStep       string `json:"moe_step"`
	// AttnTemp scales the queries by position (jlm.Config.AttnTempScale).
	AttnTemp *mjStatesTemp `json:"attn_temp"`
	// NoRopeEvery is n where every n-th layer, (il+1) % n == 0, skips the
	// rotary, a count (0 for none).
	NoRopeEvery string `json:"no_rope_every"`
	// LearnedPositions states that the input rows add a learned position
	// table, and FFNUngated that the FFN has no gate, each true or false;
	// a file's tensors are held to them (statedTensorsAgree).
	LearnedPositions string `json:"learned_positions"`
	FFNUngated       string `json:"ffn_ungated"`
	// WholeRotary states that the class turns every dimension of a head.
	WholeRotary bool `json:"whole_rotary"`
	// Router is what the class's router does whatever a file's keys say.
	Router *mjRouter `json:"router"`
	// Streams is the residual streams a block mixes, OutputGroups the
	// attention output's grouped low-rank projection, and HashLayers how many
	// lead blocks route by token id.
	Streams      *mjStatesStreams      `json:"streams"`
	OutputGroups *mjStatesOutputGroups `json:"output_groups"`
	HashLayers   string                `json:"hash_layers"`
	// AttnResBlock is how many blocks a residual-attention block spans,
	// ExpertLatent the width a mixture's routed experts run at (the model's
	// width states none), and Situ the situ activation's bounds.
	AttnResBlock string        `json:"attn_res_block"`
	ExpertLatent string        `json:"expert_latent"`
	Situ         *mjStatesSitu `json:"situ"`
	// Revision is what StateRevision states once the blocks' geometry is
	// read.
	Revision *mjStatesRevision `json:"revision"`
	// PreTokenizer is the pre-tokenizer, by name, of a file that names none
	// of its own ("default"), and RopeLayout how rotary sections lie over a
	// head (mjRopeLayouts; one section after another where unstated).
	PreTokenizer string `json:"pre_tokenizer"`
	RopeLayout   string `json:"rope_layout"`
	exprs        map[string]*jinja.Expr
	tensors      bool
}

// mjStatesLatent is latent attention's ranks: KV the key-value latent's, Q
// the query latent's (0 where the query is projected in one step), each a
// count; a file whose keys and values are not a latent states Q alone.
type mjStatesLatent struct {
	KV string `json:"kv"`
	Q  string `json:"q"`
}

// mjStatesIndexer is an indexer of Heads query heads scoring one cached key
// of KeyWidth per position, the TopK best read, each a count; all three 0
// state none.
type mjStatesIndexer struct {
	Heads    string `json:"heads"`
	KeyWidth string `json:"key_width"`
	TopK     string `json:"top_k"`
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

// mjStatesTemp is the queries' temperature by position, 1 +
// Scale*ln(1 + floor((pos+Offset)/Floor)): each a number, Offset 0 where
// unstated, and a Scale of 0 states none.
type mjStatesTemp struct {
	Scale  string `json:"scale"`
	Floor  string `json:"floor"`
	Offset string `json:"offset"`
}

// mjRouter is a mixture's router as its class computes it: Gate the
// function its scores go through (mjGates; sparsemixer ranks the raw logits
// and weighs each pick by sparsemixer at the jitter PhiMoEJitter, RouterScore
// and RouterWeight), Renormalise "always" or
// "never" for the chosen weights, Unscaled for no routed scale, Selects the
// number of experts it picks (0 for the file's).
type mjRouter struct {
	Gate        string `json:"gate"`
	Renormalise string `json:"renormalise"`
	Unscaled    bool   `json:"unscaled"`
	Selects     uint32 `json:"selects"`
}

// mjGates is each gate a router states by name, with the expert_gating_func
// value that names it (-1 for none: the class reads no gating key).
func mjGates() map[string]int64 {
	return map[string]int64{"softmax": 1, "sigmoid": 2, "sparsemixer": -1}
}

// mjStatesStreams is Count residual streams, each block's mix normalised by
// Iterations Sinkhorn rounds at Eps where both are stated.
type mjStatesStreams struct {
	Count      string `json:"count"`
	Iterations string `json:"sinkhorn_iterations"`
	Eps        string `json:"eps"`
}

// mjStatesOutputGroups is the attention output projected through Groups
// groups of heads, each to a Rank-row latent, each a count.
type mjStatesOutputGroups struct {
	Groups string `json:"groups"`
	Rank   string `json:"rank"`
}

// mjStatesSitu is the situ activation's bounds, Beta the gate's and
// LinearBeta the up projection's, each b in b*tanh(x/b).
type mjStatesSitu struct {
	Beta       string `json:"beta"`
	LinearBeta string `json:"linear_beta"`
}

// mjStatesRevision is what StateRevision states, each read off the keys
// where the statement applies: ActQuant how every projection's input is
// quantized (mjActQuants), RopeScaleSWA the sliding layers' rotary
// magnitude (a number, 0 for the model's), SWAGlobalFirst that a windowed
// model's period starts at its global layer, and PictureBidirSliding that a
// windowed model's sliding layers are the ones a picture's rows see each
// other on (each true or false).
type mjStatesRevision struct {
	ActQuant            string `json:"act_quant"`
	RopeScaleSWA        string `json:"rope_scale_swa"`
	SWAGlobalFirst      string `json:"swa_global_first"`
	PictureBidirSliding string `json:"picture_bidir_sliding"`
}

// mjNamedActivations is each activation a container statement may name that
// no flag carries, which the activations are stated by (Activation): GELU
// with erf.
func mjNamedActivations() []string { return []string{"gelu"} }

// mjRopeLayouts is each rotary layout a file may state by name
// (stateRopeTable): interleaved pair by pair, or over the elements.
func mjRopeLayouts() []string { return []string{"interleaved", "xd"} }

// mjActQuants is each activation quantization a file may state by name.
func mjActQuants() map[string]jlm.ActQuant {
	return map[string]jlm.ActQuant{"row_int8": jlm.ActQuantRowInt8}
}

// Statement is what the library states for the stages after the config's:
// StateRevision's, read off the keys, the activation by its name and the
// router's gate by its.
type Statement struct {
	actQuant                            jlm.ActQuant
	ropeScaleSWA                        float32
	swaGlobalFirst, pictureBidirSliding bool
	activation, ropeLayout              string
	routerGate                          string
	learnedPositions, ffnUngated        bool
}

// mjStatesWindow is a sliding window: Size positions (0 for none, and then
// nothing else of it applies, but a Period over none is a malformed file); every Period'th layer attends globally, or
// Pattern's where the file's attention.sliding_window_pattern does not say
// (swaPatternOf), or every layer from From on slides; RopeBase the windowed
// layers' rotary base.
type mjStatesWindow struct {
	Size     string `json:"size"`
	Period   string `json:"period"`
	Pattern  string `json:"pattern"`
	From     string `json:"from"`
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
		srcs := []string{st.NormEps, st.EmbdScale, st.AttnSoftcap, st.FinalSoftcap, st.LogitDiv,
			st.RopeBase, st.ResidualScale, st.MoEStep, st.RopeSections, st.ClampKQV, st.NoRopeEvery,
			st.LearnedPositions, st.FFNUngated}
		if tp := st.AttnTemp; tp != nil {
			if tp.Scale == "" || tp.Floor == "" {
				return fmt.Errorf("convert: %s: %s: a temperature states its scale and its floor", mj.name, arch)
			}
			srcs = append(srcs, tp.Scale, tp.Floor, tp.Offset)
		}
		if r := st.Router; r != nil {
			if _, ok := mjGates()[r.Gate]; r.Gate != "" && !ok {
				return fmt.Errorf("convert: %s: %s: router gate %q is not one the container states", mj.name, arch, r.Gate)
			}
			if r.Renormalise != "" && r.Renormalise != "always" && r.Renormalise != "never" {
				return fmt.Errorf("convert: %s: %s: router renormalise %q, not always or never", mj.name, arch, r.Renormalise)
			}
		}
		if w := st.Window; w != nil {
			if n := min(len(w.Period), 1) + min(len(w.Pattern), 1) + min(len(w.From), 1); w.Size == "" || n > 1 {
				return fmt.Errorf("convert: %s: %s: a window states its size, and one of a period, a pattern or a first layer (not both of any two)", mj.name, arch)
			}
			srcs = append(srcs, w.Size, w.Period, w.Pattern, w.From, w.RopeBase)
		}
		if l := st.Latent; l != nil {
			if l.Q == "" {
				return fmt.Errorf("convert: %s: %s: a latent states its query rank, q", mj.name, arch)
			}
			srcs = append(srcs, l.KV, l.Q)
		}
		if x := st.Indexer; x != nil {
			if x.Heads == "" || x.KeyWidth == "" || x.TopK == "" {
				return fmt.Errorf("convert: %s: %s: an indexer states its heads, key_width and top_k", mj.name, arch)
			}
			srcs = append(srcs, x.Heads, x.KeyWidth, x.TopK)
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
		if x := st.Streams; x != nil {
			if x.Count == "" || (x.Iterations == "") != (x.Eps == "") {
				return fmt.Errorf("convert: %s: %s: streams state their count, and sinkhorn_iterations with eps", mj.name, arch)
			}
			srcs = append(srcs, x.Count, x.Iterations, x.Eps)
		}
		if x := st.OutputGroups; x != nil {
			if x.Groups == "" || x.Rank == "" {
				return fmt.Errorf("convert: %s: %s: output groups state their groups and rank", mj.name, arch)
			}
			srcs = append(srcs, x.Groups, x.Rank)
		}
		if x := st.Situ; x != nil {
			if x.Beta == "" || x.LinearBeta == "" {
				return fmt.Errorf("convert: %s: %s: situ states its beta and linear_beta", mj.name, arch)
			}
			srcs = append(srcs, x.Beta, x.LinearBeta)
		}
		srcs = append(srcs, st.HashLayers, st.AttnResBlock, st.ExpertLatent)
		if r := st.Revision; r != nil {
			if _, ok := mjActQuants()[r.ActQuant]; r.ActQuant != "" && !ok {
				return fmt.Errorf("convert: %s: %s: activation quantization %q is not one the container states", mj.name, arch, r.ActQuant)
			}
			srcs = append(srcs, r.RopeScaleSWA, r.SWAGlobalFirst, r.PictureBidirSliding)
		}
		if _, ok := pretok.PreOps(st.PreTokenizer); st.PreTokenizer != "" && !ok {
			return fmt.Errorf("convert: %s: %s: pre-tokenizer %q is not one the tokenizer has", mj.name, arch, st.PreTokenizer)
		}
		if st.RopeLayout != "" && !slices.Contains(mjRopeLayouts(), st.RopeLayout) {
			return fmt.Errorf("convert: %s: %s: rotary layout %q is not one the container states", mj.name, arch, st.RopeLayout)
		}
		for _, k := range slices.Sorted(maps.Keys(st.Flags)) {
			_, ok := mjFlags()[k]
			if _, no := mjFlagsNotImplemented()[k]; !ok && !no {
				return fmt.Errorf("convert: %s: %s: flag %q is no graph feature", mj.name, arch, k)
			}
			srcs = append(srcs, st.Flags[k])
		}
		if _, ok := mjActivations()[st.Activation]; st.Activation != "" && !ok && !slices.Contains(mjNamedActivations(), st.Activation) {
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
			switch n {
			case "src", "layer_tensors":
			case "tensors":
				st.tensors = true
			default:
				return fmt.Errorf("%q reads %q, which is not the keys (src), the layers' tensors (layer_tensors) or the tensors' names (tensors)", src, n)
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
	c.Stated = &Statement{}
	if err != nil {
		return err
	}
	if st != nil {
		if err := st.apply(by, f, c); err != nil {
			return err
		}
	}
	return statedTensorsAgree(f, c)
}

// statedTensorsAgree refuses a file storing a router over no expert_count,
// a gate on an FFN the library states ungated, and one storing no position
// table where the library states the input rows add one.
func statedTensorsAgree(f *meta.File, c *Config) error {
	if _, ok := f.Get("position_embd.weight"); !ok && c.Stated.learnedPositions {
		return fmt.Errorf("convert: %s: no position_embd.weight, which the graph adds to the input rows", f.Arch())
	}
	for i := range f.Tensors {
		if n := f.Tensors[i].Name; c.Stated.ffnUngated && strings.HasPrefix(n, "blk.") && strings.HasSuffix(n, ".ffn_gate.weight") {
			return fmt.Errorf("convert: %s: a block stores %s, and its FFN is ungated: %w", f.Arch(), n, ErrNotImplemented)
		}
	}
	if f.UintKey("expert_count", 0) != 0 {
		return nil
	}
	for i := range c.NLayer {
		if _, ok := f.Get("blk." + itoa(int(i)) + ".ffn_gate_inp.weight"); ok {
			return fmt.Errorf("convert: %s: block %d stores a router, and the file states no expert_count", f.Arch(), i)
		}
	}
	return nil
}

// apply sets what st states, read off f's keys, in c; by names the file.
func (st *mjStates) apply(by string, f *meta.File, c *Config) error {
	arch := f.Arch()
	var err error
	vars := map[string]jinja.Value{"src": ggufKeys(f), "layer_tensors": layerTensors(f)}
	if st.tensors {
		names := jinja.NewDict()
		for i := range f.Tensors {
			names.AsDict().Set(f.Tensors[i].Name, jinja.NewBool(true))
		}
		vars["tensors"] = names
	}
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
	boolean := func(what, src string) (bool, error) {
		v, err := value(what, src)
		if err == nil && !v.IsBool() {
			err = fmt.Errorf("convert: %s: %s: %s is %v, not true or false", by, arch, what, v)
		}
		return err == nil && v.AsBool(), err
	}
	c.Stated = &Statement{activation: st.Activation, ropeLayout: st.RopeLayout}
	for _, b := range []struct {
		what, src string
		to        *bool
	}{{"learned_positions", st.LearnedPositions, &c.Stated.learnedPositions}, {"ffn_ungated", st.FFNUngated, &c.Stated.ffnUngated}} {
		if b.src == "" {
			continue
		}
		if *b.to, err = boolean(b.what, b.src); err != nil {
			return err
		}
	}
	if r := st.Router; r != nil {
		c.Stated.routerGate = r.Gate
	}
	if r := st.Revision; r != nil {
		c.Stated.actQuant = mjActQuants()[r.ActQuant]
		if r.RopeScaleSWA != "" {
			if c.Stated.ropeScaleSWA, err = number("revision.rope_scale_swa", r.RopeScaleSWA); err != nil {
				return err
			}
		}
		for _, b := range []struct {
			what, src string
			to        *bool
		}{{"revision.swa_global_first", r.SWAGlobalFirst, &c.Stated.swaGlobalFirst},
			{"revision.picture_bidir_sliding", r.PictureBidirSliding, &c.Stated.pictureBidirSliding}} {
			if b.src == "" {
				continue
			}
			if *b.to, err = boolean(b.what, b.src); err != nil {
				return err
			}
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
	if f, ok := mjActivations()[st.Activation]; ok {
		c.Flags = c.Flags&^mjActivationFlags | f
	}
	if st.NormEps != "" {
		if c.RMSEps, err = number("norm_eps", st.NormEps); err != nil {
			return err
		}
	}
	for _, s := range []struct {
		what, src string
		to        *float32
	}{{"embd_scale", st.EmbdScale, &c.EmbdScale}, {"attn_softcap", st.AttnSoftcap, &c.AttnSoftcap},
		{"final_softcap", st.FinalSoftcap, &c.FinalSoftcap}, {"logit_div", st.LogitDiv, &c.LogitScale},
		{"clamp_kqv", st.ClampKQV, &c.ClampKQV}} {
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
	for _, s := range []struct {
		what, src string
		to        *float32
	}{{"residual_scale", st.ResidualScale, &c.ResidualScale}} {
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
	if tp := st.AttnTemp; tp != nil {
		if err := stateAttnTemp(by, arch, tp, number, c); err != nil {
			return err
		}
	}
	if st.MoEStep != "" {
		x, err := number("moe_step", st.MoEStep)
		if err != nil {
			return err
		}
		if x < 1 || x != float32(uint32(x)) {
			return fmt.Errorf("convert: %s: %s: moe_step is %g, not a step", by, arch, x)
		}
		c.MoEStep = uint32(x)
	}
	if err := st.Router.apply(f, c); err != nil {
		return fmt.Errorf("convert: %s: %w", arch, err)
	}
	if st.WholeRotary && c.NRot != c.HeadDim {
		return fmt.Errorf("convert: %s: a rotary over %d of a head's %d dimensions, where the reference turns "+
			"the whole head (%s): %w", arch, c.NRot, c.HeadDim, by, ErrNotImplemented)
	}
	if st.RopeBase != "" {
		if c.RopeBase, err = number("rope_base", st.RopeBase); err != nil {
			return err
		}
		if x := float64(c.RopeBase); !(x > 0) || math.IsInf(x, 0) {
			return fmt.Errorf("convert: %s: %s: rope_base is %g", by, arch, x)
		}
	}
	whole := func(what, src string) (uint32, error) {
		x, err := number(what, src)
		if err == nil && (x < 0 || x != float32(uint32(x))) {
			err = fmt.Errorf("convert: %s: %s: %s is %g, not a count", by, arch, what, x)
		}
		return uint32(x), err
	}
	counts := func(set map[string]*uint32, srcs map[string]string) error {
		for _, what := range slices.Sorted(maps.Keys(set)) {
			if *set[what], err = whole(what, srcs[what]); err != nil {
				return err
			}
		}
		return nil
	}
	if l := st.Latent; l != nil {
		if c.QLoraRank, err = whole("latent.q", l.Q); err != nil {
			return err
		}
		if l.KV != "" {
			if c.KVLoraRank, err = whole("latent.kv", l.KV); err != nil {
				return err
			}
			if c.KVLoraRank == 0 {
				return fmt.Errorf("convert: %s: %s: latent attention with no key-value rank", by, arch)
			}
		}
	}
	if x := st.Indexer; x != nil {
		if err := counts(map[string]*uint32{"indexer.heads": &c.IdxHeads, "indexer.key_width": &c.IdxHeadDim,
			"indexer.top_k": &c.IdxTopK}, map[string]string{"indexer.heads": x.Heads,
			"indexer.key_width": x.KeyWidth, "indexer.top_k": x.TopK}); err != nil {
			return err
		}
		if err := indexerFits(c); err != nil {
			return fmt.Errorf("convert: %s: %s: %w", by, arch, err)
		}
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
	if x := st.Streams; x != nil {
		if c.HCMult, err = whole("streams.count", x.Count); err != nil {
			return err
		}
		if c.HCMult < 2 {
			return fmt.Errorf("convert: %s: %s: %d residual streams", by, arch, c.HCMult)
		}
		if x.Iterations != "" {
			if c.HCIters, err = whole("streams.sinkhorn_iterations", x.Iterations); err != nil {
				return err
			}
			if c.HCEps, err = number("streams.eps", x.Eps); err != nil {
				return err
			}
			if c.HCIters == 0 || !(c.HCEps > 0) {
				return fmt.Errorf("convert: %s: %s: %d streams, %d Sinkhorn rounds at %g", by, arch, c.HCMult,
					c.HCIters, c.HCEps)
			}
		}
	}
	if x := st.OutputGroups; x != nil {
		if err := counts(map[string]*uint32{"output_groups.groups": &c.OGroups, "output_groups.rank": &c.OLoraRank},
			map[string]string{"output_groups.groups": x.Groups, "output_groups.rank": x.Rank}); err != nil {
			return err
		}
		if c.OGroups == 0 || c.OLoraRank == 0 || c.NHead%c.OGroups != 0 {
			return fmt.Errorf("convert: %s: %s: %d output groups of rank %d over %d heads", by, arch, c.OGroups,
				c.OLoraRank, c.NHead)
		}
	}
	if st.AttnResBlock != "" {
		if c.AttnResBlock, err = whole("attn_res_block", st.AttnResBlock); err != nil {
			return err
		}
	}
	if st.ExpertLatent != "" {
		if c.ExpertLatent, err = whole("expert_latent", st.ExpertLatent); err != nil {
			return err
		}
		if c.ExpertLatent == c.NEmbd {
			c.ExpertLatent = 0
		}
	}
	if x := st.Situ; x != nil {
		if c.SituBeta, err = number("situ.beta", x.Beta); err != nil {
			return err
		}
		if c.SituLinearBeta, err = number("situ.linear_beta", x.LinearBeta); err != nil {
			return err
		}
		for _, b := range []float64{float64(c.SituBeta), float64(c.SituLinearBeta)} {
			if !(b > 0) || math.IsInf(b, 0) {
				return fmt.Errorf("convert: %s: %s: situ bounds %g and %g: each bounds b*tanh(x/b), so it is "+
					"positive and finite", by, arch, c.SituBeta, c.SituLinearBeta)
			}
		}
	}
	if st.HashLayers != "" {
		if c.NHashLayers, err = whole("hash_layers", st.HashLayers); err != nil {
			return err
		}
		if c.NHashLayers > c.NLayer {
			return fmt.Errorf("convert: %s: %s: %d hash-routed blocks of %d", by, arch, c.NHashLayers, c.NLayer)
		}
	}
	if w := st.Window; w != nil {
		if err := stateWindow(by, f, w, number, whole, c); err != nil {
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

// stateAttnTemp states the queries' temperature tp in c.
func stateAttnTemp(by, arch string, tp *mjStatesTemp, number func(what, src string) (float32, error), c *Config) error {
	s, err := number("attn_temp.scale", tp.Scale)
	if err != nil || s == 0 {
		return err
	}
	floor, err := number("attn_temp.floor", tp.Floor)
	if err != nil {
		return err
	}
	if floor < 1 || floor != float32(uint32(floor)) {
		return fmt.Errorf("convert: %s: %s: a temperature of %g with attn_temp.floor %g, no count to floor positions by",
			by, arch, s, floor)
	}
	off := float32(0)
	if tp.Offset != "" {
		if off, err = number("attn_temp.offset", tp.Offset); err != nil {
			return err
		}
	}
	c.AttnTempScale, c.AttnTempFloor, c.AttnTempOffset = s, uint32(floor), off
	return nil
}

// stateWindow states the sliding window w of f in c.
func stateWindow(by string, f *meta.File, w *mjStatesWindow, number func(what, src string) (float32, error),
	whole func(what, src string) (uint32, error), c *Config) error {
	arch := f.Arch()
	size, err := whole("window.size", w.Size)
	if err != nil {
		return err
	}
	if size == 0 && c.Flags.Has(jlm.FlagSWAChunked) {
		return fmt.Errorf("convert: %s: chunked attention over no chunk, every layer attending globally: %w", arch, ErrNotImplemented)
	}
	if size == 0 {
		if w.Period == "" {
			return nil
		}
		p, err := whole("window.period", w.Period)
		if err == nil && p != 0 {
			err = fmt.Errorf("convert: %s: %s: a window period over a window of 0", by, arch)
		}
		return err
	}
	if w.From != "" {
		from, err := whole("window.from", w.From)
		switch {
		case err != nil:
			return err
		case from >= c.NLayer:
			return nil
		case from > 0:
			return fmt.Errorf("convert: %s: the layers from %d on slide, which follows no period the container states: %w",
				arch, from, ErrNotImplemented)
		}
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

// indexerFits refuses an indexer missing a count, a key that is not whole
// groups of eight or does not hold the rotary, and one over latent attention
// with no query latent to project its query from; no counts state none.
func indexerFits(c *Config) error {
	switch {
	case c.IdxHeads|c.IdxHeadDim|c.IdxTopK == 0:
		return nil
	case c.IdxHeads == 0 || c.IdxHeadDim == 0 || c.IdxTopK == 0:
		return fmt.Errorf("an indexer of %d heads of %d, top %d", c.IdxHeads, c.IdxHeadDim, c.IdxTopK)
	case c.KVLoraRank != 0 && c.QLoraRank == 0:
		return fmt.Errorf("an indexer with no query latent (q_lora_rank %d, kv_lora_rank %d)",
			c.QLoraRank, c.KVLoraRank)
	case c.IdxHeadDim%8 != 0:
		return fmt.Errorf("indexer keys of %d, not whole groups of eight", c.IdxHeadDim)
	case c.NRot > c.IdxHeadDim:
		return fmt.Errorf("indexer heads of %d with a %d-wide rotary", c.IdxHeadDim, c.NRot)
	}
	return nil
}

// apply refuses a file whose keys contradict what r states its class's
// router does, and states the gate and the renormalisation in c.
func (r *mjRouter) apply(f *meta.File, c *Config) error {
	if r == nil {
		return nil
	}
	if r.Gate != "" {
		if kv, ok := f.Key("expert_gating_func"); ok {
			if v, _ := kv.Uint(); int64(v) != mjGates()[r.Gate] {
				return fmt.Errorf("expert_gating_func %d, and the class's router is %s: %w", v, r.Gate, ErrNotImplemented)
			}
		}
		if r.Gate == "sigmoid" {
			c.Flags |= jlm.FlagExpertSigmoid
		}
	}
	if kv, ok := f.Key("expert_weights_norm"); ok && r.Renormalise != "" {
		if v, _ := kv.Uint(); (v != 0) != (r.Renormalise == "always") {
			does := map[string]string{"always": "always renormalises", "never": "does not renormalise"}[r.Renormalise]
			return fmt.Errorf("expert_weights_norm %v, and the class's router %s: %w", v != 0, does, ErrNotImplemented)
		}
	}
	if r.Renormalise == "never" {
		c.Flags |= jlm.FlagNoExpertNorm
	}
	if kv, ok := f.Key("expert_weights_scale"); ok && r.Unscaled {
		if v, _ := kv.Float(); v != 0 && v != 1 {
			return fmt.Errorf("expert_weights_scale %g, and the class's router scales nothing: %w", v, ErrNotImplemented)
		}
	}
	if kv, ok := f.Key("expert_used_count"); ok && r.Selects != 0 {
		if v, _ := kv.Uint(); v != uint64(r.Selects) {
			return fmt.Errorf("expert_used_count %d, and the class's router selects %d: %w", v, r.Selects, ErrNotImplemented)
		}
	}
	return nil
}

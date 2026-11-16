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
	// LogitDiv divides the logits and ClampKQV bounds q, k and v (0 for
	// none).
	LogitDiv string `json:"logit_div"`
	ClampKQV string `json:"clamp_kqv"`
	// RopeBase is the rotary base, ResidualScale the factor each block's
	// attention and FFN output is scaled by (0 for none), and MoEStep the
	// mixture blocks' interleave (jlm.Config.MoEStep), each a number.
	RopeBase      string `json:"rope_base"`
	ResidualScale string `json:"residual_scale"`
	MoEStep       string `json:"moe_step"`
	// AttnTemp scales the queries by position (jlm.Config.AttnTempScale).
	AttnTemp *mjStatesTemp `json:"attn_temp"`
	// MRope, true, reads the rotary's sections off rope.dimension_sections
	// (mropeSectionsOf).
	MRope string `json:"mrope"`
	// WholeRotary states that the class turns every dimension of a head.
	WholeRotary bool `json:"whole_rotary"`
	// Router is what the class's router does whatever a file's keys say.
	Router  *mjRouter `json:"router"`
	exprs   map[string]*jinja.Expr
	tensors bool
}

// mjStatesTemp is the queries' temperature by position: Scale, Floor and
// Offset, each a number.
type mjStatesTemp struct {
	Scale  string `json:"scale"`
	Floor  string `json:"floor"`
	Offset string `json:"offset"`
}

// mjRouter is a mixture's router as its class computes it: Gate the
// function its scores go through (mjGates), Renormalise "always" or
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

// mjStatesWindow is a sliding window: Size positions (0 for none, and then
// nothing else of it applies); every Period'th layer attends globally, or
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
		srcs := []string{st.NormEps, st.EmbdScale, st.AttnSoftcap, st.FinalSoftcap, st.LogitDiv, st.ClampKQV,
			st.RopeBase, st.ResidualScale, st.MoEStep, st.MRope}
		if tp := st.AttnTemp; tp != nil {
			if tp.Scale == "" || tp.Floor == "" || tp.Offset == "" {
				return fmt.Errorf("convert: %s: %s: a temperature states its scale, floor and offset", mj.name, arch)
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
			switch n {
			case "src":
			case "tensors":
				st.tensors = true
			default:
				return fmt.Errorf("%q reads %q, which is not the keys (src) or the tensors' names (tensors)", src, n)
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

// statedTensorsAgree refuses a file storing a router over no expert_count.
func statedTensorsAgree(f *meta.File, c *Config) error {
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
	vars := map[string]jinja.Value{"src": ggufKeys(f)}
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
	}{{"rope_base", st.RopeBase, &c.RopeBase}, {"residual_scale", st.ResidualScale, &c.ResidualScale}} {
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
		if c.AttnTempScale, err = number("attn_temp.scale", tp.Scale); err != nil {
			return err
		}
		floor, err := number("attn_temp.floor", tp.Floor)
		if err != nil {
			return err
		}
		if floor < 1 || floor != float32(uint32(floor)) {
			return fmt.Errorf("convert: %s: %s: attn_temp.floor is %g, not a count", by, arch, floor)
		}
		c.AttnTempFloor = uint32(floor)
		if c.AttnTempOffset, err = number("attn_temp.offset", tp.Offset); err != nil {
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
	if st.MRope != "" {
		v, err := value("mrope", st.MRope)
		if err != nil {
			return err
		}
		if !v.IsBool() {
			return fmt.Errorf("convert: %s: %s: mrope is %v, not true or false", by, arch, v)
		}
		if v.AsBool() {
			if err := mropeSectionsOf(f, arch, c); err != nil {
				return err
			}
		}
	}
	if st.WholeRotary && c.NRot != c.HeadDim {
		return fmt.Errorf("convert: %s: a rotary over %d of a head's %d dimensions, where the reference turns "+
			"the whole head (%s): %w", arch, c.NRot, c.HeadDim, by, ErrNotImplemented)
	}
	w := st.Window
	if w == nil {
		return nil
	}
	whole := func(what, src string) (uint32, error) {
		x, err := number(what, src)
		if err == nil && (x < 0 || x != float32(uint32(x))) {
			err = fmt.Errorf("convert: %s: %s: %s is %g, not a count", by, arch, what, x)
		}
		return uint32(x), err
	}
	size, err := whole("window.size", w.Size)
	if err != nil {
		return err
	}
	if size == 0 && c.Flags.Has(jlm.FlagSWAChunked) {
		return fmt.Errorf("convert: %s: chunked attention over no chunk, every layer attending globally: %w", arch, ErrNotImplemented)
	}
	if size == 0 {
		if w.Period != "" {
			return fmt.Errorf("convert: %s: a window period over no attention.sliding_window: %w", arch, ErrNotImplemented)
		}
		return nil
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
	if k := f.UintKey("expert_used_count", 0); r.Selects != 0 && k != uint64(r.Selects) {
		return fmt.Errorf("expert_used_count %d, and the class's router selects %d: %w", k, r.Selects, ErrNotImplemented)
	}
	return nil
}

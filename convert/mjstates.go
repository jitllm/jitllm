package convert

import (
	"fmt"
	"maps"
	"math"
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
	// Window is the sliding window, and AttnSoftcap and FinalSoftcap the
	// caps on the scores and on the logits, each a number (0 for none).
	Window       *mjStatesWindow `json:"window"`
	AttnSoftcap  string          `json:"attn_softcap"`
	FinalSoftcap string          `json:"final_softcap"`
	// LogitDiv divides the logits (0 for none).
	LogitDiv string `json:"logit_div"`
	// Latent is latent attention's two ranks, and Indexer the indexer that
	// selects the positions attention reads.
	Latent  *mjStatesLatent  `json:"latent"`
	Indexer *mjStatesIndexer `json:"indexer"`
	exprs   map[string]*jinja.Expr
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

// mjStatesWindow is a sliding window: Size positions (0 for none, and then
// nothing else of it applies); every Period'th layer attends globally, or
// Pattern's where the file's attention.sliding_window_pattern does not say
// (swaPatternOf); RopeBase the windowed layers' rotary base.
type mjStatesWindow struct {
	Size     string `json:"size"`
	Period   string `json:"period"`
	Pattern  string `json:"pattern"`
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
		srcs := []string{st.NormEps, st.EmbdScale, st.AttnSoftcap, st.FinalSoftcap, st.LogitDiv}
		if w := st.Window; w != nil {
			if w.Size == "" || w.Period != "" && w.Pattern != "" {
				return fmt.Errorf("convert: %s: %s: a window states its size, and a period or a pattern, not both", mj.name, arch)
			}
			srcs = append(srcs, w.Size, w.Period, w.Pattern, w.RopeBase)
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
	for _, s := range []struct {
		what, src string
		to        *float32
	}{{"attn_softcap", st.AttnSoftcap, &c.AttnSoftcap}, {"final_softcap", st.FinalSoftcap, &c.FinalSoftcap},
		{"logit_div", st.LogitDiv, &c.LogitScale}} {
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
	w := st.Window
	if w == nil {
		return nil
	}
	size, err := whole("window.size", w.Size)
	if err != nil {
		return err
	}
	if size == 0 {
		if w.Period != "" {
			return fmt.Errorf("convert: %s: a window period over no attention.sliding_window: %w", arch, ErrNotImplemented)
		}
		return nil
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
	case c.IdxHeadDim%8 != 0 || c.NRot > c.IdxHeadDim:
		return fmt.Errorf("indexer heads of %d with a %d-wide rotary", c.IdxHeadDim, c.NRot)
	}
	return nil
}

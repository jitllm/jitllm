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
	// RopeBase is the rotary base of the layers no window states its own for.
	RopeBase string `json:"rope_base"`
	// Latent is latent attention's two ranks, and Indexer the indexer that
	// selects the positions attention reads.
	Latent  *mjStatesLatent  `json:"latent"`
	Indexer *mjStatesIndexer `json:"indexer"`
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
	exprs        map[string]*jinja.Expr
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

// mjStatesWindow is a sliding window: Size positions (0 for none, and then
// nothing else of it applies, but a Period over none is a malformed file); every Period'th layer attends globally, or
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
		srcs := []string{st.NormEps, st.EmbdScale, st.AttnSoftcap, st.FinalSoftcap, st.LogitDiv, st.RopeBase}
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
			return fmt.Errorf("convert: %s: %s: a window period over a window of 0", by, arch)
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
	case c.IdxHeadDim%8 != 0:
		return fmt.Errorf("indexer keys of %d, not whole groups of eight", c.IdxHeadDim)
	case c.NRot > c.IdxHeadDim:
		return fmt.Errorf("indexer heads of %d with a %d-wide rotary", c.IdxHeadDim, c.NRot)
	}
	return nil
}

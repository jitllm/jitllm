package convert

import (
	"fmt"
	"maps"
	"math"
	"path"
	"slices"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// This file is the only place the source format's key vocabulary lives. A
// container carries a typed, complete jlm.Config written once here, so the
// reader never parses another project's key names. What each architecture
// states is the library's (convert/models); an architecture it does not name
// fails here, at conversion, rather than at load.

// Config is a model's config as the converter builds it: the container's
// (jlm.Config), and Name the architecture its source declares (a GGUF's
// general.architecture, a class's container), for messages alone -- the
// container states the graph, and nothing past the converter names an
// architecture.
type Config struct {
	*jlm.Config
	Name string
	// NPredict is how many prediction blocks the source counts after the
	// trunk, the GGUF's nextn_predict_layers (predictOf): the converter's to
	// check the blocks against, never the container's.
	NPredict uint32
	// SwiGLULimits is each block's SwiGLU clamp under Flag2SwiGLUClamp, the
	// file's; nil for a container restated from its flag (Activation).
	SwiGLULimits []float32
	// Stated is what the library states for the source (stateContainer);
	// nil for a Config the library has not stated, which Activation,
	// RouterScore, RouterWeight and StateRevision refuse rather than read as
	// stating nothing (Unstated).
	Stated *Statement
	// took is the statement tables' choices reading the source made (a
	// rotary method's reading, a readout and its bands), which FeaturesOf
	// counts.
	took map[string]bool
}

// take records feature f as one reading c's source made.
func (c *Config) take(f string) {
	if c.took == nil {
		c.took = map[string]bool{}
	}
	c.took[f] = true
}

// Unstated is jc as a Config no library file states anything for.
func Unstated(jc *jlm.Config) *Config { return &Config{Config: jc, Stated: &Statement{}} }

// unstated is the refusal of a Config the library has not stated, by what
// read it.
func unstated(what string) error {
	return fmt.Errorf("convert: %s of a config the library has not stated (stateContainer, or Unstated for a source no file states)", what)
}

// Supported reports whether the converter implements a GGUF architecture
// name: the library's (Architectures), which is the list (RULE 7).
func Supported(arch string) bool {
	names, err := Architectures()
	return err == nil && slices.Contains(names, arch)
}

// Architectures is every GGUF architecture name the library's files name
// (convert/models), and the vision tower's (towerArch), sorted.
//
// It exists so the list can be checked against, not just consulted:
// library.TestEveryArchitectureIsObtainable reads it to confirm everything the
// converter implements is reachable from the download catalogue.
func Architectures() ([]string, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	out := []string{towerArch}
	for _, mj := range files {
		for _, n := range mj.GGUF {
			if !slices.Contains(out, n) {
				out = append(out, n)
			}
		}
	}
	sort.Strings(out)
	return out, nil
}

// GraphOf is the library file whose graph a GGUF architecture name or a
// Hugging Face class converts to, by its name in convert/models, and whether
// the name is one the library states a graph for. Several names share one
// graph, which each block's facts refine.
func GraphOf(name string) (string, bool) {
	files, err := libraryFiles()
	if err != nil {
		return "", false
	}
	return graphIn(files, name)
}

// GraphsOf is GraphOf for every name in names, the library read once: a name
// the library states no graph for is absent.
func GraphsOf(names []string) (map[string]string, error) {
	files, err := libraryFiles()
	if err != nil {
		return nil, err
	}
	out := make(map[string]string, len(names))
	for _, n := range names {
		if g, ok := graphIn(files, n); ok {
			out[n] = g
		}
	}
	return out, nil
}

// graphIn is GraphOf over the library's files.
func graphIn(files []*modelJSON, name string) (string, bool) {
	arch := name
	for _, mj := range files {
		if slices.Contains(mj.Classes, name) {
			arch = mj.Config.Arch
		}
	}
	for _, mj := range files {
		if _, own := mj.Container[arch]; (own || slices.Contains(mj.GGUF, arch)) && mj.Graph != nil {
			return strings.TrimSuffix(path.Base(mj.name), ".json"), true
		}
	}
	return "", false
}

func configOf(f *meta.File) (*Config, error) { return configOfAs(f, false) }

// classConfigOf is configOf for a class's statement, whose graph may be a
// container no GGUF reaches.
func classConfigOf(f *meta.File) (*Config, error) { return configOfAs(f, true) }

func configOfAs(f *meta.File, class bool) (*Config, error) {
	name, t, err := trunkOf(f)
	if err != nil {
		return nil, err
	}
	var c *Config
	if t != nil {
		c, err = t.config(name, f)
	} else {
		c, err = configOfFile(f, class)
	}
	if err != nil {
		return nil, err
	}
	return c, localAxes(c)
}

// localAxes refuses rotary sections under a second rotary, the sliding
// layers' own base or a block's own width: a table over several position
// axes on the local layers, which no statement of the library states.
func localAxes(c *Config) error {
	second := c.RopeBaseSWA != 0
	for b := range c.Geoms {
		second = second || c.Geoms[b].NRot != c.NRot
	}
	if c.RopeSections == [4]uint32{} || !second {
		return nil
	}
	return fmt.Errorf("convert: %s: rotary sections %v on the sliding layers' own rotary: %w",
		c.Name, c.RopeSections, ErrNotImplemented)
}

// headsOfProjections is the head width of a file that states no
// attention.key_length where its projections are not its heads at
// embedding/heads (Phi-tiny-MoE's sixteen heads of 128 on a 4096-wide
// model). Apart, it is q's rows over the heads, which k's rows over the kv
// heads agree with, and v's at attention.value_length where the file states
// one. Fused, where attn_qkv holds other rows than the statement cuts it
// into, it is the rows over the heads and the kv heads twice (or once, past
// a stated value width), and the statement cuts it at that width. The value
// and rotary widths the file leaves to the head follow it. A rotary stated at
// embedding/heads is the same quotient the file's writer took for the head,
// so it is the head's whole width only where the class turns the whole head;
// elsewhere the file does not say which of the two it means. A gated q
// projection carries the gate's rows beside the heads', and its files state
// key_length.
func headsOfProjections(f *meta.File, c *Config) error {
	if _, ok := f.Key("attention.key_length"); ok || c.Flags.Has(jlm.FlagAttnOutGate) || c.NHead == 0 {
		return nil
	}
	_, perKV, err := layerCount(f, "attention.head_count_kv", c.NHead)
	if err != nil {
		return err
	}
	nh, nkv := uint64(c.NHead), uint64(c.NKVHead)
	if perKV != nil {
		nkv = uint64(perKV[0])
	}
	_, vstated := f.Key("attention.value_length")
	vl := f.UintKey("attention.value_length", 0)
	var w, vw uint64
	by := "blk.0.attn_q.weight"
	if qkv, ok := f.Layer(0, "attn_qkv.weight"); ok && len(qkv.Dims) == 2 {
		if nkv == 0 || len(c.LayerKinds) > 0 && c.LayerKinds[0] == jlm.LayerLinearAttn {
			return nil
		}
		cut, stated, err := statedQKVRows(f, c)
		if err != nil || !stated || cut == qkv.Dims[1] {
			return err
		}
		by = "blk.0.attn_qkv.weight"
		r := qkv.Dims[1]
		switch {
		case vstated && r > nkv*vl && (r-nkv*vl)%(nh+nkv) == 0:
			w, vw = (r-nkv*vl)/(nh+nkv), vl
		case !vstated && r%(nh+2*nkv) == 0:
			w = r / (nh + 2*nkv)
			vw = w
		}
		if w == 0 {
			return fmt.Errorf("%s has %d rows, not the %d its statement cuts and not %d heads and %d kv heads "+
				"of one width, and the file states no attention.key_length: %w", by, r, cut, nh, nkv, ErrBadInput)
		}
	} else {
		q, ok := f.Layer(0, "attn_q.weight")
		if !ok || len(q.Dims) != 2 || q.Dims[1] == nh*uint64(c.HeadDim) {
			return nil
		}
		w = q.Dims[1] / nh
		if q.Dims[1]%nh != 0 || w == 0 {
			return fmt.Errorf("blk.0.attn_q.weight has %d rows, not whole heads of %d, and the file states no "+
				"attention.key_length: %w", q.Dims[1], c.NHead, ErrBadInput)
		}
		vw = f.UintKey("attention.value_length", w)
		k, kok := f.Layer(0, "attn_k.weight")
		v, vok := f.Layer(0, "attn_v.weight")
		if !kok || !vok || len(k.Dims) != 2 || len(v.Dims) != 2 || nkv == 0 {
			return fmt.Errorf("blk.0.attn_q.weight is %d heads of %d, not of embedding/heads %d, and the file "+
				"states no attention.key_length and no blk.0 k and v projection to hold them to: %w",
				c.NHead, w, c.HeadDim, ErrBadInput)
		}
		if k.Dims[1] != nkv*w || v.Dims[1] != nkv*vw {
			return fmt.Errorf("blk.0.attn_q.weight is %d heads of %d, and blk.0.attn_k.weight's %d rows and "+
				"blk.0.attn_v.weight's %d are not %d kv heads of %d and %d: %w",
				c.NHead, w, k.Dims[1], v.Dims[1], nkv, w, vw, ErrBadInput)
		}
	}
	quot := c.HeadDim
	c.HeadDim, c.HeadDimV = uint32(w), 0
	if vw != w {
		c.HeadDimV = uint32(vw)
	}
	if _, ok := f.Key("rope.dimension_count"); !ok || c.NRot == quot && c.Stated.wholeRotary != "" {
		c.NRot = c.HeadDim
	} else if c.NRot == quot {
		return fmt.Errorf("rope.dimension_count %d is embedding/heads, and %s is %d heads of %d: the file "+
			"does not say whether its rotary turns the whole head: %w", quot, by, c.NHead, w, ErrBadInput)
	}
	if by != "blk.0.attn_qkv.weight" {
		return nil
	}
	for n, e := range map[string]string{
		"q": fmt.Sprintf("at(attention.head_count, block) * %d", w),
		"k": fmt.Sprintf("at(attention.head_count_kv ?? (attention.head_count), block) * %d", w),
		"v": fmt.Sprintf("at(attention.head_count_kv ?? (attention.head_count), block) * %d", vw),
	} {
		if c.Stated.sizes[n], err = parseExpr(e); err != nil {
			return err
		}
	}
	return nil
}

// statedQKVRows is how many rows the statement cuts block 0's fused attn_qkv
// into, its sizes read off the file's keys; false where it cuts none.
func statedQKVRows(f *meta.File, c *Config) (uint64, bool, error) {
	if c.Stated.blockTensors["attn_qkv.weight"] == nil {
		return 0, false, nil
	}
	env := ggufEnv(f)
	at0 := func(p string) any {
		if p == "block" {
			return 0.0
		}
		return env(p)
	}
	var rows float64
	for _, n := range []string{"q", "k", "v"} {
		x, ok := c.Stated.sizes[n]
		if !ok {
			return 0, false, nil
		}
		v, err := evalNum(x, at0)
		if err != nil {
			return 0, false, fmt.Errorf("size.%s: %w", n, err)
		}
		rows += v
	}
	return uint64(rows), true, nil
}

// configOfFile is configOf for every file but the ones read through a view
// (a readout's trunk, trunk.go).
func configOfFile(f *meta.File, class bool) (*Config, error) {
	name := f.Arch()
	if name == towerArch {
		return nil, fmt.Errorf("convert: %q is a vision tower, not a language model", name)
	}
	ok := Supported(name)
	if st, _, err := ggufStates(name); !ok && class && err == nil && st != nil {
		ok = true
	}
	if !ok {
		names, err := Architectures()
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("convert: architecture %q is not implemented (%s): %w",
			name, strings.Join(names, ", "), ErrNotImplemented)
	}
	nhead, err := uniformCount(f, "attention.head_count", 0)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	c := &Config{Name: name, Config: &jlm.Config{
		NLayer:     uint32(f.UintKey("block_count", 0)),
		NEmbd:      uint32(f.UintKey("embedding_length", 0)),
		NHead:      nhead,
		NFFN:       uint32(f.UintKey("feed_forward_length", 0)),
		NCtx:       uint32(f.UintKey("context_length", 2048)),
		RMSEps:     float32(f.FloatKey("attention.layer_norm_rms_epsilon", 1e-5)),
		RopeBase:   float32(f.FloatKey("rope.freq_base", 10000)),
		EmbdScale:  1,
		AttnFactor: float32(f.FloatKey("rope.scaling.attn_factor", 1)),
	}}
	// A file stating no heads on any layer attends nowhere (plain Mamba): one
	// placeholder head as wide as the embedding keeps every attention-shaped
	// derivation finite, and no layer reads it (countsHeld refuses one that
	// does).
	if _, stated := f.Key("attention.head_count"); stated && c.NHead == 0 {
		c.NHead = 1
	}
	if c.NLayer == 0 || c.NEmbd == 0 || c.NHead == 0 {
		return nil, fmt.Errorf("convert: %s: missing block_count/embedding_length/head_count", name)
	}
	// Absent head_count_kv means MHA, not zero heads. One that differs by
	// layer is left for the blocks' own geometries to hold (countsHeld).
	nkv, perKV, err := layerCount(f, "attention.head_count_kv", c.NHead)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if c.NKVHead = nkv; perKV == nil && c.NKVHead == 0 {
		c.NKVHead = c.NHead
	}
	// key_length when present, else embd/heads. For gemma these coincide at 256,
	// so reading it from the wrong place goes unnoticed there.
	c.HeadDim = uint32(f.UintKey("attention.key_length", uint64(c.NEmbd/c.NHead)))
	c.HeadDimV = uint32(f.UintKey("attention.value_length", uint64(c.HeadDim)))
	c.NRot = uint32(f.UintKey("rope.dimension_count", uint64(c.HeadDim)))

	// On an MLA model those two keys mean something else. The source writes four
	// head widths for DeepSeek:
	//
	//	attention.key_length        576 = kv_lora_rank + qk_rope_head_dim
	//	attention.value_length      512 = kv_lora_rank
	//	attention.key_length_mla    192 = qk_nope_head_dim + qk_rope_head_dim
	//	attention.value_length_mla  128 = v_head_dim
	//
	// The first two describe the absorbed cache row; the last two are the head
	// geometry and what the softmax is scaled by. Taking key_length would load
	// and run fluently with every score scaled by sqrt(576) instead of sqrt(192).
	// The container stores the natural pair and derives the absorbed one.
	if kl, ok := f.Key("attention.key_length_mla"); ok {
		if v, ok := kl.Uint(); ok {
			c.HeadDim = uint32(v)
		}
		c.HeadDimV = uint32(f.UintKey("attention.value_length_mla", uint64(c.HeadDim)))
	}
	// Stored only when it says something HeadDim does not.
	if c.HeadDimV == c.HeadDim {
		c.HeadDimV = 0
	}
	// attention.scale is the score multiplier in place of 1/sqrt(head), for
	// every graph whose file states one; a graph that fixes its own sets it
	// below.
	if err := attnScaleKey(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}

	// What the library's file states for the architecture's container, over
	// the keys (mjStates).
	if err := stateContainer(f, c); err != nil {
		return nil, err
	}

	// A mixture on every n'th block, the rest dense, is no graph this
	// converter states.
	if n := f.UintKey("moe_every_n_layers", 0); n > 0 {
		return nil, fmt.Errorf("convert: %s: a mixture every %d layers: %w", name, n, ErrNotImplemented)
	}
	// A learned position table is added to the input rows: as wide as they
	// are, and bounding the context, since a position past its rows has no
	// embedding.
	if t, ok := f.Get("position_embd.weight"); ok {
		if len(t.Dims) < 2 || uint32(t.Dims[0]) != c.NEmbd {
			return nil, fmt.Errorf("convert: %s: position_embd is %v, the model %d wide", name, t.Dims, c.NEmbd)
		}
		c.NCtx = min(c.NCtx, uint32(t.Dims[1]))
	}

	// A per-head q/k norm under LayerNorm blocks (command-r-plus, StableLM 2
	// 12B: a weight per head) is refused by name: running without it would be
	// fluent and wrong.
	if c.Flags.Has(jlm.FlagLayerNorm) {
		if _, ok := f.Get("blk.0.attn_q_norm.weight"); ok {
			return nil, fmt.Errorf("convert: %s: a per-head q/k LayerNorm: %w", name, ErrNotImplemented)
		}
	}
	if err := statedFeaturesFit(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := stateKindGeoms(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := sparsityOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	// A rotary narrower than the head turns its leading pairs and passes the
	// rest through, whatever the architecture.
	if c.NRot != c.HeadDim {
		if err := partialRotary(c); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", name, err)
		}
	}

	// A clamp on q, k and v is read for every architecture whose file states
	// one; zero is none, and a bound that is not positive and finite clamps
	// nothing sensibly.
	if kv, ok := f.Key("attention.clamp_kqv"); ok {
		v, _ := kv.Float()
		if !(v >= 0) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("convert: %s: attention.clamp_kqv %g", name, v)
		}
		c.ClampKQV = float32(v)
	}
	// A per-head q/k norm normalises both or neither.
	_, q := f.Get("blk.0.attn_q_norm.weight")
	if _, k := f.Get("blk.0.attn_k_norm.weight"); q != k {
		return nil, fmt.Errorf("convert: %s: blk.0 carries a q norm %v and a k norm %v", name, q, k)
	}

	// An embedding model is flagged by two keys, read for every architecture:
	// pooling_type (1 mean, 2 first position, 3 last) and attention.causal=false.
	// A decoder can carry either (qwen3-embedding, EmbeddingGemma), so neither
	// makes a new architecture. Where the file states no pooling_type, the
	// statement's pooling is the readout.
	pooling := uint64(c.Stated.pooling)
	if kv, ok := f.Key("pooling_type"); ok {
		pooling, _ = kv.Uint()
	}
	switch v := pooling; v {
	case 0:
	case 1:
		c.Flags |= jlm.FlagPoolMean
	case 2:
		c.Flags |= jlm.FlagPoolCLS
	case 3:
		c.Flags |= jlm.FlagPoolLast
	default:
		// 4 is RANK, a reranker's classification head; the vector would be
		// the wrong object entirely.
		return nil, fmt.Errorf("convert: %s: pooling_type %d is not implemented: %w",
			name, v, ErrNotImplemented)
	}
	if kv, ok := f.Key("attention.causal"); ok {
		if v, ok := kv.Uint(); ok && v == 0 {
			c.Flags |= jlm.FlagNonCausal
		}
	}
	if err := denseHeadsFit(f); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}

	// block_count includes any multi-token-prediction blocks after the trunk;
	// from here on NLayer is the trunk alone. Before the layer kinds below,
	// which are the trunk's.
	if err := predictOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}

	// Rotary scaling is read for every architecture; a type this converter does
	// not implement is refused rather than dropped.
	if err := ropeScalingOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}

	// The shared expert is read off a key: its presence is the fact, not the
	// architecture.
	c.NFFNShExp = uint32(f.UintKey("expert_shared_feed_forward_length", 0))

	// The per-layer kinds are read off the tensors each layer carries, as the
	// library states them, not off an "every fourth layer" rule.
	if err := stateLayers(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := selectiveNormsFit(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := countsHeld(c, f); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}

	// A whole-projection q/k norm is each projection's width: {n_embd} for q
	// and {n_embd_gqa} for k, which is OLMo 2's k_norm (transformers'
	// Olmo2Attention). A tensor of another width is refused here, by name, rather
	// than read past or left short at run time.
	if c.Flags.Has(jlm.FlagQKNormWide) {
		for _, nw := range [2]struct {
			n string
			w uint64
		}{{"attn_q_norm", uint64(c.NHead) * uint64(c.HeadDim)}, {"attn_k_norm", uint64(c.NKVHead) * uint64(c.HeadDim)}} {
			n, w := nw.n, nw.w
			if t, ok := f.Get("blk.0." + n + ".weight"); ok && (len(t.Dims) != 1 || t.Dims[0] != w) {
				return nil, fmt.Errorf("convert: %s: blk.0.%s.weight is %v, a whole-projection norm "+
					"of %d q heads and %d kv heads wants [%d]", name, n, t.Dims, c.NHead, c.NKVHead, w)
			}
		}
	}

	if t, ok := f.Get("token_embd.weight"); ok && len(t.Dims) > 1 {
		c.NVocab = uint32(t.Dims[1])
	}
	// Tied embeddings are read off the file: qwen3-1.7B ties and qwen3-8B does
	// not, under the same arch string.
	if _, ok := f.Get("output.weight"); !ok {
		c.Flags |= jlm.FlagTiedEmbd
	}

	// A mixture is a property of the file, not the arch: Mixtral ships as arch
	// "llama" with expert_count 8.
	c.NExpert = uint32(f.UintKey("expert_count", 0))
	if c.NExpert > 0 {
		c.NExpertUsed = uint32(f.UintKey("expert_used_count", 0))
		// Not feed_forward_length: that is the dense width, which a mixture never
		// uses.
		c.NFFNExp = uint32(f.UintKey("expert_feed_forward_length", 0))
		// When the key is absent, read the bank's shape rather than guess: olmoe's
		// banks are {n_embd, n_ff, n_expert}, and a Mixtral-era file ships
		// per-expert tensors (sourceOf stacks them) where n_ff/n_used is off by 2x.
		// The first block that has a bank: Jamba's block 0 is dense.
		for i := 0; c.NFFNExp == 0 && i < int(c.NLayer); i++ {
			if t, ok := f.Get("blk." + itoa(i) + ".ffn_gate_exps.weight"); ok && len(t.Dims) > 1 {
				c.NFFNExp = uint32(t.Dims[1])
			} else if t, ok := f.Get("blk." + itoa(i) + ".ffn_gate.0.weight"); ok && len(t.Dims) > 1 {
				c.NFFNExp = uint32(t.Dims[1])
			}
		}
		if c.NFFNExp == 0 && c.NExpertUsed > 0 {
			c.NFFNExp = c.NFFN / c.NExpertUsed
		}
		if c.NExpertUsed == 0 || c.NExpertUsed > c.NExpert || c.NFFNExp == 0 {
			return nil, fmt.Errorf("convert: %s: expert_count=%d expert_used_count=%d "+
				"expert_feed_forward_length=%d", name, c.NExpert, c.NExpertUsed, c.NFFNExp)
		}
		// DeepSeek carries no shared-expert width: its GGUF writes
		// expert_shared_count, and the width is that times expert_feed_forward_length.
		// A zero here is not refused downstream; it surfaces as a misleading kernel
		// decline, since the shared expert is bound off the container's tensors.
		if c.NFFNShExp == 0 {
			if n := uint32(f.UintKey("expert_shared_count", 0)); n > 0 {
				c.NFFNShExp = n * c.NFFNExp
			}
		}
		// expert_weights_scale and expert_weights_norm are read by presence, never
		// with a default: qwen3moe carries neither, renormalises unconditionally and
		// applies no scale, so defaulting norm to false or scale to 0 would silently
		// skip the division or zero the FFN.
		if kv, ok := f.Key("expert_weights_scale"); ok {
			if v, ok := kv.Float(); ok && v != 0 {
				c.ExpertScale = float32(v)
			}
		}
		// A stated expert_weights_norm decides over the graph's default, for
		// every graph: olmoe's case leaves the weights as they are, and a file
		// that says to renormalise (OLMoE's norm_topk_prob) is obeyed.
		// meta.Value.Int decodes Bool, so Uint is the accessor for a GGUF bool
		// and 0 is false.
		if kv, ok := f.Key("expert_weights_norm"); ok {
			switch v, ok := kv.Uint(); {
			case ok && v == 0:
				c.Flags |= jlm.FlagNoExpertNorm
			case ok:
				c.Flags &^= jlm.FlagNoExpertNorm
			}
		}
		// An absent expert_weights_norm is not one answer for every arch: Qwen3-MoE's
		// config.json says norm_topk_prob: true and DeepSeek-V2-Lite's says
		// norm_topk_prob: false, so latent.json states no_expert_norm where the
		// key is absent. Renormalising it generates fluent nonsense from the first
		// token. See docs/engineering-history/model-correctness.md.
		// 1 is softmax, 2 is sigmoid; anything else is refused rather than
		// approximated, because a wrong gate is a fluent model with another mixture.
		if kv, ok := f.Key("expert_gating_func"); ok {
			switch v, _ := kv.Uint(); v {
			case 0, 1:
			case 2:
				c.Flags |= jlm.FlagExpertSigmoid
			case 4:
				// sqrt(softplus), where the graph states that router (DeepSeek V4's).
				if !c.Flags2.Has(jlm.Flag2ExpertSqrtSoftplus) {
					return nil, fmt.Errorf("convert: %s: expert_gating_func 4 (sqrt-softplus) where the "+
						"graph states no sqrt-softplus router: %w", name, ErrNotImplemented)
				}
			default:
				return nil, fmt.Errorf("convert: %s: expert_gating_func %d is not implemented: %w",
					name, v, ErrNotImplemented)
			}
		}
		// Grouped selection. Both keys or neither; one alone cannot be acted on.
		c.NExpertGroup = uint32(f.UintKey("expert_group_count", 0))
		c.NExpertGroupUsed = uint32(f.UintKey("expert_group_used_count", 0))
		// One group, all of it used, is the ungrouped selection: a Kimi-K3 GGUF
		// writes topk_group (1) as the used count and no group count beside it.
		if c.NExpertGroup == 0 && c.NExpertGroupUsed == 1 {
			c.NExpertGroupUsed = 0
		}
		if (c.NExpertGroup == 0) != (c.NExpertGroupUsed == 0) {
			return nil, fmt.Errorf("convert: %s: expert_group_count=%d with expert_group_used_count=%d",
				name, c.NExpertGroup, c.NExpertGroupUsed)
		}
		// A grouped selection picks whole groups of an even split.
		if c.NExpertGroup > 1 && (c.NExpert%c.NExpertGroup != 0 || c.NExpertGroupUsed > c.NExpertGroup) {
			return nil, fmt.Errorf("convert: %s: %d experts in %d groups, %d used", name, c.NExpert,
				c.NExpertGroup, c.NExpertGroupUsed)
		}
	}

	// The leading dense run is read for every architecture, as a property of the
	// file.
	// A dense model has no shared expert, whatever a key says: granite-4.0-micro
	// carries expert_shared_feed_forward_length beside expert_count 0.
	if c.NExpert == 0 {
		c.NFFNShExp = 0
	}
	c.NDenseLead = uint32(f.UintKey("leading_dense_block_count", 0))
	if c.NDenseLead > c.NLayer {
		return nil, fmt.Errorf("convert: %s: leading_dense_block_count %d in a %d-block model",
			name, c.NDenseLead, c.NLayer)
	}
	if err := statedMixtureFits(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := sharedExpertWidth(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if err := denseMoEFits(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	if kv, ok := f.Key("rope.scaling.finetuned"); ok && !c.Flags.Has(jlm.FlagNoPosEnc) {
		if v, ok := kv.Uint(); ok && v == 0 {
			return nil, fmt.Errorf("convert: %s: rope.scaling.finetuned false turns the rotary off, which the "+
				"architecture's graph does not state: %w", name, ErrNotImplemented)
		}
	}
	if err := decisionOf(f, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	return c, nil
}

// ropeScalingOf reads the rotary scaling keys into c: the context the
// rotary was trained to, and the method rope.scaling.type names by its GGUF
// statement in rotary.json (ggufRotary), the reads of the convention its
// class states (its yarn_magnitude) in place of the method's where it states
// one -- the magnitude and the fields the method keeps. Its divisors are
// stated with the file's own factors (stateGGUFRotary).
func ropeScalingOf(f *meta.File, c *Config) error {
	// The context the rotary was trained to: YaRN scales from it, and
	// LongRoPE turns its long factors past it (model.RopeFor).
	if v := f.UintKey("rope.scaling.original_context_length", 0); v > 0 {
		c.YarnOrigCtx = uint32(v)
	}
	convention := ""
	if c.Stated != nil {
		convention = c.Stated.yarnMagnitude
	}
	r, err := ggufRotary(f, c, convention)
	if err != nil || r == nil {
		return err
	}
	if ok, err := r.holds(c); err != nil || !ok {
		return err
	}
	_, err = r.settle(c)
	return err
}

// allLocal is the SWAPeriod that makes every layer a sliding-window one.
// Config.SWA is il%period < period-1, so NLayer+1 puts every layer in the
// local run; giving period 1 a new meaning would have made older readers see
// no local layers in a newer container.
func allLocal(nLayer uint32) uint32 { return nLayer + 1 }

// ropeSectionsFit sets an M-RoPE model's split secs of a head's rotary pairs
// between the (t, h, w) coordinates of a position in c (the rope_sections
// statement). The sum is checked because a short list leaves a tail of each
// head unrotated, which is fluent and which no self-comparing tier can see.
//
// An interleaved split (rope_layout "interleaved") is read as transformers
// reads it: pair i turns by h when i%3 == 1 and i < 3*sections[1], by w when
// i%3 == 2 and i < 3*sections[2], and by t otherwise. A reading that sends a
// pair with i%3 == 0 at or past 3*sections[0] to a fourth axis disagrees; no
// published split reaches one, and one that would is refused rather than
// resolved, since the two readings would then rotate it differently.
func ropeSectionsFit(name string, secs []int32, c *Config) error {
	if len(secs) > len(c.RopeSections) {
		return fmt.Errorf("convert: %s: %d rope sections, this format defines %d",
			name, len(secs), len(c.RopeSections))
	}
	var sum uint32
	for i, v := range secs {
		if v < 0 {
			return fmt.Errorf("convert: %s: rope section %d is %d", name, i, v)
		}
		c.RopeSections[i] = uint32(v)
		sum += uint32(v)
	}
	if sum != c.NRot/2 {
		return fmt.Errorf("convert: %s: rope sections %v sum to %d, and a head "+
			"has %d rotary pairs -- the remainder would be left unrotated",
			name, secs, sum, c.NRot/2)
	}
	if c.Stated.ropeLayout == "interleaved" {
		if c.RopeSections[3] != 0 {
			return fmt.Errorf("convert: %s: an interleaved M-RoPE with a fourth section %d: %w",
				name, c.RopeSections[3], ErrNotImplemented)
		}
		for i := uint32(0); i < sum; i += 3 {
			if i >= 3*c.RopeSections[0] {
				return fmt.Errorf("convert: %s: interleaved rope sections %v leave pair %d to "+
					"a fourth axis or, in transformers, to t: %w", name, secs, i, ErrNotImplemented)
			}
		}
	}
	return nil
}

// visionOf reads a vision tower's description. A tower is its own container.
func visionOf(f *meta.File) (*jlm.Vision, *projStates, error) {
	if f.Arch() != towerArch {
		return nil, nil, nil
	}
	v := &jlm.Vision{
		NLayer:    uint32(f.UintKey("vision.block_count", 0)),
		NEmbd:     uint32(f.UintKey("vision.embedding_length", 0)),
		NHead:     uint32(f.UintKey("vision.attention.head_count", 0)),
		NFFN:      uint32(f.UintKey("vision.feed_forward_length", 0)),
		ImageSize: uint32(f.UintKey("vision.image_size", 0)),
		PatchSize: uint32(f.UintKey("vision.patch_size", 0)),
		Scale:     uint32(f.UintKey("vision.projector.scale_factor", 1)),
		Eps:       float32(f.FloatKey("vision.attention.layer_norm_epsilon", 1e-6)),
	}
	// The projector is a list and an unknown one is refused, as an unknown
	// tokenizer.ggml.pre is.
	name, _ := f.KV["clip.projector_type"].String()
	if name == "" {
		// A vision-and-audio mmproj (Gemma 4's) names each side's projector.
		name, _ = f.KV["clip.vision.projector_type"].String()
	}
	lib, err := projectorLibrary()
	if err != nil {
		return nil, nil, err
	}
	p, ok := lib[name]
	if !ok {
		return nil, nil, fmt.Errorf("convert: projector_type %q unsupported (only %s): %w",
			name, strings.Join(slices.Sorted(maps.Keys(lib)), ", "), ErrNotImplemented)
	}
	if err := p.state(name, f, v); err != nil {
		return nil, nil, err
	}
	// Three activations, as the GGUF's keys state them: GELU for use_gelu,
	// SiLU for use_silu, and quick-GELU when neither is set (openai CLIP,
	// llava-phi-3's mmproj). Defaulting to SiLU there would be fluent and wrong.
	gelu, _ := f.KV["clip.use_gelu"].Uint()
	silu, _ := f.KV["clip.use_silu"].Uint()
	switch {
	case gelu != 0 && silu != 0:
		return nil, nil, fmt.Errorf("convert: clip: use_gelu and use_silu are both set")
	case gelu != 0:
		v.Flags |= jlm.FlagGELU
	case silu != 0:
		// SiLU is the zero value; naming the case keeps the table complete.
	default:
		v.Flags |= jlm.FlagQuickGELU
	}
	for i, key := range []string{"clip.vision.image_mean", "clip.vision.image_std"} {
		fs, err := f.KV[key].Float32s()
		if err != nil || len(fs) < 3 {
			return nil, nil, fmt.Errorf("convert: %s must have at least 3 elements", key)
		}
		if i == 0 {
			copy(v.Mean[:], fs)
		} else {
			copy(v.Std[:], fs)
		}
	}
	if v.NLayer == 0 || v.NEmbd == 0 || v.NHead == 0 {
		return nil, nil, fmt.Errorf("convert: clip: missing block_count/embedding_length/head_count")
	}
	// The FFN width comes from the matrix, not the key: mmproj-Qwen2-VL-2B
	// carries feed_forward_length 1536 (the text model's width) while its blocks
	// are 1280x5120. The width is whichever dimension is not NEmbd, since
	// Qwen2-VL's matrices are spelled the other way round.
	if ft, ok := f.Get("v.blk.0.ffn_up.weight"); ok && len(ft.Dims) >= 2 {
		a, b := uint32(ft.Dims[0]), uint32(ft.Dims[1])
		w := a
		if a == v.NEmbd {
			w = b
		}
		if w != 0 && w != v.NEmbd && w != v.NFFN {
			v.NFFN = w
		}
	}
	if v.NFFN == 0 {
		return nil, nil, fmt.Errorf("convert: clip: no feed-forward width")
	}
	p.restate(v)
	return v, p, nil
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

// sparsityOf reads activation_sparsity_scale: the standard normal's quantile
// at each layer's target sparsity, which transformers computes as
// Normal(0,1).icdf(p) and a GGUF holds per layer, -inf where a
// layer has none (icdf(0)). The published pattern is one value on a leading
// run of layers, which is what the container holds; any other shape is
// refused rather than approximated. It is read for any file stating the key.
func sparsityOf(f *meta.File, c *Config) error {
	kv, ok := f.Key("activation_sparsity_scale")
	if !ok {
		return nil
	}
	vs, err := kv.Float32s()
	if err != nil || len(vs) != int(c.NLayer) {
		return fmt.Errorf("activation_sparsity_scale: %d values for %d layers (%v)", len(vs), c.NLayer, err)
	}
	n := 0
	for n < len(vs) && !math.IsInf(float64(vs[n]), -1) {
		if vs[n] != vs[0] || math.IsNaN(float64(vs[n])) {
			return fmt.Errorf("activation_sparsity_scale %v: layer %d differs from layer 0: %w", vs, n, ErrNotImplemented)
		}
		n++
	}
	for i := n; i < len(vs); i++ {
		if !math.IsInf(float64(vs[i]), -1) {
			return fmt.Errorf("activation_sparsity_scale %v: layer %d is sparse after a dense one: %w",
				vs, i, ErrNotImplemented)
		}
	}
	c.NSparse = uint32(n)
	if n > 0 {
		c.SparseStd = vs[0]
	}
	return nil
}

// denseHeadsFit holds the widths a file states for its projection heads
// (dense_2_feat_in and the rest) to the heads' own tensors: two statements of
// one shape that disagree are refused, and so is a width for a head the file
// does not carry.
func denseHeadsFit(f *meta.File) error {
	for _, h := range [...]string{"dense_2", "dense_3"} {
		t, has := f.Get(h + ".weight")
		for i, side := range [...]string{"in", "out"} {
			kv, ok := f.Key(h + "_feat_" + side)
			if !ok {
				continue
			}
			v, _ := kv.Uint()
			if !has {
				return fmt.Errorf("%s_feat_%s is %d, and the file carries no %s.weight", h, side, v, h)
			}
			if len(t.Dims) != 2 || t.Dims[i] != v {
				return fmt.Errorf("%s_feat_%s is %d, and %s.weight is %v", h, side, v, h, t.Dims)
			}
		}
	}
	return nil
}

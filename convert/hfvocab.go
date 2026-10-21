package convert

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/pretok"
)

// The tokenizer, read from the model's own tokenizer.json (RULE 7m). A GGUF
// carries only a pre-tokenizer label (tokenizer.ggml.pre) that the GGUF path
// resolves through pretok.Table; a HuggingFace directory ships the artefact
// the weights were trained against, so this path parses the pipeline out of it
// with pretok.ParsePreTokenizer and reads as fields what the GGUF side infers
// from a family:
//
//	ignore_merges   model.ignore_merges
//	byte_fallback   model.byte_fallback
//	add_bos         the post_processor, which is what actually runs
//
// It also carries what a GGUF has no room for: added tokens with their own
// lstrip/rstrip/normalized handling, and named chat templates.

// hfTokenizer is the subset of tokenizer.json this converter reads.
type hfTokenizer struct {
	AddedTokens []struct {
		ID         int32  `json:"id"`
		Content    string `json:"content"`
		Special    bool   `json:"special"`
		LStrip     bool   `json:"lstrip"`
		RStrip     bool   `json:"rstrip"`
		Normalized bool   `json:"normalized"`
		SingleWord bool   `json:"single_word"`
	} `json:"added_tokens"`
	PostProcessor json.RawMessage `json:"post_processor"`
	Model         struct {
		Type         string            `json:"type"`
		Vocab        map[string]int32  `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
		IgnoreMerges bool              `json:"ignore_merges"`
		ByteFallback bool              `json:"byte_fallback"`
		UnkToken     *string           `json:"unk_token"`
	} `json:"model"`
}

// hfTokenizerConfig is the subset of tokenizer_config.json this converter
// reads. Every special token may be a bare string or an object with a
// "content" field, which is why they are RawMessage.
type hfTokenizerConfig struct {
	BOS          json.RawMessage `json:"bos_token"`
	EOS          json.RawMessage `json:"eos_token"`
	Unk          json.RawMessage `json:"unk_token"`
	Pad          json.RawMessage `json:"pad_token"`
	Sep          json.RawMessage `json:"sep_token"`
	Mask         json.RawMessage `json:"mask_token"`
	AddBOS       *bool           `json:"add_bos_token"`
	AddEOS       *bool           `json:"add_eos_token"`
	ChatTemplate json.RawMessage `json:"chat_template"`
	// TokenizerClass decides whether AddBOS/AddEOS mean anything: see
	// hfVocabOf.
	TokenizerClass string `json:"tokenizer_class"`
	// AddedTokensDecoder is where a tokenizer with no tokenizer.json names
	// its specials: id -> content (tiktoken.go).
	AddedTokensDecoder map[string]struct {
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens_decoder"`
}

// hfVocabOf builds the container's tokenizer section out of dir.
//
// nvocab is the model's embedding row count. A tokenizer table shorter than it
// is padded rather than left short: the sampler picks over every logit row, so
// a row with no token behind it must still have a name -- and a row that can
// never win still must not index off the end of the table.
func hfVocabOf(dir hfDir, nvocab uint32) (*jlm.Vocab, error) {
	p := dir.path("tokenizer.json")
	b, err := dir.read("tokenizer.json")
	if err != nil && dir.has("tiktoken.model") {
		return hfTiktokenVocab(dir, nvocab)
	}
	if err != nil {
		// Not optional: a container carries the tokenizer, and one with an
		// empty vocabulary would only fail at the first Encode.
		return nil, fmt.Errorf("convert: %s: a safetensors model keeps its tokenizer "+
			"in tokenizer.json beside the weights, and this one has none: %w", dir, err)
	}
	var tj hfTokenizer
	if err := json.Unmarshal(b, &tj); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", p, err)
	}
	// BPE only, refused by name: a Unigram vocabulary loaded as BPE
	// tokenizes every word differently while looking ordinary.
	if tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("convert: %s: tokenizer model %q is not implemented "+
			"from a tokenizer.json (BPE is)", p, tj.Model.Type)
	}
	var shape struct {
		Normalizer   json.RawMessage `json:"normalizer"`
		PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(b, &shape); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", p, err)
	}
	spm, prefix, err := hfSPMShape(tj.Model.ByteFallback, shape.Normalizer, shape.PreTokenizer)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", p, err)
	}
	v := &jlm.Vocab{Kind: jlm.VocabBPE, BOS: -1, EOS: -1, Unk: -1, Pad: -1, Sep: -1, Mask: -1}
	v.IgnoreMerges = tj.Model.IgnoreMerges
	v.ByteFallback = tj.Model.ByteFallback
	if spm {
		v.Kind, v.AddSpacePrefix = jlm.VocabSPM, prefix
	}

	toks, err := hfTokenTable(p, &tj, nvocab)
	if err != nil {
		return nil, err
	}
	v.Kinds = make([]jlm.TokenKind, len(toks))
	for i := range v.Kinds {
		v.Kinds[i] = jlm.TokenNormal
	}
	for i := uint32(len(tj.Model.Vocab)); i < uint32(len(toks)); i++ {
		v.Kinds[i] = jlm.TokenUnused
	}
	for _, a := range tj.AddedTokens {
		if a.ID < 0 || int(a.ID) >= len(v.Kinds) {
			return nil, fmt.Errorf("convert: %s: added token %q has id %d, outside a %d-entry table",
				p, a.Content, a.ID, len(v.Kinds))
		}
		// An added token past the model's vocabulary (qwen's <|im_end|>)
		// lands in a pad slot, and the slot takes its name so Text(id) and an
		// SPM encoder can find it.
		if int(a.ID) >= len(tj.Model.Vocab) {
			toks[a.ID] = a.Content
		}
		// Control is matched literally and never merged, which is what an added
		// token is for; a non-special added token is user-defined and gets the
		// same literal matching (tok.newBPE collects both).
		if a.Special {
			v.Kinds[a.ID] = jlm.TokenControl
		} else {
			v.Kinds[a.ID] = jlm.TokenUserDefined
		}
		v.Added = append(v.Added, jlm.AddedToken{
			ID: a.ID, Content: a.Content, Special: a.Special,
			LStrip: a.LStrip, RStrip: a.RStrip,
			Normalized: a.Normalized, SingleWord: a.SingleWord,
		})
	}
	v.Tokens = jlm.NewStrings(toks)

	merges, err := hfMerges(p, tj.Model.Merges)
	if err != nil {
		return nil, err
	}
	if spm {
		hfSPMKinds(v, int(len(tj.Model.Vocab)))
		v.Scores = hfSPMScores(toks, merges)
	} else {
		v.Merges = jlm.NewMerges(merges)
		// The pipeline comes from the artefact through the same parser
		// scripts/gentok runs; an op it cannot build is a refusal naming it.
		ops, err := pretok.ParsePreTokenizer(strings.NewReader(string(b)))
		if err != nil {
			return nil, fmt.Errorf("convert: %s: %w", p, err)
		}
		if v.Pre, err = containerPreOps(ops); err != nil {
			return nil, fmt.Errorf("convert: %s: %w", p, err)
		}
	}
	// PreName records provenance only (see format/jlm/vocab.go): here it says the
	// stages were read off the model's own artefact rather than off a name.
	v.PreName = "tokenizer.json"

	// The ids of the special tokens, and whether the tokenizer adds them.
	tc, err := readHFTokenizerConfig(dir)
	if err != nil {
		return nil, err
	}
	index := make(map[string]int32, len(toks))
	for i, s := range toks {
		if _, dup := index[s]; !dup {
			index[s] = int32(i)
		}
	}
	for _, s := range []struct {
		raw json.RawMessage
		dst *int32
	}{
		{tc.BOS, &v.BOS}, {tc.EOS, &v.EOS}, {tc.Unk, &v.Unk},
		{tc.Pad, &v.Pad}, {tc.Sep, &v.Sep}, {tc.Mask, &v.Mask},
	} {
		if content, ok := hfTokenContent(s.raw); ok {
			if id, ok := index[content]; ok {
				*s.dst = id
			}
		}
	}
	if v.Unk < 0 && tj.Model.UnkToken != nil {
		if id, ok := index[*tj.Model.UnkToken]; ok {
			v.Unk = id
		}
	}

	// The post-processor is what actually runs, so it is the base answer: a
	// TemplateProcessing whose single sequence opens with a special token adds a
	// BOS. (On the GGUF side add_bos is a family default, and getting it wrong
	// diverges a few tokens in.) tokenizer_config's explicit key still overrides
	// -- for a class that reads it. The generic PreTrainedTokenizerFast does
	// not: it runs tokenizer.json's post-processor as written, so Hunyuan's
	// "add_bos_token": true adds no BOS under transformers (or llama.cpp), and
	// obeying the key put one in front of every prompt.
	v.AddBOS, v.AddEOS = hfTemplateAdds(tj.PostProcessor)
	generic := tc.TokenizerClass == "PreTrainedTokenizerFast"
	if tc.AddBOS != nil && !generic {
		v.AddBOS = *tc.AddBOS
	}
	if tc.AddEOS != nil && !generic {
		v.AddEOS = *tc.AddEOS
	}

	if v.Templates, err = hfTemplates(dir, tc.ChatTemplate); err != nil {
		return nil, err
	}
	return v, nil
}

// hfSPMShape reports whether a tokenizer.json is sentencepiece BPE in
// HuggingFace's clothing, and whether it prepends the space mark.
//
// The shape is byte fallback, spaces escaped to ▁, and nothing splitting the
// text before the merges. HuggingFace writes it as a normalizer (Prepend "▁",
// Replace " " -> "▁") with no pre-tokenizer (llama, gemma) or as a Metaspace
// pre-tokenizer with split false (mixtral). A Metaspace that splits is a
// different algorithm (no merge crosses a ▁) and any other normalizer
// tokenizes a different string; both are refused. The prefix is the file's,
// not the family's: gemma has no Prepend.
func hfSPMShape(byteFallback bool, norm, pre json.RawMessage) (spm, prefix bool, err error) {
	null := func(r json.RawMessage) bool { return len(r) == 0 || string(r) == "null" }
	if !byteFallback {
		return false, false, nil
	}
	escaped := false
	if !null(pre) {
		var m struct {
			Type           string `json:"type"`
			Replacement    string `json:"replacement"`
			PrependScheme  string `json:"prepend_scheme"`
			AddPrefixSpace *bool  `json:"add_prefix_space"`
			Split          *bool  `json:"split"`
		}
		if err := json.Unmarshal(pre, &m); err != nil {
			return false, false, err
		}
		if m.Type != "Metaspace" {
			return false, false, nil
		}
		if m.Replacement != "\u2581" || m.Split == nil || *m.Split {
			return false, false, fmt.Errorf("a Metaspace pre-tokenizer that splits, or "+
				"replaces with %q, is not implemented", m.Replacement)
		}
		switch {
		case m.PrependScheme == "first" || m.PrependScheme == "always":
			prefix = true
		case m.PrependScheme == "never":
		case m.PrependScheme == "" && m.AddPrefixSpace != nil:
			prefix = *m.AddPrefixSpace
		default:
			return false, false, fmt.Errorf("Metaspace prepend_scheme %q", m.PrependScheme)
		}
		escaped = true
	}
	if !null(norm) {
		var n struct {
			Type        string            `json:"type"`
			Normalizers []json.RawMessage `json:"normalizers"`
		}
		if err := json.Unmarshal(norm, &n); err != nil {
			return false, false, err
		}
		nodes := []json.RawMessage{norm}
		if n.Type == "Sequence" {
			nodes = n.Normalizers
		}
		for _, raw := range nodes {
			var x struct {
				Type    string `json:"type"`
				Prepend string `json:"prepend"`
				Content string `json:"content"`
				Pattern struct {
					String *string `json:"String"`
				} `json:"pattern"`
			}
			if err := json.Unmarshal(raw, &x); err != nil {
				return false, false, err
			}
			switch {
			case x.Type == "Prepend" && x.Prepend == "\u2581":
				prefix = true
			case x.Type == "Replace" && x.Pattern.String != nil && *x.Pattern.String == " " &&
				x.Content == "\u2581":
				escaped = true
			default:
				return false, false, fmt.Errorf("a sentencepiece-style tokenizer with a %q "+
					"normalizer is not implemented", x.Type)
			}
		}
	}
	if !escaped {
		// Byte fallback with nothing that turns a space into ▁ is not a shape
		// any engine here tokenizes; say so rather than guess which one.
		return false, false, fmt.Errorf("byte_fallback is set and nothing escapes spaces " +
			"to \u2581: neither sentencepiece nor byte-level BPE")
	}
	return true, prefix, nil
}

// hfSPMKinds marks the byte tokens, which is what tok's SPM decoder keys on:
// "<0x0A>" is a newline only while its kind says it is a byte.
func hfSPMKinds(v *jlm.Vocab, nvocab int) {
	for i := 0; i < nvocab && i < v.Tokens.Len(); i++ {
		s := v.Tokens.At(i)
		if v.Kinds[i] == jlm.TokenNormal && len(s) == 6 && strings.HasPrefix(s, "<0x") && s[5] == '>' {
			v.Kinds[i] = jlm.TokenByte
		}
	}
}

// hfSPMScores recovers sentencepiece's merge priorities from the merge list.
//
// HuggingFace emits, for every piece, each split of it into two pieces,
// ordered by the piece's score, so score(t) = -(rank of the first merge that
// builds t) reproduces tok's SPM merge order. A token no merge builds gets the
// lowest score.
func hfSPMScores(toks []string, merges []jlm.Merge) []float32 {
	id := make(map[string]int, len(toks))
	for i, t := range toks {
		if _, dup := id[t]; !dup {
			id[t] = i
		}
	}
	sc := make([]float32, len(toks))
	for i := range sc {
		sc[i] = -math.MaxFloat32
	}
	for r, m := range merges {
		if i, ok := id[m.Left+m.Right]; ok && sc[i] == -math.MaxFloat32 {
			sc[i] = -float32(r + 1)
		}
	}
	return sc
}

// hfTokenTable turns {token: id} into the dense table the container stores.
//
// Tokens is indexed by id, so a hole would shift every later token. The length
// is the entry count n (not max(id), which is untrusted input); n entries with
// distinct ids all in [0,n) fill every slot, so these two checks prove density.
func hfTokenTable(path string, tj *hfTokenizer, nvocab uint32) ([]string, error) {
	n := len(tj.Model.Vocab)
	if n == 0 {
		return nil, fmt.Errorf("convert: %s: the tokenizer has no vocabulary", path)
	}
	toks := make([]string, n)
	seen := make([]bool, n)
	for s, id := range tj.Model.Vocab {
		if id < 0 || int(id) >= n {
			return nil, fmt.Errorf("convert: %s: token %q has id %d, outside a %d-entry "+
				"vocabulary, so the table would have a hole and every id above it would shift",
				path, s, id, n)
		}
		if seen[id] {
			return nil, fmt.Errorf("convert: %s: id %d is claimed by two tokens (%q and %q), "+
				"so the table would have a hole", path, id, toks[id], s)
		}
		seen[id], toks[id] = true, s
	}
	// Pad to the embedding, never truncate it: models round the embedding up,
	// and Text(id) must not index off the end for a row with no token. A table
	// longer than the embedding is a real mismatch.
	if nvocab != 0 {
		if uint32(n) > nvocab {
			return nil, fmt.Errorf("convert: %s: %d tokens against %d embedding rows",
				path, n, nvocab)
		}
		for i := uint32(n); i < nvocab; i++ {
			toks = append(toks, fmt.Sprintf("[PAD%d]", i))
		}
	}
	return toks, nil
}

// hfMerges reads the merge table, which ships in two shapes.
//
// Older tokenizers write "left right", newer ones ["left", "right"]; the
// joined form is ambiguous once a half contains a space ("Ġ Ġ"), which is why
// jlm.Merge is a pair. Splitting on the first space is convert/vocab.go's rule.
func hfMerges(path string, raw []json.RawMessage) ([]jlm.Merge, error) {
	out := make([]jlm.Merge, 0, len(raw))
	for i, rm := range raw {
		var pair []string
		if err := json.Unmarshal(rm, &pair); err == nil {
			if len(pair) != 2 {
				return nil, fmt.Errorf("convert: %s: merge %d has %d parts, want 2", path, i, len(pair))
			}
			out = append(out, jlm.Merge{Left: pair[0], Right: pair[1]})
			continue
		}
		var joined string
		if err := json.Unmarshal(rm, &joined); err != nil {
			return nil, fmt.Errorf("convert: %s: merge %d is neither a pair nor a string", path, i)
		}
		sp := strings.IndexByte(joined, ' ')
		if sp <= 0 {
			return nil, fmt.Errorf("convert: %s: merge %d (%q) has no left half", path, i, joined)
		}
		out = append(out, jlm.Merge{Left: joined[:sp], Right: joined[sp+1:]})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("convert: %s: a byte-level BPE tokenizer with no merges", path)
	}
	return out, nil
}

func readHFTokenizerConfig(dir hfDir) (*hfTokenizerConfig, error) {
	tc := &hfTokenizerConfig{}
	b, err := dir.read("tokenizer_config.json")
	if err != nil {
		// Optional: tokenizer.json alone is a complete tokenizer. What is lost
		// is the special-token names and the chat template.
		return tc, nil
	}
	if err := json.Unmarshal(b, tc); err != nil {
		return nil, fmt.Errorf("convert: %s/tokenizer_config.json: %w", dir, err)
	}
	return tc, nil
}

// hfTokenContent reads a special-token field, which is a bare string in most
// files and an AddedToken object in the rest.
func hfTokenContent(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, s != ""
	}
	var o struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &o); err == nil && o.Content != "" {
		return o.Content, true
	}
	return "", false
}

// hfTemplateAdds reads a post_processor and reports whether it prepends and
// appends a special token to a single sequence.
//
// It walks a Sequence of processors, because a ByteLevel post-processor sits
// beside the TemplateProcessing in most files and only the second one adds
// anything.
func hfTemplateAdds(raw json.RawMessage) (bos, eos bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return false, false
	}
	var node any
	if err := json.Unmarshal(raw, &node); err != nil {
		return false, false
	}
	var walk func(any) (bool, bool)
	walk = func(n any) (bool, bool) {
		m, ok := n.(map[string]any)
		if !ok {
			return false, false
		}
		switch m["type"] {
		case "Sequence":
			b, e := false, false
			if lst, ok := m["processors"].([]any); ok {
				for _, p := range lst {
					pb, pe := walk(p)
					b, e = b || pb, e || pe
				}
			}
			return b, e
		case "TemplateProcessing":
			single, ok := m["single"].([]any)
			if !ok || len(single) == 0 {
				return false, false
			}
			isSpecial := func(v any) bool {
				e, ok := v.(map[string]any)
				if !ok {
					return false
				}
				_, ok = e["SpecialToken"]
				return ok
			}
			return isSpecial(single[0]), len(single) > 1 && isSpecial(single[len(single)-1])
		}
		return false, false
	}
	return walk(node)
}

// hfTemplates reads chat_template, which is a string in most files and a list
// of named templates in the rest.
//
// A GGUF has room for exactly one unnamed template, so a model shipping a
// default and a tool-calling variant keeps both only through this path.
func hfTemplates(dir hfDir, raw json.RawMessage) ([]jlm.ChatTemplate, error) {
	if len(raw) == 0 || string(raw) == "null" {
		// Some repos keep it in its own file instead.
		if b, err := dir.read("chat_template.jinja"); err == nil && len(b) > 0 {
			return []jlm.ChatTemplate{{Body: string(b)}}, nil
		}
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return nil, nil
		}
		return []jlm.ChatTemplate{{Body: s}}, nil
	}
	var lst []struct {
		Name     string `json:"name"`
		Template string `json:"template"`
	}
	if err := json.Unmarshal(raw, &lst); err != nil {
		return nil, fmt.Errorf("convert: %s: chat_template is neither a string nor a named list", dir)
	}
	out := make([]jlm.ChatTemplate, 0, len(lst))
	for _, t := range lst {
		out = append(out, jlm.ChatTemplate{Name: t.Name, Body: t.Template})
	}
	// Deterministic: a map-ordered template list would make two conversions of
	// one directory produce two different containers.
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// hfStops is the stop set transformers' generate ends on besides the
// tokenizer's EOS: generation_config.json's eos_token_id, one id or a list
// (HunyuanOCR's [120007, 120020]: <｜hy_Assistant｜> closes its turn, and the
// tokenizer's eos is the other). With no generation_config.json, transformers
// builds one from config.json, whose eos_token_id a vision-language model
// keeps under text_config.
func hfStops(dir hfDir, v *jlm.Vocab) ([]int32, error) {
	name := "generation_config.json"
	b, err := dir.read(name)
	if err != nil {
		name = "config.json"
		if b, err = dir.read(name); err != nil {
			return nil, nil
		}
	}
	var c struct {
		EOS        json.RawMessage `json:"eos_token_id"`
		TextConfig *struct {
			EOS json.RawMessage `json:"eos_token_id"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", dir.path(name), err)
	}
	null := func(r json.RawMessage) bool { return len(r) == 0 || string(r) == "null" }
	raw := c.EOS
	if null(raw) && c.TextConfig != nil {
		raw = c.TextConfig.EOS
	}
	if null(raw) {
		return nil, nil
	}
	var ids []int64
	if err := json.Unmarshal(raw, &ids); err != nil {
		var one int64
		if err := json.Unmarshal(raw, &one); err != nil {
			return nil, fmt.Errorf("convert: %s: eos_token_id %s is neither an id nor a list of ids",
				dir.path(name), raw)
		}
		ids = []int64{one}
	}
	var stop []int32
	for _, id := range ids {
		if id < 0 || id >= int64(v.Tokens.Len()) {
			return nil, fmt.Errorf("convert: %s: eos_token_id %d is outside a %d-token vocabulary",
				dir.path(name), id, v.Tokens.Len())
		}
		stop = addStop(stop, v.EOS, int32(id))
	}
	return stop, nil
}

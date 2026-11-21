package convert

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok"
	"github.com/jitllm/jitllm/tok/pretok"
)

// The source's tokenizer keys, and the only place they appear. A container
// carries jlm.Vocab: the token table, the merge pairs, the special ids and the
// pre-tokenizer ops. The source stores only a pre-tokenizer name, which every
// engine resolves through a hand-maintained table; resolving it here, once,
// means a missing name is a conversion-time question, IgnoreMerges and
// ByteFallback are stored rather than inferred per load, and PreName (what the
// source claimed) sits beside Pre (what will run).
func vocabOf(f *meta.File) (*jlm.Vocab, error) { return VocabOf(f) }

// VocabOf is vocabOf, exported so a test that needs a container's tokenizer
// section without writing a container can build one.
func VocabOf(f *meta.File) (*jlm.Vocab, error) {
	model, _ := f.KV["tokenizer.ggml.model"].String()
	if model == "" {
		return nil, nil // no tokenizer in this file; a tower, say
	}
	tms, err := tokenizerModels()
	if err != nil {
		return nil, err
	}
	tm, ok := tms[model]
	if !ok {
		return nil, fmt.Errorf("convert: tokenizer model %q is not one this container defines", model)
	}
	v := &jlm.Vocab{
		BOS: -1, EOS: -1, Unk: -1, Pad: -1, Sep: -1, Mask: -1,
		Kind: vocabKinds()[tm.Kind], ByteFallback: tm.ByteFallback, AddSpacePrefix: tm.SpacePrefix, AddBOS: tm.AddBOS,
	}

	toks, ok := f.KV["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("convert: a tokenizer with no token list")
	}
	v.Tokens = make([]string, 0, toks.Len())
	if err := toks.EachString(func(i int, s []byte) bool {
		v.Tokens = append(v.Tokens, string(s))
		return true
	}); err != nil {
		return nil, err
	}
	if sc, err := f.KV["tokenizer.ggml.scores"].Float32s(); err == nil {
		v.Scores = sc
	}
	if ty, err := f.KV["tokenizer.ggml.token_type"].Int32s(); err == nil {
		v.Kinds = make([]jlm.TokenKind, len(ty))
		for i, t := range ty {
			// The source numbers these 1..6 in the same order; anything else
			// becomes Normal rather than zero, which would read as unset.
			if t >= 1 && t <= 6 {
				v.Kinds[i] = jlm.TokenKind(t)
			} else {
				v.Kinds[i] = jlm.TokenNormal
			}
		}
	}
	for _, s := range []struct {
		key string
		dst *int32
	}{
		{"tokenizer.ggml.bos_token_id", &v.BOS},
		{"tokenizer.ggml.eos_token_id", &v.EOS},
		{"tokenizer.ggml.unknown_token_id", &v.Unk},
		{"tokenizer.ggml.padding_token_id", &v.Pad},
		{"tokenizer.ggml.seperator_token_id", &v.Sep}, // sic: the source spells it this way
		{"tokenizer.ggml.mask_token_id", &v.Mask},
	} {
		if n, ok := f.KV[s.key].Uint(); ok {
			*s.dst = int32(n)
		}
	}
	// The ids the file states end a generation besides EOS: its end of turn
	// and end of message. A GGUF has no key for generation_config.json's
	// eos_token_id list; HunyuanOCR's <｜hy_Assistant｜> reaches its file only
	// as eot_token_id, and the rest of a family's list is its library
	// statement's stop pieces.
	for _, k := range []string{"eot", "eom"} {
		n, ok := f.KV["tokenizer.ggml."+k+"_token_id"].Uint()
		if !ok {
			continue
		}
		if n >= uint64(len(v.Tokens)) {
			return nil, fmt.Errorf("convert: tokenizer.ggml.%s_token_id %d is outside a %d-token vocabulary",
				k, n, len(v.Tokens))
		}
		v.Stop = addStop(v.Stop, v.EOS, int32(n))
	}
	if v.Stop, err = statedStops(f.Arch(), v); err != nil {
		return nil, err
	}
	for _, s := range []struct {
		key string
		dst *bool
	}{
		{"tokenizer.ggml.add_bos_token", &v.AddBOS},
		{"tokenizer.ggml.add_eos_token", &v.AddEOS},
		{"tokenizer.ggml.add_space_prefix", &v.AddSpacePrefix},
	} {
		if n, ok := f.KV[s.key].Uint(); ok {
			*s.dst = n != 0
		}
	}

	// Merges become pairs here, split on the first space: a merge whose left
	// half is the space character is written "Ġ Ġ".
	if m, ok := f.KV["tokenizer.ggml.merges"]; ok {
		v.Merges = make([]jlm.Merge, 0, m.Len())
		if err := m.EachString(func(i int, s []byte) bool {
			str := string(s)
			if sp := strings.IndexByte(str, ' '); sp > 0 {
				v.Merges = append(v.Merges, jlm.Merge{Left: str[:sp], Right: str[sp+1:]})
			}
			return true
		}); err != nil {
			return nil, err
		}
	}

	v.Templates = ggufTemplates(f.KV)

	if v.Kind == jlm.VocabWPM {
		return wordPiece(f, v, tm.Pre), nil
	}
	if v.Kind != jlm.VocabBPE {
		return v, nil
	}
	v.PreName, _ = f.KV["tokenizer.ggml.pre"].String()
	if v.PreName == "" {
		v.PreName = "default"
	}
	// A file naming no pre-tokenizer of its own ("default") splits with the
	// one the library states for its architecture (a Llama 4 GGUF that says
	// "default", where its tokenizer.json splits with the o200k pattern,
	// RULE 7m); a real name is kept.
	if v.PreName == "default" {
		st, _, err := ggufStates(f.Arch())
		if err != nil {
			return nil, err
		}
		if st != nil && st.PreTokenizer != "" {
			v.PreName = st.PreTokenizer
		}
	}
	// A name with no entry is not an error here (the legacy family splitters
	// cover several), but it is recorded as an empty pipeline, so the reader
	// can tell "no stages" from "stages nobody verified".
	ops, ok := pretok.PreOps(v.PreName)
	if !ok && v.PreName == "default" {
		ops, ok = pretok.DefaultBPE(), true
	}
	if ok {
		pre, err := containerPreOps(ops)
		if err != nil {
			return nil, fmt.Errorf("pre-tokenizer %q: %w", v.PreName, err)
		}
		v.Pre = pre
	}
	// AddBOS and IgnoreMerges default from the pre-tokenizer's entry in
	// pretokenizers.json and are stored as properties of the tokenizer rather
	// than looked up by name at load. The add_bos_token key overrides the
	// first; it is often absent (Llama-3.2 ships none and wants its BOS, qwen3
	// ships an explicit 0), and a wrong BOS is a fluent divergence a few
	// tokens in.
	pres, err := preTokenizers()
	if err != nil {
		return nil, err
	}
	p := pres[v.PreName]
	v.AddBOS, v.IgnoreMerges = p.AddBOS, p.IgnoreMerges
	if n, ok := f.KV["tokenizer.ggml.add_bos_token"].Uint(); ok {
		v.AddBOS = n != 0
	}
	return v, nil
}

// containerPreOps translates a parsed pipeline into the container's own stage
// codes.
//
// It is the one translation for both input formats: the GGUF path gets its
// []pretok.Op through pretok.Table() by name, the safetensors path from
// pretok.ParsePreTokenizer. A stage with no container code is a refusal rather
// than routed to the nearest-looking one, and so is a stage the tokenizer
// could not build, a pattern that does not compile among them.
func containerPreOps(ops []pretok.Op) ([]jlm.PreOp, error) {
	if len(ops) > 0 {
		if err := tok.CheckPreOps(ops); err != nil {
			return nil, fmt.Errorf("convert: %w", err)
		}
	}
	out := make([]jlm.PreOp, 0, len(ops))
	for _, o := range ops {
		p := jlm.PreOp{
			UseRegex: o.UseRegex, AddPrefixSpace: o.AddPrefixSpace,
			Individual: o.Individual, Behavior: o.Behavior,
			Pattern: o.Pattern, Invert: o.Invert,
			Contractions: o.Split.Contractions, CaseFold: o.Split.CaseFold,
			CaseRuns: o.Split.CaseRuns, SuffixContract: o.Split.SuffixContract,
			PunctSlash: o.Split.PunctSlash, HanRuns: o.Split.HanRuns, PunctNoNewline: o.Split.PunctNoNewline,
			LetterMarks: o.Split.LetterMarks, DigitGroup: uint8(o.Split.DigitGroup),
		}
		switch o.Kind {
		case pretok.OpSplit:
			p.Kind = jlm.PreSplit
		case pretok.OpByteLevel:
			p.Kind = jlm.PreByteLevel
		case pretok.OpDigits:
			p.Kind = jlm.PreDigits
		case pretok.OpPunctuation:
			p.Kind = jlm.PrePunctuation
		case pretok.OpDigitTriples:
			p.Kind = jlm.PreDigitTriples
		case pretok.OpPunctDefault:
			p.Kind = jlm.PrePunctDefault
		case pretok.OpDigitGroups:
			p.Kind = jlm.PreDigitGroups
		case pretok.OpKanaHanRuns:
			p.Kind = jlm.PreKanaHanRuns
		case pretok.OpHunyuanSplit:
			p.Kind = jlm.PreHunyuanSplit
		case pretok.OpSparkSplit:
			p.Kind = jlm.PreSparkSplit
		case pretok.OpPattern:
			p.Kind = jlm.PrePattern
		default:
			return nil, fmt.Errorf("convert: stage %v is one this container does not define", o.Kind)
		}
		out = append(out, p)
	}
	return out, nil
}

// preTokenizersJSON is each pre-tokenizer name's add_bos and ignore_merges
// where they are not false: llama3's regex group (the pre names that share its
// pattern) and Llama 4, whose own tokenizer.json says ignore_merges: true and
// prepends <|begin_of_text|> (RULE 7m). "default" is not in it: an absent or
// "default" name gets its own pipeline with neither boolean
// (pretok.DefaultBPE()). It is resolved at conversion; nothing on the
// load path consults it.
//
//go:embed pretokenizers.json
var preTokenizersJSON []byte

// preTokenizer is a pre-tokenizer name's two booleans.
type preTokenizer struct {
	AddBOS       bool `json:"add_bos"`
	IgnoreMerges bool `json:"ignore_merges"`
}

//go:embed tokenizers.json
var tokenizerModelsJSON []byte

// tokenizerModel is what a GGUF's tokenizer.ggml.model states
// (tokenizers.json): the container's vocabulary kind (vocabKinds), whether
// a piece no token covers falls back to byte tokens, the space prefix and the
// BOS the file's own keys override, and the pre-tokenizer a WordPiece
// vocabulary names (it runs one stage, BertPreTokenizer's).
type tokenizerModel struct {
	Kind         string `json:"kind"`
	ByteFallback bool   `json:"byte_fallback"`
	SpacePrefix  bool   `json:"add_space_prefix"`
	AddBOS       bool   `json:"add_bos"`
	Pre          string `json:"pre"`
}

// vocabKinds is each vocabulary kind a tokenizer model states, by its name.
func vocabKinds() map[string]jlm.VocabKind {
	return map[string]jlm.VocabKind{"spm": jlm.VocabSPM, "bpe": jlm.VocabBPE, "wpm": jlm.VocabWPM, "bpe_spm": jlm.VocabBPESPM}
}

// tokenizerModels is tokenizers.json, by model (tokenizerModelsOf).
func tokenizerModels() (map[string]tokenizerModel, error) {
	return tokenizerModelsOf(tokenizerModelsJSON)
}

// tokenizerModelsOf is a tokenizers.json, by model, each of a kind vocabKinds
// names and a WordPiece one naming its pre-tokenizer.
func tokenizerModelsOf(b []byte) (map[string]tokenizerModel, error) {
	var out map[string]tokenizerModel
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&out); err != nil || len(out) == 0 {
		return nil, fmt.Errorf("convert: tokenizers.json: %v, %d models", err, len(out))
	}
	for name, tm := range out {
		k, ok := vocabKinds()[tm.Kind]
		if !ok || (k == jlm.VocabWPM) != (tm.Pre != "") {
			return nil, fmt.Errorf("convert: tokenizers.json: %s is of kind %q with pre-tokenizer %q", name, tm.Kind, tm.Pre)
		}
	}
	return out, nil
}

// preTokenizers is pretokenizers.json, by name.
func preTokenizers() (map[string]preTokenizer, error) {
	var out map[string]preTokenizer
	if err := json.Unmarshal(preTokenizersJSON, &out); err != nil || len(out) == 0 {
		return nil, fmt.Errorf("convert: pretokenizers.json: %v, %d names", err, len(out))
	}
	return out, nil
}

// wordPiece resolves a WordPiece vocabulary's special ids and its one
// pre-tokenizer stage.
//
// A BERT file's [CLS] opens the sequence and [SEP] closes it. Files spell
// them two ways (bos/eos_token_id, or only cls/seperator_token_id), and
// reading one spelling leaves the other with no opener, which a CLS-pooled
// model reads its embedding from. Both are added, as BertProcessing does.
func wordPiece(f *meta.File, v *jlm.Vocab, pre string) *jlm.Vocab {
	if n, ok := f.KV["tokenizer.ggml.cls_token_id"].Uint(); ok && v.BOS < 0 {
		v.BOS = int32(n)
	}
	if v.EOS < 0 {
		v.EOS = v.Sep
	}
	if v.Sep < 0 {
		v.Sep = v.EOS
	}
	v.AddBOS, v.AddEOS, v.AddSpacePrefix = true, true, false
	// BertNormalizer's two options. Absent keys mean an uncased model
	// (BertNormalizer's default), and strip_accents follows lowercase unless the
	// file says otherwise, as in HuggingFace.
	lower := true
	if n, ok := f.KV["tokenizer.ggml.normalizer.lowercase"].Uint(); ok {
		lower = n != 0
	}
	strip := lower
	if n, ok := f.KV["tokenizer.ggml.normalizer.strip_accents"].Uint(); ok {
		strip = n != 0
	}
	v.Pre = []jlm.PreOp{{Kind: jlm.PreBert, CaseFold: lower, LetterMarks: !strip}}
	v.PreName = pre
	return v
}

// addStop appends id to a stop list unless it is already there or is EOS,
// which every generation loop stops on anyway.
func addStop(stop []int32, eos, id int32) []int32 {
	if id == eos || slices.Contains(stop, id) {
		return stop
	}
	return append(stop, id)
}

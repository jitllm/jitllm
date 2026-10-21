package convert

import (
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
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
	v := &jlm.Vocab{
		BOS: -1, EOS: -1, Unk: -1, Pad: -1, Sep: -1, Mask: -1,
	}
	switch model {
	case "llama":
		v.Kind, v.ByteFallback, v.AddSpacePrefix, v.AddBOS = jlm.VocabSPM, true, true, true
	case "gpt2":
		v.Kind = jlm.VocabBPE
	case "bert", "wordpiece":
		v.Kind = jlm.VocabWPM
	case "gemma4":
		// SentencePiece-style BPE (llama.cpp's LLAMA_VOCAB_TYPE_BPE under the
		// gemma4 pre-type): merges, no byte-level alphabet, byte fallback.
		// add_space_prefix and add_bos are read below, as the file states them.
		v.Kind, v.ByteFallback = jlm.VocabBPESPM, true
	case "":
		return nil, nil // no tokenizer in this file; a tower, say
	default:
		return nil, fmt.Errorf("convert: tokenizer model %q is not one this container defines", model)
	}

	toks, ok := f.KV["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("convert: a tokenizer with no token list")
	}
	tokens := make([]string, 0, toks.Len())
	if err := toks.EachString(func(i int, s []byte) bool {
		tokens = append(tokens, string(s))
		return true
	}); err != nil {
		return nil, err
	}
	v.Tokens = jlm.NewStrings(tokens)
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
	// The ids the file states end a generation besides EOS: llama.cpp's
	// special_eot_id and special_eom_id, and the FIM pad, repo and file
	// separator it also stops on (llama-vocab.cpp, special_eog_ids). A GGUF has
	// no list key; HunyuanOCR's <｜hy_Assistant｜> reaches its file only as
	// eot_token_id.
	for _, k := range []string{"eot", "eom", "fim_pad", "fim_rep", "fim_sep"} {
		n, ok := f.KV["tokenizer.ggml."+k+"_token_id"].Uint()
		if !ok {
			continue
		}
		if n >= uint64(v.Tokens.Len()) {
			return nil, fmt.Errorf("convert: tokenizer.ggml.%s_token_id %d is outside a %d-token vocabulary",
				k, n, v.Tokens.Len())
		}
		v.Stop = addStop(v.Stop, v.EOS, int32(n))
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
		merges := make([]jlm.Merge, 0, m.Len())
		if err := m.EachString(func(i int, s []byte) bool {
			str := string(s)
			if sp := strings.IndexByte(str, ' '); sp > 0 {
				merges = append(merges, jlm.Merge{Left: str[:sp], Right: str[sp+1:]})
			}
			return true
		}); err != nil {
			return nil, err
		}
		v.Merges = jlm.NewMerges(merges)
	}

	v.Templates = ggufTemplates(f.KV)

	if v.Kind == jlm.VocabWPM {
		return wordPiece(f, v), nil
	}
	if v.Kind != jlm.VocabBPE {
		return v, nil
	}
	v.PreName, _ = f.KV["tokenizer.ggml.pre"].String()
	if v.PreName == "" {
		v.PreName = "default"
	}
	// Ollama's Llama 4 says "default", which is llama3's split. The model's
	// own tokenizer.json (RULE 7m) splits with the o200k pattern, which
	// llama.cpp's converter names "llama4", so a llama4 vocabulary carrying the
	// fallback name gets that pipeline; a real name is kept.
	if v.PreName == "default" && f.Arch() == "llama4" {
		v.PreName = "llama4"
	}
	// A name with no entry is not an error here (the legacy family splitters
	// cover several), but it is recorded as an empty pipeline, so the reader
	// can tell "no stages" from "stages nobody verified".
	ops, ok := pretok.PreOps(v.PreName)
	if !ok && v.PreName == "default" {
		ops, ok = pretok.LlamaCppDefault, true
	}
	if ok {
		pre, err := containerPreOps(ops)
		if err != nil {
			return nil, fmt.Errorf("pre-tokenizer %q: %w", v.PreName, err)
		}
		v.Pre = pre
	}
	// AddBOS and IgnoreMerges default from the llama3 family and are stored as
	// properties of the tokenizer rather than looked up by name at load. The
	// add_bos_token key overrides the first; it is often absent (Llama-3.2 ships
	// none and wants its BOS, qwen3 ships an explicit 0), and a wrong BOS is a
	// fluent divergence a few tokens in.
	v.AddBOS = llama3Family[v.PreName]
	v.IgnoreMerges = llama3Family[v.PreName]
	// Llama 4 is not in that family but shares both booleans: its own
	// tokenizer.json says ignore_merges: true and prepends <|begin_of_text|>,
	// where llama.cpp's gpt-4o branch sets ignore_merges false (RULE 7m).
	if v.PreName == "llama4" {
		v.AddBOS, v.IgnoreMerges = true, true
	}
	if n, ok := f.KV["tokenizer.ggml.add_bos_token"].Uint(); ok {
		v.AddBOS = n != 0
	}
	return v, nil
}

// containerPreOps translates a parsed pipeline into the container's own stage
// codes.
//
// It is the one translation for both input formats: the GGUF path gets its
// []pretok.Op through pretok.Table by name, the safetensors path from
// pretok.ParsePreTokenizer. A stage with no container code is a refusal rather
// than routed to the nearest-looking one.
func containerPreOps(ops []pretok.Op) ([]jlm.PreOp, error) {
	out := make([]jlm.PreOp, 0, len(ops))
	for _, o := range ops {
		p := jlm.PreOp{
			UseRegex: o.UseRegex, AddPrefixSpace: o.AddPrefixSpace,
			Individual: o.Individual, Behavior: o.Behavior,
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
		default:
			return nil, fmt.Errorf("convert: stage %v is one this container does not define", o.Kind)
		}
		out = append(out, p)
	}
	return out, nil
}

// llama3Family is the pre-tokenizer group that takes the llama3 regex, and with
// it add_bos and ignore_merges (llama.cpp's list, name for name). "default" is
// not in it: llama.cpp gives an absent or "default" name its own pipeline with
// neither boolean (pretok.LlamaCppDefault). This source-format heuristic is
// resolved at conversion; nothing on the load path consults it.
var llama3Family = map[string]bool{
	"llama3": true, "llama-v3": true, "llama-bpe": true,
	"falcon3": true, "falcon-h1": true, "pixtral": true,
	"midm-2.0": true, "lfm2": true, "jina-v5-nano": true,
}

// wordPiece resolves a WordPiece vocabulary's special ids and its one
// pre-tokenizer stage.
//
// A BERT file's [CLS] opens the sequence and [SEP] closes it. Files spell
// them two ways (bos/eos_token_id, or only cls/seperator_token_id), and
// reading one spelling leaves the other with no opener, which a CLS-pooled
// model reads its embedding from. Both are added, as llama.cpp does for every
// WPM vocabulary and as BertProcessing does.
func wordPiece(f *meta.File, v *jlm.Vocab) *jlm.Vocab {
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
	// (llama.cpp's default), and strip_accents follows lowercase unless the
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
	v.PreName = "bert"
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

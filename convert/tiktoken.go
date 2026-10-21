package convert

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/pretok"
)

// A tiktoken vocabulary is the third shape a tokenizer arrives in, and the one
// that carries the least: tiktoken.model is base64 bytes and a rank per line.
// Moonshot's models ship it with no tokenizer.json: the pre-tokenizer is a
// pat_str in Python source, and the merges are implicit in the ranks.

// tiktokenPre names the pre-tokenizer a tiktoken vocabulary was trained with,
// keyed by the vocabulary file's sha256.
//
// It is keyed by hash because the file carries no name and the pattern lives
// in Python source (tokenization_kimi.py's pat_str): one vocabulary identity
// pairs with one pattern (RULE 7m). An unlisted hash is refused, naming it.
var tiktokenPre = map[string]string{
	// Kimi-K2, Moonlight-16B-A3B and Kimi-Linear-48B-A3B share one file,
	// checked against the Hub's x-linked-etag for all three.
	"b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103": "kimi-k2",
}

// hfTiktokenVocab builds the container's tokenizer from tiktoken.model and
// tokenizer_config.json.
func hfTiktokenVocab(dir hfDir, nvocab uint32) (*jlm.Vocab, error) {
	p := dir.path("tiktoken.model")
	raw, err := dir.read("tiktoken.model")
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(raw)
	pre, ok := tiktokenPre[hex.EncodeToString(sum[:])]
	if !ok {
		return nil, fmt.Errorf("convert: %s (sha256 %x) is a tiktoken vocabulary this "+
			"converter has not paired with a pre-tokenizer; its pattern lives in the "+
			"model's Python tokenizer and must be identified before it is used", p, sum)
	}
	ranks, byRank, err := readTiktoken(p, raw)
	if err != nil {
		return nil, err
	}
	if uint32(len(byRank)) > nvocab {
		return nil, fmt.Errorf("convert: %s has %d tokens for a %d-row embedding", p, len(byRank), nvocab)
	}

	v := &jlm.Vocab{Kind: jlm.VocabBPE, BOS: -1, EOS: -1, Unk: -1, Pad: -1, Sep: -1, Mask: -1,
		// tiktoken looks the whole piece up before merging (byte_pair_encode),
		// which is what ignore_merges means. On Kimi's file it changes nothing,
		// but it is the reference's algorithm.
		IgnoreMerges: true, PreName: "tiktoken.model"}
	ops, ok := pretok.PreOps(pre)
	if !ok {
		return nil, fmt.Errorf("convert: pre-tokenizer %q is not in the table", pre)
	}
	if v.Pre, err = containerPreOps(ops); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", p, err)
	}
	tokens := make([]string, nvocab)
	v.Kinds = make([]jlm.TokenKind, nvocab)
	for r, t := range byRank {
		tokens[r], v.Kinds[r] = pretok.ByteLevel(t), jlm.TokenNormal
	}

	// The merges are recovered from the ranks, as llama.cpp's converter does
	// (QwenModel.bpe): a token of rank r is the merge of the two parts that
	// BPE with every rank below r leaves it in, and only two can be left.
	var merges []jlm.Merge
	for r, t := range byRank {
		if len(t) < 2 {
			continue
		}
		parts := bpeParts(ranks, t, r)
		if len(parts) != 2 {
			return nil, fmt.Errorf("convert: %s: token %d (%q) is not the merge of two "+
				"lower-ranked tokens; BPE with ranks below it leaves %d parts", p, r, t, len(parts))
		}
		merges = append(merges, jlm.Merge{Left: pretok.ByteLevel(parts[0]), Right: pretok.ByteLevel(parts[1])})
	}
	v.Merges = jlm.NewMerges(merges)

	tc, err := readHFTokenizerConfig(dir)
	if err != nil {
		return nil, err
	}
	// The specials fill every row past the base vocabulary, named from
	// added_tokens_decoder and "<|reserved_token_N|>" elsewhere -- which is
	// what tokenization_kimi.py builds -- and tiktoken matches every one of
	// them literally in text (allowed_special="all").
	for id := uint32(len(byRank)); id < nvocab; id++ {
		content := fmt.Sprintf("<|reserved_token_%d|>", id)
		if a, ok := tc.AddedTokensDecoder[strconv.Itoa(int(id))]; ok {
			content = a.Content
		}
		tokens[id], v.Kinds[id] = content, jlm.TokenControl
		v.Added = append(v.Added, jlm.AddedToken{ID: int32(id), Content: content, Special: true})
	}
	v.Tokens = jlm.NewStrings(tokens)
	// tokenization_kimi.py reads added_tokens_decoder only for its 256
	// reserved rows after the base vocabulary. An entry past those and past
	// the embedding names a row neither the tokenizer nor the model has --
	// transformers appends its additional_special_tokens there when it saves
	// the tokenizer (Kimi-K3-0.40B's <|im_end|> at 163840 of 163840 rows) --
	// so it names nothing and is passed over. One the reference would make a
	// special without an embedding row behind it is still refused.
	const tiktokenReserved = 256 // TikTokenTokenizer.num_reserved_special_tokens
	for k := range tc.AddedTokensDecoder {
		id, err := strconv.Atoi(k)
		if err == nil && uint32(id) >= nvocab && id >= len(byRank)+tiktokenReserved {
			continue
		}
		if err != nil || id < len(byRank) || uint32(id) >= nvocab {
			return nil, fmt.Errorf("convert: tokenizer_config.json names special %q, "+
				"outside rows %d..%d", k, len(byRank), nvocab-1)
		}
	}
	index := make(map[string]int32, len(v.Added))
	for _, a := range v.Added {
		index[a.Content] = a.ID
	}
	for _, s := range []struct {
		raw []byte
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
	// tokenization_kimi.py's encode adds nothing; a key in the config wins.
	if tc.AddBOS != nil {
		v.AddBOS = *tc.AddBOS
	}
	if tc.AddEOS != nil {
		v.AddEOS = *tc.AddEOS
	}
	if v.Templates, err = hfTemplates(dir, tc.ChatTemplate); err != nil {
		return nil, err
	}
	return v, nil
}

// readTiktoken parses "base64 rank" lines. Ranks must be exactly 0..n-1:
// a gap is a row with no token, and a repeat is two tokens claiming one id.
func readTiktoken(p string, raw []byte) (map[string]int, []string, error) {
	ranks := map[string]int{}
	var byRank []string
	sc := bufio.NewScanner(bytes.NewReader(raw))
	for line := 1; sc.Scan(); line++ {
		f := strings.Fields(sc.Text())
		if len(f) == 0 {
			continue
		}
		if len(f) != 2 {
			return nil, nil, fmt.Errorf("convert: %s:%d: want \"base64 rank\"", p, line)
		}
		b, err := base64.StdEncoding.DecodeString(f[0])
		if err != nil {
			return nil, nil, fmt.Errorf("convert: %s:%d: %w", p, line, err)
		}
		r, err := strconv.Atoi(f[1])
		if err != nil || r != len(byRank) {
			return nil, nil, fmt.Errorf("convert: %s:%d: rank %q, want %d", p, line, f[1], len(byRank))
		}
		if _, dup := ranks[string(b)]; dup {
			return nil, nil, fmt.Errorf("convert: %s:%d: token %q twice", p, line, b)
		}
		ranks[string(b)] = r
		byRank = append(byRank, string(b))
	}
	return ranks, byRank, sc.Err()
}

// bpeParts byte-pair encodes t using only the ranks below maxRank.
func bpeParts(ranks map[string]int, t string, maxRank int) []string {
	parts := make([]string, len(t))
	for i := 0; i < len(t); i++ { // bytes, not runes: a token is a byte string
		parts[i] = t[i : i+1]
	}
	for {
		best, bestRank := -1, maxRank
		for i := 0; i+1 < len(parts); i++ {
			if r, ok := ranks[parts[i]+parts[i+1]]; ok && r < bestRank {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			return parts
		}
		parts[best] += parts[best+1]
		parts = append(parts[:best+1], parts[best+2:]...)
	}
}

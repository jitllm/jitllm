package server

import (
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/tok"
)

// *Engine is the LlamaBackend of the llama.cpp routes (compat_llamacpp.go).
var _ LlamaBackend = (*Engine)(nil)

// fimNames are the spellings of each fill-in-the-middle token the vocabularies
// here carry.
var fimNames = struct{ pre, suf, mid, repo, sep []string }{
	pre:  []string{"<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<PRE>", "▁<PRE>", "<｜fim▁begin｜>"},
	suf:  []string{"<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<SUF>", "▁<SUF>", "<｜fim▁hole｜>"},
	mid:  []string{"<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<MID>", "▁<MID>", "<｜fim▁end｜>"},
	repo: []string{"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"},
	sep:  []string{"<|file_sep|>", "<fim-sep>", "<SEP>"},
}

func vocabID(v *tok.Vocab, names []string) int32 {
	for _, n := range names {
		if id, ok := v.ID(n); ok {
			return id
		}
	}
	return -1
}

func (e *Engine) llamaVocab(name string) (*LoadedModel, *tok.Vocab, error) {
	lm, err := e.resolveModel(name)
	if err != nil {
		return nil, nil, err
	}
	if lm.m.Vocab == nil {
		return nil, nil, fmt.Errorf("%w: model %q has no tokenizer (%v)", ErrInvalid, lm.id, lm.m.TokErr)
	}
	return lm, lm.m.Vocab, nil
}

func (e *Engine) LlamaModel(name string) (LlamaModelInfo, error) {
	lm, err := e.resolveModel(name)
	if err != nil {
		return LlamaModelInfo{}, err
	}
	info := LlamaModelInfo{ID: lm.id, Name: lm.name, Path: lm.path, NCtx: e.defaultMaxSeq(lm),
		Vision: lm.m.Tower() != nil, Embedding: lm.m.IsEmbedding(), Version: e.cfg.Version}
	info.ChatTemplate, _ = lm.m.ChatTemplate("")
	if v := lm.m.Vocab; v != nil {
		f := FIMTokens{Pre: vocabID(v, fimNames.pre), Suf: vocabID(v, fimNames.suf), Mid: vocabID(v, fimNames.mid),
			Repo: vocabID(v, fimNames.repo), Sep: vocabID(v, fimNames.sep)}
		if f.Pre >= 0 && f.Suf >= 0 && f.Mid >= 0 {
			info.FIM = &f
		}
	}
	return info, nil
}

func (e *Engine) Tokenize(name, text string, addSpecial, parseSpecial bool) ([]int32, error) {
	_, v, err := e.llamaVocab(name)
	if err != nil {
		return nil, err
	}
	if parseSpecial {
		return v.EncodeSpecial(text, addSpecial), nil
	}
	ids := v.Encode(text, addSpecial)
	// A byte-level BPE vocabulary matches its control tokens in the text
	// whatever the flag says (tok.Vocab.Encode), so plain text cannot be
	// honoured when one is there: refuse rather than answer with the token.
	for i, id := range ids {
		if i == 0 && addSpecial && v.AddBOS {
			continue
		}
		if _, ok := v.Piece(id); !ok && v.Literal(id) != "" && strings.Contains(text, v.Literal(id)) {
			return nil, fmt.Errorf("%w: parse_special=false cannot be honoured: this tokenizer matches %q in the text "+
				"as a special token", ErrInvalid, v.Literal(id))
		}
	}
	return ids, nil
}

func (e *Engine) Piece(name string, id int32) ([]byte, error) {
	_, v, err := e.llamaVocab(name)
	if err != nil {
		return nil, err
	}
	return []byte(v.Literal(id)), nil
}

func (e *Engine) Detokenize(name string, ids []int32) (string, error) {
	lm, v, err := e.llamaVocab(name)
	if err != nil {
		return "", err
	}
	for _, id := range ids {
		if id < 0 || int(id) >= lm.m.Cfg.NVocab {
			return "", fmt.Errorf("%w: token id %d is outside the vocabulary of %d", ErrInvalid, id, lm.m.Cfg.NVocab)
		}
	}
	return v.Decode(ids), nil
}

func (e *Engine) ApplyTemplate(name string, msgs []model.ChatMessage, tools []byte) (string, error) {
	lm, err := e.resolveModel(name)
	if err != nil {
		return "", err
	}
	if !lm.m.ChatCapable() {
		return "", fmt.Errorf("%w: model %q carries no chat template", ErrInvalid, lm.called(name))
	}
	text, err := lm.m.ChatPromptTools(msgs, tools, true)
	if err != nil {
		return "", err
	}
	// llama-server drops the BOS the template wrote when the tokenizer adds
	// one, so the prompt can go straight to /completion without a double BOS.
	if v := lm.m.Vocab; v != nil && v.AddBOS {
		if bos := v.Text(v.BOS); bos != "" {
			text = strings.TrimPrefix(text, bos)
		}
	}
	return text, nil
}

func (e *Engine) Slots(name string) []LlamaSlot {
	lm, err := e.resolveModel(name)
	if err != nil {
		return nil
	}
	var out []LlamaSlot
	for _, s := range e.Sessions() {
		if s.lm != lm {
			continue
		}
		// A session serialises its own work on mu: held means generating.
		busy := !s.mu.TryLock()
		if !busy {
			s.mu.Unlock()
		}
		out = append(out, LlamaSlot{ID: s.id, NCtx: s.maxSeq, Processing: busy})
	}
	slices.SortFunc(out, func(a, b LlamaSlot) int { return strings.Compare(a.ID, b.ID) })
	return out
}

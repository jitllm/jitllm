package convert

import (
	"fmt"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// A decision model's GGUF states its readout as llama.cpp's converter writes
// it (conversion/{lev,lfm2,bert}.py): <arch>.decision.type names the readout,
// <arch>.decision.temperature.<type>[.<band>] the fitted temperatures, and the
// prompt is the template tokenizer.chat_template.systemone, which the vocab
// reader already carries by name. docs/design/decision-models.md has the
// family.

// decisionBands are the option-count buckets the temperatures were fitted on,
// by the names llama.cpp's converter gives them: rl_common.temp_bucket's
// 2 / 3-5 / 6-10 / 11+ (Laya and the models that copy it) and lev's
// calibration bands small <= 8, mid <= 26, large.
var decisionBands = map[string][2]uint32{
	"2": {0, 2}, "3_5": {3, 5}, "6_10": {6, 10}, "11": {11, 0},
	"small": {0, 8}, "mid": {9, 26}, "large": {27, 0},
}

var questionTypes = map[string]jlm.QuestionType{
	"choice": jlm.QuestionChoice, "score": jlm.QuestionScore, "noul": jlm.QuestionNoul,
}

// decisionOf fills c's decision fields from f. A file with no decision type
// is not a decision model and is left alone; a type this engine has no readout
// for is refused, since the model would otherwise convert into the backbone
// and answer a decision request by generating text.
func decisionOf(f *meta.File, c *jlm.Config) error {
	v, ok := f.Key("decision.type")
	if !ok {
		return nil
	}
	name, _ := v.String()
	kind, ok := jlm.DecisionKindOf(name)
	if !ok {
		return fmt.Errorf("decision type %q is not implemented: %w", name, ErrNotImplemented)
	}
	if _, ok := f.KV["tokenizer.chat_template.systemone"]; !ok {
		return fmt.Errorf("decision model %q carries no systemone template, which is its prompt", name)
	}
	c.Decision = kind
	c.DecisionBlocks = uint32(f.UintKey("decision.block_count", 0))
	if kind == jlm.DecisionLaya {
		if c.Arch != jlm.ArchModernBERT || c.DecisionBlocks == 0 {
			return fmt.Errorf("a laya readout is a ModernBERT encoder with head blocks, and this is %v with %d", c.Arch, c.DecisionBlocks)
		}
		if c.DecisionHeadTokens = uint32(f.UintKey("decision.max_head_tokens", 0)); c.DecisionHeadTokens == 0 {
			return fmt.Errorf("a laya readout needs decision.max_head_tokens, the budget its prompt is cut to")
		}
		// rl_common.DecisionModel builds its head with max(1, d // 64) heads,
		// whatever the encoder's count; llama.cpp's graph runs the encoder's,
		// which agrees on every published checkpoint (1024 wide, 16 heads)
		// and on no smaller one. A file that states it is taken at its word.
		c.DecisionHeads = uint32(f.UintKey("decision.head_count", uint64(max(1, c.NEmbd/64))))
		if c.DecisionHeads == 0 || c.NEmbd%c.DecisionHeads != 0 {
			return fmt.Errorf("%d decision heads do not divide a %d-wide residual", c.DecisionHeads, c.NEmbd)
		}
	} else if kind == jlm.DecisionClef {
		if err := clefOf(f, c); err != nil {
			return err
		}
	} else if c.DecisionBlocks != 0 {
		return fmt.Errorf("decision type %q with %d head blocks: only laya's and clef's heads are implemented: %w", name, c.DecisionBlocks, ErrNotImplemented)
	}

	prefix := f.Arch() + ".decision.temperature."
	for key, val := range f.KV {
		rest, ok := strings.CutPrefix(key, prefix)
		if !ok {
			continue
		}
		t, ok := val.Float()
		if !ok || !(t > 0) {
			return fmt.Errorf("decision temperature %s is %s, not a positive number", key, val.Short())
		}
		typ, band, banded := strings.Cut(rest, ".")
		qt, ok := questionTypes[typ]
		if !ok {
			return fmt.Errorf("decision temperature %s names no question type", key)
		}
		d := jlm.DecisionTemp{Type: qt, T: float32(t)}
		if banded {
			b, ok := decisionBands[band]
			if !ok {
				return fmt.Errorf("decision temperature %s names band %q, which no reference fits", key, band)
			}
			d.MinOptions, d.MaxOptions = b[0], b[1]
		}
		c.DecisionTemps = append(c.DecisionTemps, d)
	}
	// Bands before the type's own entry, which is what a lookup that takes the
	// first band holding the count needs; then a fixed order, so a container is
	// a function of its source.
	sort.Slice(c.DecisionTemps, func(i, j int) bool {
		a, b := c.DecisionTemps[i], c.DecisionTemps[j]
		if a.Type != b.Type {
			return a.Type < b.Type
		}
		ab, bb := a.MinOptions != 0 || a.MaxOptions != 0, b.MinOptions != 0 || b.MaxOptions != 0
		if ab != bb {
			return ab
		}
		return a.MinOptions < b.MinOptions
	})
	return nil
}

// modernBERTConfig is ModernBERT's encoder, and the decision head llama.cpp
// appends to it as trailing blocks (conversion/bert.py,
// ModernBertDecisionModel): block_count counts both, decision.block_count the
// head, and feed_forward_length is per block because the head's MLP is 4x
// wide. From here on NLayer is the encoder; the head is blocks NLayer.. and
// Config.DecisionBlocks says how many.
func modernBERTConfig(f *meta.File, c *jlm.Config) error {
	c.RMSEps = float32(f.FloatKey("attention.layer_norm_epsilon", 1e-5))
	c.Flags |= jlm.FlagNonCausal | jlm.FlagRopeNeox | jlm.FlagGELU
	if v, ok := f.Key("hidden_activation"); ok {
		// IBM's Granite Embedding R2 runs the same graph under SiLU; nothing
		// here has a fixture for it.
		if s, _ := v.String(); s != "gelu" {
			return fmt.Errorf("hidden_activation %q is not implemented: %w", s, ErrNotImplemented)
		}
	}
	head := uint32(f.UintKey("decision.block_count", 0))
	if head >= c.NLayer {
		return fmt.Errorf("decision.block_count %d in a %d-block file", head, c.NLayer)
	}
	c.NLayer -= head
	ffn, ok := f.Key("feed_forward_length")
	if !ok {
		return fmt.Errorf("no feed_forward_length")
	}
	if v, ok := ffn.Uint(); ok {
		c.NFFN = uint32(v)
	} else {
		vs, err := ffn.Int32s()
		if err != nil {
			return fmt.Errorf("feed_forward_length: %w", err)
		}
		if uint32(len(vs)) != c.NLayer+head {
			return fmt.Errorf("feed_forward_length has %d entries for %d blocks", len(vs), c.NLayer+head)
		}
		for i := uint32(0); i < c.NLayer; i++ {
			if vs[i] != vs[0] || vs[i] <= 0 {
				return fmt.Errorf("feed_forward_length differs between encoder blocks: %w", ErrNotImplemented)
			}
		}
		c.NFFN = uint32(vs[0])
	}
	// Every SWAPeriod-th layer from the first attends to everything; the
	// others within a symmetric window. An absent window is every layer
	// global.
	c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 0))
	if c.SWAWindow != 0 {
		if err := swaPatternOf(f, c, 3); err != nil {
			return err
		}
	}
	c.RopeBaseSWA = float32(f.FloatKey("rope.freq_base_swa", float64(c.RopeBase)))
	return nil
}

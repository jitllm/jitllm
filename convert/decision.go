package convert

import (
	"fmt"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// A decision model's GGUF states its readout: <arch>.decision.type names the readout,
// <arch>.decision.temperature.<type>[.<band>] the fitted temperatures, and the
// prompt is the template tokenizer.chat_template.systemone, which the vocab
// reader already carries by name. What each readout reads is its statement
// in readouts.json; docs/design/decision-models.md has the family.

func questionTypes(name string) (jlm.QuestionType, bool) {
	switch name {
	case "choice":
		return jlm.QuestionChoice, true
	case "score":
		return jlm.QuestionScore, true
	case "noul":
		return jlm.QuestionNoul, true
	}
	return 0, false
}

// decisionType is the readout f's decision.type names, "" for none.
func decisionType(f *meta.File) string {
	v, ok := f.Key("decision.type")
	if !ok {
		return ""
	}
	name, _ := v.String()
	return name
}

// decisionOf fills c's decision fields from f, as the library states its
// readout. A file with no decision type is not a decision model and is left
// alone; a type the library states no readout for is refused, since the
// model would otherwise convert into the backbone and answer a decision
// request by generating text.
func decisionOf(f *meta.File, c *Config) error {
	v, ok := f.Key("decision.type")
	if !ok {
		return nil
	}
	name, _ := v.String()
	lib, err := readoutLibrary()
	if err != nil {
		return err
	}
	st := lib.Readouts[name]
	if st == nil {
		return fmt.Errorf("decision type %q is not implemented: %w", name, ErrNotImplemented)
	}
	if _, ok := f.KV["tokenizer.chat_template.systemone"]; !ok {
		return fmt.Errorf("decision model %q carries no systemone template, which is its prompt", name)
	}
	c.DecisionBlocks = uint32(f.UintKey("decision.block_count", 0))
	if err := st.state(name, f, c); err != nil {
		return err
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
		qt, ok := questionTypes(typ)
		if !ok {
			return fmt.Errorf("decision temperature %s names no question type", key)
		}
		d := jlm.DecisionTemp{Type: qt, T: float32(t)}
		if banded {
			b, ok := lib.Bands[band]
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

// stateHeadBlocks splits a decision head a GGUF appends to an encoder as
// trailing blocks off c (the head_blocks statement; ModernBertDecisionModel):
// block_count counts both, head the head, and feed_forward_length may be per
// block because the head's MLP is wider. From here on NLayer is the encoder;
// the head is blocks NLayer.. and Config.DecisionBlocks says how many.
func stateHeadBlocks(f *meta.File, c *Config, head uint32) error {
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
		return nil
	}
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
	return nil
}

// headSeam is where a marker head's model-level tensor goes once its blocks
// are the model's: the trunk's final norm closes the trunk's last block, a
// norm over its residual (RoleLayerOutNorm), and the question types' table
// is the first head block's, whose input adds the question's row.
func headSeam(role jlm.Role, block int32, c *Config) (jlm.Role, int32) {
	if !c.Flags2.Has(jlm.Flag2HeadBlocks) || block != jlm.DenseBlock {
		return role, block
	}
	switch role {
	case jlm.RoleOutputNorm:
		return jlm.RoleLayerOutNorm, int32(c.NLayer) - 1
	case jlm.RoleOutputNormBias:
		return jlm.RoleLayerOutNormBias, int32(c.NLayer) - 1
	case jlm.RoleTokenTypes:
		return role, int32(c.NLayer)
	}
	return role, block
}

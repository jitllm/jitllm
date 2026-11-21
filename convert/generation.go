package convert

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// statedStops is v's stop ids with the pieces the library states for a GGUF
// of architecture arch (mjStates.Stop) that v's token table holds.
func statedStops(arch string, v *jlm.Vocab) ([]int32, error) {
	st, _, err := ggufStates(arch)
	if err != nil || st == nil || len(st.Stop) == 0 {
		return v.Stop, err
	}
	want := map[string]bool{}
	for _, n := range st.Stop {
		want[n] = true
	}
	found := map[string]int32{}
	for i, t := range v.Tokens {
		if _, seen := found[t]; want[t] && !seen {
			found[t] = int32(i)
		}
	}
	stop := v.Stop
	for _, n := range st.Stop {
		if id, ok := found[n]; ok {
			stop = addStop(stop, v.EOS, id)
		}
	}
	return stop, nil
}

//go:embed generation.json
var generationJSON []byte

// generationFields is the draw's fields as generation_config.json names
// them, each nil where unstated.
type generationFields struct {
	DoSample          *bool    `json:"do_sample"`
	Temperature       *float64 `json:"temperature"`
	TopK              *int     `json:"top_k"`
	TopP              *float64 `json:"top_p"`
	MinP              *float64 `json:"min_p"`
	TypicalP          *float64 `json:"typical_p"`
	RepetitionPenalty *float64 `json:"repetition_penalty"`
}

// over is g with each sampling field gf states in place of g's.
func (gf generationFields) over(g jlm.Generation) jlm.Generation {
	if gf.TopK != nil {
		g.TopK = *gf.TopK
	}
	for _, f := range []struct {
		from *float64
		to   *float64
	}{{gf.Temperature, &g.Temperature}, {gf.TopP, &g.TopP}, {gf.MinP, &g.MinP}, {gf.TypicalP, &g.TypicalP},
		{gf.RepetitionPenalty, &g.RepetitionPenalty}} {
		if f.from != nil {
			*f.to = *f.from
		}
	}
	return g
}

// sampling reports whether gf states any field a draw cuts or scales by.
func (gf generationFields) sampling() bool {
	return gf.Temperature != nil || gf.TopK != nil || gf.TopP != nil || gf.MinP != nil || gf.TypicalP != nil
}

// set states the field named name (generation.json's) at x.
func (gf *generationFields) set(name string, x float64) error {
	switch name {
	case "temperature":
		gf.Temperature = &x
	case "top_k":
		k := int(x)
		gf.TopK = &k
	case "top_p":
		gf.TopP = &x
	case "min_p":
		gf.MinP = &x
	case "typical_p":
		gf.TypicalP = &x
	case "repetition_penalty":
		gf.RepetitionPenalty = &x
	default:
		return fmt.Errorf("convert: generation.json: %q is no generation_config.json sampling field", name)
	}
	return nil
}

// hfGeneration is the draw a safetensors model's generate runs by default:
// generation_config.json's fields over GenerationConfig's defaults, or with
// no generation_config.json, config.json's, as transformers builds one from
// it. do_sample is GenerationConfig's default, false, where the file leaves
// it out: a file stating a temperature and not do_sample (Seed-OSS) draws
// greedily, as its generate does.
func hfGeneration(dir hfDir) (*jlm.Generation, error) {
	name := "generation_config.json"
	b, err := dir.read(name)
	if err != nil {
		name = "config.json"
		if b, err = dir.read(name); err != nil {
			g := jlm.ReferenceGeneration()
			return &g, nil
		}
	}
	var c generationFields
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", dir.path(name), err)
	}
	g := c.over(jlm.ReferenceGeneration())
	g.DoSample = c.DoSample != nil && *c.DoSample
	if err := g.Valid(); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", dir.path(name), err)
	}
	return &g, nil
}

// generationKeys is generation.json: the GGUF key of each sampling field.
func generationKeys() (map[string]string, error) {
	var keys struct {
		GGUF map[string]string `json:"gguf"`
	}
	if err := json.Unmarshal(generationJSON, &keys); err != nil {
		return nil, fmt.Errorf("convert: generation.json: %w", err)
	}
	return keys.GGUF, nil
}

// ggufGeneration is the draw a GGUF states, over GenerationConfig's
// defaults: the family's statement's fields (mjStates.Generation), then
// each field the file states under the key generation.json names for it.
// A GGUF has no key for do_sample: the family's statement states it where
// its reference says, and otherwise a file or statement stating a sampling
// field samples at its temperature (0 the greedy draw), one stating none
// is greedy. Each field the file states is added to taken where that is not
// nil.
func ggufGeneration(f *meta.File, taken map[string]bool) (*jlm.Generation, error) {
	keys, err := generationKeys()
	if err != nil {
		return nil, err
	}
	st, _, err := ggufStates(f.Arch())
	if err != nil {
		return nil, err
	}
	var family, file generationFields
	if st != nil && st.Generation != nil {
		family = *st.Generation
	}
	for _, field := range slices.Sorted(maps.Keys(keys)) {
		x, ok := f.KV[keys[field]].Float()
		if !ok {
			continue
		}
		take(taken, "generation "+field)
		// The keys are f32: the decimal the writer meant is the shortest one
		// that rounds to them.
		if d, err := strconv.ParseFloat(strconv.FormatFloat(x, 'g', -1, 32), 64); err == nil {
			x = d
		}
		if err := file.set(field, x); err != nil {
			return nil, err
		}
	}
	g := file.over(family.over(jlm.ReferenceGeneration()))
	g.DoSample = (family.sampling() || file.sampling()) && g.Temperature > 0
	if family.DoSample != nil {
		g.DoSample = *family.DoSample && g.Temperature > 0
	}
	if err := g.Valid(); err != nil {
		return nil, fmt.Errorf("convert: the file's general.sampling keys: %w", err)
	}
	return &g, nil
}

package jlm

import (
	"fmt"
	"math"
)

// Generation is the draw the model's reference runs when a caller asks
// nothing: transformers' GenerationConfig fields, as a model's
// generation_config.json (or a GGUF's general.sampling keys) states them and
// ReferenceGeneration where it states none. DoSample false is the greedy
// draw, whatever the other fields hold.
type Generation struct {
	DoSample          bool    `json:"do_sample"`
	Temperature       float64 `json:"temperature"`
	TopK              int     `json:"top_k"`
	TopP              float64 `json:"top_p"`
	MinP              float64 `json:"min_p"`
	TypicalP          float64 `json:"typical_p"`
	RepetitionPenalty float64 `json:"repetition_penalty"`
}

// ReferenceGeneration is GenerationConfig's own defaults: the greedy draw,
// and where a model samples without stating more, temperature 1, top-k 50,
// top-p 1, typical-p 1, no min-p and no repetition penalty.
func ReferenceGeneration() Generation {
	return Generation{Temperature: 1, TopK: 50, TopP: 1, TypicalP: 1, RepetitionPenalty: 1}
}

// Valid refuses a field GenerationConfig refuses: a number that is not
// finite, a negative one, a top-p, typical-p or min-p past 1, or a
// repetition penalty of 0.
func (g *Generation) Valid() error {
	for _, f := range []struct {
		name string
		v    float64
	}{{"temperature", g.Temperature}, {"top_p", g.TopP}, {"min_p", g.MinP}, {"typical_p", g.TypicalP},
		{"repetition_penalty", g.RepetitionPenalty}} {
		if math.IsNaN(f.v) || math.IsInf(f.v, 0) || f.v < 0 {
			return fmt.Errorf("jlm: generation %s %v is not a number at least 0", f.name, f.v)
		}
	}
	switch {
	case g.TopP > 1 || g.TypicalP > 1 || g.MinP > 1:
		return fmt.Errorf("jlm: generation top_p %v, typical_p %v or min_p %v is past 1", g.TopP, g.TypicalP, g.MinP)
	case g.RepetitionPenalty == 0:
		return fmt.Errorf("jlm: generation repetition_penalty is 0")
	case g.TopK < 0:
		return fmt.Errorf("jlm: generation top_k %d is negative", g.TopK)
	}
	return nil
}

func encodeGeneration(w *wbuf, g *Generation) {
	w.bool(g.DoSample)
	w.u64(math.Float64bits(g.Temperature))
	w.u32(uint32(g.TopK))
	for _, f := range []float64{g.TopP, g.MinP, g.TypicalP, g.RepetitionPenalty} {
		w.u64(math.Float64bits(f))
	}
}

func decodeGeneration(r *rbuf) *Generation {
	g := &Generation{DoSample: r.bool(), Temperature: math.Float64frombits(r.u64()), TopK: int(r.u32())}
	for _, f := range []*float64{&g.TopP, &g.MinP, &g.TypicalP, &g.RepetitionPenalty} {
		*f = math.Float64frombits(r.u64())
	}
	return g
}

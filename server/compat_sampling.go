package server

import (
	"encoding/json"
	"fmt"
	"maps"
	"math/rand/v2"
	"slices"
	"strconv"

	"github.com/jitllm/jitllm/engine"
)

// oaSampling is every sampling field the two shims read, decoded in one place
// so the OpenAI routes and the Anthropic route build the same sampler from the
// same request.
//
// A request that names no sampling field gets the API's documented defaults,
// temperature 1 and top_p 1, not the session's greedy sampler: an HTTP
// benchmark sends one request body to every engine, and every other engine
// samples it at temperature 1. Greedy is temperature 0, as the API spells it.
// The one exception is a request naming a jitllm_session and no sampling field,
// which keeps the sampler that session was opened with.
//
//	temperature         0 = greedy; default 1
//	top_p               nucleus; default 1 (disabled)
//	top_k               0 or -1 = disabled (vLLM spells disabled -1)
//	min_p               0 = disabled
//	repetition_penalty  1 = disabled; llama.cpp's repeat_penalty over the last 64
//	seed                the draw's seed; without one each request draws its own
//	presence_penalty,
//	frequency_penalty   0 = disabled, in [-2, 2]; llama.cpp's over the same window
//	n                   continuations of one prompt, prefilled once; read by
//	                    the OpenAI routes (compat_choices.go)
//	logprobs,
//	top_logprobs        the raw distribution's log-probabilities; read by the
//	                    OpenAI routes (compat_choices.go)
//	logit_bias          token id to a bias in [-100, 100], added to the logits
//	                    before the penalties (GenerateOptions.LogitBias)
type oaSampling struct {
	Temperature       *float64           `json:"temperature"`
	TopP              *float64           `json:"top_p"`
	TopK              *int               `json:"top_k"`
	MinP              *float64           `json:"min_p"`
	RepetitionPenalty *float64           `json:"repetition_penalty"`
	PresencePenalty   *float64           `json:"presence_penalty"`
	FrequencyPenalty  *float64           `json:"frequency_penalty"`
	Seed              *int64             `json:"seed"`
	N                 *int               `json:"n"`
	Logprobs          json.RawMessage    `json:"logprobs"`
	TopLogprobs       *int               `json:"top_logprobs"`
	LogitBias         map[string]float64 `json:"logit_bias"`
}

// logitBias is logit_bias as the engine takes it, in token order: a key that
// is not a token id, or a bias outside [-100, 100], is refused.
func (q *oaSampling) logitBias() ([]TokenBias, error) {
	var out []TokenBias
	for _, k := range slices.Sorted(maps.Keys(q.LogitBias)) {
		id, err := strconv.ParseInt(k, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("%w: logit_bias key %q is not a token id", ErrInvalid, k)
		}
		v := q.LogitBias[k]
		if v < -100 || v > 100 {
			return nil, fmt.Errorf("%w: logit_bias %v for token %d is outside [-100, 100]", ErrInvalid, v, id)
		}
		out = append(out, TokenBias{ID: int32(id), Bias: float32(v)})
	}
	return out, nil
}

// samplingFields are the sampling values both surfaces validate, held as
// pointers so an absent field is not a zero. The OpenAI routes fill them from
// JSON and the Connect InferenceService from SamplingParams, and validate is
// the one check, so a value either refuses the other refuses in the same
// words.
type samplingFields struct {
	Temperature, TopP, MinP, RepetitionPenalty, PresencePenalty, FrequencyPenalty *float64
	TopK                                                                          *int
}

func (q *oaSampling) fields() samplingFields {
	return samplingFields{q.Temperature, q.TopP, q.MinP, q.RepetitionPenalty, q.PresencePenalty,
		q.FrequencyPenalty, q.TopK}
}

// validate refuses a value outside its range.
func (q samplingFields) validate() error {
	if q.PresencePenalty != nil && (*q.PresencePenalty < -2 || *q.PresencePenalty > 2) {
		return fmt.Errorf("presence_penalty %v is outside [-2, 2]", *q.PresencePenalty)
	}
	if q.FrequencyPenalty != nil && (*q.FrequencyPenalty < -2 || *q.FrequencyPenalty > 2) {
		return fmt.Errorf("frequency_penalty %v is outside [-2, 2]", *q.FrequencyPenalty)
	}
	if q.Temperature != nil && *q.Temperature < 0 {
		return fmt.Errorf("temperature %v is negative", *q.Temperature)
	}
	if q.TopP != nil && (*q.TopP < 0 || *q.TopP > 1) {
		return fmt.Errorf("top_p %v is outside [0, 1]", *q.TopP)
	}
	if q.TopK != nil && *q.TopK < -1 {
		return fmt.Errorf("top_k %d is below -1", *q.TopK)
	}
	if q.MinP != nil && (*q.MinP < 0 || *q.MinP > 1) {
		return fmt.Errorf("min_p %v is outside [0, 1]", *q.MinP)
	}
	if q.RepetitionPenalty != nil && *q.RepetitionPenalty <= 0 {
		return fmt.Errorf("repetition_penalty %v is not positive", *q.RepetitionPenalty)
	}
	return nil
}

// sampler validates the fields and builds the sampler a request runs with. nil
// means the session's own, which only a named session with no sampling field
// gets.
func (q *oaSampling) sampler(session string) (*engine.Sampler, error) {
	if err := q.fields().validate(); err != nil {
		return nil, err
	}
	if session != "" && q.Temperature == nil && q.TopP == nil && q.TopK == nil &&
		q.MinP == nil && q.RepetitionPenalty == nil && q.PresencePenalty == nil && q.FrequencyPenalty == nil && q.Seed == nil {
		return nil, nil
	}
	s := &engine.Sampler{Temp: 1, TopP: 1}
	if q.Temperature != nil {
		s.Temp = *q.Temperature
	}
	if q.TopP != nil {
		s.TopP = *q.TopP
	}
	if q.TopK != nil {
		s.TopK = max(*q.TopK, 0)
	}
	if q.MinP != nil {
		s.MinP = *q.MinP
	}
	if q.RepetitionPenalty != nil {
		s.RepeatPen = *q.RepetitionPenalty
	}
	if q.PresencePenalty != nil {
		s.PresencePen = *q.PresencePenalty
	}
	if q.FrequencyPenalty != nil {
		s.FreqPen = *q.FrequencyPenalty
	}
	if q.Seed != nil {
		s.Seed = *q.Seed
	} else {
		// Unseeded requests must not all draw the same sequence, which a zero
		// seed would make them do.
		s.Seed = rand.Int64()
	}
	return s, nil
}

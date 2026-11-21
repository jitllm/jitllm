package jlm

import "fmt"

// A decision model answers typed questions about a state in one forward pass
// and generates nothing (.notes/design/decision-models.md). What makes a model
// one is not its graph -- Lev is qwen35 and d1 is lfm2 -- but how its answer is
// read out and calibrated, which the reference hardcodes per model and the
// container therefore states: a label readout (DecisionLabels), a marker
// head's prompt (MarkerPrompt) or a joint head (JointHead). The engine reads
// those statements alone; no readout has a code of its own.

// Decides reports whether the container states a decision readout.
func (c *Config) Decides() bool {
	return c.DecisionLabels != nil || c.MarkerPrompt != nil || c.JointHead != nil
}

// QuestionType is a decision question's type, in TypeSafe's names.
type QuestionType uint8

const (
	QuestionChoice QuestionType = iota
	QuestionScore
	QuestionNoul
)

func (t QuestionType) String() string {
	switch t {
	case QuestionChoice:
		return "choice"
	case QuestionScore:
		return "score"
	case QuestionNoul:
		return "noul"
	}
	return fmt.Sprintf("question(%d)", uint8(t))
}

// DecisionLabels is a label readout's statement: how a question's options
// become the codes its prompt shows, and the token groups whose next-token
// logits at the prompt's last row score them, a group by its largest logit.
// The converter writes it per kind (convert/decision.go); the engine reads
// it alone.
type DecisionLabels struct {
	// Codes are the options' codes in order: option i takes its own key where
	// KeyCodes holds and every key is one letter, else Codes[i], or
	// WideCodes[i] where the question has more options than Codes holds. A
	// code that is not one token, or whose token another option took, gives
	// way to the first of Codes, then Pool, that is neither. Spaced adds
	// each code's " "+code token to its group. MaxOptions is the most
	// options a question may have.
	Codes, WideCodes, Pool []string
	KeyCodes, Spaced       bool
	MaxOptions             uint32
	// ShowCodes hands the template each option's code as its label; a
	// template that writes its own letters is handed none.
	ShowCodes bool
	// A noul's groups: NoulTrue and NoulFalse, each form one token; or, with
	// NoulRatings, that many levels read at the first codes, the answer the
	// expected level over NoulRatings-1. NoulTrueFirst shows true first.
	NoulTrue, NoulFalse []string
	NoulRatings         uint32
	NoulTrueFirst       bool
	// ScoreByKey reads a score question's levels at their keys' own tokens.
	ScoreByKey bool
	// Reversed asks a choice of several options again in reversed order and
	// averages the two, in the original order.
	Reversed bool
	// SortKeys sorts every object's keys in what the template is handed, and
	// IDInstructions stands a question's id in for absent instructions.
	SortKeys, IDInstructions bool
	// EntropyConfidence gives a choice's and a score's confidence as one
	// minus the distribution's entropy over log of its option count.
	EntropyConfidence bool
	// SortedLegend writes a score's level description that is not text, in
	// the response's legend, as its compact JSON with every object's keys
	// sorted, a string.
	SortedLegend bool
}

// DecisionTemp is one fitted calibration temperature: it divides the scores of
// a question of Type with MinOptions to MaxOptions options before the softmax
// (MaxOptions zero: no upper bound). A band is the bucket the temperature was
// fitted on, and a question outside every band of its type takes the type's
// own entry, MinOptions and MaxOptions both zero, which the table holds after
// the bands.
type DecisionTemp struct {
	Type                   QuestionType
	MinOptions, MaxOptions uint32
	T                      float32
}

// Temperature is the temperature a question of type t with n options is
// calibrated by: the first entry whose band holds n, else 1.
func (c *Config) Temperature(t QuestionType, n int) float32 {
	for _, d := range c.DecisionTemps {
		if d.Type == t && uint32(n) >= d.MinOptions && (d.MaxOptions == 0 || uint32(n) <= d.MaxOptions) {
			return d.T
		}
	}
	return 1
}

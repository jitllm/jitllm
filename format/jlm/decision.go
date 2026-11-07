package jlm

import "fmt"

// A decision model answers typed questions about a state in one forward pass
// and generates nothing (docs/design/decision-models.md). What makes a model
// one is not its graph -- Lev is qwen35 and d1 is lfm2 -- but how its answer is
// read out and calibrated, which the reference hardcodes per model and the
// container therefore states.

// A kind's code is written into every container, so codes never move: a new
// kind takes a number of its own, stated, past the iota run that ends at
// DecisionVllmD3 (7). Taken: 8 nimble, 9 pplx-decider, 10 kev.

// DecisionKind is a decision model's readout: which prompt shape and which
// scores the answer is read from. Zero is not a decision model.
type DecisionKind uint8

const (
	DecisionNone DecisionKind = iota
	// DecisionLev reads the next-token logits of single-token option codes
	// (A..Z then AA..ZZ) at the prompt's last row; a choice is asked twice,
	// options forward and reversed, and a noul is a nine-level scale read at
	// the first nine codes (interfaze-ai/lev).
	DecisionLev
	// DecisionLFM2D1 reads the next-token logits of each option's code group
	// at the prompt's last row: the code and " "+code, scored by the larger
	// (LiquidAI/d1-3B).
	DecisionLFM2D1
	// DecisionLaya reads a scorer at each option's [MASK] marker after the
	// encoder and Config.DecisionBlocks head blocks, the question's type
	// embedding added to every row before them (convaiinnovations/laya).
	DecisionLaya
	// DecisionClef reads Clef's joint schema head over the backbone's final
	// hidden states: every question and option of a request mean-pooled over
	// its span of one prompt, Config.DecisionRouting routing blocks and
	// Config.DecisionBlocks joint blocks, and one logit per option of every
	// question in one pass (Cloudflare/clef-flash; decisionroles.go has its
	// tensors).
	DecisionClef
	// DecisionD1Omni is d1-omni's readout (LiquidAI/d1-omni-600M): an LFM2
	// encoder over one question's prompt (ArchLFM2Enc), the question type's
	// row added to every row and Config.DecisionBlocks head blocks -- Laya's
	// head -- and a scorer at each option's <|mask|>, a noul read as [false,
	// true]. Images or a voice clip enter as a prefix of embedding rows.
	DecisionD1Omni
	// DecisionOpenJev reads the next-token logits of one letter per option
	// (A..Z then a..z, each one token) at the prompt's last row, a noul
	// shown true first (openjev/openjev).
	DecisionOpenJev // 6
	// DecisionVllmD3 is lev's codes (A..Z then AA..ZZ, each one token) read
	// at the prompt's last row over a qwen35 whose attention is not causal
	// (vllm-sr/d3), one prompt per question; its confidence is one minus the
	// normalised entropy, and it reads a null state and null instructions.
	DecisionVllmD3 // 7
	// DecisionKev is Kev's pointer head (jaredpalmer/kev): cls.output over
	// the normed final hidden state, the prompt's last row's first half
	// dotted with each option's <|box_end|> row's second half.
	DecisionKev DecisionKind = 10
)

func decisionName(k DecisionKind) (string, bool) {
	switch k {
	case DecisionLev:
		return "lev", true
	case DecisionLFM2D1:
		return "lfm2-d1", true
	case DecisionLaya:
		return "laya", true
	case DecisionClef:
		return "clef", true
	case DecisionD1Omni:
		return "lfm2-d1-omni", true
	case DecisionOpenJev:
		return "openjev", true
	case DecisionVllmD3:
		return "vllm-d3", true
	case DecisionKev:
		return "kev", true
	}
	return "", false
}

// String is the name llama.cpp writes as <arch>.decision.type.
func (k DecisionKind) String() string {
	if s, ok := decisionName(k); ok {
		return s
	}
	if k == DecisionNone {
		return "none"
	}
	return fmt.Sprintf("decision(%d)", uint8(k))
}

// DecisionKindOf maps llama.cpp's <arch>.decision.type name to a kind.
func DecisionKindOf(name string) (DecisionKind, bool) {
	for k := DecisionKind(1); k != 0; k++ {
		if s, ok := decisionName(k); ok && s == name {
			return k, true
		}
	}
	return DecisionNone, false
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

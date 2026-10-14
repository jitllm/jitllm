package jlm

import "fmt"

// A decision model answers typed questions about a state in one forward pass
// and generates nothing (docs/design/decision-models.md). What makes a model
// one is not its graph -- Lev is qwen35 and d1 is lfm2 -- but how its answer is
// read out and calibrated, which the reference hardcodes per model and the
// container therefore states.

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
)

var decisionNames = map[DecisionKind]string{
	DecisionLev:    "lev",
	DecisionLFM2D1: "lfm2-d1",
	DecisionLaya:   "laya",
	DecisionClef:   "clef",
	DecisionD1Omni: "lfm2-d1-omni",
}

// String is the name llama.cpp writes as <arch>.decision.type.
func (k DecisionKind) String() string {
	if s, ok := decisionNames[k]; ok {
		return s
	}
	if k == DecisionNone {
		return "none"
	}
	return fmt.Sprintf("decision(%d)", uint8(k))
}

// DecisionKindOf maps llama.cpp's <arch>.decision.type name to a kind.
func DecisionKindOf(name string) (DecisionKind, bool) {
	for k, s := range decisionNames {
		if s == name {
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

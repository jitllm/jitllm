package convert

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

//go:embed readouts.json
var readoutsJSON []byte

// readoutFile is readouts.json: the decision readouts by the name a GGUF
// writes as <arch>.decision.type, and the option-count bands their
// temperatures were fitted on, by the names the converter gives them.
type readoutFile struct {
	Bands    map[string][2]uint32      `json:"bands"`
	Readouts map[string]*readoutStates `json:"readouts"`
}

// readoutStates is a readout as the library states it: its head (each
// field an expression over the file's keys, src, and the model, c), what the
// head and the model must satisfy (over head and c), the label readout's
// rules, the joint head, the container flags it sets, a marker head's
// prompt (promptstates.go), and the trunk a file read through a view of it
// carries (trunk.go). Backbone is a
// readout only a backbone whose own statement names it carries.
type readoutStates struct {
	Backbone bool           `json:"backbone"`
	Head     *readoutHead   `json:"head"`
	Require  []mjRequire    `json:"require"`
	Labels   *readoutLabels `json:"labels"`
	Joint    *readoutJoint  `json:"joint"`
	Flags    []string       `json:"flags"`
	Prompt   *readoutPrompt `json:"prompt"`
	Trunk    *readoutTrunk  `json:"trunk"`
}

// readoutHead is a decision head's geometry: routing blocks before its
// blocks (decision.block_count's, which every readout reads), its attention
// heads, the token budget its prompt is cut to, and its LayerNorms' epsilon.
type readoutHead struct {
	Routing string `json:"routing"`
	Heads   string `json:"heads"`
	Tokens  string `json:"tokens"`
	Eps     string `json:"eps"`
}

// readoutLabels is a label readout's rules (jlm.DecisionLabels), each code
// list a run of runs.
type readoutLabels struct {
	Codes             []labelRun `json:"codes"`
	WideCodes         []labelRun `json:"wide_codes"`
	Pool              []labelRun `json:"pool"`
	KeyCodes          bool       `json:"key_codes"`
	Spaced            bool       `json:"spaced"`
	MaxOptions        uint32     `json:"max_options"`
	ShowCodes         bool       `json:"show_codes"`
	NoulTrue          []string   `json:"noul_true"`
	NoulFalse         []string   `json:"noul_false"`
	NoulRatings       uint32     `json:"noul_ratings"`
	NoulTrueFirst     bool       `json:"noul_true_first"`
	ScoreByKey        bool       `json:"score_by_key"`
	Reversed          bool       `json:"reversed"`
	SortKeys          bool       `json:"sort_keys"`
	IDInstructions    bool       `json:"id_instructions"`
	EntropyConfidence bool       `json:"entropy_confidence"`
	SortedLegend      bool       `json:"sorted_legend"`
}

// labelRun is a run of codes: each letter from Letters' first to its second,
// each pair of them (Pairs), or Format of 0 to Count-1.
type labelRun struct {
	Letters string `json:"letters"`
	Pairs   string `json:"pairs"`
	Format  string `json:"format"`
	Count   int    `json:"count"`
}

// readoutJoint is a joint head's reading of a request (jlm.JointHead).
type readoutJoint struct {
	TypeRows       [3]uint8 `json:"type_rows"`
	SortKeys       bool     `json:"sort_keys"`
	SortChoices    bool     `json:"sort_choices"`
	NoulTrueFirst  bool     `json:"noul_true_first"`
	IDInstructions bool     `json:"id_instructions"`
	Confidence     string   `json:"confidence"`
}

func readoutFlags() map[string]jlm.Flags2 {
	return map[string]jlm.Flags2{"head_blocks": jlm.Flag2HeadBlocks, "marker_pointer": jlm.Flag2MarkerPointer,
		"media_prefix": jlm.Flag2MediaPrefix, "media_uncalibrated": jlm.Flag2MediaUncalibrated}
}

func readoutConfidences() map[string]jlm.PromptConfidence {
	return map[string]jlm.PromptConfidence{"": jlm.ConfidenceOfType, "entropy": jlm.ConfidenceEntropy,
		"pmax": jlm.ConfidencePMax}
}

// readoutLibrary is readouts.json, each readout's flags and confidence
// checked.
func readoutLibrary() (*readoutFile, error) {
	var lib readoutFile
	if err := json.Unmarshal(readoutsJSON, &lib); err != nil {
		return nil, fmt.Errorf("convert: readouts.json: %w", err)
	}
	for name, r := range lib.Readouts {
		for _, f := range r.Flags {
			if _, ok := readoutFlags()[f]; !ok {
				return nil, fmt.Errorf("convert: readouts.json: %s: %q is no readout flag", name, f)
			}
		}
		if j := r.Joint; j != nil {
			if _, ok := readoutConfidences()[j.Confidence]; !ok {
				return nil, fmt.Errorf("convert: readouts.json: %s: %q is no confidence", name, j.Confidence)
			}
		}
		if pr := r.Prompt; pr != nil {
			if _, ok := readoutConfidences()[pr.Confidence]; !ok {
				return nil, fmt.Errorf("convert: readouts.json: %s: %q is no confidence", name, pr.Confidence)
			}
		}
		if l := r.Labels; l != nil {
			for _, runs := range [][]labelRun{l.Codes, l.WideCodes, l.Pool} {
				for _, run := range runs {
					if _, err := run.codes(); err != nil {
						return nil, fmt.Errorf("convert: readouts.json: %s: %w", name, err)
					}
				}
			}
		}
	}
	return &lib, nil
}

// libraryReadout is the readout the library states by name, nil for none.
func libraryReadout(name string) *readoutStates {
	lib, err := readoutLibrary()
	if err != nil {
		return nil
	}
	return lib.Readouts[name]
}

// codes is the run's codes.
func (r labelRun) codes() ([]string, error) {
	span := func(s string) (rune, rune, error) {
		rs := []rune(s)
		if len(rs) != 2 || rs[0] > rs[1] {
			return 0, 0, fmt.Errorf("a run of %q is not two letters, the first before the second", s)
		}
		return rs[0], rs[1], nil
	}
	var out []string
	switch {
	case r.Letters != "":
		from, to, err := span(r.Letters)
		if err != nil {
			return nil, err
		}
		for c := from; c <= to; c++ {
			out = append(out, string(c))
		}
	case r.Pairs != "":
		from, to, err := span(r.Pairs)
		if err != nil {
			return nil, err
		}
		for a := from; a <= to; a++ {
			for b := from; b <= to; b++ {
				out = append(out, string([]rune{a, b}))
			}
		}
	case r.Format != "" && r.Count > 0:
		for i := range r.Count {
			out = append(out, fmt.Sprintf(r.Format, i))
		}
	default:
		return nil, fmt.Errorf("a run of codes states no letters, pairs or format and count")
	}
	return out, nil
}

// labels is l as the container's label readout.
func (l *readoutLabels) labels() *jlm.DecisionLabels {
	if l == nil {
		return nil
	}
	runs := func(rs []labelRun) []string {
		var out []string
		for _, r := range rs {
			c, _ := r.codes()
			out = append(out, c...)
		}
		return out
	}
	return &jlm.DecisionLabels{Codes: runs(l.Codes), WideCodes: runs(l.WideCodes), Pool: runs(l.Pool),
		KeyCodes: l.KeyCodes, Spaced: l.Spaced, MaxOptions: l.MaxOptions, ShowCodes: l.ShowCodes,
		NoulTrue: l.NoulTrue, NoulFalse: l.NoulFalse, NoulRatings: l.NoulRatings, NoulTrueFirst: l.NoulTrueFirst,
		ScoreByKey: l.ScoreByKey, Reversed: l.Reversed, SortKeys: l.SortKeys, IDInstructions: l.IDInstructions,
		EntropyConfidence: l.EntropyConfidence, SortedLegend: l.SortedLegend}
}

// state fills c's readout fields from f as st states readout name: the
// labels, the head and what it must satisfy, the joint head and the flags.
// A file that states head blocks for a readout with no head is refused: the
// blocks would convert into a model nothing reads them in.
func (st *readoutStates) state(name string, f *meta.File, c *Config) error {
	if st.Backbone && !slices.Contains(c.Stated.readouts, name) {
		return fmt.Errorf("a %s readout on %s, whose library statement names no %s readout", name, f.Arch(), name)
	}
	c.DecisionLabels = st.Labels.labels()
	if st.Head == nil {
		if c.DecisionBlocks != 0 {
			return fmt.Errorf("decision type %q with %d head blocks: its readout states no head: %w", name, c.DecisionBlocks, ErrNotImplemented)
		}
	} else {
		model := jinja.NewDict()
		model.AsDict().Set("embd", jinja.NewInt(int64(c.NEmbd)))
		model.AsDict().Set("non_causal", jinja.NewBool(c.Flags.Has(jlm.FlagNonCausal)))
		vars := map[string]jinja.Value{"src": ggufKeys(f), "c": model}
		by := "readouts.json: " + name
		num := func(what, expr string) (float64, error) {
			if expr == "" {
				return 0, nil
			}
			v, err := libraryEval(by, what, expr, vars)
			if err != nil {
				return 0, err
			}
			switch {
			case v.IsInt():
				return float64(v.AsInt()), nil
			case v.IsFloat():
				return v.AsFloat(), nil
			}
			return 0, fmt.Errorf("convert: %s: %s is not a number", by, what)
		}
		head := jinja.NewDict()
		head.AsDict().Set("blocks", jinja.NewInt(int64(c.DecisionBlocks)))
		for _, h := range []struct {
			name, expr string
			to         *uint32
		}{{"routing", st.Head.Routing, &c.DecisionRouting}, {"heads", st.Head.Heads, &c.DecisionHeads},
			{"tokens", st.Head.Tokens, &c.DecisionHeadTokens}} {
			x, err := num(h.name, h.expr)
			if err != nil {
				return err
			}
			*h.to = uint32(x)
			head.AsDict().Set(h.name, jinja.NewInt(int64(*h.to)))
		}
		eps, err := num("eps", st.Head.Eps)
		if err != nil {
			return err
		}
		c.DecisionEps = float32(eps)
		head.AsDict().Set("eps", jinja.NewFloat(eps))
		vars["head"] = head
		for _, r := range st.Require {
			v, err := libraryEval(by, "require", r.That, vars)
			if err != nil {
				return err
			}
			if !v.IsTruthy() {
				return fmt.Errorf("%s", r.Else)
			}
		}
	}
	if j := st.Joint; j != nil {
		c.JointHead = &jlm.JointHead{TypeRows: j.TypeRows, SortKeys: j.SortKeys, SortChoices: j.SortChoices,
			NoulTrueFirst: j.NoulTrueFirst, IDInstructions: j.IDInstructions, Confidence: readoutConfidences()[j.Confidence]}
	}
	for _, fl := range st.Flags {
		c.Flags2 |= readoutFlags()[fl]
	}
	return nil
}

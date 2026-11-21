package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/jitllm/jitllm/engine"
)

// specFlags are -spec and its knobs, shared by run and speed.
type specFlags struct {
	kind     *string
	k        *int
	minP     *float64
	rollback *string
}

func addSpecFlags(fs *flag.FlagSet) specFlags {
	return specFlags{
		kind: fs.String("spec", "", "speculative decoding: \"mtp\" drafts with the model's own "+
			"multi-token-prediction block, \"lookup\" with the tokens that followed the sequence's last "+
			"n-gram earlier in it (any model; greedy only), \"auto\" the first where the model has one; "+
			"the drafts are verified in one pass of the model"),
		k: fs.Int("spec-k", 0, "tokens drafted a round with -spec; 0 chooses each round from this "+
			"session's measured acceptance and timing"),
		minP: fs.Float64("spec-min-p", 0, "stop a round's drafting at the first draft the prediction "+
			"block gives less than this probability; 0 drafts to the chosen count"),
		rollback: fs.String("spec-rollback", "auto", "how a hybrid's recurrent state is taken back "+
			"after a rejected draft: auto, rows (a copy per verified row; host blocks only) or replay "+
			"(the accepted rows again at the head of the next verification)"),
	}
}

// options turns the flags into the Speculator's options; nil without -spec.
func (f specFlags) options() ([]engine.SpecOption, error) {
	var dr engine.SpecDrafter
	switch *f.kind {
	case "":
		return nil, nil
	case "mtp":
		dr = engine.SpecDraftMTP
	case "lookup":
		dr = engine.SpecDraftLookup
	case "auto":
		dr = engine.SpecDraftAuto
	default:
		return nil, fmt.Errorf("-spec %q: mtp, lookup or auto", *f.kind)
	}
	var rb engine.SpecRollback
	switch *f.rollback {
	case "auto":
		rb = engine.SpecRollbackAuto
	case "rows":
		rb = engine.SpecRollbackRows
	case "replay":
		rb = engine.SpecRollbackReplay
	default:
		return nil, fmt.Errorf("-spec-rollback %q: auto, rows or replay", *f.rollback)
	}
	if *f.k < 0 || *f.minP < 0 || *f.minP > 1 {
		return nil, fmt.Errorf("-spec-k %d, -spec-min-p %g: a count and a probability", *f.k, *f.minP)
	}
	return []engine.SpecOption{engine.WithSpecDrafter(dr), engine.WithSpecDraft(*f.k), engine.WithSpecMinP(*f.minP),
		engine.WithSpecRollback(rb)}, nil
}

// printSpec reports what a Speculator did: how many tokens a verification
// pass yielded is the number speculation exists to raise.
func printSpec(w io.Writer, sp *engine.Speculator) {
	s := sp.Stats()
	pct := 0.0
	if s.Drafted > 0 {
		pct = 100 * float64(s.Accepted) / float64(s.Drafted)
	}
	on := "; the prediction block on the host"
	switch d := sp.Draft(); {
	case d == nil:
		on = ""
	case d.GPULayers() > 0:
		on = "; the prediction block on the device"
	}
	fmt.Fprintf(w, "spec     %v: %d pass(es), %d drafted, %d accepted (%.1f%%), %.2f tokens a pass; "+
		"draft %.0f ms, verify %.0f ms, catch-up %.0f ms; rollback %v (%d restored, %d row(s) replayed, %d replay pass(es))"+
		"%s\n",
		s.Drafter, s.Rounds, s.Drafted, s.Accepted, pct, s.TokensPerRound(),
		float64(s.Draft.Microseconds())/1e3, float64(s.Verify.Microseconds())/1e3,
		float64(s.CatchUp.Microseconds())/1e3, s.Rollback, s.Restored, s.Replayed, s.Commits, on)
	if s.Retired {
		fmt.Fprintf(w, "spec     retired: no draft count beat decode here even with every draft accepted; "+
			"the rest decoded plainly\n")
	}
}

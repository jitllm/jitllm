package model

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestSessionsShareAnInstance is the design's batching gate: two sessions
// leasing one instance, their prompts and tokens run a call at a time and
// as rows of one step, decode what each decodes alone on an instance of its
// own. The rows that ran together are counted, so a step that went one
// session after another fails it; reading one session's history for both
// rows (stepShare) must fail it too.
func TestSessionsShareAnInstance(t *testing.T) {
	for _, src := range []string{models["stories260K"], testmodels.Path("SmolLM2-360M-Instruct-Q8_0.gguf"),
		models["qwen3moe"]} {
		t.Run(filepath.Base(src), func(t *testing.T) {
			path, ok := existingModel(src)
			if !ok {
				t.Fatalf("model not present: %s (nor its container)", src)
			}
			// The reduction order pinned as hostStepNMSE says: the step runs the
			// rows' forms and alone runs the one-row forms.
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false), WithJITOptions(nn.WithGEMMExact(true)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			prompts := [][]int32{{1, 45, 79, 33}, {1, 65, 108, 250, 205, 63}}
			const gen = 12
			alone := make([][]int32, len(prompts))
			for i, p := range prompts {
				st := m.NewInstance(64)
				lg, err := st.Prefill(p)
				if err != nil {
					t.Fatal(err)
				}
				for range gen {
					tok := int32(argmax(lg))
					alone[i] = append(alone[i], tok)
					if lg, err = st.Forward(tok); err != nil {
						t.Fatal(err)
					}
				}
				st.Close()
			}
			got, joint := sharedDecode(t, m, prompts, gen)
			for i := range prompts {
				if !equalIDs(got[i], alone[i]) {
					t.Errorf("session %d decoded %v on the shared instance, %v alone", i, got[i], alone[i])
				}
			}
			if joint == 0 {
				t.Errorf("no step ran the sessions' rows together")
			}
			stepShare = true
			bad, _ := sharedDecode(t, m, prompts, gen)
			stepShare = false
			if equalIDs(bad[1], alone[1]) {
				t.Errorf("rows reading the first session's history decoded session 1's tokens: the gate cannot see a shared history")
			}
		})
	}
}

// sharedDecode runs prompts as sessions leasing one instance: each prefill a
// call of its own, then gen tokens, odd tokens a call per session and even
// ones every session's row in one step. It returns each session's tokens
// and the rows that stepped together.
func sharedDecode(t *testing.T, m *Model, prompts [][]int32, gen int) ([][]int32, int64) {
	return sharedDecodeArms(t, m, prompts, gen, 2)
}

// sharedDecodeArms is sharedDecode with every token's arm chosen by every:
// 1 a step every token, 0 a call per session every token, 2 alternating.
func sharedDecodeArms(t *testing.T, m *Model, prompts [][]int32, gen, every int) ([][]int32, int64) {
	t.Helper()
	ctx := context.Background()
	in := m.NewInstance(64)
	defer in.Close()
	if err := m.Serve(in, 0); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := m.Unserve(in); err != nil {
			t.Error(err)
		}
	}()
	sess := make([]*Session, len(prompts))
	last := make([][]float32, len(prompts))
	for i, p := range prompts {
		s, err := m.NewSession(64)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		sess[i] = s
		if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
			lg, err := in.Prefill(p)
			last[i] = lg
			return err
		}); err != nil {
			t.Fatal(err)
		}
	}
	out := make([][]int32, len(prompts))
	before := m.HostStepRows()
	runs := make([]Run, len(prompts))
	for k := range gen {
		for i := range sess {
			out[i] = append(out[i], int32(argmax(last[i])))
		}
		if every == 0 || every == 2 && k%2 == 1 {
			for i, s := range sess {
				tok := out[i][len(out[i])-1]
				if err := s.Run(ctx, PhaseGenerate, func(in *Instance) error {
					lg, err := in.Forward(tok)
					last[i] = lg
					return err
				}); err != nil {
					t.Fatal(err)
				}
			}
			continue
		}
		for i, s := range sess {
			if err := s.Acquire(ctx, PhaseGenerate); err != nil {
				t.Fatal(err)
			}
			runs[i] = Run{Session: s, Tokens: out[i][len(out[i])-1:], Logits: true}
		}
		lg, err := StepRuns(runs)
		if err != nil {
			t.Fatal(err)
		}
		for i := range sess {
			last[i] = lg[i]
		}
	}
	return out, m.HostStepRows() - before
}

// TestSessionsStepWithinTheBandOnF16KV is TestSessionsShareAnInstance's
// binary16 history, the default it pins away: two sessions stepped together
// on one instance, teacher-forced with the tokens each decodes alone, hold
// every step's logits to hostStepNMSE against each alone. A session's rows
// read a binary16 history through the rows' form and alone through the
// one-row form, which round apart; a row reading the other session's
// history (stepShare) is whole logits off and must fail it.
func TestSessionsStepWithinTheBandOnF16KV(t *testing.T) {
	for _, src := range []string{testmodels.Path("SmolLM2-360M-Instruct-Q8_0.gguf"), models["qwen3moe"]} {
		t.Run(filepath.Base(src), func(t *testing.T) {
			path, ok := existingModel(src)
			if !ok {
				t.Fatalf("model not present: %s (nor its container)", src)
			}
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(true), WithJITOptions(nn.WithGEMMExact(true)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			prompts := [][]int32{{1, 45, 79, 33}, {1, 65, 108, 250, 205, 63}}
			const gen = 10
			toks := make([][]int32, len(prompts))
			want := make([][][]float32, len(prompts))
			for i, p := range prompts {
				st := m.NewInstance(64)
				lg, err := st.Prefill(p)
				if err != nil {
					t.Fatal(err)
				}
				for range gen {
					tok := int32(argmax(lg))
					toks[i] = append(toks[i], tok)
					if lg, err = st.Forward(tok); err != nil {
						t.Fatal(err)
					}
					want[i] = append(want[i], append([]float32(nil), lg...))
				}
				st.Close()
			}
			got := forcedSteps(t, m, prompts, toks)
			for i := range prompts {
				if w, at := worstStep(got[i], want[i]); w > hostStepNMSE {
					t.Errorf("session %d stepped together reads NMSE %.3e at token %d against alone", i, w, at)
				} else {
					t.Logf("session %d: worst NMSE %.3e (token %d)", i, w, at)
				}
			}
			stepShare = true
			bad := forcedSteps(t, m, prompts, toks)
			stepShare = false
			if w, _ := worstStep(bad[1], want[1]); w < hostStepNMSE {
				t.Errorf("rows reading the first session's history read NMSE %.3e: the band cannot see a shared history", w)
			}
		})
	}
}

// forcedSteps runs prompts as sessions leasing one instance and steps them
// together, token by token as toks says, returning each step's logits.
func forcedSteps(t *testing.T, m *Model, prompts, toks [][]int32) [][][]float32 {
	t.Helper()
	ctx := context.Background()
	in := m.NewInstance(64)
	defer in.Close()
	if err := m.Serve(in, 0); err != nil {
		t.Fatal(err)
	}
	defer m.Unserve(in)
	sess := make([]*Session, len(prompts))
	for i, p := range prompts {
		s, err := m.NewSession(64)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		sess[i] = s
		if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
			_, err := in.Prefill(p)
			return err
		}); err != nil {
			t.Fatal(err)
		}
		if err := s.Acquire(ctx, PhaseGenerate); err != nil {
			t.Fatal(err)
		}
	}
	out := make([][][]float32, len(prompts))
	runs := make([]Run, len(prompts))
	for k := range toks[0] {
		for i, s := range sess {
			runs[i] = Run{Session: s, Tokens: toks[i][k : k+1], Logits: true}
		}
		lg, err := StepRuns(runs)
		if err != nil {
			t.Fatal(err)
		}
		for i := range sess {
			out[i] = append(out[i], append([]float32(nil), lg[i]...))
		}
	}
	return out
}

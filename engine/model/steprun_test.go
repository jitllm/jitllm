package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestStepRunsAdmitsAPromptBesideDecoding is the gate on a step across
// sessions carrying a prompt chunk: two sessions decode while a third session's
// prompt rides beside them in chunks of three tokens -- rows of its own
// sequence at consecutive positions in the same step as the other sessions'
// decoding rows -- and then all three decode together. Teacher-forced against
// each session alone (Prefill and Forward on the same device): every decoding
// row's logits, the late session's logits at the end of its prompt, and its
// decoding after.
//
// It selects what it gates -- every row ran across sessions
// (Stats.SessionRows), and on a hybrid a chunk's rows chained through the
// late session's recurrent state on the device (Stats.RecChainRows) -- and it
// runs the violation: each chunk laid out at mirrored positions must part from
// the session alone by more than the bound.
//
// A dense model and a hybrid, on every device named in JITLLM_TEST_DEVICES
// (default "cuda:0,vulkan:0,metal").
func TestStepRunsAdmitsAPromptBesideDecoding(t *testing.T) {
	for _, c := range []struct {
		name string
		open func(*testing.T) *Model
	}{{"dense", openStepDense}, {"hybrid", openStepHybrid}} {
		t.Run(c.name, func(t *testing.T) {
			m := c.open(t)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					g := stepTier(t, m, dev, false)
					defer g.Close()
					plan := runsPlan{early: []string{"The capital of France is", "Water boils at a temperature of"},
						late:  "The clerk counted barrels of salt on the upper floor, and",
						chunk: 3, steps: 4}
					want := plan.alone(t, m, g)
					s0 := g.Stats()
					got, rows := plan.together(t, m, g)
					s1 := g.Stats()
					if n := s1.SessionRows - s0.SessionRows; n != rows {
						t.Fatalf("%d rows ran across sessions, want %d: StepRuns went one instance after another (%s)",
							n, rows, g.Err())
					}
					if chained := s1.RecChainRows - s0.RecChainRows; m.Cfg.Hybrid() && chained == 0 {
						t.Fatal("no row of the late prompt's chunks chained a device linear block's state: " +
							"the chunks did not ride as runs")
					} else if m.Cfg.Hybrid() {
						t.Logf("%d row-steps of linear blocks chained a chunk's state", chained)
					}
					w, at := worstStep(got, want)
					t.Logf("%d rows, %d logit rows compared: worst NMSE %.3e (row %d) against each session alone",
						rows, len(want), w, at)
					if !(w < mlaDeviceNMSE) {
						for i := range want {
							if e := logitNMSE(got[i], want[i]); !(e < mlaDeviceNMSE) {
								t.Logf("logit row %d: NMSE %.3e", i, e)
							}
						}
						t.Fatalf("a step carrying a prompt chunk disagrees with each session alone: NMSE %.3e at row %d", w, at)
					}

					// The violation: the late prompt's chunks back to front.
					m.eng.stepPosFault = func(base, j, k int) int { return base + k - 1 - j }
					bad, _ := plan.together(t, m, g)
					m.eng.stepPosFault = nil
					bw, bat := worstStep(bad, want)
					if bw < mlaDeviceNMSE {
						t.Fatalf("chunks at mirrored positions still matched each session alone (NMSE %.3e): "+
							"this gate cannot see a chunk's positions", bw)
					}
					t.Logf("violation: mirrored chunks read NMSE %.3e at row %d", bw, bat)
				})
			}
		})
	}
}

// openStepDense opens the dense model the step gates run.
func openStepDense(t *testing.T) *Model {
	t.Helper()
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// runsPlan is a schedule for StepRuns: sessions decoding from the start, their
// prompts prefilled alone, and a late session whose prompt rides beside them
// in chunks, after which every session decodes together for steps more.
type runsPlan struct {
	early []string
	late  string
	chunk int
	steps int
	// The teacher: each session's tokens alone, which both arms feed.
	earlyToks [][]int32
	lateToks  []int32
}

// chunks is how many steps the late prompt takes.
func (p *runsPlan) chunks(n int) int { return (n + p.chunk - 1) / p.chunk }

// alone runs every session by itself on g, and returns the logits the joint
// arm must give, in its order: step by step, each early session's, then the
// late session's where its run wanted them.
func (p *runsPlan) alone(t *testing.T, m *Model, g *gpu.GPU) [][]float32 {
	t.Helper()
	late := m.Vocab.Encode(p.late, true)
	total := p.chunks(len(late)) + p.steps
	p.earlyToks = make([][]int32, len(p.early))
	var early [][][]float32
	for i, pr := range p.early {
		st := p.placed(t, m, g, len(m.Vocab.Encode(pr, true))+total+2)
		lg, err := st.Prefill(m.Vocab.Encode(pr, true))
		if err != nil {
			t.Fatal(err)
		}
		var lgs [][]float32
		for range total {
			next := Greedy(lg)
			p.earlyToks[i] = append(p.earlyToks[i], next)
			if lg, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
			lgs = append(lgs, append([]float32(nil), lg...))
		}
		early = append(early, lgs)
		st.Close()
	}
	st := p.placed(t, m, g, len(late)+p.steps+2)
	defer st.Close()
	lg, err := st.Prefill(late)
	if err != nil {
		t.Fatal(err)
	}
	lateLg := [][]float32{append([]float32(nil), lg...)}
	p.lateToks = nil
	for range p.steps {
		next := Greedy(lg)
		p.lateToks = append(p.lateToks, next)
		if lg, err = st.Forward(next); err != nil {
			t.Fatal(err)
		}
		lateLg = append(lateLg, append([]float32(nil), lg...))
	}
	var want [][]float32
	nc := p.chunks(len(late))
	for s := range total {
		for i := range p.early {
			want = append(want, early[i][s])
		}
		if s >= nc-1 {
			want = append(want, lateLg[s-(nc-1)])
		}
	}
	return want
}

// together runs the plan through StepRuns on g, teacher-forced by alone's
// tokens, and returns the logits in alone's order and how many rows ran.
func (p *runsPlan) together(t *testing.T, m *Model, g *gpu.GPU) ([][]float32, int) {
	t.Helper()
	late := m.Vocab.Encode(p.late, true)
	nc := p.chunks(len(late))
	total := nc + p.steps
	var sts []*Instance
	for _, pr := range p.early {
		ids := m.Vocab.Encode(pr, true)
		st := p.placed(t, m, g, len(ids)+total+2)
		defer st.Close()
		if _, err := st.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		sts = append(sts, st)
	}
	ls := p.placed(t, m, g, len(late)+p.steps+2)
	defer ls.Close()
	var got [][]float32
	rows := 0
	for s := range total {
		var runs []Run
		for i, st := range sts {
			runs = append(runs, Run{Instance: st, Tokens: p.earlyToks[i][s : s+1], Logits: true})
		}
		if s < nc {
			lo, hi := s*p.chunk, min(len(late), (s+1)*p.chunk)
			runs = append(runs, Run{Instance: ls, Tokens: late[lo:hi], Logits: hi == len(late)})
		} else {
			runs = append(runs, Run{Instance: ls, Tokens: p.lateToks[s-nc : s-nc+1], Logits: true})
		}
		out, err := StepRuns(runs)
		if err != nil {
			t.Fatal(err)
		}
		for i, r := range runs {
			rows += len(r.Tokens)
			if r.Logits != (out[i] != nil) {
				t.Fatalf("step %d run %d: logits wanted %v, returned %v", s, i, r.Logits, out[i] != nil)
			}
			if out[i] != nil {
				got = append(got, append([]float32(nil), out[i]...))
			}
		}
	}
	return got, rows
}

// placed is an instance of maxSeq positions with every block and the head on g,
// or the gate skips naming the card; with g nil, an instance on the host.
func (p *runsPlan) placed(t *testing.T, m *Model, g *gpu.GPU, maxSeq int) *Instance {
	t.Helper()
	st := m.NewInstance(maxSeq)
	// No tier is the host arm (hoststep_test.go): the instance stays home.
	if g == nil {
		return st
	}
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
		st.Close()
		t.Skipf("CARD TOO SMALL: %d of %d blocks, head %v -- this gate proved nothing here (%s)",
			st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
	}
	return st
}

// TestStepRunsStepsTheSeamTuner: a session decoding as a row of a joint step
// steps its seam tuner once a token, as Forward does, so a server's step loop
// can tune a seam. The warm-up is set past the steps so no arm migrates. The
// selection check is that every row ran across sessions.
func TestStepRunsStepsTheSeamTuner(t *testing.T) {
	m := openStepDense(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g := stepTier(t, m, dev, false)
			defer g.Close()
			p := &runsPlan{}
			const steps = 5
			var sts []*Instance
			for _, pr := range []string{"The capital of France is", "Water boils at"} {
				ids := m.Vocab.Encode(pr, true)
				st := p.placed(t, m, g, len(ids)+steps+2)
				defer st.Close()
				if _, err := st.Prefill(ids); err != nil {
					t.Fatal(err)
				}
				st.SetSeamTuning(true)
				if st.seam == nil {
					t.Skip("SEAM TUNER NOT ARMED: no device seam -- this gate proved nothing here")
				}
				st.SetSeamSchedule(1000, 0, 0)
				sts = append(sts, st)
			}
			s0 := g.Stats()
			for range steps {
				runs := []Run{{Instance: sts[0], Tokens: []int32{1}, Logits: true}, {Instance: sts[1], Tokens: []int32{1}, Logits: true}}
				if _, err := StepRuns(runs); err != nil {
					t.Fatal(err)
				}
			}
			if n := g.Stats().SessionRows - s0.SessionRows; n != 2*steps {
				t.Fatalf("%d rows ran across sessions, want %d (%s)", n, 2*steps, g.Err())
			}
			for i, st := range sts {
				if st.seam.tok != steps {
					t.Fatalf("session %d's seam tuner counted %d tokens of %d joint steps", i, st.seam.tok, steps)
				}
			}
		})
	}
}

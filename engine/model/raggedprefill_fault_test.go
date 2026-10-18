//go:build jitllmfault

package model

import (
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestRaggedStepAttendsAsAPrefill holds a batch session's prompt chunks,
// attended through the run table's flash prefill (tier raggedprefill.go,
// docs/design/ragged-prefill.md), to each prompt's own prefill on the same
// device: each sequence's first-token logits in the device band. The launches
// must have been the run table's (Stats.RagPrefills), the decode-plan arm
// (Config.NoRagPrefill) is held to the same band as the control, and the
// violation -- the table's rows reading their keys through the emptied
// descriptors (rows fault "ragtab") -- must part.
func TestRaggedStepAttendsAsAPrefill(t *testing.T) {
	// LFM2-350M, and gpt-oss's fixture for the sinks (one split, the mapped
	// merge adding the sink).
	for _, name := range []string{"lfm2/LFM2-350M-Q4_K_M.gguf", "synth-gptoss.gguf"} {
		t.Run(name, func(t *testing.T) {
			path := testmodels.Path(name)
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) (RULE 11)", err)
			}
			m, err := Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			vocab := min(m.Cfg.NVocab, 30000)
			prompts := [][]int32{linPrompt(130, vocab), linPrompt(170, vocab)[3:], linPrompt(90, vocab)[5:]}
			for _, arm := range []struct {
				name  string
				tiles bool
			}{{"default", false}, {"tiles", true}} {
				t.Run(arm.name, func(t *testing.T) { raggedAsPrefill(t, m, prompts, arm.tiles) })
			}
		})
	}
}

// raggedAsPrefill is TestRaggedStepAttendsAsAPrefill on one form: the
// device's own choice, or the collective tiles (Config.PreferTiles,
// FlashPrefillTile's run table) where they lower.
func raggedAsPrefill(t *testing.T, m *Model, prompts [][]int32, tiles bool) {
	const seq = 400
	open := func(t *testing.T, noRag bool) *tier.GPU {
		o := append(testTierOpts(t), tier.WithDeviceTune(tier.TuneOff),
			tier.WithConfig(func(c *tier.Config) { c.NoRagPrefill, c.PreferTiles = noRag, tiles }))
		if testdevices.Selected() {
			o = append(o, tier.WithDevices(testdevices.One("")))
		}
		g, err := tier.OpenWith(o...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		return g
	}
	place := func(t *testing.T, st *State, g *tier.GPU) {
		if err := st.SetDeviceLayers(g, -1); err != nil || st.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("%v: %d of %d blocks placed: %s", err, st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
	}
	// Each prompt alone on the device: the reference.
	g := open(t, false)
	defer g.Close()
	want := make([][]float32, len(prompts))
	for i, p := range prompts {
		st := m.NewState(seq)
		place(t, st, g)
		lg, err := st.Prefill(p)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = append([]float32(nil), lg...)
		st.Close()
	}
	// A device whose flash prefill has no run-table form yet (sm_70's
	// FlashPrefill70, the staged kernels) keeps the decode plan in ragged
	// steps.
	if g.Stats().RagForms == 0 {
		t.Skipf("%s's prefill form has no run table yet: this gate proved nothing here", g.Name())
	}
	batch := func(t *testing.T, g *tier.GPU, fault string) ([][]float32, tier.Stats) {
		sc := NewScheduler(m, len(prompts), seq)
		defer sc.Close()
		place(t, sc.st, g)
		got := make([][]float32, len(prompts))
		seqs := make([]Seq, len(prompts))
		for i := range seqs {
			seqs[i] = Seq{Prompt: prompts[i], MaxTokens: 1, Sample: func(lg []float32) int32 {
				got[i] = append([]float32(nil), lg...)
				return 0
			}}
			sc.Submit(&seqs[i])
		}
		g.SetFault("rows", fault)
		defer g.SetFault("rows", "")
		s0 := g.Stats()
		for sc.Pending() > 0 || sc.Live() > 0 {
			done, err := sc.Step()
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range done {
				if s.Err != nil {
					t.Fatal(s.Err)
				}
			}
		}
		s1 := g.Stats()
		s1.RagPrefills -= s0.RagPrefills
		return got, s1
	}
	worst := func(got [][]float32) float64 {
		w := 0.0
		for i := range want {
			w = max(w, logitNMSE(want[i], got[i]))
		}
		return w
	}
	// The band is the batch step's other reduction orders (its GEMMs are
	// the row kernels, not the prefill's tiles): the decode-plan control
	// reads 2.9e-3 on an sm_86 card, the run table the same.
	const band = 1e-2
	got, st := batch(t, g, "")
	if st.RagPrefills == 0 {
		t.Fatal("no run-table launch: the ragged step's chunks never attended as a prefill")
	}
	e := worst(got)
	t.Logf("run table: %d launches, NMSE %.3g", st.RagPrefills, e)
	if e > band {
		t.Fatalf("the run table's batch against each prompt alone: NMSE %.3g (band %g)", e, band)
	}
	msg := fmt.Sprintf("%d run-table launches; NMSE %.3g against each alone", st.RagPrefills, e)

	gc := open(t, true)
	defer gc.Close()
	got, st = batch(t, gc, "")
	if st.RagPrefills != 0 {
		t.Fatalf("NoRagPrefill still launched the run table %d times", st.RagPrefills)
	}
	ec := worst(got)
	t.Logf("decode-plan control: NMSE %.3g", ec)
	if ec > band {
		t.Fatalf("the decode-plan control against each prompt alone: NMSE %.3g (band %g)", ec, band)
	}
	got, _ = batch(t, g, "ragtab")
	ev := worst(got)
	t.Logf("violation: NMSE %.3g", ev)
	if ev < 10*band {
		t.Fatalf("the violation (the table's rows reading the emptied descriptors) moved the logits only %.3g", ev)
	}
	t.Logf("%s; the decode-plan control %.3g", msg, ec)
}

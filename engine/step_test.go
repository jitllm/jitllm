package engine

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestStepAcrossSessionsMatchesEachAlone decodes three instances -- three
// sessions on one device, three different prompts -- once a token at a time
// each, and once together through Step, and demands the same logits: each row
// of the joint step must read its own session's history and no other.
// Stats.SessionRows is the selection check: a Step that fell back to one
// Forward after another would compare the arm with itself.
func TestStepAcrossSessionsMatchesEachAlone(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// Every backend this host has: the step's matvecs are chosen per device
	// (an integer matrix unit, sm_70's tensor cores, neither), and a gate that
	// asks for one backend covers none of the others (RULE 11d).
	// JITLLM_TEST_DEVICES names them where a host's default index is not the
	// card to use.
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) { stepAcrossSessions(t, m, spec, instOpts...) })
	}
}

func stepAcrossSessions(t *testing.T, m *model.Model, spec string, io ...Option) {
	ctl := orderControl(t, m, spec, io...)
	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff)},
		testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	r := stepGate(t, m, g, false, io...)
	// The configuration this gates: the composer serves the step's rows, every
	// sequence's, as one call of its blocks.
	if n, d := g.Stats().ServedPart(nn.ComposedBlock); n == 0 || d != 0 {
		t.Fatalf("the composer served %d of the step's block steps and declined %d (%s)", n, d, g.Err())
	}
	if bound := denseStepBand * ctl; !(r.worst < bound) {
		t.Fatalf("a step across sessions disagrees with each session alone: NMSE %.3e at row %d, bound %.3e "+
			"(%dx the order-only control)", r.worst, r.at, bound, denseStepBand)
	}
}

// TestStepAcrossSessionsHybridMatchesEachAlone is the same gate on a hybrid:
// each row's linear blocks must step its own session's recurrent state. A
// device keeps every session's state for a block in one pool and a row finds
// its own through its seat (tier/recpool.go), so the failure it guards is two
// sessions reading one summary -- fluent text on every row, from a history
// that is not the row's.
//
// Halfway through, the first session decodes one token alone, in both arms,
// so its recurrent states sit in the other half of every pool from the other
// sessions': the next joint step must bring them to one half first
// (Stats.RecAligns), and no other step after the first may copy anything,
// since sessions stepping together stay aligned. Where the halves stand after
// the prompts is the prompts' business (a device prefilling a row at a time
// flips once a token), so the first joint step may or may not align.
// Stats.SessionLinear is the selection check that every linear block ran in
// every joint step.
//
// Both forms of the pools run: two halves of every state, and the shared form
// (gpu.Config.SharedRec: one state a slot, the step writing the device's
// next-state scratch and a copy taking each row's home).
//
// It runs on every device named in JITLLM_TEST_DEVICES (comma-separated,
// default "cuda:0,vulkan:0,metal"); one that does not open here is skipped by name.
// JITLLM_STEP_HYBRID names another hybrid.
func TestStepAcrossSessionsHybridMatchesEachAlone(t *testing.T) {
	m, openOpts := openStepHybrid(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		for _, shared := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/shared=%v", dev, shared), func(t *testing.T) {
				g := stepTier(t, m, dev, shared)
				defer g.Close()
				r := stepGate(t, m, g, true, openOpts...)
				hybridStepChecks(t, m, r)
				if got := g.Stats().RecShared; (got > 0) != shared {
					t.Fatalf("%d pool(s) sized in the shared form with shared=%v: the arm ran the other form",
						got, shared)
				}
				if !(r.worst < mlaDeviceNMSE) {
					t.Fatalf("a hybrid's step across sessions disagrees with each session alone: NMSE %.3e at row %d",
						r.worst, r.at)
				}
			})
		}
	}
}

// TestStepAcrossSessionsMixtureMatchesEachAlone is the same gate on a mixture:
// a step of two or more rows routes on the device and runs its experts in the
// composer's rows form, where each session alone runs decode's one-row
// mixture. A joint step needs every block placed, so a 4 GiB card skips olmoe
// by name and only a larger one covers it. Stats.ServedPart(ComposedBlock) is
// the selection check: every block of every joint step ran on the composer.
//
// JITLLM_STEP_MIXTURE names the mixtures (comma-separated); the default is
// Qwen3-MoE-4x0.6B and olmoe.
func TestStepAcrossSessionsMixtureMatchesEachAlone(t *testing.T) {
	for _, name := range stepMixtures() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m, openOpts := openStepMixture(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					g := stepTier(t, m, dev, false)
					defer g.Close()
					r := stepGate(t, m, g, false, openOpts...)
					mixtureStepChecks(t, m, r)
					if !(r.worst < mixtureStepNMSE) {
						t.Fatalf("a mixture's step across sessions disagrees with each session alone: NMSE %.3e "+
							"at row %d", r.worst, r.at)
					}
				})
			}
		})
	}
}

// mixtureStepNMSE is the mixture step gate's bound on its worst row. A joint
// step runs the experts grouped (int8 by dp4a, or binary16 on sm_70's tensor
// cores) where each session alone runs decode's one-row mixture, and a router
// near a tie amplifies that arithmetic: on a V100 and on a 3050 Ti the clean
// arms read 1.2e-3..1.04e-2, past the dense gates' 1e-2, while every down
// reading the gate's output ("downsrc") reads 1.46..1.94.
const mixtureStepNMSE = 5e-2

// stepMixtures is the mixtures the mixture step gates run.
func stepMixtures() []string {
	if v := os.Getenv("JITLLM_STEP_MIXTURE"); v != "" {
		return strings.Split(v, ",")
	}
	return []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "olmoe-1b-7b-0924-instruct-Q4_K_M.gguf",
		"synth-glm4moe.gguf", "synth-qwen2moe.gguf", "synth-ernie45moe.gguf", "synth-hunyuanmoe.gguf",
		"synth-minimaxm2.gguf", "synth-gemma4-moe.gguf", "synth-minimaxm3.gguf",
		"synth-bailingmoe2.gguf", "synth-dots1.gguf", "synth-phimoe.gguf",
		"synth-kimik3.gguf"}
}

// openStepMixture opens a mixture for the step gates, refusing anything that
// is not one: a dense model would pass with nothing grouped in it.
func openStepMixture(t *testing.T, name string) (*model.Model, []Option) {
	t.Helper()
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(jlmOf(t, testmodels.Path(name)))
	if err != nil {
		t.Fatal(err)
	}
	if m.Cfg.NExpert == 0 {
		m.Close()
		t.Fatalf("%s is not a mixture -- this gate would prove nothing", name)
	}
	return m, instOpts
}

// mixtureStepChecks is the selection a mixture step gate demands: every
// block of every joint step, its mixtures among them, ran on the composer,
// none declined. A dense model would pass it, so the gates name mixtures.
func mixtureStepChecks(t *testing.T, m *model.Model, r stepResult) {
	t.Helper()
	want := r.steps * m.Cfg.NLayer
	t.Logf("%d block steps on the composer across %d joint steps, %d declined", r.blocks, r.steps, r.blockDeclined)
	if r.blocks < want || r.blockDeclined > 0 {
		t.Fatalf("%d block steps on the composer across %d joint steps (%d declined), want %d: the step took another path",
			r.blocks, r.steps, r.blockDeclined, want)
	}
}

// stepTier opens dev for a step gate on m, its recurrent pools in the shared
// form when asked, or skips naming the device.
func stepTier(t *testing.T, m *model.Model, dev string, shared bool, extra ...gpu.Option) *gpu.GPU {
	t.Helper()
	g, err := openGPU(t, append(append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) { c.SharedRec = shared })}, testTierOpts(t)...), extra...)...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	t.Logf("%s is %s", dev, g.Name())
	return g
}

// openStepHybrid opens the hybrid the step gates run, refusing anything that
// is not one: a dense model would pass every check here with nothing linear in
// it.
func openStepHybrid(t *testing.T) (*model.Model, []Option) {
	t.Helper()
	name := os.Getenv("JITLLM_STEP_HYBRID")
	if name == "" {
		name = "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"
	}
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.Hybrid() {
		m.Close()
		t.Fatalf("%s is not a hybrid -- this gate would prove nothing", name)
	}
	return m, instOpts
}

// stepDevices is the backends a device gate runs on: JITLLM_TEST_DEVICES, or
// one of each backend at index 0. Every gate that sweeps backends reads it, so
// a shared box confines them all to the card it was given.
func stepDevices() []string { return testdevices.Specs("cuda:0", "vulkan:0", "metal") }

// hybridStepChecks is the selection a hybrid step gate demands: every linear
// block of every joint step ran across sessions, the session that stepped
// apart had every block's state aligned at the next joint step, and no other
// joint step after the first copied anything.
func hybridStepChecks(t *testing.T, m *model.Model, r stepResult) {
	t.Helper()
	linear := 0
	for li := 0; li < m.Cfg.NLayer; li++ {
		if m.Cfg.LayerKind(li).Recurrent() {
			linear++
		}
	}
	t.Logf("%d linear block(s) of %d; %d run across sessions; state(s) aligned: %d at the first joint step, "+
		"%d after the first session stepped apart, %d at any other", linear, m.Cfg.NLayer, r.linear,
		r.alignsFirst, r.alignsApart, r.alignsOther)
	switch {
	case r.linear != r.steps*linear:
		t.Fatalf("%d linear block(s) ran across sessions, want %d: the recurrence went somewhere else",
			r.linear, r.steps*linear)
	case r.alignsApart != linear:
		t.Fatalf("%d state(s) aligned after the first session stepped apart, want %d (one a linear block): "+
			"the gate did not reach the alignment it exists to test", r.alignsApart, linear)
	case r.alignsOther != 0:
		t.Fatalf("%d state(s) copied at joint steps whose sessions had stepped together: they must stay "+
			"aligned, or every step pays a copy of every state", r.alignsOther)
	}
}

// stepResult is what stepGate measured.
type stepResult struct {
	worst  float64
	at     int
	steps  int
	linear int // linear blocks run across sessions
	// Block steps the composer ran across the joint steps, served or
	// replayed, and declined.
	blocks, blockDeclined int
	// instances aligned at the first joint step, at the one after the first
	// session stepped apart, and at every other.
	alignsFirst, alignsApart, alignsOther int
	// refFinite is every logit of the sessions alone finite: a reference
	// that is not makes every comparison against it degenerate.
	refFinite bool
}

// stepPrompts is the sessions stepGate steps together, one prompt each.
func stepPrompts() []string {
	return []string{"The capital of France is", "Water boils at a temperature of",
		"The clerk counted barrels of salt on the upper floor, and"}
}

// stepGate decodes three prompts on g once an instance at a time and once
// together through Step, teacher-forced, and returns the worst logit NMSE of
// the joint arm against each session alone. With apart, the first session
// decodes one token on its own halfway through, in both arms. It fails if
// Step did not run every row across sessions.
func stepGate(t *testing.T, m *model.Model, g *gpu.GPU, apart bool, io ...Option) stepResult {
	t.Helper()
	prompts := stepPrompts()
	const steps, mid = 8, 4
	// arm opens an instance per prompt and prefills it. It returns the instances
	// and each one's next token. The alone arm's are closed as soon as its
	// logits are taken, so a card holds one arm's histories at a time.
	closed := map[*Instance]bool{}
	arm := func() ([]*Instance, []int32) {
		var sts []*Instance
		var next []int32
		for _, pr := range prompts {
			ids := m.Vocab.Encode(pr, true)
			st := testInstance(t, m, len(ids)+steps+2, io...)
			t.Cleanup(func() {
				if !closed[st] {
					st.Close()
				}
			})
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
				t.Skipf("CARD TOO SMALL: %d of %d blocks, head %v -- this gate proved nothing here (%s)",
					st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
			}
			lg, err := st.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			sts, next = append(sts, st), append(next, Greedy(lg))
		}
		return sts, next
	}
	alone, next := arm()
	// Teacher-forced: the joint step reads the tokens each session chose
	// alone. Letting each arm follow its own greedy choice turns one near-tie
	// (a 0.045 margin at step 2 of the first prompt) into every later row
	// comparing two different tokens' logits. solo is the first session's
	// token on its own, and its logits are compared too.
	var want [][]float32
	var fed [][]int32
	var solo int32
	toks := append([]int32(nil), next...)
	for s := range steps {
		if apart && s == mid {
			solo = toks[0]
			lg, err := alone[0].Forward(solo)
			if err != nil {
				t.Fatal(err)
			}
			want = append(want, append([]float32(nil), lg...))
			toks[0] = Greedy(lg)
		}
		fed = append(fed, append([]int32(nil), toks...))
		for i, st := range alone {
			lg, err := st.Forward(toks[i])
			if err != nil {
				t.Fatal(err)
			}
			want = append(want, append([]float32(nil), lg...))
			toks[i] = Greedy(lg)
		}
	}
	for _, st := range alone {
		st.Close()
		closed[st] = true
	}
	together, _ := arm()
	// The selection named before anything runs: an instance StepRuns would take
	// alone says why, where the row count below could only say that it did.
	for i, st := range together {
		switch err := st.StepRefusal(); err {
		case nil:
		case errStepNoHead, errStepSplit:
			// Placed whole and then moved: a device that failed the prompt
			// is a failure, the prompts' history taking the room
			// (relocation) a card too small for this gate.
			if n := st.DeviceDemotions(); n > 0 {
				t.Fatalf("session %d's prompt failed on the device %d time(s) and moved the model home: %v (%s)",
					i, n, err, g.Err())
			}
			t.Skipf("CARD TOO SMALL: session %d's prompt moved part of the model home: %v (%s) -- this gate "+
				"proved nothing here", i, err, g.Err())
		default:
			t.Fatalf("session %d cannot step across sessions: %v (%s)", i, err, g.Err())
		}
	}
	s0 := g.Stats()
	var got [][]float32
	r := stepResult{steps: steps}
	for s := range steps {
		if apart && s == mid {
			lg, err := together[0].Forward(solo)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, append([]float32(nil), lg...))
		}
		before := g.Stats().RecAligns
		lgs, err := Step(together, fed[s])
		if err != nil {
			t.Fatal(err)
		}
		switch n := g.Stats().RecAligns - before; {
		case s == 0:
			r.alignsFirst = n
		case apart && s == mid:
			r.alignsApart = n
		default:
			r.alignsOther += n
		}
		for _, lg := range lgs {
			got = append(got, append([]float32(nil), lg...))
		}
	}
	s1 := g.Stats()
	if rows := s1.SessionRows - s0.SessionRows; rows != steps*len(prompts) {
		t.Fatalf("%d rows ran across sessions, want %d: Step went one Forward after another (%s)",
			rows, steps*len(prompts), g.Err())
	}
	r.linear = s1.SessionLinear - s0.SessionLinear
	// A recording's replay runs the composer's parts as its capture did.
	b1, d1 := s1.ServedPart(nn.ComposedBlock)
	b0, d0 := s0.ServedPart(nn.ComposedBlock)
	r.blocks = b1 - b0 + s1.ReplayedPart(nn.ComposedBlock) - s0.ReplayedPart(nn.ComposedBlock)
	r.blockDeclined = d1 - d0
	r.worst, r.at = worstStep(got, want)
	r.refFinite = true
	for _, w := range want {
		for _, v := range w {
			r.refFinite = r.refFinite && !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
		}
	}
	t.Logf("%d sessions x %d steps together: worst logit NMSE %.3e (row %d) against each alone; "+
		"rows %s", len(prompts), steps, r.worst, r.at, stepRows(got, want))
	return r
}

// stepRows is every row's logit NMSE between two arms, in row order, so a
// worst row reads against the rows before it: a step that moves every row
// past its control differs from one in which a single near-tie row amplifies.
func stepRows(got, want [][]float32) string {
	var b strings.Builder
	for i := range want {
		fmt.Fprintf(&b, " %.1e", logitNMSE(got[i], want[i]))
	}
	return b.String()
}

// denseStepBand is how many times the order-only control (orderControl) a
// dense model's step across sessions may sit from each session alone. The
// step sums in another order from decode -- its few-sequence matvec and its
// ragged attention are other kernels -- and how far an order change carries
// is the model's own property: Llama-3.2-1B's decode moved by a split pin
// reads a worst row of 8.6e-03 over 256 on a V100 and Qwen3-4B's 6.7e-02,
// and the step sits inside each (6.9e-03, 2.6e-02), so no one constant fits
// both. Measured as the gate measures it, 37 (model, device) pairs on the
// V100, the 3050 Ti, the Iris Xe and the M4 read step/control at a geometric
// mean of 0.90 and a worst of 2.44, with log10 spread 0.24: 8 is 3.9 spreads
// above the mean, where 4 would fail one sweep in seven on noise. Every
// violation measured reads 6x or more above 8x its own control. Those 37
// were against the split-4 control alone; SmolLM3-3B on the M4 read 8.8x it
// at one row that order changes alone move 5x apart, which is why the
// control is now the worst of three (orderControl).
const denseStepBand = 8

// orderControl is the dense step gate's order-only control: stepGate's
// sessions decoded alone on dev for as many steps, on tiers that differ only
// in whether the composer cuts its sums into segments (Config.ComposedUnsplit)
// -- the same arithmetic summed in another order and nothing else -- the
// unsplit arm fed the split arm's greedy tokens, and the worst logit NMSE
// between them. It runs before the gate opens its own tier, so a card holds
// one placement of the model at a time. It fails rather than return a control
// that moved nothing, which could size no band.
//
// The composer has this one order change to offer; the three split pins it
// replaces were main's matvec's, sampled because a model's most sensitive
// row answers each differently (SmolLM3-3B's row 22 on the M4).
func orderControl(t *testing.T, m *model.Model, dev string, io ...Option) float64 {
	t.Helper()
	const steps = 8 // stepGate's
	var fed [][]int32
	arm := func(unsplit bool) [][]float32 {
		g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
			gpu.WithConfig(func(c *gpu.Config) { c.ComposedUnsplit = unsplit })}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			noDevice(t, dev, err)
		}
		defer g.Close()
		var out [][]float32
		sts := make([]*Instance, len(stepPrompts()))
		next := make([]int32, len(stepPrompts()))
		for i, pr := range stepPrompts() {
			ids := m.Vocab.Encode(pr, true)
			sts[i] = testInstance(t, m, len(ids)+steps+2, io...)
			defer sts[i].Close()
			if err := sts[i].SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if sts[i].GPULayers() != m.Cfg.NLayer || !sts[i].HeadOnDevice() {
				t.Skipf("CARD TOO SMALL for the order-only control: %d of %d blocks, unsplit %v (%s)",
					sts[i].GPULayers(), m.Cfg.NLayer, unsplit, g.Err())
			}
			lg, err := sts[i].Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			next[i] = Greedy(lg)
		}
		lead := fed == nil
		for s := range steps {
			if lead {
				fed = append(fed, slices.Clone(next))
			}
			for i, st := range sts {
				lg, err := st.Forward(fed[s][i])
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, slices.Clone(lg))
				next[i] = Greedy(lg)
			}
		}
		return out
	}
	a1 := arm(false)
	ax := arm(true)
	w, at := worstStep(a1, ax)
	if w == 0 || math.IsNaN(w) {
		t.Fatalf("the order-only control moved nothing (%v): it cannot size the band", w)
	}
	t.Logf("order-only control (split sums against unsplit, each session alone): worst logit NMSE %.3e (row %d); "+
		"rows %s", w, at, stepRows(a1, ax))
	return w
}

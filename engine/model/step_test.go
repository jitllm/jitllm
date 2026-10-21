package model

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestStepAcrossSessionsMatchesEachAlone decodes three States -- three
// sessions on one device, three different prompts -- once a token at a time
// each, and once together through Step, and demands the same logits: each row
// of the joint step must read its own session's history and no other.
// Stats.SessionRows is the selection check: a Step that fell back to one
// Forward after another would compare the arm with itself.
func TestStepAcrossSessionsMatchesEachAlone(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
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
		t.Run(spec, func(t *testing.T) { stepAcrossSessions(t, m, spec) })
	}
}

func stepAcrossSessions(t *testing.T, m *Model, spec string) {
	ctl := orderControl(t, m, spec)
	g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)},
		testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	r := stepGate(t, m, g, false)
	// The configuration this gates: a step of up to four sequences runs the
	// decode matvec with every token in one thread (tier.groupMV).
	if g.Stats().RagGroupMV == 0 {
		t.Fatalf("no few-sequence decode matvec ran: the step took the tiled twins (%s)", g.Err())
	}
	t.Logf("%d few-sequence decode matvecs, %d sm_70 tensor-core matvecs", g.Stats().RagGroupMV, g.Stats().VoltaMV)
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
// (tier.Config.SharedRec: one state a slot, the step writing the device's
// next-state scratch and a copy taking each row's home).
//
// It runs on every device named in JITLLM_TEST_DEVICES (comma-separated,
// default "cuda:0,vulkan:0,metal"); one that does not open here is skipped by name.
// JITLLM_STEP_HYBRID names another hybrid.
func TestStepAcrossSessionsHybridMatchesEachAlone(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		for _, shared := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/shared=%v", dev, shared), func(t *testing.T) {
				g := stepTier(t, m, dev, shared)
				defer g.Close()
				r := stepGate(t, m, g, true)
				hybridStepChecks(t, m, r)
				checkLinearRowsForm(t, m, g)
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
// a step of two or more rows routes on the device and runs every expert matvec
// grouped by expert (tier's emitGroupedMoE: dp4a, or sm_70's tensor cores),
// where each session alone runs decode's one-row mixture. A joint step needs
// every block placed, so a 4 GiB card skips olmoe by name and only a larger
// one covers it. Stats.GroupedMoE is the selection check: every mixture block
// of every joint step ran grouped.
//
// JITLLM_STEP_MIXTURE names the mixtures (comma-separated); the default is
// Qwen3-MoE-4x0.6B and olmoe.
func TestStepAcrossSessionsMixtureMatchesEachAlone(t *testing.T) {
	for _, name := range stepMixtures() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			m := openStepMixture(t, name)
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					g := stepTier(t, m, dev, false)
					defer g.Close()
					r := stepGate(t, m, g, false)
					mixtureStepChecks(t, m, r)
					t.Logf("%d grouped mixture block(s), %d grouped expert matvec(s) on tensor cores",
						r.grouped, r.groupedVolta)
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
		"synth-bailingmoe2.gguf", "synth-dots1.gguf", "synth-phimoe.gguf", "synth-deepseek4.gguf",
		"synth-kimik3.gguf"}
}

// openStepMixture opens a mixture for the step gates, refusing anything that
// is not one: a dense model would pass with nothing grouped in it.
func openStepMixture(t *testing.T, name string) *Model {
	t.Helper()
	m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	if m.Cfg.NExpert == 0 {
		m.Close()
		t.Fatalf("%s is not a mixture -- this gate would prove nothing", name)
	}
	return m
}

// mixtureStepChecks is the selection a mixture step gate demands: every
// mixture block of every joint step ran grouped by expert.
func mixtureStepChecks(t *testing.T, m *Model, r stepResult) {
	t.Helper()
	moe := 0
	for li := 0; li < m.Cfg.NLayer; li++ {
		if m.Cfg.MoEAt(li) {
			moe++
		}
	}
	if r.grouped != r.steps*moe {
		t.Fatalf("%d mixture block(s) ran grouped across %d joint steps, want %d: the step took another path",
			r.grouped, r.steps, r.steps*moe)
	}
}

// stepTier opens dev for a step gate on m, its recurrent pools in the shared
// form when asked, or skips naming the device.
func stepTier(t *testing.T, m *Model, dev string, shared bool, extra ...tier.Option) *tier.GPU {
	t.Helper()
	g, err := openTier(t, append(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.SharedRec = shared })}, testTierOpts(t)...), extra...)...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	t.Logf("%s is %s", dev, g.Name())
	return g
}

// checkLinearRowsForm is the selection check on the form a hybrid's ragged
// linear steps took on g: the scan's one-thread form where the device
// guarantees no 32-lane subgroup (llvmpipe; a Vulkan driver that will not pin
// the width) or JITLLM_SCALAR_LINEAR_ROWS forced it (testTierOpts), and a lane
// group a state row everywhere else. Whether a step ran at all is the calling
// gate's own check; this one says which kernel it was. Which device an index
// such as "vulkan:0" names depends on the drivers a process finds, so it asks
// the device rather than the name.
func checkLinearRowsForm(t *testing.T, m *Model, g *tier.GPU) {
	t.Helper()
	// A short convolution (LFM2) has no scan, so neither form of it runs.
	if !m.Cfg.Hybrid() || m.Cfg.ShortConv() {
		return
	}
	s := g.Stats()
	scalar := scalarLinearRows() || s.Lanes != ir.SubgroupLanes
	t.Logf("%s: lanes %d (%s), forced %v: %d linear block(s) stepped on the one-thread form",
		g.Name(), s.Lanes, s.LanesWhy, scalarLinearRows(), s.LinearRowsScalar)
	switch {
	case scalar && s.LinearRowsScalar == 0:
		t.Fatalf("%s should step a linear block's rows on the scan's one-thread form and never did", g.Name())
	case !scalar && s.LinearRowsScalar != 0:
		t.Fatalf("%s guarantees %d lanes and stepped %d linear block(s) on the one-thread form",
			g.Name(), ir.SubgroupLanes, s.LinearRowsScalar)
	}
}

// openStepHybrid opens the hybrid the step gates run, refusing anything that
// is not one: a dense model would pass every check here with nothing linear in
// it.
func openStepHybrid(t *testing.T) *Model {
	t.Helper()
	name := os.Getenv("JITLLM_STEP_HYBRID")
	if name == "" {
		name = "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"
	}
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.Hybrid() {
		m.Close()
		t.Fatalf("%s is not a hybrid -- this gate would prove nothing", name)
	}
	return m
}

// stepDevices is the backends a device gate runs on: JITLLM_TEST_DEVICES, or
// one of each backend at index 0. Every gate that sweeps backends reads it, so
// a shared box confines them all to the card it was given.
func stepDevices() []string { return testdevices.Specs("cuda:0", "vulkan:0", "metal") }

// hybridStepChecks is the selection a hybrid step gate demands: every linear
// block of every joint step ran across sessions, the session that stepped
// apart had every block's state aligned at the next joint step, and no other
// joint step after the first copied anything.
func hybridStepChecks(t *testing.T, m *Model, r stepResult) {
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
	// Mixture blocks the joint steps ran grouped by expert, and the grouped
	// expert matvecs among them on sm_70's tensor cores.
	grouped, groupedVolta int
	// States aligned at the first joint step, at the one after the first
	// session stepped apart, and at every other.
	alignsFirst, alignsApart, alignsOther int
}

// stepPrompts is the sessions stepGate steps together, one prompt each.
var stepPrompts = []string{"The capital of France is", "Water boils at a temperature of",
	"The clerk counted barrels of salt on the upper floor, and"}

// stepGate decodes three prompts on g once a State at a time and once
// together through Step, teacher-forced, and returns the worst logit NMSE of
// the joint arm against each session alone. With apart, the first session
// decodes one token on its own halfway through, in both arms. It fails if
// Step did not run every row across sessions.
func stepGate(t *testing.T, m *Model, g *tier.GPU, apart bool) stepResult {
	t.Helper()
	prompts := stepPrompts
	const steps, mid = 8, 4
	// arm opens a State per prompt and prefills it. It returns the States
	// and each one's next token. The alone arm's are closed as soon as its
	// logits are taken, so a card holds one arm's histories at a time.
	closed := map[*State]bool{}
	arm := func() ([]*State, []int32) {
		var sts []*State
		var next []int32
		for _, pr := range prompts {
			ids := m.Vocab.Encode(pr, true)
			st := m.NewState(len(ids) + steps + 2)
			t.Cleanup(func() {
				if !closed[st] {
					st.Close()
				}
			})
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
				// A block the device does not implement is not a small card.
				if why := g.DeclinePlan(st.planFor(0)); why != "" {
					t.Skipf("THE DEVICE DECLINES THE BLOCK: %s -- this gate proved nothing here", why)
				}
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
	// The selection named before anything runs: a State StepRuns would take
	// alone says why, where the row count below could only say that it did.
	for i, st := range together {
		switch err := st.StepRefusal(); err {
		case nil:
		case errStepNoHead, errStepSplit:
			// Placed whole and then moved: the prompts' history took the room
			// (relocation), which is a card too small for this gate.
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
	r.grouped, r.groupedVolta = s1.GroupedMoE-s0.GroupedMoE, s1.GroupedVolta-s0.GroupedVolta
	r.worst, r.at = worstStep(got, want)
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
// in the matvec split they pin -- the same arithmetic summed in another order
// and nothing else -- each fed the split-1 arm's greedy tokens, and the worst
// logit NMSE of split 2, 4 and 8 against split 1. It runs before the gate
// opens its own tier, so a card holds one placement of the model at a time.
// It fails rather than return a control that moved nothing, which could size
// no band.
//
// Three order changes and not one, because a model's most sensitive row
// answers each differently: SmolLM3-3B on the M4 moves its row 22 by 9.2e-03,
// 7.1e-03 and 3.5e-02 at splits 2, 4 and 8 while its other 23 rows stay within
// 2x of each other, and its step reads 6.2e-02 there -- 8.8x the split-4
// control alone, and 1.27x it in geometric mean over the rows. One sample of
// an order change is not the model's order sensitivity at its worst row.
func orderControl(t *testing.T, m *Model, dev string) float64 {
	t.Helper()
	const steps = 8 // stepGate's
	var fed [][]int32
	arm := func(split int) [][]float32 {
		g, err := openTier(t, append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
			tier.WithSplit(split)}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			noDevice(t, dev, err)
		}
		defer g.Close()
		var out [][]float32
		sts := make([]*State, len(stepPrompts))
		next := make([]int32, len(stepPrompts))
		for i, pr := range stepPrompts {
			ids := m.Vocab.Encode(pr, true)
			sts[i] = m.NewState(len(ids) + steps + 2)
			defer sts[i].Close()
			if err := sts[i].SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if sts[i].GPULayers() != m.Cfg.NLayer || !sts[i].HeadOnDevice() {
				t.Skipf("CARD TOO SMALL for the order-only control: %d of %d blocks at split %d (%s)",
					sts[i].GPULayers(), m.Cfg.NLayer, split, g.Err())
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
	a1 := arm(1)
	w := 0.0
	for _, sp := range []int{2, 4, 8} {
		ax := arm(sp)
		wx, at := worstStep(a1, ax)
		if wx == 0 || math.IsNaN(wx) {
			t.Fatalf("the order-only control at split %d moved nothing (%v): it cannot size the band", sp, wx)
		}
		t.Logf("order-only control (split 1 against %d, each session alone): worst logit NMSE %.3e (row %d); "+
			"rows %s", sp, wx, at, stepRows(a1, ax))
		w = max(w, wx)
	}
	return w
}

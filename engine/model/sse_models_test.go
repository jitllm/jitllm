//go:build amd64 && linux && jitllmtest

package model

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The SSE tier at the model level: every gate here forces the tier before a
// model opens (nn.NewJIT builds its kernels at construction), asserts the probe
// took it, and uses cpu.MappedByTier to assert no AVX2 kernel was mapped while
// it was on. Until every SSE family has landed they fail naming the missing
// ops rather than skip.

// ssemBand is the floor of the teacher-forced max|dlogit| between the SSE and
// AVX2 transcriptions (no FMA and a 4-lane reduction make them differ): the
// host/device reduction-order band. The per-model bar is measured in the same
// process against a control nobody calls a bug, the AVX2 tier's f16 KV cache
// against its f32 one, since how far a rounding step travels depends on the
// model.
const ssemBand = 0.94

// ssemMargin is how far past the same-process control the SSE arm may land.
const ssemMargin = 1.25

// ssemCap is the ceiling on the control-derived band. The control is a proxy,
// and on some models the f16 KV control is far larger than the tier
// difference, so without a cap that gate could not fail. 6.0 leaves the worst
// observed tier difference 1.4x headroom.
const ssemCap = 6.0

// ssemForce forces the SSE tier for the rest of t and checks it took.
func ssemForce(t *testing.T) {
	t.Helper()
	old := cpu.ForceTierForTest(cpu.TierSSE)
	t.Cleanup(func() { cpu.ForceTierForTest(old) })
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("forced SSE and the probe reports %v -- the force reached nothing", cpu.HostTier())
	}
}

// ssemRequireComplete fails t naming every SSE-tier op with no kernel.
func ssemRequireComplete(t *testing.T) {
	t.Helper()
	if p := cpu.SSEPending(); len(p) > 0 {
		t.Fatalf("the SSE tier is incomplete: %d ops have no kernel (%s). This gate runs models "+
			"on that tier and is red by design until every family has landed.",
			len(p), strings.Join(p, ", "))
	}
}

// ssemSmallModels is every language model under 2 GiB, sorted by size
// (languageModels leaves the projectors out).
func ssemSmallModels(t testing.TB) []string {
	var out []string
	for _, p := range languageModels(t) {
		if fi, err := os.Stat(p); err == nil && fi.Size() <= 2<<30 {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		a, _ := os.Stat(out[i])
		b, _ := os.Stat(out[j])
		return a.Size() < b.Size()
	})
	return out
}

// TestOpenRefusesAnIncompleteTierByName is model.Open's floor on an SSE host,
// in both directions and in every state of the tree: while any SSE-tier op is
// pending, Open refuses before building anything and names every pending op;
// once none is, Open succeeds.
func TestOpenRefusesAnIncompleteTierByName(t *testing.T) {
	models := ssemSmallModels(t)
	if len(models) == 0 {
		t.Fatal("no model under 2 GiB -- this gate proved nothing")
	}
	path := jlmOf(t, models[0]) // convert on the host's own tier, before the force
	ssemForce(t)
	before := cpu.MappedByTier()
	m, err := Open(path)
	after := cpu.MappedByTier()
	pending := cpu.SSEPending()
	if len(pending) == 0 {
		if err != nil {
			t.Fatalf("every SSE-tier op has a kernel and Open refused: %v", err)
		}
		m.Close()
		t.Logf("%s opened on the complete SSE tier", filepath.Base(path))
	} else {
		if err == nil {
			m.Close()
			t.Fatalf("%d SSE-tier ops are pending and Open succeeded: %s", len(pending), strings.Join(pending, ", "))
		}
		if !errors.Is(err, cpu.ErrNoSSEKernel) {
			t.Errorf("the refusal %v does not wrap cpu.ErrNoSSEKernel", err)
		}
		for _, p := range pending {
			if !strings.Contains(err.Error(), p) {
				t.Errorf("the refusal does not name pending op %s: %v", p, err)
			}
		}
		if after[cpu.TierSSE] != before[cpu.TierSSE] {
			t.Errorf("the refused Open mapped %d SSE kernels first -- it refused late", after[cpu.TierSSE]-before[cpu.TierSSE])
		}
		t.Logf("refused by name: %v", err)
	}
	if after[cpu.TierAVX2] != before[cpu.TierAVX2] {
		t.Errorf("%d AVX2 kernels were mapped while the SSE tier was forced", after[cpu.TierAVX2]-before[cpu.TierAVX2])
	}
}

// ssemArm is one tier's transcription of a model: teacher-forced logits at every
// position of ids, the prefill's last logits, and a greedy continuation with
// the top-1 margin at each generated step.
type ssemArm struct {
	forced  [][]float32
	prefill []float32
	greedy  []int32
	margins []float32
	steps   [][]float32 // the logits greedy[i] was chosen from
	// ksliced is how many decode matvecs summed k in slices, over workers.
	ksliced int64
	workers int
}

func ssemArgmax(l []float32) (int32, float32) {
	best, second := 0, -1
	for i := 1; i < len(l); i++ {
		switch {
		case l[i] > l[best]:
			best, second = i, best
		case second < 0 || l[i] > l[second]:
			second = i
		}
	}
	if second < 0 {
		return int32(best), float32(math.Inf(1))
	}
	return int32(best), l[best] - l[second]
}

// ssemRun opens path on the current tier and transcribes it, also driving
// ForwardBatch so all three graph transcriptions run.
func ssemRun(t *testing.T, path string, ids []int32, gen int, opts ...Option) ssemArm {
	t.Helper()
	m, err := Open(path, opts...)
	if err != nil {
		t.Fatalf("open on tier %v: %v", cpu.HostTier(), err)
	}
	defer m.Close()
	var a ssemArm
	s := m.NewState(len(ids) + gen + 1)
	for _, id := range ids {
		l, err := s.Forward(id)
		if err != nil {
			t.Fatalf("Forward on tier %v: %v", cpu.HostTier(), err)
		}
		a.forced = append(a.forced, append([]float32(nil), l...))
	}
	step := a.forced[len(a.forced)-1]
	next, margin := ssemArgmax(step)
	for i := 0; i < gen; i++ {
		a.greedy, a.margins = append(a.greedy, next), append(a.margins, margin)
		a.steps = append(a.steps, step)
		l, err := s.Forward(next)
		if err != nil {
			t.Fatalf("greedy Forward on tier %v: %v", cpu.HostTier(), err)
		}
		step = append([]float32(nil), l...)
		next, margin = ssemArgmax(step)
	}
	a.ksliced, a.workers = s.jit.KSliced(), s.jit.Workers()
	s.Close()

	p := m.NewState(len(ids) + 1)
	l, err := p.Prefill(ids)
	if err != nil {
		t.Fatalf("Prefill on tier %v: %v", cpu.HostTier(), err)
	}
	a.prefill = append([]float32(nil), l...)
	p.Close()

	b := m.NewBatch(2, 8)
	if _, err := b.ForwardBatch([]int32{ids[0], ids[1]}); err != nil {
		t.Fatalf("ForwardBatch on tier %v: %v", cpu.HostTier(), err)
	}
	b.Close()
	return a
}

func ssemMaxAbs(a, b []float32) float64 {
	var d float64
	for i := range a {
		x := float64(a[i]) - float64(b[i])
		if math.IsNaN(x) {
			return math.Inf(1)
		}
		d = math.Max(d, math.Abs(x))
	}
	return d
}

// TestEveryModelRunsSSE decodes, prefills and batches every model under 2 GiB
// on the forced SSE tier, and holds it to the AVX2 tier's transcription of the
// same model in the same process. The bar is the reduction band, not token
// equality: teacher-forced max|dlogit| inside the band at every position,
// prefill against the SSE tier's own decode, and greedy agreement up to the
// first step whose AVX2 margin is inside twice the observed difference.
func TestEveryModelRunsSSE(t *testing.T) {
	// First, before converting or opening anything: on an incomplete tier
	// every model would be refused, and the conversions would be wasted.
	ssemRequireComplete(t)
	models := ssemSmallModels(t)
	if len(models) == 0 {
		t.Fatal("no model under 2 GiB -- this gate proved nothing")
	}
	ids := make([]int32, 12)
	for i := range ids {
		ids[i] = int32(1 + i)
	}
	// The KV width is pinned per Open: it is a per-tier default (f16 on AVX2,
	// f32 on SSE), so unpinned the arms would run different caches. Both run
	// the f32 cache an SSE host ships with.
	//
	// And the k-split, for the host's sake rather than the arms': nn's
	// fusedSlices sums a large matrix over k in two slices on a pool of even
	// width and whole on an odd one, for both tiers alike (432 split matvecs
	// in each arm on four workers, none in either on three) -- so the parity
	// of the pool decided what the tiers were compared at, and gemma-2b's
	// part by 2.890 summed whole (pools of 3, 5, 7 and 9)
	// against 0.96-1.88 in slices and a KV control of 1.69-1.99 that does not
	// move with the pool. Pinned, every host compares the same summation.
	f32 := WithKVF16(false)
	split := WithJITOptions(nn.WithFusedKSlices(2))
	const gen = 12
	ran := 0
	for _, src := range models {
		t.Run(filepath.Base(src), func(t *testing.T) {
			path := jlmOf(t, src)
			var ref *ssemArm
			band := ssemBand
			if cpu.HostTier() == cpu.TierAVX2 {
				r := ssemRun(t, path, ids, gen, f32, split)
				ref = &r
				// The control is compared over exactly what the SSE arm is:
				// every prompt position, then the greedy steps while the two
				// chains share a history, the step where they part included.
				c := ssemRun(t, path, ids, gen, WithKVF16(true), split)
				var ctl float64
				for pos := range ids {
					ctl = math.Max(ctl, ssemMaxAbs(c.forced[pos], r.forced[pos]))
				}
				for i := range c.greedy {
					ctl = math.Max(ctl, ssemMaxAbs(c.steps[i], r.steps[i]))
					if c.greedy[i] != r.greedy[i] {
						break
					}
				}
				// A second control nobody calls a bug: the AVX2 tier's own
				// prefill against its own decode, a GEMM's summation order against
				// a matvec's. On gemma-3-1b it is 1.854 where the KV control is
				// 13, on Qwen3-1.7B 0.945 where the SSE tier's is 1.295; the
				// band is set by the larger of the two, for the teacher-forced
				// arm and the SSE tier's prefill alike.
				pd := ssemMaxAbs(r.prefill, r.forced[len(r.forced)-1])
				band = math.Max(ssemBand, math.Min(ssemMargin*math.Max(ctl, pd), ssemCap))
				t.Logf("control: the AVX2 tier's f16 KV cache against its f32 one moves the logits %.4f, "+
					"its prefill against its decode %.4f; band %.4f (the controls alone would allow %.4f, the cap is %.2f)",
					ctl, pd, band, ssemMargin*math.Max(ctl, pd), ssemCap)
			}

			ssemForce(t)
			ssemRequireComplete(t)
			before := cpu.MappedByTier()
			got := ssemRun(t, path, ids, gen, f32, split)
			after := cpu.MappedByTier()
			if d := after[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
				t.Fatalf("%d AVX2 kernels were mapped during the SSE arm -- it did not run the SSE tier", d)
			}
			if after[cpu.TierSSE] == before[cpu.TierSSE] {
				t.Fatal("the SSE arm mapped no SSE-tier kernel -- it ran nothing it claims to")
			}

			for pos, l := range got.forced {
				for _, v := range l {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("pos %d: a non-finite logit on the SSE tier", pos)
					}
				}
			}
			if d := ssemMaxAbs(got.prefill, got.forced[len(got.forced)-1]); !(d <= band) {
				t.Errorf("SSE prefill against SSE decode at the last position: max|dlogit| %.3f (band %.3f)", d, band)
			}
			if ref == nil {
				t.Logf("no AVX2 arm on this host; the SSE tier ran all three transcriptions and stayed finite")
				ran++
				return
			}
			// Every compared position in order: the prompt's, then the greedy
			// steps while the chains share a history (the step where they part
			// included). Step 0 is the prompt's last position again, so it is
			// not counted twice.
			var seq []float64
			var what []string
			for pos := range ids {
				seq = append(seq, ssemMaxAbs(got.forced[pos], ref.forced[pos]))
				what = append(what, "pos "+itoa(pos))
			}
			// Where the greedy chains part, the flip is arithmetic when the
			// AVX2 margin is inside twice that step's difference d: no argmax
			// leading by more than 2d can change.
			agreed := 0
			for i := range got.greedy {
				d := ssemMaxAbs(got.steps[i], ref.steps[i])
				if i > 0 {
					seq = append(seq, d)
					what = append(what, "greedy step "+itoa(i))
				}
				if got.greedy[i] == ref.greedy[i] {
					agreed++
					continue
				}
				if float64(ref.margins[i]) > 2*d {
					t.Errorf("greedy step %d: SSE %d, AVX2 %d at an AVX2 margin of %.3f -- more than twice "+
						"the %.3f the arms differ by at that step, so this is not reduction order",
						i, got.greedy[i], ref.greedy[i], ref.margins[i], d)
				}
				break
			}
			// The verdict is persistence, as TestKVF16IsSelectedAndRuns's:
			// a transcription defect reaches every position after the one it
			// hits, while a mixture can amplify the tiers' rounding at one
			// position through a routing decision and pass it no further.
			// Qwen3-MOE-4x0.6B's router logits run to 30-170, and the AVX2
			// tier's own f16-KV control moves them 2-8 per block and flips its
			// top-2 selection at half the positions; at position 10 the SSE
			// tier flips block 13's (AVX2 margin 5.09, the router moved 3.33
			// there and 3.23 under the control) and reads 1.667 against a
			// 1.569 band, with position 11 back at 0.94. So more than a
			// quarter of the positions past the band, three in a row, or the
			// last fail, and a lone excursion is still held to the cap.
			var worst float64
			var over []int
			for i, d := range seq {
				switch {
				case !(d <= ssemCap):
					t.Errorf("%s: max|dlogit| %.3f against the AVX2 tier, past the %.2f cap", what[i], d, ssemCap)
				case d > band:
					over = append(over, i)
				}
				worst = math.Max(worst, d)
			}
			var at []string
			for _, i := range over {
				at = append(at, what[i]+" ("+strconv.FormatFloat(seq[i], 'f', 3, 64)+")")
			}
			switch {
			case carried(over, len(seq)):
				t.Errorf("%s past the %.3f band against the AVX2 tier: damage that carries, not a rounding a "+
					"routing decision amplified", strings.Join(at, ", "), band)
			case len(over) > 0:
				t.Logf("%s past the %.3f band and every other position inside it: an amplified rounding that "+
					"did not carry", strings.Join(at, ", "), band)
			}
			t.Logf("teacher-forced max|dlogit| %.4f over %d positions and the shared greedy prefix; "+
				"greedy agreed for %d of %d steps; k-sliced matvecs AVX2 %d, SSE %d, on %d workers",
				worst, len(ids), agreed, gen, ref.ksliced, got.ksliced, got.workers)
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no model ran on the SSE tier -- this gate proved nothing")
	}
}

// ssemKVF16 runs fn with the f16 KV cache pinned. fn opens its own models, so
// the pin goes through greedyLoadOpts (forward_test.go), a test-only variable,
// rather than a package-wide default.
func ssemKVF16(fn func(*testing.T)) func(*testing.T) {
	return func(t *testing.T) {
		was := greedyLoadOpts
		greedyLoadOpts = append(append([]Option(nil), was...), WithKVF16(true))
		defer func() { greedyLoadOpts = was }()
		fn(t)
	}
}

// TestSSEModelGates runs the architecture gates -- the hybrid's four, the
// batch transcriptions, the llama.cpp and transformers goldens, the tower --
// with the SSE tier forced for all of them. Each is the unchanged gate: what
// it checks on the AVX2 tier it checks here.
func TestSSEModelGates(t *testing.T) {
	ssemForce(t)
	ssemRequireComplete(t)
	before := cpu.MappedByTier()
	for _, g := range []struct {
		name string
		fn   func(*testing.T)
	}{
		{"HybridRuns", TestHybridRuns},
		{"HybridPrefillMatchesForward", TestHybridPrefillMatchesForward},
		{"HybridBatchMatchesForward", TestHybridBatchMatchesForward},
		{"HybridStateIsConstantInContext", TestHybridStateIsConstantInContext},
		// llama.cpp's goldens use its default f16 KV cache, and one (qwen2vl)
		// has a first token the KV width alone decides, so this arm runs the
		// f16 cache. That also puts the SSE f16 attention kernels under a
		// model-level gate.
		{"GreedyMatchesLlamaCpp", ssemKVF16(TestGreedyMatchesLlamaCpp)},
		{"BatchMatchesForward", TestBatchMatchesForward},
		{"RaggedBatchMatchesForward", TestRaggedBatchMatchesForward},
		{"BiasedBatchMatchesForward", TestBiasedBatchMatchesForward},
		{"MoEFFNRunsBatched", TestMoEFFNRunsBatched},
		{"TowerEncodes", TestTowerEncodes},
		{"TowerMatchesLlamaCpp", TestTowerMatchesLlamaCpp},
		{"SafetensorsMatchTransformers", TestSafetensorsMatchTransformers},
		{"GPTOSSMatchesLlamaCppIntermediates", TestGPTOSSMatchesLlamaCppIntermediates},
		{"SynthGPTOSSMatchesTransformers", TestSynthGPTOSSMatchesTransformers},
		{"PrefillMatchesForward", TestPrefillMatchesForward},
		{"PrefillSeqMatchesForward", TestPrefillSeqMatchesForward},
		{"PrefillMixedMatchesPrefill", TestPrefillMixedMatchesPrefill},
		{"EveryModelRunsGenerated", TestEveryModelRunsGenerated},
		{"KVF16IsSelectedAndRuns", TestKVF16IsSelectedAndRuns},
		{"PagedKVIsTokenIdenticalOnEveryModel", TestPagedKVIsTokenIdenticalOnEveryModel},
		{"LlavaTowerEncodes", TestLlavaTowerEncodes},
	} {
		t.Run(g.name, g.fn)
	}
	if d := cpu.MappedByTier()[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernels were mapped while the SSE tier was forced", d)
	}
}

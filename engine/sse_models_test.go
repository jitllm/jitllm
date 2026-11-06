//go:build amd64 && linux && jitllmtest

package engine

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/nn"
)

// The SSE tier at the model level: every gate here opens its models on the CPU
// device at the SSE tier (cpu:sse, sseDevice), and uses that device's
// Target.MappedByTier to assert no AVX2 kernel was mapped. Until every SSE family has landed they fail naming the missing
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

// sseDevice is the device list of a CPU at the SSE tier.
func sseDevice() Option { return WithDevices("cpu:sse") }

// ssemTook fails t unless an instance opened with sseDevice has its CPU device
// at the SSE tier: an arm that ran AVX2 code would prove nothing.
func ssemTook(t *testing.T, path string) {
	t.Helper()
	instOpts := []Option{sseDevice()}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := testInstance(t, m, 8, instOpts...)
	defer s.Close()
	if got := cpuOf(s).Target.Tier(); got != cpu.TierSSE {
		t.Fatalf("an instance opened on cpu:sse has its CPU device at %v", got)
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

// ssemArm is one tier's transcription of a model: teacher-forced logits at every
// position of ids, the prefill's last logits, and a greedy continuation with
// the top-1 margin at each generated step.
type ssemArm struct {
	forced  [][]float32
	prefill []float32
	greedy  []int32
	margins []float32
	steps   [][]float32 // the logits greedy[i] was chosen from
	workers int
	mapped  map[cpu.Tier]int64 // the arm's CPU device's codes by tier
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
	// One CPU device for the three paths, whose mapped kernels the arm counts.
	instOpts := append(append([]Option(nil), opts...), sharedCPU(t, opts...))
	m, err := model.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer m.Close()
	var a ssemArm
	s := testInstance(t, m, len(ids)+gen+1, instOpts...)
	for _, id := range ids {
		l, err := s.Forward(id)
		if err != nil {
			t.Fatalf("Forward on tier %v: %v", cpuOf(s).Target.Tier(), err)
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
			t.Fatalf("greedy Forward on tier %v: %v", cpuOf(s).Target.Tier(), err)
		}
		step = append([]float32(nil), l...)
		next, margin = ssemArgmax(step)
	}
	a.workers = s.Workers()
	s.Close()

	p := testInstance(t, m, len(ids)+1, instOpts...)
	l, err := p.Prefill(ids)
	if err != nil {
		t.Fatalf("Prefill on tier %v: %v", cpuOf(s).Target.Tier(), err)
	}
	a.prefill = append([]float32(nil), l...)
	p.Close()

	b := newSessBatch(m, 2, 8, instOpts...)
	if _, err := b.ForwardBatch([]int32{ids[0], ids[1]}); err != nil {
		t.Fatalf("ForwardBatch on tier %v: %v", cpuOf(s).Target.Tier(), err)
	}
	b.Close()
	a.mapped = cpuOf(s).Target.MappedByTier()
	return a
}

// ssemComposedServes is the SSE gate's composed CPU arm: the composer forms
// every block for the CPU four floats a vector in the legacy encodings, so on
// the SSE tier every block step is served, none declined by the ISA, no AVX2
// kernel is mapped, and the teacher-forced logits are held to the SSE arm's
// (forced) within band at every position.
func ssemComposedServes(t *testing.T, path string, ids []int32, forced [][]float32, band float64, opts ...Option) {
	t.Helper()
	instOpts := opts
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := testInstance(t, m, len(ids)+1, instOpts...)
	defer s.Close()
	var worst float64
	for pos, id := range ids {
		l, err := s.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		d := ssemMaxAbs(l, forced[pos])
		if !(d <= band) {
			t.Fatalf("pos %d: the composed CPU's logits are %.4f from the SSE arm's (band %.4f)", pos, d, band)
		}
		worst = math.Max(worst, d)
	}
	mapped := cpuOf(s).Target.MappedByTier()
	if d := mapped[cpu.TierAVX2]; d != 0 {
		t.Fatalf("the composed CPU mapped %d AVX2 kernels on the SSE tier", d)
	}
	if mapped[cpu.TierSSE] == 0 {
		t.Fatal("the composed CPU mapped no SSE-tier kernel")
	}
	served, declined := s.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
	if declined[nn.DeclineISA] != 0 || served == 0 {
		t.Fatalf("on the SSE tier the composed CPU served %d block steps and declined %d by %q", served, declined[nn.DeclineISA], nn.DeclineISA)
	}
	if err := s.Account(nn.CPU).FormErr; err != nil {
		t.Errorf("forming for the CPU on the SSE tier: %v", err)
	}
	var named []string
	for d, n := range declined {
		if n > 0 {
			named = append(named, fmt.Sprintf("%d by %q", n, nn.Decline(d)))
		}
	}
	t.Logf("composed CPU on the SSE tier: %d of %d block steps served, declined %v; worst max|dlogit| %.4g against its own SSE arm (band %.4f)",
		served, len(ids)*m.Cfg.NLayer, named, worst, band)
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
	f32 := WithKVF16(false)
	const gen = 12
	ran := 0
	for _, src := range models {
		t.Run(filepath.Base(src), func(t *testing.T) {
			path := jlmOf(t, src)
			var ref *ssemArm
			band := ssemBand
			if cpu.New().Tier() == cpu.TierAVX2 {
				r := ssemRun(t, path, ids, gen, f32)
				ref = &r
				// The control is compared over exactly what the SSE arm is:
				// every prompt position, then the greedy steps while the two
				// chains share a history, the step where they part included.
				c := ssemRun(t, path, ids, gen, WithKVF16(true))
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

			got := ssemRun(t, path, ids, gen, f32, sseDevice())
			if d := got.mapped[cpu.TierAVX2]; d != 0 {
				t.Fatalf("%d AVX2 kernels were mapped during the SSE arm -- it did not run the SSE tier", d)
			}
			if got.mapped[cpu.TierSSE] == 0 {
				t.Fatal("the SSE arm mapped no SSE-tier kernel -- it ran nothing it claims to")
			}
			ssemComposedServes(t, path, ids, got.forced, band, f32, sseDevice())

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
				what = append(what, "pos "+model.Itoa(pos))
			}
			// Where the greedy chains part, the flip is arithmetic when the
			// AVX2 margin is inside twice that step's difference d: no argmax
			// leading by more than 2d can change.
			agreed := 0
			for i := range got.greedy {
				d := ssemMaxAbs(got.steps[i], ref.steps[i])
				if i > 0 {
					seq = append(seq, d)
					what = append(what, "greedy step "+model.Itoa(i))
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
				"greedy agreed for %d of %d steps, on %d workers",
				worst, len(ids), agreed, gen, got.workers)
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no model ran on the SSE tier -- this gate proved nothing")
	}
}

// ssemKVF16 runs fn with the f16 KV cache pinned. fn opens its own models, so
// the pin is a load option it is handed, not a package-wide default.
func ssemKVF16(fn func(*testing.T, ...Option)) func(*testing.T, ...Option) {
	return func(t *testing.T, opts ...Option) { fn(t, append([]Option{WithKVF16(true)}, opts...)...) }
}

// TestSSEModelGates runs the architecture gates -- the hybrid's four, the
// batch transcriptions, the llama.cpp and transformers goldens, the tower --
// with every model they open on the CPU device at the SSE tier. Each is the
// unchanged gate: what it checks on the AVX2 tier it checks here.
func TestSSEModelGates(t *testing.T) {
	// Every model the gates open maps onto a Target at the SSE tier, whose
	// gate refuses an AVX2 kernel by name: none can be mapped silently.
	ssemTook(t, jlmOf(t, benchModel(t)))
	for _, g := range []struct {
		name string
		fn   func(*testing.T, ...Option)
	}{
		{"HybridRuns", hybridRuns},
		{"HybridPrefillMatchesForward", hybridPrefillMatchesForward},
		{"HybridBatchMatchesForward", hybridBatchMatchesForward},
		{"HybridStateIsConstantInContext", hybridStateIsConstantInContext},
		// llama.cpp's goldens use its default f16 KV cache, and one (qwen2vl)
		// has a first token the KV width alone decides, so this arm runs the
		// f16 cache. That also puts the SSE f16 attention kernels under a
		// model-level gate.
		{"GreedyMatchesLlamaCpp", ssemKVF16(greedyMatchesLlamaCpp)},
		{"BatchMatchesForward", batchMatchesForward},
		{"RaggedBatchMatchesForward", raggedBatchMatchesForward},
		{"BiasedBatchMatchesForward", biasedBatchMatchesForward},
		{"SafetensorsMatchTransformers", safetensorsMatchTransformers},
		{"GPTOSSMatchesLlamaCppIntermediates", gptossMatchesLlamaCppIntermediates},
		{"SynthGPTOSSMatchesTransformers", synthGPTOSSMatchesTransformers},
		{"PrefillMatchesForward", prefillMatchesForward},
		{"PrefillSeqMatchesForward", prefillSeqMatchesForward},
		{"PrefillMixedMatchesPrefill", prefillMixedMatchesPrefill},
		{"EveryModelRunsGenerated", everyModelRunsGenerated},
		{"KVF16IsSelectedAndRuns", kvF16IsSelectedAndRuns},
		{"PagedKVIsTokenIdenticalOnEveryModel", pagedKVIsTokenIdenticalOnEveryModel},
	} {
		t.Run(g.name, func(t *testing.T) { g.fn(t, sseDevice()) })
	}
}

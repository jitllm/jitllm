package engine

import (
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// centredModel is LFM2 350M restated non-causal (jlm.FlagNonCausal): its
// blocks attend both ways and its short convolutions are centred across a
// call's rows (graph.OpRowConv), as LFM2.5-Encoder's trunk is. No converted
// model reaches a card's pieces with such a block -- d1-omni's trunk runs as
// one non-causal call, its marker head making every block non-causal -- so
// the gate restates one.
func centredModel(t *testing.T) *model.Model {
	t.Helper()
	src := testmodels.Path("lfm2/LFM2-350M-Q4_K_M.gguf")
	if !sourcePresent(src) {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", src)
	}
	path := filepath.Join(t.TempDir(), "centred"+model.Ext)
	in, err := os.Open(jlmOf(t, src))
	if err != nil {
		t.Fatal(err)
	}
	out, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.Copy(out, in)
	in.Close()
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		t.Fatal(err)
	}
	if changed, err := jlm.Restate(path, func(_ *jlm.File, c *jlm.Config) (bool, error) {
		if c.Flags.Has(jlm.FlagNonCausal) {
			t.Fatal("LFM2 350M states itself non-causal already: the gate would restate nothing")
		}
		c.Flags |= jlm.FlagNonCausal
		// The blocks build from the restated config: the stated graphs are
		// the causal ones.
		c.Blocks = jlm.BlockGraphs{}
		return true, nil
	}); err != nil || !changed {
		t.Fatalf("restating the mask: changed %v, %v", changed, err)
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	centred := 0
	for _, g := range m.FrontGraphs().Blocks {
		for _, o := range g.Ops {
			if o.Kind == graph.OpRowConv {
				centred++
			}
		}
	}
	if centred == 0 {
		m.Close()
		t.Fatal("the restated model convolves nothing centred: the gate would hold no halo")
	}
	return m
}

// centredRunRows is the prompt's rows: past two submissions of a card,
// inside three.
const centredRunRows = 2*nn.MaxDevicePrefillChunk + 126

// centredPrompt is one bidirectional run of centredRunRows rows (a
// non-causal decoder's prompt, as engine.Decider asks one) and of its first
// rows that one submission holds.
func centredPrompt(t *testing.T, m *model.Model) (long, short []model.Span) {
	t.Helper()
	ids := m.Vocab.Encode(strings.Repeat("The ledger lists the parcels by weight, the weights by day, and the days by the "+
		"van that carried them; a parcel that missed its van waits on the shelf by the door. ", 40), true)
	if len(ids) < centredRunRows {
		t.Fatalf("the prompt is %d rows, short of %d", len(ids), centredRunRows)
	}
	return []model.Span{{Tokens: ids[:centredRunRows], Bidir: true}},
		[]model.Span{{Tokens: ids[:nn.MaxDevicePrefillChunk-112], Bidir: true}}
}

// centredLogits is the last row's logits of spans, layers of the blocks on
// g where g is not nil, the CPU on its float forms over an f32 history.
func centredLogits(t *testing.T, m *model.Model, g *gpu.GPU, layers int, spans []model.Span) []float32 {
	t.Helper()
	in := testInstance(t, m, model.SpanPositions(spans, m.Cfg.NEmbd)+1, WithKVF16(false), withCPUFloat())
	defer in.Close()
	if g != nil {
		if err := in.SetDeviceLayers(g, layers); err != nil {
			t.Fatal(err)
		}
		if in.GPULayers() != layers {
			t.Fatalf("the card took %d of %d blocks: %s", in.GPULayers(), layers, g.Err())
		}
	}
	lg, err := in.PrefillMixed(spans...)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), lg...)
}

// haloPieces is the pieces with a halo g's devices ran.
func haloPieces(g *gpu.GPU) int {
	n := 0
	for _, s := range g.DevStats() {
		n += s.HaloPieces
	}
	return n
}

// centredBound is how many times the band of a run one submission holds
// the long run's logits may stand from the CPU's.
const centredBound = 3.0

// centredArm is the CPU's logits of the long run and of the short one.
type centredArm struct {
	m                 *model.Model
	long, short       []model.Span
	cpuLong, cpuShort []float32
}

func newCentredArm(t *testing.T, m *model.Model) *centredArm {
	long, short := centredPrompt(t, m)
	return &centredArm{m: m, long: long, short: short,
		cpuLong: centredLogits(t, m, nil, 0, long), cpuShort: centredLogits(t, m, nil, 0, short)}
}

// openCentredCard is spec's tier, its reserved width rows where rows is
// past 0 (Config.PromptRows).
func openCentredCard(t *testing.T, spec string, rows int, extra ...gpu.Option) *gpu.GPU {
	t.Helper()
	opts := append([]gpu.Option{gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)
	if rows > 0 {
		opts = append(opts, gpu.WithConfig(func(c *gpu.Config) { c.PromptRows = rows }))
	}
	g, err := openGPU(t, append(opts, extra...)...)
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	return g
}

// TestCentredRunMatchesTheCPU prefills a non-causal decoder whose short
// convolutions are centred (centredModel) over one bidirectional run of
// 1150 rows: on the CPU, which runs the run in one call, and with every
// block on each card, where a submission holds 512 rows -- and 64 with the
// reserved width narrowed (Config.PromptRows), where the run of 400 rows
// one submission holds as placed goes in pieces too -- so the run goes in
// pieces, block by block (gpu.GPU.phased), each piece of a centred block
// reading the rows its convolution needs past its edges (the halo); and
// with half the blocks there, the rest on the CPU. The card's logits are
// held to the CPU's within three times the band a run one submission holds
// reads on the same card; pieces with a halo ran. The violation, a halo of
// no rows, is TestCentredRunHaloIsSeen, under jitllmfault.
func TestCentredRunMatchesTheCPU(t *testing.T) {
	m := centredModel(t)
	defer m.Close()
	a := newCentredArm(t, m)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g := openCentredCard(t, spec, 0)
			band, _ := nmseOf(centredLogits(t, m, g, m.Cfg.NLayer, a.short), a.cpuShort)
			g.Close()
			t.Logf("%s: a run of %d rows, one submission, against the CPU's: logit NMSE %.3e", spec, len(a.short[0].Tokens), band)
			if math.IsNaN(band) {
				t.Fatalf("a run one submission holds reads NMSE %v on %s", band, spec)
			}
			type arm struct {
				spans  []model.Span
				cpu    []float32
				layers int
			}
			for _, rows := range []int{0, 64} {
				g := openCentredCard(t, spec, rows)
				arms := []arm{{a.long, a.cpuLong, m.Cfg.NLayer}, {a.long, a.cpuLong, m.Cfg.NLayer / 2}}
				if rows > 0 {
					arms = append(arms, arm{a.short, a.cpuShort, m.Cfg.NLayer})
				}
				for _, c := range arms {
					n := len(c.spans[0].Tokens)
					halos, phases := haloPieces(g), runPhases(g)
					got := centredLogits(t, m, g, c.layers, c.spans)
					halos, phases = haloPieces(g)-halos, runPhases(g)-phases
					nm, worst := nmseOf(got, c.cpu)
					t.Logf("%s, reserved %d, a run of %d rows, %d of %d blocks: against the CPU's, logit NMSE %.3e (max|d| %.4f); band %.3e; %d phase calls, %d pieces with a halo",
						spec, rows, n, c.layers, m.Cfg.NLayer, nm, worst, band, phases, halos)
					if halos == 0 || phases == 0 {
						t.Fatalf("%d phase calls and %d pieces with a halo: the run did not go in pieces", phases, halos)
					}
					if math.IsNaN(nm) || nm > centredBound*max(band, 1e-6) {
						t.Errorf("a run of %d rows on %s, reserved %d, %d blocks, reads NMSE %.3e against the CPU's, past %gx the band %.3e",
							n, spec, rows, c.layers, nm, centredBound, band)
					}
				}
				g.Close()
			}
		})
	}
}

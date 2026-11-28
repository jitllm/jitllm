package engine

import (
	"math"
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
	return centredModelAs(t, 0)
}

// centredModelAs is centredModel restated with flags2 too: with
// jlm.Flag2MediaPrefix its attention and its convolutions read each row's
// window (graph.Attrs.RowWindows), as d1-omni's trunk does.
func centredModelAs(t *testing.T, flags2 jlm.Flags2) *model.Model {
	t.Helper()
	src := testmodels.Path("lfm2/LFM2-350M-Q4_K_M.gguf")
	if !sourcePresent(src) {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", src)
	}
	was := false
	m := statedModel(t, src, func(s *jlm.Source) {
		was = s.Config.Flags.Has(jlm.FlagNonCausal)
		s.Config.Flags |= jlm.FlagNonCausal
		s.Config.Flags2 |= flags2
	})
	if was {
		m.Close()
		t.Fatal("LFM2 350M states itself non-causal already: the gate would restate nothing")
	}
	centred, windowed := 0, map[graph.OpKind]int{}
	for _, g := range m.FrontGraphs().Blocks {
		for _, o := range g.Ops {
			if o.Kind == graph.OpRowConv {
				centred++
			}
			if o.Attr.RowWindows {
				windowed[o.Kind]++
			}
		}
	}
	if centred == 0 {
		m.Close()
		t.Fatal("the restated model convolves nothing centred: the gate would hold no halo")
	}
	if flags2.Has(jlm.Flag2MediaPrefix) && (windowed[graph.OpRowConv] == 0 || windowed[graph.OpAttend] == 0) {
		m.Close()
		t.Fatalf("the restated model reads windows in %v: the gate would window nothing", windowed)
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

// cardForms is a way a windowed gate's card forms its blocks: as it ships,
// where a call of few rows runs a projection's vector form over int8
// activations (composer's bands), or with float activations everywhere
// (Config.ComposedFloat), where nothing but the pieces' cutting stands
// between the card's rows and the CPU's float rows.
type cardForms struct {
	name  string
	float bool
}

func (f cardForms) opts() []gpu.Option {
	if !f.float {
		return nil
	}
	return []gpu.Option{gpu.WithConfig(func(c *gpu.Config) { c.ComposedFloat = true })}
}

// windowForms is the forms a windowed gate holds each card's runs in.
func windowForms() []cardForms {
	return []cardForms{{name: "as shipped"}, {name: "float", float: true}}
}

// formBands is the band a windowed gate holds spec's runs in each form to:
// measure, a run one submission holds, on a card opened in the form, and as
// shipped the most of that and measure on a card whose every projection
// takes the int8 vector form (Config.NoMMA), the class a call of few rows
// runs -- an arm whose pieces leave a ragged call is held to the forms it
// ran. The float band must stand below the shipped one, or the float forms
// were not the ones the card ran.
func formBands(t *testing.T, spec string, measure func(*gpu.GPU) float64) map[string]float64 {
	t.Helper()
	at := func(opts ...gpu.Option) float64 {
		g := openCentredCard(t, spec, 0, opts...)
		defer g.Close()
		v := measure(g)
		if math.IsNaN(v) {
			t.Fatalf("a run one submission holds reads NMSE %v on %s", v, spec)
		}
		return v
	}
	shipped, int8 := at(), at(gpu.WithConfig(func(c *gpu.Config) { c.NoMMA = true }))
	float := at(windowForms()[1].opts()...)
	t.Logf("%s: a run one submission holds, against the CPU's: NMSE %.3e as shipped, %.3e with every projection on the int8 vector form, %.3e float",
		spec, shipped, int8, float)
	if !(float < shipped) {
		t.Fatalf("the float forms read %.3e on %s, the shipped ones %.3e: the card did not run float activations", float, spec, shipped)
	}
	return map[string]float64{windowForms()[0].name: max(shipped, int8), windowForms()[1].name: float}
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
// reads on the same card in the same forms (formBands), as shipped and
// float; pieces with a halo ran. The violation, a halo of
// no rows, is TestCentredRunHaloIsSeen, under jitllmfault.
func TestCentredRunMatchesTheCPU(t *testing.T) {
	m := centredModel(t)
	defer m.Close()
	a := newCentredArm(t, m)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			bands := formBands(t, spec, func(g *gpu.GPU) float64 {
				v, _ := nmseOf(centredLogits(t, m, g, m.Cfg.NLayer, a.short), a.cpuShort)
				return v
			})
			type arm struct {
				spans  []model.Span
				cpu    []float32
				layers int
			}
			for _, f := range windowForms() {
				band := bands[f.name]
				for _, rows := range []int{0, 64} {
					g := openCentredCard(t, spec, rows, f.opts()...)
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
						t.Logf("%s %s, reserved %d, a run of %d rows, %d of %d blocks: against the CPU's, logit NMSE %.3e (max|d| %.4f); band %.3e; %d phase calls, %d pieces with a halo",
							spec, f.name, rows, n, c.layers, m.Cfg.NLayer, nm, worst, band, phases, halos)
						if halos == 0 || phases == 0 {
							t.Fatalf("%d phase calls and %d pieces with a halo: the run did not go in pieces", phases, halos)
						}
						if math.IsNaN(nm) || nm > centredBound*max(band, 1e-6) {
							t.Errorf("a run of %d rows on %s %s, reserved %d, %d blocks, reads NMSE %.3e against the CPU's, past %gx the band %.3e",
								n, spec, f.name, rows, c.layers, nm, centredBound, band)
						}
					}
					g.Close()
				}
			}
		})
	}
}

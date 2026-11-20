package engine

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// embedPrincipleModels are the embedding models the five principles' gates
// run beside the generators: every encoder the library states (bert,
// nomic-bert, gemma-embedding) and the decoders read out pooled
// (EmbeddingGemma, qwen3-embedding), each a real file or a fixture under
// embed/. An embedding has no decode and no history between calls, so each
// gate's arm for them holds the vectors of a run of texts.
func embedPrincipleModels() []string {
	return []string{"embed/MiniLM-L6-v2.Q8_0.gguf", "embed/all-minilm-l6.gguf", "embed/snowflake-arctic-embed-22m.gguf",
		"embed/mxbai-embed-large.gguf", "embed/nomic-embed-text.gguf", "embed/synth-gemma-embedding.gguf",
		"embed/embeddinggemma-300m.gguf", "embed/embeddinggemma-300M-BF16-lcpp.gguf", "embed/qwen3-embedding-0.6b.gguf"}
}

// eachEmbedModel runs fn on every embedding principle model's container, as
// the subtest named by its file under embed/.
func eachEmbedModel(t *testing.T, fn func(t *testing.T, path string)) {
	for _, name := range embedPrincipleModels() {
		t.Run(name, func(t *testing.T) { fn(t, jlmOf(t, testmodels.Path(name))) })
	}
}

// embedPrincipleTexts are the run of inputs every arm embeds: one a batch
// of the server's would carry, of three lengths.
func embedPrincipleTexts() []string {
	return []string{
		"The capital of France is Paris.",
		"search_query: how does a page budget below the model change the vectors it gives?",
		"A quick brown fox jumps over the lazy dog, and the dog, unbothered, goes back to sleep in the sun " +
			"while the fox runs on across the field towards the river at the edge of the wood.",
	}
}

// openEmbedding opens the embedding container at path.
func openEmbedding(t *testing.T, path string) *model.Model {
	t.Helper()
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if !m.IsEmbedding() {
		m.Close()
		t.Fatalf("%s: the container sets no pooling", path)
	}
	return m
}

// embedIDs is every principle text's ids under m's own tokenizer.
func embedIDs(m *model.Model) [][]int32 {
	texts := embedPrincipleTexts()
	out := make([][]int32, len(texts))
	for i, s := range texts {
		out[i] = m.EmbedIDs(s)
	}
	return out
}

// embedOne is ids' vector from e, copied and checked finite.
func embedOne(t *testing.T, e *Embedder, ids []int32) []float32 {
	t.Helper()
	v, err := e.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatal("a non-finite component: a degenerate vector is a failure, not a pass")
		}
	}
	return append([]float32(nil), v...)
}

// embedEach is every principle text's vector from e.
func embedEach(t *testing.T, m *model.Model, e *Embedder) [][]float32 {
	t.Helper()
	var out [][]float32
	for _, ids := range embedIDs(m) {
		out = append(out, embedOne(t, e, ids))
	}
	return out
}

// sameVectors fails unless got is want bit for bit, vector by vector.
func sameVectors(t *testing.T, what string, want, got [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d vectors, want %d", what, len(got), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if math.Float32bits(want[i][j]) != math.Float32bits(got[i][j]) {
				t.Fatalf("%s: text %d component %d is %v, %v unpaged and unmoved", what, i, j, got[i][j], want[i][j])
			}
		}
	}
}

// embedBand is how close a device's vector must sit to the host's
// (TestEmbeddingsOnEveryDevice's bounds): the f32 reduction-order band, or
// the int8 activations' where the blocks' matrices are quantized.
func embedBand(m *model.Model) float64 {
	for _, w := range m.Layers[0].Mats {
		switch w.T.Typ {
		case quant.F32, quant.F16, quant.BF16:
		default:
			return embedDevCosQuant
		}
	}
	return embedDevCos
}

// nearVectors fails unless every vector of got sits within bound of want's.
func nearVectors(t *testing.T, what string, want, got [][]float32, bound float64) float64 {
	t.Helper()
	c := worstCos(got, want)
	if !(c >= bound) {
		t.Errorf("%s: worst cosine to the host %.9f < %v", what, c, bound)
	}
	return c
}

// embedPagesWithTheSameAnswer is TestPagingDoesNotChangeTheAnswer on an
// embedding model: the texts under a budget of one frame, half the blocks'
// and all but one block's, each arm's vectors the resident run's bit for bit,
// and each arm reading pages again (a budget that evicted nothing tested
// nothing).
func embedPagesWithTheSameAnswer(t *testing.T, path string) {
	run := func(frames int) ([][]float32, int64, int, int) {
		m := openEmbedding(t, path)
		defer m.Close()
		var opts []Option
		if frames > 0 {
			opts = append(opts, WithPageBudget(uint64(frames)*m.PageSize()))
		}
		e, err := NewEmbedder(m, opts...)
		if err != nil {
			t.Fatal(err)
		}
		defer e.Close()
		out := embedEach(t, m, e)
		_, reads, _ := m.PageStats(e.st.fr)
		return out, reads, m.Container.NPages(), m.Cfg.NLayer
	}
	whole, once, pages, nb := run(0)
	tried := 0
	for _, frames := range []int{1, nb / 2, nb - 1} {
		if frames < 1 || frames >= nb || frames == tried {
			continue
		}
		tried = frames
		paged, reads, _, _ := run(frames)
		if reads <= once {
			t.Fatalf("a budget of %d of the %d block pages read %d, the resident run %d: nothing was evicted, so nothing was tested",
				frames, pages, reads, once)
		}
		sameVectors(t, "paged", whole, paged)
		t.Logf("%d frame(s) of %d block pages: %d page reads against %d resident, %d vectors alike", frames, pages, reads, once, len(whole))
	}
	if tried == 0 {
		t.Fatalf("%d block(s): no budget below the model evicts, so this gate proved nothing", nb)
	}
}

// embedOpenHonoursAPageBudget is TestOpenHonoursAPageBudget on an embedding
// model: opening reads no block page, an embedder of one frame binds its
// budget where the unbudgeted one holds the whole model, the composed CPU
// serves the paged run's blocks, and the vectors are the resident ones bit
// for bit.
func embedOpenHonoursAPageBudget(t *testing.T, path string) {
	full := openEmbedding(t, path)
	defer full.Close()
	tight := openEmbedding(t, path)
	defer tight.Close()
	if nb := int(tight.Container.H.NBlocks); nb < 2 {
		t.Fatalf("%d block(s): a budget cannot bind", nb)
	}
	if in := tight.PageFaults(); in != 0 {
		t.Errorf("%d page-ins during open: a block was read before the budget reached the reader", in)
	}
	var allPages uint64
	for i := 0; i < tight.Container.NPages(); i++ {
		allPages += tight.Container.PageBytes(i)
	}
	a, err := NewEmbedder(full)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := NewEmbedder(tight, WithPageBudget(1))
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if got := b.st.fr.Budget(); got == 0 || got >= allPages {
		t.Fatalf("budget is %d bytes against %d of pages: it did not bind, so this compares two resident models", got, allPages)
	}
	if got := a.st.fr.Budget(); got != 0 && got < allPages {
		t.Fatalf("the unbudgeted embedder has a %d-byte budget against %d of pages: both arms are paged", got, allPages)
	}
	sameVectors(t, "under a one-page budget", embedEach(t, full, a), embedEach(t, tight, b))
	served, declined := b.st.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
	if served == 0 {
		t.Error("the composed CPU served no block step of the paged model")
	}
	if declined[nn.DeclineNoRoom] > 0 {
		t.Errorf("the composed CPU declined %d block step(s): %s", declined[nn.DeclineNoRoom], nn.DeclineName(int(nn.DeclineNoRoom)))
	}
	_, reads, _ := tight.PageStats(b.st.fr)
	t.Logf("budget %d of %d page bytes: %d page reads, %d block steps served", b.st.fr.Budget(), allPages, reads, served)
}

// embedLeaseFaultsItsBlocksIn is TestForwardBatchFaultsItsBlocksIn on an
// embedding model, through the entry point a server's batch of inputs takes:
// two sessions on one served instance of one frame, each input embedded on a
// lease (EmbedderFor, Session.Run), the sessions taking turns, every vector the
// resident embedder's bit for bit.
func embedLeaseFaultsItsBlocksIn(t *testing.T, path string) {
	full := openEmbedding(t, path)
	ref, err := NewEmbedder(full)
	if err != nil {
		t.Fatal(err)
	}
	want := embedEach(t, full, ref)
	ref.Close()
	full.Close()

	m := openEmbedding(t, path)
	defer m.Close()
	if !budgetEvicts(m, 1) {
		t.Fatalf("%d block(s): one frame holds the model, so this gate proved nothing", m.Cfg.NLayer)
	}
	in, err := NewInstance(m, min(m.Cfg.NCtx, model.EmbedMaxSeq), WithPageBudget(1))
	if err != nil {
		t.Fatal(err)
	}
	pool := NewPool()
	defer pool.Close()
	if err := pool.Serve(in, 0); err != nil {
		in.Close()
		t.Fatal(err)
	}
	const nsess = 2
	var sess [nsess]*Session
	var emb [nsess]*Embedder
	for k := range sess {
		if emb[k], err = EmbedderFor(m); err != nil {
			t.Fatal(err)
		}
		if sess[k], err = pool.NewSession(emb[k].MaxTokens()); err != nil {
			t.Fatal(err)
		}
		defer sess[k].Close()
	}
	ids := embedIDs(m)
	got := make([][]float32, len(ids))
	for i := range ids {
		k := i % nsess
		if err := sess[k].Run(context.Background(), PhasePrefill, func(in *Instance) error {
			emb[k].Use(in)
			got[i] = embedOne(t, emb[k], ids[i])
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	all := uint64(0)
	for i := 0; i < m.Container.NPages(); i++ {
		all += m.Container.PageBytes(i)
	}
	if b := in.fr.Budget(); b == 0 || b >= all {
		t.Fatalf("the served instance's frames hold a %d-byte budget against %d of pages: its quota did not bind", b, all)
	}
	_, reads, _ := m.PageStats(in.fr)
	if reads <= int64(m.Container.NPages()) {
		t.Fatalf("%d page reads for %d block pages: nothing was evicted, so nothing was tested", reads, m.Container.NPages())
	}
	sameVectors(t, "on leases of a one-frame instance", want, got)
	t.Logf("%d inputs over %d sessions on one frame: %d page reads for %d block pages", len(ids), nsess, reads, m.Container.NPages())
}

// embedAllocs is TestDecodeDoesNotAllocate on an embedding model: a warm run
// of the texts makes zero engine allocations, on the host (g nil) and with
// every block on g, whose submissions show the blocks ran there.
func embedAllocs(t *testing.T, m *model.Model, g *gpu.GPU) {
	e, err := NewEmbedder(m, WithKVF16(false), WithPageBudget(1<<40))
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if g != nil {
		if err := e.SetDeviceLayers(g, -1); err != nil {
			t.Fatal(err)
		}
		if e.DeviceBlocks() != m.Cfg.NLayer {
			t.Fatalf("the card took %d of %d blocks: %v %s", e.DeviceBlocks(), m.Cfg.NLayer, e.DeviceDeclines(), g.Err())
		}
	}
	ids := embedIDs(m)
	one := func() {
		for _, x := range ids {
			if _, err := e.Embed(x); err != nil {
				t.Fatal(err)
			}
		}
	}
	one()
	one()
	var s0 int
	if g != nil {
		s0 = g.Stats().Submits
	}
	if w := countAllocs(one); w.engine != 0 {
		t.Errorf("%d engine allocations over %d warm embeddings:\n%s", w.engine, len(ids), w.where)
	}
	if g != nil {
		if s := g.Stats().Submits - s0; s < len(ids) {
			t.Fatalf("%d submission(s) for %d texts: the blocks did not run on the card", s, len(ids))
		}
	}
}

// embedDoesNotAllocate runs embedAllocs on the host and on each of specs.
func embedDoesNotAllocate(t *testing.T, path string, specs []string) {
	m := openEmbedding(t, path)
	defer m.Close()
	t.Run("host", func(t *testing.T) { embedAllocs(t, m, nil) })
	for _, spec := range specs {
		t.Run(spec, func(t *testing.T) {
			g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, spec, err)
			}
			defer g.Close()
			embedAllocs(t, m, g)
		})
	}
}

// encoderRelocModels are the models the relocation gates run as encoders:
// every embedding principle model, and Laya's fixture, a modern-bert trunk
// read out through a marker head (its paging and allocation gates are
// TestLayaPagesWithTheSameAnswer and TestMarkerDoesNotAllocate).
func encoderRelocModels() []string {
	return append(embedPrincipleModels(), "laya/fixture/laya-fixture.gguf")
}

// eachEncoder runs fn on every encoder relocation model, as the subtest named
// by its file.
func eachEncoder(t *testing.T, fn func(t *testing.T, path string)) {
	for _, name := range encoderRelocModels() {
		t.Run(name, func(t *testing.T) { fn(t, jlmOf(t, testmodels.Path(name))) })
	}
}

// encoderRun is an encoder's readout on whatever instance runs it: an
// embedding model's vector of each principle text, or a marker head's scores
// of each question.
type encoderRun struct {
	m *model.Model
	// n is the inputs, run the readout of input i on in.
	n   int
	run func(t *testing.T, in *Instance, i int) []float32
	// long runs an input of about n positions on in.
	long func(t *testing.T, in *Instance, n int)
	// tol bounds a device's readout's distance from the host's (near).
	tol float64
	emb bool
}

// markTol is how far a card's marker scores may sit from the host's: the
// f32 reduction-order band of the fixture's raw logits, inside the 2e-3
// TestMarkerHeadMatchesTheReference holds it to the reference by.
const markTol = 2e-4

// encoderOf is m's readout: its embedder's where it pools, its marker head's
// where it reads out at markers.
func encoderOf(t *testing.T, m *model.Model) *encoderRun {
	t.Helper()
	if m.IsEmbedding() {
		e, err := EmbedderFor(m)
		if err != nil {
			t.Fatal(err)
		}
		ids := embedIDs(m)
		return &encoderRun{m: m, n: len(ids), tol: embedBand(m), emb: true,
			run: func(t *testing.T, in *Instance, i int) []float32 {
				e.Use(in)
				return embedOne(t, e, ids[i])
			},
			long: func(t *testing.T, in *Instance, n int) {
				e.Use(in)
				embedOne(t, e, m.EmbedIDs(longEmbedText(m.EmbedIDs, n))[:n])
			}}
	}
	if !m.Cfg.Marks() {
		t.Fatal("the container neither pools nor reads out at markers")
	}
	return &encoderRun{m: m, n: 3, tol: markTol,
		run: func(t *testing.T, in *Instance, i int) []float32 { return markerScores(t, in, i) },
		long: func(t *testing.T, in *Instance, n int) {
			ids := make([]int32, n)
			for k := range ids {
				ids[k] = int32(1 + (4*k+1)%(m.Cfg.NVocab-1))
			}
			out := make([]float32, 3)
			if err := in.MarkScores(ids, 0, []int{1, n / 2, n - 1}, out); err != nil {
				t.Fatalf("MarkScores: %v", err)
			}
		}}
}

// instance is an instance of r's model for its inputs, with opts.
func (r *encoderRun) instance(t *testing.T, opts ...Option) *Instance {
	t.Helper()
	in, err := NewInstance(r.m, min(r.m.Cfg.NCtx, model.EmbedMaxSeq), append([]Option{WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return in
}

// host is every input's readout with every block on the host.
func (r *encoderRun) host(t *testing.T) [][]float32 {
	t.Helper()
	in := r.instance(t)
	defer in.Close()
	out := make([][]float32, r.n)
	for i := range out {
		out[i] = r.run(t, in, i)
	}
	return out
}

// near fails unless got sits within r's bound of want, input by input: the
// worst cosine of an embedding, the largest difference of a score. It is the
// worst figure.
func (r *encoderRun) near(t *testing.T, what string, want, got [][]float32) float64 {
	t.Helper()
	if r.emb {
		return nearVectors(t, what, want, got, r.tol)
	}
	d := 0.0
	for i := range want {
		for j := range want[i] {
			d = math.Max(d, math.Abs(float64(got[i][j]-want[i][j])))
		}
	}
	if !(d <= r.tol) {
		t.Errorf("%s: scores %.3e from the host's, past %v", what, d, r.tol)
	}
	return d
}

// embedRelocatesToAnotherDevice is TestLayerRelocatesToAnotherDevice on an
// encoder: the inputs run with every block on tier a, then with the instance
// moved to tier b after the first input and back to a after the second, each
// readout the static run's bit for bit (both tiers one card's, tuning off)
// and the host's within the device band; each tier ran blocks.
func embedRelocatesToAnotherDevice(t *testing.T, path string) {
	a, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || a == nil {
		noDevice(t, "device", err)
	}
	defer a.Close()
	b, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || b == nil {
		noDevice(t, "second tier", err)
	}
	defer b.Close()
	m := openEncoder(t, path)
	defer m.Close()
	r := encoderOf(t, m)
	host := r.host(t)
	nb := m.Cfg.NLayer
	run := func(hop bool) [][]float32 {
		in := r.instance(t)
		defer in.Close()
		var out [][]float32
		var was *gpu.GPU
		for i := range r.n {
			on := a
			if hop && i%2 == 1 {
				on = b
			}
			if on != was {
				if err := in.SetDevice(on); err != nil {
					t.Fatal(err)
				}
				was = on
			}
			if in.GPULayers() != nb {
				t.Fatalf("input %d: %s took %d of %d blocks: %v %s", i, on.Name(), in.GPULayers(), nb, in.DeviceDeclines(), on.Err())
			}
			s0 := on.Stats().Submits
			out = append(out, r.run(t, in, i))
			if on.Stats().Submits == s0 {
				t.Fatalf("input %d: no submission on %s: the blocks did not run there", i, on.Name())
			}
		}
		return out
	}
	static := run(false)
	hopped := run(true)
	sameVectors(t, "moved to a second tier", static, hopped)
	d := r.near(t, "moved to a second tier", host, hopped)
	t.Logf("%d inputs hopping %s -> %s -> %s: the static run's bits; worst against the host %.9f", r.n, a.Name(), b.Name(), a.Name(), d)
}

// openEncoder opens the container at path, which pools or reads out at
// markers.
func openEncoder(t *testing.T, path string) *model.Model {
	t.Helper()
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// embedBlocksMoveBetweenDevices is TestBlocksMoveBetweenDevices on an
// encoder: every block on card a, then moved between inputs (each move
// prepared while an input runs and committed after it) to card b, the upper
// half back to a, every block to the CPU and back to a. Each move lands its
// blocks where it sent them; an input run with every block on the cards is
// the unmoved control's bit for bit (two cards running the same kernels),
// and every readout sits within the device band of the host's.
func embedBlocksMoveBetweenDevices(t *testing.T, path string) {
	a, b := moveDevices(t)
	g, err := openGPU(t, gpu.WithDevices(a+","+b), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	distinctCards(t, g)
	m := openEncoder(t, path)
	defer m.Close()
	r := encoderOf(t, m)
	host := r.host(t)
	nb := m.Cfg.NLayer
	open := func() *Instance {
		in := r.instance(t, WithPlacement(Placement{Rest: &Place{On: a}, Head: &Place{On: a}}))
		if err := in.SetDevice(g); err != nil {
			in.Close()
			t.Fatal(err)
		}
		if n := len(in.placedOn(a)); n != nb {
			in.Close()
			t.Fatalf("%s took %d of %d blocks: %v %s", a, n, nb, in.DeviceDeclines(), g.Err())
		}
		return in
	}
	ctl := open()
	control := make([][]float32, r.n)
	for i := range control {
		control[i] = r.run(t, ctl, i)
	}
	ctl.Close()

	in := open()
	defer in.Close()
	moves := []struct {
		lo, hi int
		to     string
	}{{0, nb, b}, {nb / 2, nb, a}, {0, nb, nn.CPU}, {0, nb, a}}
	for _, mv := range moves {
		what := fmt.Sprintf("blocks %d..%d moved to %s", mv.lo, mv.hi-1, mv.to)
		pm, err := in.PrepareMove(mv.lo, mv.hi, mv.to)
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		before := r.run(t, in, 0)
		if _, err := pm.Commit(); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		want := mv.to
		if want == nn.CPU {
			want = ""
		}
		cards := true
		for li := 0; li < nb; li++ {
			at, _ := in.DeviceOf(li)
			if li >= mv.lo && li < mv.hi && at != want {
				t.Fatalf("block %d is on %q, sent to %s", li, at, mv.to)
			}
			cards = cards && at != ""
		}
		got := [][]float32{before}
		for i := 1; i < r.n; i++ {
			got = append(got, r.run(t, in, i))
		}
		if cards {
			sameVectors(t, what, control[1:], got[1:])
		}
		d := r.near(t, what, host, got)
		t.Logf("%s: worst against the host %.9f", what, d)
	}
}

// embedBlocksComeBack is TestRelocatedBlocksComeBack on an encoder: a card
// with room for every block and the history a short input writes (and at
// least 8 MiB), and a long input that outgrows it. A causal readout's
// history grows on the card, so the long input sends blocks home
// (SetRelocate) and the short input after it brings every one back before
// its run. A non-causal block's keys and values live only inside its call,
// in the one pair the card reserved for the instance's context at placement
// (gpu's transientKV): nothing grows, so the long input sends nothing home
// and every block stays. Either way every readout sits within the device
// band of the host's.
func embedBlocksComeBack(t *testing.T, path string) {
	m := openEncoder(t, path)
	defer m.Close()
	r := encoderOf(t, m)
	host := r.host(t)
	nb := m.Cfg.NLayer
	n := min(m.Cfg.NCtx, 2*relocPage) - 8
	probe, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || probe == nil {
		noDevice(t, "device", err)
	}
	pin := r.instance(t, WithKVPrewarm(false))
	if err := pin.SetDevice(probe); err != nil || pin.GPULayers() != nb {
		t.Fatalf("placed %d of %d blocks unconstrained: %v %s", pin.GPULayers(), nb, err, probe.Err())
	}
	r.run(t, pin, 0)
	used := probe.Bytes()
	var grow uint64
	causal := false
	for li := 0; li < nb; li++ {
		perPos, _, _ := pin.BlockHistory(li)
		grow += perPos * uint64(n)
		causal = causal || !pin.planFor(li).NonCausal
	}
	pin.Close()
	probe.Close()
	if grow == 0 {
		t.Fatal("no block holds a history: nothing can outgrow the card")
	}
	budget := used + max(grow/4, 8<<20)
	g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff), gpu.WithBudget(budget))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	in := r.instance(t, WithKVPrewarm(false))
	defer in.Close()
	if err := in.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	in.SetRelocate(true)
	if in.GPULayers() != nb {
		t.Fatalf("placed %d of %d blocks under a budget of %d: %v %s", in.GPULayers(), nb, budget, in.DeviceDeclines(), g.Err())
	}
	r.long(t, in, n)
	went := in.Relocations()
	switch {
	case causal && (went == 0 || in.GPULayers() == nb):
		t.Fatalf("a %d-position input on a card with %d bytes past its blocks for %d of history: %d relocation(s), %d of %d blocks on the card",
			n, budget-used, grow, went, in.GPULayers(), nb)
	case !causal && (went != 0 || in.GPULayers() != nb):
		t.Fatalf("a %d-position input of a non-causal readout, whose history the card reserved at placement: %d relocation(s), %d of %d blocks on the card",
			n, went, in.GPULayers(), nb)
	}
	got := [][]float32{r.run(t, in, 0)}
	if in.Reclaims() != went || in.GPULayers() != nb {
		t.Fatalf("after the short input: %d of %d relocated blocks reclaimed, %d of %d on the card (%q)",
			in.Reclaims(), went, in.GPULayers(), nb, g.Err())
	}
	for i := 1; i < r.n; i++ {
		got = append(got, r.run(t, in, i))
	}
	d := r.near(t, "after the long input", host, got)
	t.Logf("causal %v: %d block(s) home for a %d-position input, every one on the card for the short ones; worst against the host %.9f", causal, went, n, d)
}

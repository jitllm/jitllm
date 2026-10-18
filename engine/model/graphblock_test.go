package model

import (
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

func graphModel(t *testing.T, name string) string {
	p := testmodels.Path(name + ".jlm")
	if src := testmodels.Path(name + ".gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	return p
}

// graphDecode greedily decodes n tokens after the prompt and returns every
// position's logits, the State's graph block count and its warm allocations.
func graphDecode(t *testing.T, m *Model, n, frames int) ([][]float32, int, float64) {
	if frames != 0 {
		m.SetPageBudget(uint64(frames) * m.PageSize())
	}
	s := m.NewState(64)
	defer s.Close()
	var out [][]float32
	tok := m.Vocab.Encode("The capital of France is", true)
	var next int32
	for i := 0; i < len(tok)+n; i++ {
		in := next
		if i < len(tok) {
			in = tok[i]
		}
		lg, err := s.Forward(in)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
		next = Greedy(lg)
	}
	allocs := testing.AllocsPerRun(4, func() {
		if _, err := s.Forward(next); err != nil {
			t.Fatal(err)
		}
	})
	return out, s.GraphBlocks(), allocs
}

// TestBlockGraphRefusesWhatItDoesNotCover: a block whose configuration states
// what the op language does not (a value head of its own width beside a GQA
// attention) refuses the model at Open, naming why, rather than run through a
// lowering that would drop it -- there is no other body to run it.
func TestBlockGraphRefusesWhatItDoesNotCover(t *testing.T) {
	m, err := Open(graphModel(t, "Qwen3-0.6B-Q8_0"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	was := m.Cfg.HeadDimV
	m.Cfg.HeadDimV = m.Cfg.HeadDim + 2
	err = m.buildGraphs()
	m.Cfg.HeadDimV = was
	if err == nil {
		t.Fatal("a value head the op language does not state built a graph")
	}
	named := false
	for _, r := range UncoveredReasons {
		named = named || strings.HasSuffix(err.Error(), r)
	}
	if !named {
		t.Fatalf("refused without a named structural reason: %v", err)
	}
	t.Logf("refused: %v", err)
}

// TestBlockGraphOfTheRealContainersVerifies: the facts are read from the
// container, and a graph without its q/k norm is refused against them.
func TestBlockGraphOfTheRealContainersVerifies(t *testing.T) {
	m, err := Open(graphModel(t, "Qwen3-0.6B-Q8_0"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	f, err := m.blockFacts(5)
	if err != nil {
		t.Fatal(err)
	}
	if err := graph.Verify(f.Config.Build(), f); err != nil {
		t.Fatal(err)
	}
	d := f.Config
	d.QKNorm = false
	if err := graph.Verify(d.Build(), f); err == nil || !strings.Contains(err.Error(), "attn_q_norm") {
		t.Fatalf("a graph without qwen3's q/k norm: %v", err)
	}
	// A latent attention: its graph verifies, and the violations fire -- the
	// latent norms held to the model's epsilon, the rotary parts at the
	// head's width, the reference form without its up-projections.
	dm, err := Open(graphModel(t, "synth-deepseek"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	li := dm.Cfg.NLayer - 1
	mf, err := dm.blockFacts(li)
	if err != nil {
		t.Fatal(err)
	}
	if mf.Config.MLA == nil {
		t.Fatal("synth-deepseek's block states no latent attention")
	}
	if err := graph.Verify(mf.Config.Build(), mf); err != nil {
		t.Fatal(err)
	}
	for name, mut := range map[string]func(*graph.MLA){
		"epsilon":   func(a *graph.MLA) { a.LatentEps *= 2 },
		"rotary":    func(a *graph.MLA) { a.NRot /= 2 },
		"latent":    func(a *graph.MLA) { a.KVLoraRank-- },
		"no rotary": func(a *graph.MLA) { a.NoRope = !a.NoRope },
	} {
		d := mf.Config
		a := *d.MLA
		mut(&a)
		d.MLA = &a
		if err := graph.Verify(d.Build(), mf); err == nil {
			t.Errorf("a latent attention with its %s changed verified", name)
		}
	}
	d = mf.Config
	d.MLA = nil
	if u := graph.Unbound(d.Build(), mf); len(u) == 0 {
		t.Error("a dense attention over an MLA block's tensors left none of them unbound")
	}
}

// graphPrompt prefills a prompt of n tokens on a fresh State over a page
// budget of frames (0 keeps the model's), decodes after it, and returns the
// logits after the prompt and after each decoded token, the block-chunks the
// graph ran and the allocations of a warm second prompt on another slot.
func graphPrompt(t *testing.T, m *Model, n, frames int) ([][]float32, int) {
	t.Helper()
	if frames != 0 {
		m.SetPageBudget(uint64(frames) * m.PageSize())
	}
	s := m.NewState(n + 16)
	defer s.Close()
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32((100 + (i*7919)%3000) % m.Cfg.NVocab)
	}
	lg, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for i := 0; i < 4; i++ {
		if lg, err = s.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out, s.GraphChunks()
}

// planFields is the part of a device plan a dense block's operations state.
type planFields struct {
	NEmbd, NHead, NKVHead, HeadDim, NRot, NFFN int
	RMSEps, RopeBase, AttnScale                float64
	AttnSoftcap                                float32
	RopeNeox, QKNorm                           bool
	Act                                        nn.ActKind
}

func fieldsOf(p *nn.LayerPlan) planFields {
	return planFields{p.NEmbd, p.NHead, p.NKVHead, p.HeadDim, p.NRot, p.NFFN, p.RMSEps, p.RopeBase, p.AttnScale,
		p.AttnSoftcap, p.RopeNeox, p.QKNorm, p.Act}
}

// flipRope swaps every rotary op of g to the other pair layout.
func flipRope(g *graph.Graph) {
	for i := range g.Ops {
		if g.Ops[i].Kind != graph.OpRope {
			continue
		}
		if g.Ops[i].Attr.Layout == graph.RopeNeox {
			g.Ops[i].Attr.Layout = graph.RopeNorm
		} else {
			g.Ops[i].Attr.Layout = graph.RopeNeox
		}
	}
}

// TestBlockGraphLowersTheDevicePlan is Slice B's device half: a graphed
// block's device plan takes its operations' fields from the graph
// (lowerPlan), and they equal what the hand-written planFor derives from the
// configuration for every block; a graph whose rotary layout is the other one
// gives another plan (the lowering reads the graph, not the config). On a
// CUDA device the graphed model decodes bit for bit as the model without the
// graph with every block placed and at a partial seam.
func TestBlockGraphLowersTheDevicePlan(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0", "synth-stablelm", "synth-dbrx", "synth-glm4", "synth-olmo2", "synth-gptoss", "synth-gemma", "gemma-3-1b-it-Q4_K_M"} {
		t.Run(name, func(t *testing.T) {
			p := graphModel(t, name)
			// The configuration's plan: the same model with no graph to lower
			// it from, every block on the device (a host block would need its
			// graph).
			ctl, err := Open(p, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer ctl.Close()
			ctl.graphs = make([]*denseLowering, len(ctl.graphs))
			gm, err := Open(p, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer gm.Close()
			sc, sg := ctl.NewState(8), gm.NewState(8)
			defer sc.Close()
			defer sg.Close()
			sc.plan, sg.plan = sc.basePlan(), sg.basePlan()
			for li := 0; li < gm.Cfg.NLayer; li++ {
				hp, gp := sc.planFor(li), sg.planFor(li)
				if !gp.Graphed || hp.Graphed {
					t.Fatalf("block %d: graphed %v, control graphed %v", li, gp.Graphed, hp.Graphed)
				}
				if fieldsOf(hp) != fieldsOf(gp) {
					t.Fatalf("block %d: the graph's plan %+v, the configuration's %+v", li, fieldsOf(gp), fieldsOf(hp))
				}
			}
			// The violation: the other rotary layout in the graph reaches the plan.
			flipRope(gm.graphs[0].g)
			if fieldsOf(sg.planFor(0)) == fieldsOf(sc.planFor(0)) {
				t.Fatal("a graph with the other rotary layout lowered to the configuration's plan: the plan does not read the graph")
			}
			flipRope(gm.graphs[0].g)
			ids := gm.Vocab.Encode("The capital of France is", true)
			toks := []int32{ids[1], ids[2], ids[3], ids[1], ids[2], ids[3]}
			for _, seam := range []int{-1} {
				// The tuners pinned (RULE 11c): an equality gate pins every
				// knob its equality depends on, and the split and prologue
				// tuners pick a reduction order per tier.
				want := planDecode(t, ctl, ids, toks, seam)
				if want == nil {
					t.Skip("NO CUDA DEVICE: the device half proved nothing")
				}
				got := planDecode(t, gm, ids, toks, seam)
				for i := range want {
					for j := range want[i] {
						if got[i][j] != want[i][j] {
							t.Fatalf("seam %d step %d logit %d: graph %v, without %v", seam, i, j, got[i][j], want[i][j])
						}
					}
				}
				t.Logf("seam %d: device decode bit-identical over %d steps", seam, len(toks))
			}
		})
	}
}

// planDecode decodes toks teacher-forced after ids on a fresh CUDA tier with
// its tuners off, seam blocks placed (-1 every block); nil without CUDA.
func planDecode(t *testing.T, m *Model, ids, toks []int32, seam int) [][]float32 {
	t.Helper()
	g, err := openTier(t, tier.WithDevices("cuda"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.NoFuse = true }))
	if err != nil || g == nil {
		return nil
	}
	defer g.Close()
	s := m.NewState(len(ids) + len(toks) + 4)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if seam >= 0 {
		s.SetGPULayers(seam)
	}
	if want := seam; want >= 0 && s.GPULayers() != want || want < 0 && s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("%d blocks placed, want seam %d", s.GPULayers(), seam)
	}
	l, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	var out [][]float32
	for _, tok := range toks {
		out = append(out, append([]float32(nil), l...))
		if l, err = s.Forward(tok); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// graphBatch runs steps batch steps of nseq sequences (ForwardBatch) and then
// steps more through Step over nseq single-sequence States, the rows owned by
// different sessions, returning every step's logits and the ragged-step
// graph block count of both.
func graphBatch(t *testing.T, m *Model, frames int) ([][]float32, int) {
	if frames != 0 {
		m.SetPageBudget(uint64(frames) * m.PageSize())
	}
	const nseq, steps = 3, 6
	b := m.NewBatch(nseq, 64)
	defer b.Close()
	var out [][]float32
	tok := make([]int32, nseq)
	for j := 0; j < steps; j++ {
		for i := range tok {
			tok[i] = int32((7 + i*101 + j*13) % m.Cfg.NVocab)
		}
		lg, err := b.ForwardBatch(tok)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	ran := b.GraphRaggedBlocks()
	states := make([]*State, nseq)
	for i := range states {
		states[i] = m.NewState(64)
		defer states[i].Close()
	}
	for j := 0; j < steps; j++ {
		for i := range tok {
			tok[i] = int32((11 + i*37 + j*29) % m.Cfg.NVocab)
		}
		lgs, err := Step(states, tok)
		if err != nil {
			t.Fatal(err)
		}
		for _, lg := range lgs {
			out = append(out, append([]float32(nil), lg...))
		}
	}
	for _, s := range states {
		ran += s.GraphRaggedBlocks()
	}
	return out, ran
}

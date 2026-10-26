package model

import (
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestPlacementComposesOnce holds the composer's census on Llama-3.2-1B with
// every part composed and a reserved prompt width of 512 rows
// (JITLLM_COMPOSE_ONCE_ROWS another): once a
// State's blocks are placed, prompts of 1, 5, 33, 100, 511, 512, 513 and 2000
// tokens, each followed by decode steps, a second session's prompt and
// decode, and a ragged batch of three sequences compose nothing, compile no
// kernel and build no scratch -- Stats.PartsFormed, ScratchBuilds and
// backend.Compiles stay where the placement left them, and the composer
// serves every chunk's every block. A prompt longer than the reserved width
// runs in chunks of it.
//
// JITLLM_VIOLATE_COMPOSE_ONCE=1 forms a prompt at its chunk's grain width
// (Config.ComposedWidths), a form keyed by rows, which must fail.
func TestPlacementComposesOnce(t *testing.T) {
	violate := os.Getenv("JITLLM_VIOLATE_COMPOSE_ONCE") != ""
	const name = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
			testmodels.Missing(t, "MODEL MISSING: "+name)
		}
	}
	parts := tier.ComposedStep | tier.ComposedPrompt
	// JITLLM_COMPOSE_ONCE_ROWS reserves another prompt width, for a card
	// whose budget does not hold a 512-row composed block beside the model.
	width := 512
	if v := os.Getenv("JITLLM_COMPOSE_ONCE_ROWS"); v != "" {
		w, err := strconv.Atoi(v)
		if err != nil || w < 32 {
			t.Fatalf("JITLLM_COMPOSE_ONCE_ROWS=%q: a width of at least 32 rows", v)
		}
		width = w
	}
	g, err := tier.OpenWith(tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) {
		c.Composed, c.PromptRows, c.ComposedWidths = parts, width, violate
	}))
	if err != nil || g == nil {
		t.Fatalf("no GPU device: %v", err)
	}
	defer g.Close()
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	lengths := []int{1, 5, 33, 100, 511, 512, 513, 2000}
	const decode = 2
	total := 0
	for _, n := range lengths {
		total += n + decode
	}
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 400)
	ids := m.Vocab.Encode(text, true)
	if len(ids) < total {
		t.Fatalf("%d prompt tokens, %d wanted", len(ids), total)
	}
	s := m.NewState(total + 64)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("the device took %d of %d blocks: %s", s.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	st0, c0 := g.Stats(), backend.Compiles()
	t.Logf("at placement: %d forms (%d kernels) composed, %d scratches built", st0.PartsFormed, st0.PartKernels, st0.ScratchBuilds)
	if st0.PartsFormed == 0 {
		t.Fatalf("the placement composed nothing: %s", g.Err())
	}
	held := func(what string) {
		t.Helper()
		t.Logf("%s: %s", what, g.Err())
		st := g.Stats()
		if f, k, sb, c := st.PartsFormed-st0.PartsFormed, st.PartKernels-st0.PartKernels, st.ScratchBuilds-st0.ScratchBuilds,
			backend.Compiles()-c0; f != 0 || k != 0 || sb != 0 || c != 0 {
			t.Fatalf("%s composed %d form(s) of %d kernel(s), built %d scratch(es) and compiled %d kernel(s) after the placement",
				what, f, k, sb, c)
		}
	}
	served0, _, _ := g.Stats().ServedPart(tier.ComposedPrompt)
	steps, off := 0, 0
	for _, n := range lengths {
		l, err := s.Prefill(ids[off : off+n])
		if err != nil {
			t.Fatalf("a prompt of %d: %v", n, err)
		}
		off += n
		for range decode {
			if l, err = s.Forward(argmaxID(l)); err != nil {
				t.Fatal(err)
			}
			off++
		}
		held("a prompt of " + strconv.Itoa(n) + " tokens and its decode")
		if n > 1 {
			steps += (n + width - 1) / width * m.Cfg.NLayer
		}
	}
	np, dp, _ := g.Stats().ServedPart(tier.ComposedPrompt)
	if np-served0 < steps || dp != 0 {
		t.Fatalf("the composer served %d prompt block steps of the %d the chunks hold and declined %d: %s",
			np-served0, steps, dp, g.Err())
	}
	t.Logf("%d prompt block steps served after the placement", np-served0)

	s2 := m.NewState(512)
	defer s2.Close()
	if err := s2.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	held("a second session's placement (" + g.Err() + ")")
	l, err := s2.Prefill(ids[:100])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s2.Forward(argmaxID(l)); err != nil {
		t.Fatal(err)
	}
	held("a second session's prompt and decode")

	sc := NewScheduler(m, 4, 512)
	defer sc.Close()
	if err := sc.st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	seqs := []*Seq{{Prompt: ids[:7], MaxTokens: 4}, {Prompt: ids[50:83], MaxTokens: 6}, {Prompt: ids[200:300], MaxTokens: 5}}
	for _, q := range seqs {
		sc.Submit(q)
	}
	for done, n := 0, 0; done < len(seqs); n++ {
		if n > 100 {
			t.Fatalf("the batch made no progress: %d of %d retired", done, len(seqs))
		}
		d, err := sc.Step()
		if err != nil {
			t.Fatal(err)
		}
		done += len(d)
	}
	for i, q := range seqs {
		if q.Err != nil {
			t.Fatalf("sequence %d: %v", i, q.Err)
		}
	}
	held("a ragged batch of three sequences")
	if testing.Verbose() {
		st := g.Stats()
		t.Logf("after every call: %d forms, %d scratches, %d compiles since the placement", st.PartsFormed-st0.PartsFormed,
			st.ScratchBuilds-st0.ScratchBuilds, backend.Compiles()-c0)
	}
}

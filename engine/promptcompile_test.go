package engine

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestPlacedPromptsCompileNothing holds the ready boundary for a prompt: once
// a instance's blocks are placed, prompts of several chunk widths (each run in a
// scratch of its own grain width) and the decode after them compile no
// kernel. Every width's kernels were built at the end of the placement
// (tier.prewarmPromptWidths); a compile here would land in a time to first
// token.
//
// JITLLM_VIOLATE_PROMPT_PREWARM=1 switches the prewarm off (Config.
// NoPromptPrewarm), which must fail. Both tuning modes run: the shipping
// default (TuneAuto) retunes the decode's splits, compiling candidates, and
// that belongs to the placement too. With the composer serving the blocks
// and the head, their forms are built at placement and at PrepHead, which a
// model calls after placing its blocks.
func TestPlacedPromptsCompileNothing(t *testing.T) {
	violate := os.Getenv("JITLLM_VIOLATE_PROMPT_PREWARM") != ""
	ran := 0
	for _, arm := range []struct {
		name     string
		composed nn.Parts
	}{
		{"newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", nn.ComposedStep},
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", nn.ComposedStep},
	} {
		name := arm.name
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err != nil {
			if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
				t.Logf("MODEL MISSING: %s", name)
				continue
			}
		}
		for _, tune := range []gpu.TuneMode{gpu.TuneOff, gpu.TuneAuto} {
			t.Run(fmt.Sprintf("%s/tune%d/composed%d", name, tune, arm.composed), func(t *testing.T) {
				g, err := openGPU(t, gpu.WithDeviceTune(tune),
					gpu.WithConfig(func(c *gpu.Config) {
						// The composer forms a prompt block once, at the
						// reserved width: the widest chunk here is 224 rows.
						c.NoPromptPrewarm, c.PromptRows = violate, 256
					}))
				if err != nil || g == nil {
					t.Skipf("no GPU device: %v", err)
				}
				defer g.Close()
				m, err := model.Open(jlmOf(t, p))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				text := strings.Repeat("The river rose every spring until the bridges were islands, "+
					"and the clerk counted barrels of salt on the upper floor. ", 40)
				ids := m.Vocab.Encode(text, true)
				chunks := []int{37, 200, 150} // grain widths 64, 224 and 160
				total := 0
				for _, n := range chunks {
					total += n
				}
				if len(ids) < total {
					t.Fatalf("%d prompt tokens, %d wanted", len(ids), total)
				}
				s := testInstance(t, m, total+8)
				defer s.Close()
				if err := s.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				if s.GPULayers() != m.Cfg.NLayer {
					t.Fatalf("the device took %d of %d blocks: %s", s.GPULayers(), m.Cfg.NLayer, g.Err())
				}
				c0 := g.Counters().Compiles()
				g.Counters().CompileLog = func(n string) { t.Logf("compiled after placement: %s", n) }
				defer func() { g.Counters().CompileLog = nil }()
				off := 0
				var l []float32
				for _, n := range chunks {
					if l, err = s.Prefill(ids[off : off+n]); err != nil {
						t.Fatal(err)
					}
					off += n
					t.Logf("after %d: exact %d, %d placed, demotions %d, err %q", off, g.Stats().ExactPrompts,
						s.GPULayers(), s.DeviceDemotions(), g.Err())
				}
				for i := 0; i < 2; i++ {
					if l, err = s.Forward(argmaxID(l)); err != nil {
						t.Fatal(err)
					}
				}
				st := g.Stats()
				got := g.Counters().Compiles() - c0
				t.Logf("%d compile(s) after placement; %d exact prompt chunk(s), %d width(s) warmed",
					got, st.ExactPrompts, st.PromptWidthsWarmed)
				// A prompt the composer serves runs in the reserved width, its
				// rows the call's, and no other width is warmed.
				if st.ExactPrompts != 0 || st.PromptWidthsWarmed != 0 {
					t.Fatalf("with the composer serving the prompt, %d chunks ran at their own width and %d widths were warmed",
						st.ExactPrompts, st.PromptWidthsWarmed)
				}
				if got != 0 {
					t.Fatalf("%d kernel(s) compiled after the blocks were placed", got)
				}
				if served, _ := st.ServedPart(arm.composed); served == 0 {
					t.Fatalf("the composer served none of the parts asked of it: %s", g.Err())
				}
				// Every chunk's every block step the composer's, each at its
				// own width's form.
				{
					np, dp := st.ServedPart(nn.ComposedPrompt)
					if np != len(chunks)*m.Cfg.NLayer || dp != 0 {
						t.Fatalf("the composer served %d block steps of %d chunks of %d blocks and declined %d: %s",
							np, len(chunks), m.Cfg.NLayer, dp, g.Err())
					}
					t.Logf("the composer served the prompt's %d block steps", np)
				}
				ran++
			})
		}
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}

package model

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDeviceProgramMatchesTheEncoder: a block the device program lowers
// (tier's devprog.go) decodes through it, every block placed, and gives the
// tokens the block encoder gives on the same card, its logits within the
// device band of the host's. The program arm counts its block-steps and records
// "program" for every block; the encoder arm (Config.NoProgram) runs none.
//
// VIOLATION SIGNATURE. Drop the KV write from runProg and every token after
// the first parts from the encoder's; point a matvec at the wrong slot and the
// first token parts; let the hook skip the program and ProgBlocks reads 0.
func TestDeviceProgramMatchesTheEncoder(t *testing.T) {
	for _, name := range []string{"stories15M-q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"} {
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(name)), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			type arm struct {
				st    tier.Stats
				forms []string
				prog  string
				first map[string]string
			}
			run := func(noProg bool) ([]int32, [][]float32, arm) {
				opts := append([]tier.Option{tier.WithDevices("cuda"),
					tier.WithConfig(func(c *tier.Config) {
						c.NoProgram = noProg
						// The FFN recipe is timed per tier: pinned, so both arms
						// take the same one.
						c.FFNForm = tier.FFNGate2In
					})}, testTierOpts(t)...)
				g, err := tier.OpenWith(opts...)
				if err != nil || g == nil {
					noDevice(t, "cuda", err)
				}
				// One arm's tier at a time: two would share the card.
				defer g.Close()
				s := m.NewState(64)
				defer s.Close()
				if err := s.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				if s.GPULayers() != m.Cfg.NLayer {
					t.Skipf("CARD TOO SMALL: %d of %d blocks placed", s.GPULayers(), m.Cfg.NLayer)
				}
				lg, err := s.Prefill(m.Vocab.Encode("Once upon a time", true))
				if err != nil {
					t.Fatal(err)
				}
				var toks []int32
				var logits [][]float32
				for i := 0; i < 12; i++ {
					tok := Greedy(lg)
					toks = append(toks, tok)
					if lg, err = s.Forward(tok); err != nil {
						t.Fatal(err)
					}
					logits = append(logits, slices.Clone(lg))
				}
				a := arm{st: g.Stats()}
				for li := 0; li < m.Cfg.NLayer; li++ {
					a.forms = append(a.forms, g.BlockForms(li)["block"])
				}
				a.prog, _ = g.BlockProgram(0)
				a.first = g.BlockForms(0)
				return toks, logits, a
			}
			pt, pl, pa := run(false)
			et, el, ea := run(true)
			// The host's logits for the encoder's tokens, teacher-forced: the
			// reference both device arms are held to.
			hs := m.NewState(64)
			defer hs.Close()
			hl, err := hs.Prefill(m.Vocab.Encode("Once upon a time", true))
			if err != nil {
				t.Fatal(err)
			}
			var host [][]float32
			for _, tok := range et {
				if hl, err = hs.Forward(tok); err != nil {
					t.Fatal(err)
				}
				host = append(host, slices.Clone(hl))
			}
			far := func(a [][]float32) float64 {
				w := 0.0
				for i := range a {
					for j := range a[i] {
						w = math.Max(w, math.Abs(float64(a[i][j]-host[i][j])))
					}
				}
				return w
			}
			ps, es := pa.st, ea.st
			if ps.ProgBlocks == 0 {
				t.Fatalf("the program arm ran no block-step as a device program (forms %v)\n%s", pa.forms, pa.prog)
			}
			if es.ProgBlocks != 0 {
				t.Fatalf("the encoder arm ran %d block-steps as a device program", es.ProgBlocks)
			}
			for li, f := range pa.forms {
				if f != "program" {
					t.Fatalf("block %d ran as %q, not the program", li, f)
				}
			}
			worst := 0.0
			for i := range pl {
				for j := range pl[i] {
					if v := pl[i][j]; math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("token %d: the program's logit %d is %v", i, j, v)
					}
					worst = math.Max(worst, math.Abs(float64(pl[i][j]-el[i][j])))
				}
			}
			prog := pa.prog
			t.Logf("%d block-steps as programs; worst logit difference from the encoder %.4g over %d tokens\n%s",
				ps.ProgBlocks, worst, len(pt), prog)
			if !strings.HasPrefix(prog, "rewrites: ") || strings.HasPrefix(prog, "rewrites: \n") {
				t.Fatalf("block 0's program took no rewrite:\n%s", prog)
			}
			if !slices.Equal(pt, et) {
				t.Fatalf("the program's tokens %v part from the encoder's %v", pt, et)
			}
			// Where both arms launch the same kernels they agree exactly; where
			// a block takes another form the band grows with the depth
			// (RULE 11c), so the bar is the host: the program no farther from
			// it than the encoder, with a margin of a tenth of a logit.
			ph, eh := far(pl), far(el)
			t.Logf("block 0's forms: program %v, encoder %v; worst logit difference from the host: program %.4g, encoder %.4g",
				pa.first, ea.first, ph, eh)
			if ph > eh*1.25+0.1 {
				t.Fatalf("the program is %.4g from the host where the encoder is %.4g", ph, eh)
			}
		})
	}
}

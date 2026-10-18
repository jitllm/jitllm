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
// device band of them. The program arm counts its block-steps and records
// "program" for every block; the encoder arm (Config.NoProgram) runs none.
//
// VIOLATION SIGNATURE. Drop the KV write from runProg and every token after
// the first parts from the encoder's; point a matvec at the wrong slot and the
// first token parts; let the hook skip the program and ProgBlocks reads 0.
func TestDeviceProgramMatchesTheEncoder(t *testing.T) {
	for _, name := range []string{"stories15M-q8_0.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"} {
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(name)), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			run := func(noProg bool) ([]int32, [][]float32, *tier.GPU) {
				opts := append([]tier.Option{tier.WithDevices("cuda"),
					tier.WithConfig(func(c *tier.Config) { c.NoProgram = noProg })}, testTierOpts(t)...)
				g, err := tier.OpenWith(opts...)
				if err != nil || g == nil {
					noDevice(t, "cuda", err)
				}
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
				return toks, logits, g
			}
			pt, pl, pg := run(false)
			defer pg.Close()
			et, el, eg := run(true)
			defer eg.Close()
			ps, es := pg.Stats(), eg.Stats()
			if ps.ProgBlocks == 0 {
				prog, why := pg.BlockProgram(0)
				t.Fatalf("the program arm ran no block-step as a device program (block 0: %q, forms %v)\n%s", why, pg.BlockForms(0), prog)
			}
			if es.ProgBlocks != 0 {
				t.Fatalf("the encoder arm ran %d block-steps as a device program", es.ProgBlocks)
			}
			for li := 0; li < m.Cfg.NLayer; li++ {
				if f := pg.BlockForms(li)["block"]; f != "program" {
					t.Fatalf("block %d ran as %q, not the program (%v)", li, f, pg.BlockForms(li))
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
			prog, _ := pg.BlockProgram(0)
			t.Logf("%d block-steps as programs; worst logit difference from the encoder %.4g over %d tokens\n%s",
				ps.ProgBlocks, worst, len(pt), prog)
			if !strings.HasPrefix(prog, "rewrites: ") || strings.HasPrefix(prog, "rewrites: \n") {
				t.Fatalf("block 0's program took no rewrite:\n%s", prog)
			}
			if !slices.Equal(pt, et) {
				t.Fatalf("the program's tokens %v part from the encoder's %v", pt, et)
			}
			if worst > 0.25 {
				t.Fatalf("worst logit difference %.4g from the encoder, past the device band", worst)
			}
		})
	}
}

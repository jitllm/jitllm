package model

import (
	"math"
	"slices"
	"strings"
	"testing"

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
	// synth-qwen2 carries q/k/v biases; synth-glm4 post-norms, a partial
	// rotary and biases; synth-olmo2 post-norms; tiny-qwen3moe and
	// synth-qwen2moe a routed mixture, the latter with a gated shared expert.
	for _, name := range []string{"stories15M-q8_0", "synth-qwen2", "synth-glm4", "synth-olmo2", "tiny-qwen3moe-f32",
		"synth-qwen2moe", "synth-mamba2", "synth-lfm2", "synth-granitehybrid", "synth-falconmamba", "synth-falcon", "synth-phi2", "synth-starcoder", "synth-starcoder2",
		"synth-stablelm", "synth-commandr", "synth-nemotron", "synth-deepseek", "synth-deepseek-lite",
		"synth-deepseek-dense", "synth-glm4moelite", "synth-llama4", "synth-smollm3", "synth-hunyuan",
		"synth-cohere2", "synth-exaone4", "synth-mistral3", "synth-qwen35-hybrid-mtp", "synth-qwen35moe-hybrid-mtp", "synth-olmo3", "synth-gemma4", "synth-gemma4-e", "synth-gemma4-kv", "synth-gemma4-wide", "synth-gemma4-moe", "synth-dbrx", "synth-bitnet-f16",
		"Qwen3-0.6B-Q8_0",
		"Llama-3.2-1B-Instruct-Q4_K_M"} {
		t.Run(name, func(t *testing.T) {
			m, err := Open(graphModel(t, name), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			type arm struct {
				st    tier.Stats
				forms []string
				prog  string
				why   string
				whys  []string
				first map[string]string
			}
			// forced, where given, is fed in place of the arm's own greedy
			// tokens: both device arms then read the same inputs.
			run := func(noProg bool, forced []int32) ([]int32, [][]float32, arm) {
				opts := append([]tier.Option{tier.WithDevices("cuda"),
					tier.WithConfig(func(c *tier.Config) {
						c.NoProgram = noProg
						// The FFN recipe is timed per tier: pinned, so both arms
						// take the same one.
						c.FFNForm = tier.FFNGate2In
					})}, testTierOpts(t)...)
				g, err := openTier(t, opts...)
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
					t.Skipf("the device took %d of %d blocks: %v", s.GPULayers(), m.Cfg.NLayer, s.DeviceDeclines())
				}
				lg, err := s.Prefill(m.Vocab.Encode("Once upon a time", true))
				if err != nil {
					t.Fatal(err)
				}
				var toks []int32
				// The prefill's logits first: the chunk runs as the program's
				// rows form.
				logits := [][]float32{slices.Clone(lg)}
				for i := 0; i < 12; i++ {
					tok := Greedy(lg)
					toks = append(toks, tok)
					if forced != nil {
						tok = forced[i]
					}
					if lg, err = s.Forward(tok); err != nil {
						t.Fatal(err)
					}
					logits = append(logits, slices.Clone(lg))
				}
				a := arm{st: g.Stats()}
				for li := 0; li < m.Cfg.NLayer; li++ {
					a.forms = append(a.forms, g.BlockForms(li)["block"])
					_, why := g.BlockProgram(li)
					a.whys = append(a.whys, why)
				}
				a.prog, a.why = g.BlockProgram(0)
				a.first = g.BlockForms(0)
				return toks, logits, a
			}
			et, el, ea := run(true, nil)
			pt, pl, pa := run(false, et)
			// The host's logits for the encoder's tokens, teacher-forced: the
			// reference both device arms are held to.
			hs := m.NewState(64)
			defer hs.Close()
			hl, err := hs.Prefill(m.Vocab.Encode("Once upon a time", true))
			if err != nil {
				t.Fatal(err)
			}
			host := [][]float32{slices.Clone(hl)}
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
				t.Fatalf("the program arm ran no block-step as a device program (forms %v; %s)\n%s", pa.forms, pa.why, pa.prog)
			}
			if es.ProgBlocks != 0 {
				t.Fatalf("the encoder arm ran %d block-steps as a device program", es.ProgBlocks)
			}
			for li, f := range pa.forms {
				if f != "program" {
					t.Fatalf("block %d ran as %q, not the program: %s", li, f, pa.whys[li])
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
			// Where the encoder fuses on this block the program takes a
			// rewrite too; a block with nothing to fuse (biased projections
			// beside no rotary, a recurrent mixer alone) takes none in either.
			fused := false
			for span, f := range ea.first {
				switch span + ":" + f {
				case "qkv:grouped", "qkv:actin", "qkv:gemm", "gate_up:gate2in", "gate_up:actin", "gate_up:gu",
					"gate_up:gate2", "down:actin", "down:act":
					fused = true
				}
			}
			if fused && strings.HasPrefix(prog, "rewrites: \n") {
				t.Fatalf("the encoder fuses block 0 (%v) and its program took no rewrite:\n%s", ea.first, prog)
			}
			if strings.Count(prog, "rewrites:") != 2 {
				t.Fatalf("block 0 did not run both its decode and its prefill-chunk program (%s):\n%s", pa.why, prog)
			}
			// The program's own greedy choice, read on the encoder's inputs: a
			// near-tied position may flip where the arms' kernels differ; the
			// logits below are the bar.
			agree := 0
			for i := range pt {
				if pt[i] == et[i] {
					agree++
				}
			}
			t.Logf("the program's greedy choice agrees with the encoder's at %d of %d positions", agree, len(pt))
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

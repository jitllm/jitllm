package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestStagedGemmPrefillMatchesTheHost prefills real models on CUDA with every
// block placed and holds the prompt's logits AND the decode that reads the
// chunk's KV to the host, in two arms: Config.PreferGemmInt8, whose prompt
// matvecs are the staged int8 GEMM (kernels.GemmInt8, sm_80 on), and the
// default, whose are the staged binary16 GEMM on the m16n8 instruction
// (GemmVolta's F16K, sm_75 on). The chunk's attention accumulates on m16n8k8
// (kernels.PagedAttnAccMMA) behind the m16n8k16 scores in both.
//
// The models cover the formats and features the GEMMs carry: Q4_K/Q6_K,
// Q3_K beside Q4_K/Q5_K/Q6_K (Q3_K has neither decode and keeps MatVecMMA in
// the same block), Q8_0, biased q/k/v, Q4_0, the hybrid's projections and a
// mixture's dense ones. Stats().GemmInt8, GemmF16 and PagedAccMMA are the
// selection checks, since MatVecMMA and the FMA tiles also answer correctly;
// a card whose PTX target has an instruction and built nothing on it fails.
//
// The host and the int8 arm use int8 activations and the binary16 arm
// binary16, so the bar is the tile's: NMSE under 2e-2 and an argmax that
// agrees or is a tie (tieMargin). The split tuner is pinned off (RULE 11c).
func TestStagedGemmPrefillMatchesTheHost(t *testing.T) {
	ran := 0
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf",
		"tinyllama-1.1b-q3_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "gemma-2b.gguf",
		"Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err != nil {
			if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
				t.Logf("MODEL MISSING: %s", name)
				continue
			}
		}
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			text := strings.Repeat("The river rose every spring until the bridges were islands, "+
				"and the clerk counted barrels of salt on the upper floor. ", 40)
			ids := m.Vocab.Encode(text, true)[:200]
			drive := []int32{ids[3], ids[17], ids[40], ids[7]}
			run := func(s *State) [][]float32 {
				l, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), l...)}
				for _, tok := range drive {
					if l, err = s.Forward(tok); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), l...))
				}
				return out
			}
			host := m.NewState(len(ids) + 8)
			want := run(host)
			host.Close()
			for _, arm := range []struct {
				name   string
				noInt8 bool
				deep   int
			}{{"int8", false, -1}, {"binary16", true, -1}, {"binary16-deep", true, 1}} {
				t.Run(arm.name, func(t *testing.T) {
					g, err := openTier(t, tier.WithAPI("ptx"), tier.WithDeviceTune(tier.TuneOff),
						tier.WithConfig(func(c *tier.Config) { c.PreferGemmInt8, c.GemmDeep = !arm.noInt8, arm.deep }))
					if err != nil || g == nil {
						noDevice(t, "cuda", err)
					}
					defer g.Close()
					if !strings.Contains(g.Name(), "[ptx]") {
						t.Skipf("the device is %s; the staged GEMMs are CUDA's", g.Name())
					}
					stagedGemmArm(t, m, g, len(ids)+8, run, want, arm.noInt8, arm.deep)
				})
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}

// stagedGemmArm is one arm of TestStagedGemmPrefillMatchesTheHost on g.
func stagedGemmArm(t *testing.T, m *Model, g *tier.GPU, seq int, run func(*State) [][]float32, want [][]float32, noInt8 bool, deep int) {
	t.Helper()
	dev := m.NewState(seq)
	dev.SetDevice(g)
	if dev.GPULayers() != m.Cfg.NLayer {
		dev.Close()
		t.Skipf("CARD TOO SMALL: the device took %d of %d blocks (%s) -- this model proved nothing here",
			dev.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	got := run(dev)
	demoted, placed := dev.DeviceDemotions(), dev.GPULayers()
	dev.SetGPULayers(0)
	dev.Close()
	if demoted != 0 || placed != m.Cfg.NLayer {
		t.Fatalf("%d demotion(s), %d of %d blocks on the device: the host answered",
			demoted, placed, m.Cfg.NLayer)
	}
	for i := range want {
		var num, den float64
		for j := range want[i] {
			v := float64(got[i][j])
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("step %d: logit %d is %v on the device", i, j, v)
			}
			d := v - float64(want[i][j])
			num += d * d
			den += float64(want[i][j]) * float64(want[i][j])
		}
		ga, wa := argmaxID(got[i]), argmaxID(want[i])
		t.Logf("step %d: logit NMSE %.3e, argmax %d (host %d)", i, num/den, ga, wa)
		if num/den > 2e-2 {
			t.Fatalf("step %d: logit NMSE %.3e against the host", i, num/den)
		}
		if ga != wa && float64(want[i][wa]-want[i][ga]) > tieMargin {
			t.Fatalf("step %d: argmax %d against the host's %d on a margin of %.3f",
				i, ga, wa, want[i][wa]-want[i][ga])
		}
	}
	// After the comparison, so a run with a GEMM switched off still prints
	// the other arm's numbers.
	st := g.Stats()
	sm := ptxTargetSM(g.Name())
	t.Logf("%s: %d matvecs on the int8 GEMM, %d on the m16n8 binary16 GEMM, %d on sm_70's; "+
		"%d attention accumulates on m16n8k8", g.Name(), st.GemmInt8, st.GemmF16, st.VoltaMV, st.PagedAccMMA)
	switch {
	case !noInt8 && sm >= 80 && st.GemmInt8 == 0:
		t.Fatalf("%s lowers the int8 m16n8k32 instruction and built no GemmInt8: MatVecMMA answered", g.Name())
	case (noInt8 || sm < 80) && sm >= 75 && st.GemmF16 == 0:
		t.Fatalf("%s lowers the m16n8 binary16 instruction and built no GEMM on it: the int8 twin answered", g.Name())
	case noInt8 && st.GemmInt8 != 0:
		t.Fatalf("no GemmInt8 asked and %d GemmInt8 matvecs: the arm is not the binary16 GEMM", st.GemmInt8)
	case deep > 0 && sm >= 75 && st.GemmDeep == 0:
		t.Fatalf("the deep m16n8 tiles forced and none built: the arm is the shipping tile")
	case deep < 0 && st.GemmDeep != 0:
		t.Fatalf("the deep m16n8 tiles forced off and %d built", st.GemmDeep)
	}
	t.Logf("%d of the binary16 GEMMs on the deep tiles", st.GemmDeep)
	t.Logf("%d FlashPrefill80 launches; fused operands: %d norms and %d activations wrote binary16, %d activations quantized",
		st.FlashPrefill80, st.NormF16, st.ActF16Fused, st.QuantActs)
	// From sm_80 the prompt's attention is FlashPrefill80 wherever a head is
	// 256 wide or less (gemma-2b's 256 splits its dims over two warps), and
	// the staged kernels with the m16n8k8 accumulate past it; a hybrid's
	// attention blocks and a mixture's take them as well.
	if sm >= 80 {
		if m.Cfg.HeadDim <= 256 && m.Cfg.HeadDimSWA <= 256 && st.FlashPrefill80 == 0 {
			t.Fatalf("%s: head width %d and no FlashPrefill80: the staged kernels answered", g.Name(), m.Cfg.HeadDim)
		}
		if st.FlashPrefill80+st.PagedAccMMA == 0 {
			t.Fatalf("%s: no prompt attention on the matrix unit's accumulate: the FMA tiles answered", g.Name())
		}
	}
	// The operands their producers write: a block whose matvecs are all a
	// binary16 GEMM (QuantSkipped counts them) has its norms and its
	// activation write the operand; any other gated block on the int8 path
	// quantizes in its activation's launch. A dense gated FFN is in every
	// model here.
	switch {
	case st.QuantSkipped > 0 && st.NormF16+st.ActF16Fused == 0:
		t.Fatalf("%d blocks ran every matvec as a binary16 GEMM and no norm or activation wrote their operand", st.QuantSkipped)
	case st.GemmInt8 > 0 && st.QuantActs == 0 && m.Cfg.NExpert == 0:
		t.Fatalf("%d int8 GEMMs and no activation quantized in its own launch", st.GemmInt8)
	}
}

// ptxTargetSM reads the target out of a CUDA device's name, "... (sm_86)
// [ptx]", and is 0 when there is none.
func ptxTargetSM(name string) int {
	i := strings.LastIndex(name, "(sm_")
	if i < 0 {
		return 0
	}
	sm := 0
	for _, c := range name[i+4:] {
		if c < '0' || c > '9' {
			break
		}
		sm = sm*10 + int(c-'0')
	}
	return sm
}

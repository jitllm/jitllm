//go:build jitllmfault

package model

import (
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// pagedFaultCase is one violation TestPagedKVGateDiscriminates runs.
type pagedFaultCase struct {
	file, fault string
	repeat      int
	// arm sets a fault of the kernels package (a package switch of its own);
	// nil says the fault is the tier's and rides on the arm's tier options.
	arm func(string)
	// prefill runs the paged arm's prompt batched, through the paged prefill
	// kernels, at pass keys a launch (0 the default).
	prefill bool
	pass    int
	// staged decodes the paged arm through the staged kernels
	// (tier.Config.StagedDecode), row at a time.
	staged bool
}

// TestPagedKVGateDiscriminates runs the paged gates' comparison under each
// violation, in the kernels and in the tier's wiring, and demands the
// difference lands outside the band the gate allows (RULE 10): a gate that
// passes a wrong page, a lost window or two sequences reading one table
// proves nothing.
//
//	go test -tags jitllmfault ./engine/model -run TestPagedKVGateDiscriminates
func TestPagedKVGateDiscriminates(t *testing.T) {
	const gemma = "gemma-3-1b-it-Q4_K_M.gguf|google_gemma-3-1b-it-Q4_K_M.gguf"
	cases := []pagedFaultCase{
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "(t-1)/P", 6, oldkernels.SetPagedFault, false, 0, false},
		{gemma, "unaligned", 60, oldkernels.SetPagedFault, false, 0, false},
		{gemma, "window", 60, nil, false, 0, false},
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "(t-1)/P", 6, oldkernels.SetPagedFault, true, 0, false},
		{gemma, "unaligned", 60, oldkernels.SetPagedFault, true, 0, false},
		{gemma, "window", 60, nil, true, 0, false},
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "passclip", 6, nil, true, 64, false},
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "normpart", 6, oldkernels.SetPagedFault, false, 0, true},
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "(t-1)/P", 6, oldkernels.SetPagedFault, false, 0, true},
	}
	for _, c := range cases {
		name := c.fault
		if c.prefill {
			name += "/prefill"
		}
		if c.staged {
			name += "/staged"
		}
		t.Run(name, func(t *testing.T) {
			p := firstModel(c.file)
			if _, err := os.Stat(p); err != nil {
				t.Skipf("MODEL MISSING: %v -- this gate proved nothing", err)
			}
			m, err := Open(jlmOf(t, p), noTune, noGEMM)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			prompt := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog, and then it runs. ", c.repeat), true)
			seq := len(prompt) + 8
			forced := []int32{prompt[1], prompt[2], prompt[3]}
			rowwise := func(o []tier.Option, split int) []tier.Option {
				return append(o, tier.WithConfig(func(c *tier.Config) { c.NoBatch, c.FlashSplit = true, split }))
			}
			// The clean paged run row at a time, and the order-only control
			// beside it (two splits), which sizes the band.
			want, _ := pagedRun(t, m, seq, prompt, forced, rowwise(pagedOpts(pagedKVPage), 1))
			ctl, _ := pagedRun(t, m, seq, prompt, forced, rowwise(pagedOpts(pagedKVPage), 2))
			arm := rowwise(pagedOpts(pagedKVPage), 1)
			if c.staged {
				arm = append(arm, tier.WithConfig(func(cf *tier.Config) { cf.StagedDecode = true }))
			}
			if c.prefill {
				arm = append(pagedOpts(pagedKVPage), tier.WithConfig(func(cf *tier.Config) { cf.PrefillPass = c.pass }))
				// The clean arm first: the same batched configuration must sit
				// inside the band, or the violation's distance means nothing.
				clean, st := pagedRun(t, m, seq, prompt, forced, arm)
				if st.PagedPrefillLaunches == 0 || c.pass > 0 && st.PagedPrefillPasses == 0 {
					t.Fatalf("the prefill arm did not select the paged prefill path (%d launches, %d passes)",
						st.PagedPrefillLaunches, st.PagedPrefillPasses)
				}
				var ec, bo float64
				for i := range want {
					ec = max(ec, logitNMSE(clean[i], want[i]))
					bo = max(bo, logitNMSE(ctl[i], want[i]))
				}
				if ec > pagedBand*bo {
					t.Fatalf("the clean prefill arm sits %.3e from the clean row-at-a-time one, outside its %.3e band", ec, pagedBand*bo)
				}
			}
			if c.arm != nil {
				c.arm(c.fault)
			} else {
				arm = append(arm[:len(arm):len(arm)], tier.WithFault("paged", c.fault))
			}
			got, _ := pagedRun(t, m, seq, prompt, forced, arm)
			if c.arm != nil {
				c.arm("")
			}
			var wo, e float64
			for i := range want {
				wo = max(wo, logitNMSE(ctl[i], want[i]))
				e = max(e, logitNMSE(got[i], want[i]))
			}
			if e <= pagedBand*wo {
				t.Fatalf("under %q the paged arm sits %.3e from the clean one, inside the %.3e band: the gate cannot see it",
					c.fault, e, pagedBand*wo)
			}
			t.Logf("%q: NMSE %.3e against a band of %.3e (%.0fx)", c.fault, e, pagedBand*wo, e/(pagedBand*wo))
		})
	}
	t.Run("alias", func(t *testing.T) {
		p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
		if _, err := os.Stat(p); err != nil {
			t.Skipf("MODEL MISSING: %v -- this gate proved nothing", err)
		}
		m, err := Open(jlmOf(t, p), noTune, noGEMM)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		ids := [][]int32{m.Vocab.Encode("Once upon a time, in a small village by the sea, there lived", true),
			m.Vocab.Encode("1, 2, 3,", true), m.Vocab.Encode("Water boils at", true)}
		got, _, _ := batchGreedy(t, m, ids, 8, 256, append([]tier.Option{tier.WithDevices("gpu:0"), tier.WithFault("paged", "alias")}, pagedOpts(pagedKVPage)...))
		for i := 1; i < len(ids); i++ {
			want := hostGreedy(t, m, ids[i], 8)
			for j := range want {
				if got[i][j] != want[j] {
					if gap := hostGap(t, m, ids[i], want[:j], want[j], got[i][j]); gap > batchTie {
						t.Logf("alias: row %d parts from the host at token %d, %.3f apart -- the batch gate fires", i, j, gap)
						return
					}
					break
				}
			}
		}
		t.Fatal("with every row reading row 0's pages, every row still matched the host: the batch gate cannot see aliasing")
	})
}

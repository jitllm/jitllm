package model

import (
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// pagedKVPage is the page the gates run at: the smallest the kernels take, so
// a short prompt crosses several page edges and a window starts mid-page.
const pagedKVPage = 64

// pagedOpts is the tier configuration of a paged arm at page size page.
func pagedOpts(page int) []tier.Option {
	return []tier.Option{tier.WithConfig(func(c *tier.Config) { c.KVPage = page })}
}

// pagedRun prefills prompt on a device and then feeds the forced ids one at a
// time, returning every step's logits (the prefill's first) and the tier's
// counters. forced is the same for both arms, so the logits are comparable
// position by position rather than only while the greedy chains agree.
func pagedRun(t *testing.T, m *Model, seq int, prompt, forced []int32, opt []tier.Option) ([][]float32, tier.Stats) {
	t.Helper()
	g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff)}, opt...)...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	st := m.NewState(seq)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.devCount() != m.Cfg.NLayer || !st.HeadOnDevice() {
		t.Fatalf("placed %d of %d blocks, head on the card %v: %s", st.devCount(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
	}
	lg, err := st.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{slices.Clone(lg)}
	for _, id := range forced {
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
	}
	if st.devCount() != m.Cfg.NLayer {
		t.Fatalf("%d of %d blocks left the card during the run: %s", m.Cfg.NLayer-st.devCount(), m.Cfg.NLayer, g.Err())
	}
	return out, g.Stats()
}

// logitNMSE is |a-b|^2 / |b|^2 over one logit row.
func logitNMSE(a, b []float32) float64 {
	var num, den float64
	for i := range b {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(b[i]) * float64(b[i])
	}
	return num / den
}

// pagedBand is how many times the order-only control a device arm may sit from
// the host. The device reads the same history through kernels that sum in
// another order (the prefill scores in binary16 on the matrix units, split
// partials merged, a window's tiles from the aligned position below it), and
// how far an order change carries is the model's own property: measured, the
// paged arm against itself at two split counts moves Llama-3.2-1B's logits by
// NMSE ~2e-03 and gemma-3-1b's by ~6e-03. So the bound is that control,
// measured in the same run, not a constant.
const pagedBand = 4

// pagedCase is one model TestPagedKVMatchesTheHost runs.
type pagedCase struct {
	file   string
	repeat int  // prompt repetitions: gemma's must pass its 512-key window
	f16    bool // V stored as packed binary16
}

// TestPagedKVMatchesTheHost runs one sequence -- a prefill crossing several
// pages, then decode -- with the history in the device's pool, feeding the
// host's own greedy ids, and holds every step's logits to the host's: batched,
// row at a time, prefilled in passes and decoded through the staged kernels.
// It asserts each arm took the path it names (Stats).
//
// gemma-3-1b's local layers keep a 512-key window, so a prompt past it makes
// the windowed descriptors (keyStart mid-page) load-bearing; Llama-3.2-1B is
// the dense GQA shape every board row runs.
func TestPagedKVMatchesTheHost(t *testing.T) {
	cases := []pagedCase{
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", 6, false},
		{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", 6, true},
		{"gemma-3-1b-it-Q4_K_M.gguf|google_gemma-3-1b-it-Q4_K_M.gguf", 60, false},
	}
	ran := 0
	for _, c := range cases {
		name := strings.TrimSuffix(strings.Split(c.file, "|")[0], ".gguf")
		if c.f16 {
			name += "/f16V"
		}
		t.Run(name, func(t *testing.T) {
			// The arms' options, with the V width of this case in each.
			f16 := tier.WithConfig(func(cf *tier.Config) { cf.KVF16 = c.f16 })
			pagedOpts := func(page int) []tier.Option { return append(pagedOpts(page), f16) }
			p := firstModel(c.file)
			if _, err := os.Stat(p); err != nil {
				t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
			}
			m, err := Open(jlmOf(t, p), noTune, noGEMM)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			prompt := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog, and then it runs. ", c.repeat), true)
			const gen = 24
			seq := len(prompt) + gen + 8
			if m.Cfg.SWAWindow > 0 && len(prompt) <= m.Cfg.SWAWindow {
				t.Fatalf("a %d-token prompt does not pass the %d-key window, so the windowed descriptors are not exercised",
					len(prompt), m.Cfg.SWAWindow)
			}
			// The ids every arm is fed: the host's own greedy chain, so the
			// logits are compared position by position.
			forced := hostGreedy(t, m, prompt, gen)
			host := hostRun(t, m, seq, prompt, forced)
			got, ps := pagedRun(t, m, seq, prompt, forced, pagedOpts(pagedKVPage))
			// Prefill in passes of 64 keys, so the fold runs on every case:
			// its default width is 4096 and no prompt here reaches it. The
			// passes are the staged kernels', so the flash forms are off: Metal
			// and sm_70 otherwise take FlashPrefillTile and FlashPrefill70,
			// which fold no pass, and the arm would select nothing.
			gotP, pps := pagedRun(t, m, seq, prompt, forced, append(pagedOpts(pagedKVPage),
				tier.WithConfig(func(c *tier.Config) { c.PrefillPass, c.NoFlashPrefill = 64, true })))
			// Decode through the staged kernels, which the plan takes only at a
			// depth no prompt here reaches.
			gotS, sps := pagedRun(t, m, seq, prompt, forced, append(pagedOpts(pagedKVPage),
				tier.WithConfig(func(c *tier.Config) { c.StagedDecode = true })))
			rowwise := func(o []tier.Option, split int) []tier.Option {
				return append(o, tier.WithConfig(func(c *tier.Config) { c.NoBatch, c.FlashSplit = true, split }))
			}
			gotR, _ := pagedRun(t, m, seq, prompt, forced, rowwise(pagedOpts(pagedKVPage), 1))
			// The order-only control: row at a time at two splits, which moves
			// nothing but the summation order. It sizes the band.
			ctl, _ := pagedRun(t, m, seq, prompt, forced, rowwise(pagedOpts(pagedKVPage), 2))
			var wo float64
			for i := range host {
				wo = max(wo, logitNMSE(ctl[i], gotR[i]))
			}
			if ps.PagedLaunches == 0 {
				t.Fatal("the paged arm launched no paged attention kernel: it did not read the pool")
			}
			if ps.PagedPrefillLaunches == 0 {
				t.Fatal("the paged arm prefilled through no paged prefill kernel: the chunk path was not selected")
			}
			if pps.PagedPrefillPasses == 0 {
				t.Fatal("the pass arm ran no prefill pass: the fold was not selected")
			}
			if sps.PagedStagedLaunches == 0 {
				t.Fatal("the staged arm decoded through no staged kernel: the path was not selected")
			}
			if wo == 0 || math.IsNaN(wo) {
				t.Fatalf("the order-only control moved nothing (%v): it cannot size the band", wo)
			}
			bound := pagedBand * wo
			worst, flips := 0.0, 0
			for i := range host {
				e := max(logitNMSE(got[i], host[i]), logitNMSE(gotR[i], host[i]), logitNMSE(gotP[i], host[i]),
					logitNMSE(gotS[i], host[i]))
				if math.IsNaN(e) || math.IsInf(e, 0) {
					t.Fatalf("step %d: non-finite NMSE %v", i, e)
				}
				worst = max(worst, e)
				if Greedy(got[i]) != Greedy(host[i]) {
					flips++
				}
				if e > bound {
					t.Errorf("step %d: logit NMSE %.3e against the host, bound %.3e (%dx the order-only control)",
						i, e, bound, pagedBand)
				}
			}
			t.Logf("%d prompt tokens, %d decoded, page %d: worst logit NMSE %.3e, %d argmax flip(s), %d paged launches (%d prefill, %d passes at 64 keys, %d staged decode)",
				len(prompt), gen, pagedKVPage, worst, flips, ps.PagedLaunches, ps.PagedPrefillLaunches, pps.PagedPrefillPasses,
				sps.PagedStagedLaunches)
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "%s", "no case ran -- this gate proved nothing")
	}
}

// TestPagedKVBatchMatchesHost runs three sequences of different lengths as one
// batch with every row's history in its own pages, and holds each row's
// greedy tokens to the host's up to a tie -- the check that rows of one
// session never read each other's pages.
func TestPagedKVBatchMatchesHost(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune, noGEMM)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := []string{
		strings.Repeat("Once upon a time, in a small village by the sea, there lived ", 3),
		"1, 2, 3,",
		"Water boils at",
	}
	ids := make([][]int32, len(prompts))
	for i, s := range prompts {
		ids[i] = m.Vocab.Encode(s, true)
	}
	const gen, seq = 12, 256
	got, after, _ := batchGreedy(t, m, ids, gen, seq, append([]tier.Option{tier.WithDevices("gpu:0")}, pagedOpts(pagedKVPage)...))
	if lastBatchStats.PagedLaunches == 0 {
		t.Fatal("the batch launched no paged attention kernel")
	}
	for i := range ids {
		want := hostGreedy(t, m, ids[i], gen)
		for j := range want {
			if got[i][j] == want[j] {
				continue
			}
			if gap := hostGap(t, m, ids[i], want[:j], want[j], got[i][j]); gap > batchTie {
				t.Fatalf("row %d token %d: %d paged in a batch, %d on the host, %.4f apart", i, j, got[i][j], want[j], gap)
			}
			break
		}
	}
	want0 := hostGreedy(t, m, ids[0], gen)
	for j := range want0 {
		if after[j] != want0[j] {
			if gap := hostGap(t, m, ids[0], want0[:j], want0[j], after[j]); gap > batchTie {
				t.Fatalf("the single State after the batch, token %d: %d, host %d, %.4f apart", j, after[j], want0[j], gap)
			}
			break
		}
	}
	t.Logf("%d rows, %d tokens each, %d paged launches", len(ids), gen, lastBatchStats.PagedLaunches)
}

// firstModel is the first of |-separated file names present in the model
// directory, or the first name's path when none is (the caller's Stat reports
// it): hosts name the same checkpoint differently.
func firstModel(names string) string {
	for _, n := range strings.Split(names, "|") {
		if p := testmodels.Path(n); fileExists(p) {
			return p
		}
	}
	return testmodels.Path(strings.Split(names, "|")[0])
}

// hostRun is pagedRun on the host: the prompt one token at a time, then the
// forced ids, every step's logits.
func hostRun(t *testing.T, m *Model, seq int, prompt, forced []int32) [][]float32 {
	t.Helper()
	st := m.NewState(seq)
	defer st.Close()
	var lg []float32
	var err error
	for _, id := range prompt {
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	out := [][]float32{slices.Clone(lg)}
	for _, id := range forced {
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
	}
	return out
}

// hostGreedy is prompt's first gen greedy tokens on the host.
func hostGreedy(t *testing.T, m *Model, prompt []int32, gen int) []int32 {
	t.Helper()
	st := m.NewState(len(prompt) + gen + 1)
	defer st.Close()
	var next int32
	var err error
	for _, id := range prompt {
		if next, err = st.ForwardGreedy(id); err != nil {
			t.Fatal(err)
		}
	}
	out := make([]int32, 0, gen)
	for len(out) < gen {
		out = append(out, next)
		if next, err = st.ForwardGreedy(next); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

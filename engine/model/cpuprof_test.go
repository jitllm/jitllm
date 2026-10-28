package model

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestComposedCPUProfile decomposes a decode token with every block on the
// CPU: main's per-op time against the composed kernels' time, launches and
// calls, by kernel, with the pool's regions a token and the bytes a token
// against the CPU's read rate measured in the same process. A measurement:
// JITLLM_COMPOSED_PROF=1, the model JITLLM_COMPOSED_MODEL.
func TestComposedCPUProfile(t *testing.T) {
	if os.Getenv("JITLLM_COMPOSED_PROF") == "" {
		t.Skip("a measurement: set JITLLM_COMPOSED_PROF=1")
	}
	const warm, toks = 32, 64
	path := os.Getenv("JITLLM_COMPOSED_MODEL")
	if path == "" {
		path = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
	}
	path = jlmOf(t, testmodels.Resolve(path))
	host, err := probe.Host()
	if err != nil {
		t.Fatal(err)
	}
	for _, composed := range []bool{false, true} {
		m, err := Open(path, WithComposedCPU(composed), WithProfile(true))
		if err != nil {
			t.Fatal(err)
		}
		st := m.NewState(warm + toks + 64)
		l, err := st.Prefill(m.Vocab.Encode("The river rose every spring until", true))
		if err != nil {
			t.Fatal(err)
		}
		tok := int32(argmax(l))
		run := func(n int) time.Duration {
			t0 := time.Now()
			for range n {
				lg, err := st.Forward(tok)
				if err != nil {
					t.Fatal(err)
				}
				tok = int32(argmax(lg))
			}
			return time.Since(t0)
		}
		run(warm)
		ResetProfile()
		m.cpuForms.mu.Lock()
		for _, f := range m.cpuForms.m {
			for i := range f.codes {
				f.codes[i].nanos, f.codes[i].runs = 0, 0
			}
		}
		m.cpuForms.mu.Unlock()
		st.jit.ResetRegions()
		d := run(toks)
		par, ser := st.jit.Regions()
		per := float64(d.Microseconds()) / toks
		bpt := float64(m.BytesPerToken())
		gb := bpt / (per * 1e-6) / 1e9
		served, declined := st.CPUBlockSteps()
		var b strings.Builder
		fmt.Fprintf(&b, "composed=%v: %.0f us a token (%.1f tok/s), %.1f MB a token = %.1f GB/s = %.0f%% of the %.1f GB/s wall; regions a token %.1f parallel %.1f serial; CPU block steps %d declined %v\n",
			composed, per, 1e6/per, bpt/1e6, gb, 100*gb*1e9/host.Bandwidth, host.Bandwidth/1e9, float64(par)/toks, float64(ser)/toks, served, declined)
		names, ns := OpProfile()
		for i, n := range ns {
			if n > 0 {
				fmt.Fprintf(&b, "  main %-12s %8.1f us a token\n", names[i], float64(n)/1e3/toks)
			}
		}
		type row struct {
			name               string
			nanos, runs, calls int64
			bytes              int
		}
		var rows []row
		m.cpuForms.mu.Lock()
		for _, f := range m.cpuForms.m {
			for i := range f.codes {
				c := &f.codes[i]
				if c.runs > 0 {
					rows = append(rows, row{c.name, c.nanos, c.runs, int64(c.n), c.code.Size})
				}
			}
		}
		m.cpuForms.mu.Unlock()
		slices.SortFunc(rows, func(a, b row) int { return int(b.nanos - a.nanos) })
		var tot int64
		for _, r := range rows {
			tot += r.nanos
		}
		if len(rows) > 0 {
			fmt.Fprintf(&b, "  composed kernels %.1f us a token (%.0f%% of the token)\n", float64(tot)/1e3/toks, 100*float64(tot)/float64(d.Nanoseconds()))
		}
		for _, r := range rows {
			fmt.Fprintf(&b, "  %-60.60s %8.1f us/tok %5.1f%% launches/tok %5.1f calls %6d  %.2f us/launch  %d B\n",
				r.name, float64(r.nanos)/1e3/toks, 100*float64(r.nanos)/float64(tot), float64(r.runs)/toks, r.calls, float64(r.nanos)/1e3/float64(r.runs), r.bytes)
		}
		t.Log(b.String())
		st.Close()
		m.Close()
	}
}

package model

import (
	"fmt"
	"os"
	"slices"
	"strconv"
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
// JITLLM_COMPOSED_PROF=1, the model JITLLM_COMPOSED_MODEL. With
// JITLLM_COMPOSED_PP=n it decomposes a prompt of n tokens instead, each
// figure a prompt token's.
func TestComposedCPUProfile(t *testing.T) {
	if os.Getenv("JITLLM_COMPOSED_PROF") == "" {
		t.Skip("a measurement: set JITLLM_COMPOSED_PROF=1")
	}
	warm, toks, pp := 32, 64, 0
	if v := os.Getenv("JITLLM_COMPOSED_PP"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			t.Fatal(err)
		}
		warm, toks, pp = n, 2*n, n
	}
	ft := float64(toks)
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
		m, err := Open(path, WithProfile(true))
		if err != nil {
			t.Fatal(err)
		}
		st := m.NewInstance(warm + toks + 64)
		prompt := m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands. ", 1+pp/8), true)[:pp]
		l, err := st.Prefill(m.Vocab.Encode("The river rose every spring until", true))
		if err != nil {
			t.Fatal(err)
		}
		tok := int32(argmax(l))
		each := make([]time.Duration, 0, toks)
		run := func(n int) time.Duration {
			t0 := time.Now()
			if pp > 0 {
				for range n / pp {
					st.Reset()
					if _, err := st.Prefill(prompt); err != nil {
						t.Fatal(err)
					}
				}
				return time.Since(t0)
			}
			for range n {
				t1 := time.Now()
				lg, err := st.Forward(tok)
				if err != nil {
					t.Fatal(err)
				}
				if len(each) < cap(each) {
					each = append(each, time.Since(t1))
				}
				tok = int32(argmax(lg))
			}
			return time.Since(t0)
		}
		run(warm)
		each = each[:0]
		ResetProfile()
		m.cpuForms.mu.Lock()
		for _, f := range m.cpuForms.m {
			for i := range f.codes {
				f.codes[i].nanos, f.codes[i].runs = 0, 0
				f.codes[i].pcalls, f.codes[i].pcallNs, f.codes[i].pstartNs, f.codes[i].pmaxNs = 0, 0, 0, 0
			}
		}
		m.cpuForms.mu.Unlock()
		st.jit.ResetRegions()
		ph0 := st.jit.Phases()
		d := run(toks)
		par, ser := st.jit.Regions()
		phases := st.jit.Phases() - ph0
		per := float64(d.Microseconds()) / ft
		bpt := float64(m.BytesPerToken())
		gb := bpt / (per * 1e-6) / 1e9
		served, declined := st.CPUBlockSteps()
		var b strings.Builder
		fmt.Fprintf(&b, "composed=%v: %.0f us a token (%.1f tok/s), %.1f MB a token = %.1f GB/s = %.0f%% of the %.1f GB/s wall; regions a token %.1f parallel %.1f serial, %.1f phases; CPU block steps %d declined %v\n",
			composed, per, 1e6/per, bpt/1e6, gb, 100*gb*1e9/host.Bandwidth, host.Bandwidth/1e9, float64(par)/ft, float64(ser)/ft, float64(phases)/ft, served, declined)
		if len(each) > 0 {
			slices.Sort(each)
			fmt.Fprintf(&b, "  a token's time: min %v p10 %v median %v p90 %v max %v\n", each[0], each[len(each)/10], each[len(each)/2], each[len(each)*9/10], each[len(each)-1])
		}
		names, ns := OpProfile()
		for i, n := range ns {
			if n > 0 {
				fmt.Fprintf(&b, "  main %-12s %8.1f us a token\n", names[i], float64(n)/1e3/ft)
			}
		}
		type row struct {
			name                    string
			calls                   []int
			serial                  bool
			nanos, runs, n          int64
			bytes                   int
			pc, pcall, pstart, pmax int64
		}
		var rows []row
		m.cpuForms.mu.Lock()
		for _, f := range m.cpuForms.m {
			for i := range f.codes {
				c := &f.codes[i]
				if c.runs > 0 {
					rows = append(rows, row{c.name + c.desc, c.calls, c.serial, c.nanos, c.runs, int64(c.n), c.code.Size, c.pcalls, c.pcallNs, c.pstartNs, c.pmaxNs})
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
			fmt.Fprintf(&b, "  composed kernels %.1f us a token (%.0f%% of the token)\n", float64(tot)/1e3/ft, 100*float64(tot)/float64(d.Nanoseconds()))
		}
		for _, r := range rows {
			fmt.Fprintf(&b, "  %-70.70s %8.1f us/tok %5.1f%% launches/tok %5.1f calls %v serial %v  %.2f us/launch  %d B\n",
				r.name, float64(r.nanos)/1e3/ft, 100*float64(r.nanos)/float64(tot), float64(r.runs)/ft, r.calls, r.serial, float64(r.nanos)/1e3/float64(r.runs), r.bytes)
			if r.pc > 0 {
				fmt.Fprintf(&b, "      calls/launch %.1f, a call %.2f us, starts %.2f us after the launch, the last %.2f us\n",
					float64(r.pc)/float64(r.runs), float64(r.pcall)/1e3/float64(r.pc), float64(r.pstart)/1e3/float64(r.pc), float64(r.pmax)/1e3/float64(r.runs))
			}
		}
		if composed {
			if cb := st.cpu.blocks[st.lo]; cb != nil && cb.f != nil {
				b.WriteString("  block 0's kernels in order:\n")
				for _, c := range cb.f.codes {
					fmt.Fprintf(&b, "    %-60.60s calls %v n %d serial %v join %d chain %d", c.name+c.desc, c.calls, c.n, c.serial, c.join, c.chain)
					for _, a := range c.args {
						if a.src == cpuArgWeight || a.src == cpuArgCoord {
							fmt.Fprintf(&b, " [src %d region %d per %v elem %d off %d]", a.src, a.region, a.per, a.elem, a.off)
						}
					}
					b.WriteString("\n")
				}
			}
		}
		t.Log(b.String())
		st.Close()
		m.Close()
	}
}

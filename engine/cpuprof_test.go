package engine

import (
	"fmt"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
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
		instOpts := []Option{WithProfile(true)}
		m, err := model.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		st := testInstance(t, m, warm+toks+64, instOpts...)
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
		st.ResetProfile()
		for _, f := range st.eng.cpu.Forms(m.Id).All() {
			f.ResetProfile()
		}
		cpudev.PoolOf(st.Devices()).ResetRegions()
		ph0 := cpudev.PoolOf(st.Devices()).Phases()
		d := run(toks)
		par, ser := cpudev.PoolOf(st.Devices()).Regions()
		phases := cpudev.PoolOf(st.Devices()).Phases() - ph0
		per := float64(d.Microseconds()) / ft
		bpt := float64(m.BytesPerToken())
		gb := bpt / (per * 1e-6) / 1e9
		served, declined := st.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
		var b strings.Builder
		fmt.Fprintf(&b, "composed=%v: %.0f us a token (%.1f tok/s), %.1f MB a token = %.1f GB/s = %.0f%% of the %.1f GB/s wall; regions a token %.1f parallel %.1f serial, %.1f phases; CPU block steps %d declined %v\n",
			composed, per, 1e6/per, bpt/1e6, gb, 100*gb*1e9/host.Bandwidth, host.Bandwidth/1e9, float64(par)/ft, float64(ser)/ft, float64(phases)/ft, served, declined)
		if len(each) > 0 {
			slices.Sort(each)
			fmt.Fprintf(&b, "  a token's time: min %v p10 %v median %v p90 %v max %v\n", each[0], each[len(each)/10], each[len(each)/2], each[len(each)*9/10], each[len(each)-1])
		}
		names, ns := st.OpProfile()
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
		for _, f := range st.eng.cpu.Forms(m.Id).All() {
			for i := range f.Codes {
				c := f.Codes[i].Profile()
				if c.Runs > 0 {
					rows = append(rows, row{c.Name, c.Calls, c.Serial, c.Nanos, c.Runs, int64(c.N), c.Bytes, c.CallCount, c.CallNs, c.StartNs, c.MaxNs})
				}
			}
		}
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
			if cb, _ := st.exec.x.Block(st.lo); cb != nil && cb.F != nil {
				b.WriteString("  block 0's kernels in order:\n")
				for i := range cb.F.Codes {
					c := &cb.F.Codes[i]
					p := c.Profile()
					fmt.Fprintf(&b, "    %-60.60s calls %v n %d serial %v join %d chain %d%s\n", p.Name, p.Calls, p.N, p.Serial, p.Join, p.Chain, c.Layout())
				}
			}
		}
		t.Log(b.String())
		st.Close()
		m.Close()
	}
}

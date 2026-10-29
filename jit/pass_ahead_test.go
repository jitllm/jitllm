package jit

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/x86"
)

// TestAheadPrefetchesTheWeightStream holds the CPU's decode matvec over a
// weight whose plane stride crosses a page to prefetching it (Ahead), with
// and without VNNI, where a worker's run of a plane is shorter than the
// hardware prefetcher's reach (2048 rows over twelve), and to none where it
// covers it (4096, but for Q8_0's f16 scale planes, two bytes a row: 684 B a
// worker) or a trip moves it less than Min (16 rows): each prefetch is PREFETCHT0
// (0F 18 /1), counted in the code.
func TestAheadPrefetchesTheWeightStream(t *testing.T) {
	host := uint64(cpu.CPU().ISA())
	for _, typ := range []quant.Type{quant.Q4_K, quant.Q8_0, quant.Q6_K} {
		jt, _ := jlm.TypeOf(typ)
		for _, isa := range []uint64{host, host &^ uint64(x86.ISAVNNI)} {
			for _, rows := range []int{2048, 4096, 16} {
				count := func(trips int) (int, bool) {
					r := Default()
					p := r.Profiles[DeviceCPU]
					p.Passes = slices.Clone(p.Passes)
					for i, ps := range p.Passes {
						if a, ok := ps.(Ahead); ok {
							a.Trips = trips
							p.Passes[i] = a
						}
					}
					r.Profiles[DeviceCPU] = p
					_, codes, err := r.Form(Request{
						Graph:   graph.MatVec(jlm.RoleOutput, rows, 2048),
						Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
						Step:    Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
						Device:  Device{Kind: DeviceCPU, Slots: 12, Lanes: 8, ISA: isa, ExactDiv: true},
					})
					if err != nil {
						t.Fatalf("%v at %d rows: %v", typ, rows, err)
					}
					n, ran := 0, false
					for _, c := range codes {
						ran = ran || slices.Contains(c.Passes, "ahead")
						n += prefetchesIn(c.CPU)
					}
					return n, ran
				}
				on, ran := count(2)
				off, _ := count(0)
				want := rows == 2048 || rows == 4096 && typ == quant.Q8_0
				if want != ran || want != (on > 0) || off != 0 {
					t.Errorf("%v at %d rows, isa %#x: ahead ran %v with %d prefetches, %d at no trips; want prefetches %v", typ, rows, isa, ran, on, off, want)
				}
			}
		}
	}
}

// prefetchesIn counts the PREFETCHT0 encodings in code: 0F 18 and a ModRM
// byte whose reg field is 1.
func prefetchesIn(code []byte) int {
	n := 0
	for i := 0; i+2 < len(code); i++ {
		if code[i] == 0x0F && code[i+1] == 0x18 && code[i+2]>>3&7 == 1 && code[i+2]>>6 != 3 {
			n++
		}
	}
	return n
}

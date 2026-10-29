package jit

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

// TestStreamsRefusedPerKernel forms a part of several matvecs on the CPU
// where the registers refuse one kernel's streamed form: that kernel alone
// is formed without Streams, every other kernel of the part keeps it.
func TestStreamsRefusedPerKernel(t *testing.T) {
	q4k, _ := jlm.TypeOf(quant.Q4_K)
	b := graph.New()
	x := b.In("x", 2048)
	b.Out(b.MatVec(x, jlm.RoleAttnQ, 2048), b.MatVec(x, jlm.RoleAttnK, 512), b.MatVec(x, jlm.RoleAttnV, 512))
	_, codes, err := Default().Form(Request{
		Graph:   b.G,
		Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: q4k, jlm.RoleAttnK: q4k, jlm.RoleAttnV: q4k},
		Step:    Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
		Device:  Device{Kind: DeviceCPU, Slots: 14, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	streamed, unstreamed := 0, 0
	for _, c := range codes {
		if c.Share == 0 {
			continue
		}
		t.Logf("%s calls %v passes %v", c.Name, c.Calls, c.Passes)
		if slices.Contains(c.Passes, Streams{}.Name()) {
			streamed++
		} else {
			unstreamed++
		}
	}
	if unstreamed == 0 {
		t.Fatal("the registers refused no streamed form: the part does not exercise a refusal")
	}
	if streamed == 0 {
		t.Fatalf("one kernel's refusal formed every matvec of the part without Streams (%d)", unstreamed)
	}
}

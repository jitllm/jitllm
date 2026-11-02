package composer

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// TestStreamsRefusedPerKernel forms a part of a 2048-row and two 512-row
// Q4_K matvecs on a CPU of 14 workers. The 512-row kernels' calls are too
// short to take their planes apart, and the registers refuse their walk of
// every plane in one rows loop: those kernels alone are formed again, their
// planes apart (Crowded), and every matvec of the part streams.
func TestStreamsRefusedPerKernel(t *testing.T) {
	q4k, _ := jlm.TypeOf(quant.Q4_K)
	b := graph.New()
	x := b.In("x", 2048)
	b.Out(b.MatVec(x, jlm.RoleAttnQ, 2048), b.MatVec(x, jlm.RoleAttnK, 512), b.MatVec(x, jlm.RoleAttnV, 512))
	req := Request{
		Graph:   b.G,
		Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: q4k, jlm.RoleAttnK: q4k, jlm.RoleAttnV: q4k},
		Step:    Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
		Device:  Device{Kind: DeviceCPU, Slots: 14, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true},
	}
	reg := Default()
	ks, err := reg.formKernels(req)
	if err != nil {
		t.Fatal(err)
	}
	streamed, crowded := 0, 0
	for _, k := range ks {
		if !k.readsWeight() {
			continue
		}
		t.Logf("%s crowded %v passes %v", k.Name, k.Crowded, k.Passes)
		if k.streamed() {
			streamed++
		}
		if k.Crowded {
			crowded++
		}
		if !k.streamed() {
			t.Errorf("%s reading %v is formed without Streams", k.Name, k.at().Role)
		}
	}
	if streamed != 3 {
		t.Fatalf("%d of the part's 3 matvecs stream", streamed)
	}
	if crowded == 0 {
		t.Fatal("the registers refused no walk of every plane: the part does not exercise a refusal")
	}
	if crowded == 3 {
		t.Fatal("every matvec was formed again for one kernel's refusal")
	}
	if _, _, err := reg.Form(req); err != nil {
		t.Fatal(err)
	}
}

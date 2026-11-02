//go:build amd64

package composer

import (
	"testing"

	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// TestCPUWideWindowAddsNoWork holds a token's and a prompt's int8 matvec
// on the CPU at a k-quant super-block's window, 256, to no more code than
// at 32: the window's scale is read where the sub-block's index scales to.
func TestCPUWideWindowAddsNoWork(t *testing.T) {
	host := uint64(cpu.CPU().ISA())
	bytes := func(typ quant.Type, isa uint64, rows, win int) int {
		jt, _ := jlm.TypeOf(typ)
		_, codes, err := Default().Form(Request{
			Graph:   graph.MatVec(jlm.RoleOutput, 2048, 2048),
			Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
			Step:    Step{Rows: rows, Page: 64, Keys: 1, ActWindow: win},
			Device:  Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: isa, ExactDiv: true},
		})
		if err != nil {
			t.Fatalf("%v at window %d: %v", typ, win, err)
		}
		n := 0
		for _, c := range codes {
			n += len(c.CPU)
		}
		return n
	}
	for _, typ := range []quant.Type{quant.Q4_K, quant.Q5_K, quant.Q6_K, quant.Q3_K} {
		for _, isa := range []uint64{host, host &^ uint64(x86.ISAVNNI)} {
			for _, rows := range []int{1, 512} {
				narrow, wide := bytes(typ, isa, rows, 32), bytes(typ, isa, rows, 256)
				t.Logf("%v vnni %v, %d rows: %d bytes at window 32, %d at 256", typ, isa&uint64(x86.ISAVNNI) != 0, rows, narrow, wide)
				if wide > narrow {
					t.Errorf("%v vnni %v, %d rows: window 256 forms %d bytes of code against window 32's %d", typ, isa&uint64(x86.ISAVNNI) != 0, rows, wide, narrow)
				}
			}
		}
	}
}

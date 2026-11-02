package composer

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// a64Registry is Default lowering the CPU for arm64 whatever the host.
func a64Registry() *Registry {
	r := withCPUInt8()
	r.Lowerings = map[DeviceKind]Lowering{DeviceGPU: gpuLowering{}, DeviceCPU: cpuLowering{Shuffles: true, Arm64: true}}
	return r
}

// a64Words counts the words of code that match want under mask.
func a64Words(code []byte, mask, want uint32) int {
	n := 0
	for i := 0; i+4 <= len(code); i += 4 {
		if w := uint32(code[i]) | uint32(code[i+1])<<8 | uint32(code[i+2])<<16 | uint32(code[i+3])<<24; w&mask == want {
			n++
		}
	}
	return n
}

// TestA64LoweringStatesTheDevicesDot forms the int8 matvec of a Q8_0 weight
// for an arm64 CPU stated with FEAT_DotProd and without: with it the byte
// dots are SDOT and nothing widens them, without it there is no SDOT, which
// faults on a Cortex-A72, and the dots are widened (SMULL, SADALP). Every
// kernel ends in RET.
func TestA64LoweringStatesTheDevicesDot(t *testing.T) {
	const (
		sdotMask, sdot     = 0xBF20FC00, 0x0E009400
		smullMask, smull   = 0xBF20FC00, 0x0E20C000
		sadalpMask, sadalp = 0xBF3FFC00, 0x0E206800
		ret                = 0xD65F03C0
	)
	r := a64Registry()
	jt, ok := jlm.TypeOf(quant.Q8_0)
	if !ok {
		t.Fatal("Q8_0 has no container type")
	}
	for _, dp := range []bool{true, false} {
		isa := uint64(0)
		if dp {
			isa = uint64(cpu.ISADotProd)
		}
		_, codes, err := r.Form(Request{
			Graph:   graph.MatVec(jlm.RoleOutput, 64, 256),
			Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
			Step:    Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
			Device:  Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: isa, ExactDiv: true},
		})
		if err != nil {
			t.Fatalf("dotprod %v: %v", dp, err)
		}
		var sd, wide int
		for _, c := range codes {
			if len(c.CPU) == 0 || len(c.CPU)%4 != 0 || a64Words(c.CPU[len(c.CPU)-4:], ^uint32(0), ret) != 1 {
				t.Fatalf("dotprod %v: %s is %d bytes and does not end in RET", dp, c.Name, len(c.CPU))
			}
			sd += a64Words(c.CPU, sdotMask, sdot)
			wide += a64Words(c.CPU, smullMask, smull) + a64Words(c.CPU, sadalpMask, sadalp)
		}
		t.Logf("dotprod %v: %d SDOT, %d widening words over %d kernels", dp, sd, wide, len(codes))
		switch {
		case dp && (sd == 0 || wide != 0):
			t.Errorf("with dotprod: %d SDOT and %d widening words", sd, wide)
		case !dp && (sd != 0 || wide == 0):
			t.Errorf("without dotprod: %d SDOT and %d widening words", sd, wide)
		}
	}
}

// TestA64LoweringDividesByARegister forms a picture's patch gather for an
// arm64 CPU: the grid's width is the request's word, known only at the
// call, so the gather divides by a register (UDIV), which arm64 once had
// no form of and the entry did not form.
func TestA64LoweringDividesByARegister(t *testing.T) {
	const udivMask, udiv = 0xFFE0FC00, 0x9AC00800
	b := graph.NewBlock(-1)
	px := b.In("pixels", 3*14*14)
	b.Out(b.Emit(graph.OpPatchify, []graph.ValueID{px}, []graph.Result{{Name: "patches", Width: 3 * 14 * 14}}, nil,
		graph.Attrs{Span: 14, Offset: 1}, nil, nil)[0])
	_, codes, err := a64Registry().Form(Request{Graph: b.G, Step: Step{Rows: 65, Page: 64, Keys: 1, ActWindow: 32},
		Device: Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.ISADotProd), ExactDiv: true}, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, c := range codes {
		n += a64Words(c.CPU, udivMask, udiv)
	}
	t.Logf("%d kernels, %d UDIV", len(codes), n)
	if n == 0 {
		t.Fatal("the gather divides by the grid's width with no UDIV")
	}
}

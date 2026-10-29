package jit_test

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// TestPrologueAdmitsWhatALaunchCosts forms Llama-3.1-8B's and
// Llama-3.2-1B's decode blocks on the V100's probe row: the 8B's down
// projection runs its activation apart, every one of its workgroups
// recomputing 14336 products costing more than the launch it saves; its o
// projection, and the 1B's down, take theirs in. A device that measured no
// launch takes every prologue.
func TestPrologueAdmitsWhatALaunchCosts(t *testing.T) {
	p, ok := cost.Known("Tesla V100-SXM2-16GB", "ptx")
	if !ok {
		t.Fatal("the hardware table has no V100 row")
	}
	v100 := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, Regs: 32, Launch: p.Device.LaunchWords()}
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp} {
		formats[r] = jlm.TypeQ4S
	}
	formats[jlm.RoleFFNDown] = jlm.TypeQ6S
	formats[jlm.RoleAttnNorm], formats[jlm.RoleFFNNorm] = jlm.TypeF32, jlm.TypeF32
	inside := func(d jit.Device, embd, heads, kv, dim, ffn int, role jlm.Role) bool {
		t.Helper()
		_, codes, err := jit.Default().Form(jit.Request{Graph: blockOf(embd, heads, kv, dim, ffn), Formats: formats,
			Step: jit.Step{Rows: 1, Page: 64, Keys: 1024, ActWindow: 32, Seqs: 1}, Device: d,
			History: jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: true}}})
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range codes {
			for _, b := range c.Binds {
				if b.Operand.Kind == jit.OperandWeight && b.Operand.Role == role {
					return slices.Contains(c.Passes, "prologue")
				}
			}
		}
		t.Fatalf("no kernel reads %v", role)
		return false
	}
	free := v100
	free.Launch = 0
	for _, c := range []struct {
		name                      string
		embd, heads, kv, dim, ffn int
		role                      jlm.Role
		want                      bool
	}{
		{"8B down", 4096, 32, 8, 128, 14336, jlm.RoleFFNDown, false},
		{"8B o", 4096, 32, 8, 128, 14336, jlm.RoleAttnOut, true},
		{"1B down", 2048, 32, 8, 64, 8192, jlm.RoleFFNDown, true},
	} {
		if got := inside(v100, c.embd, c.heads, c.kv, c.dim, c.ffn, c.role); got != c.want {
			t.Errorf("%s on the V100 (%d words a launch): prologue %v, want %v", c.name, v100.Launch, got, c.want)
		}
		if !inside(free, c.embd, c.heads, c.kv, c.dim, c.ffn, c.role) {
			t.Errorf("%s with no launch measured: no prologue", c.name)
		}
	}
}

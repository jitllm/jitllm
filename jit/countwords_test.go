package jit_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/cpu"
)

// TestCountsBeyondTheWordsAreRefused forms a Llama-shaped block on the CPU
// with one count word fewer than its kernels' loops need: the form is
// refused by name, never a kernel whose two loop counts share a word.
func TestCountsBeyondTheWordsAreRefused(t *testing.T) {
	q4k, _ := jlm.TypeOf(quant.Q4_K)
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp, jlm.RoleFFNDown} {
		formats[r] = q4k
	}
	form := func(counts int) ([]jit.Code, error) {
		reg := jit.Default()
		p := reg.Profiles[jit.DeviceCPU]
		p.Passes = slices.Clone(p.Passes)
		for i, ps := range p.Passes {
			if a, ok := ps.(jit.Args); ok {
				a.Fields.Counts = a.Fields.Counts[:counts]
				p.Passes[i] = a
			}
		}
		reg.Profiles[jit.DeviceCPU] = p
		_, codes, err := reg.Form(jit.Request{Graph: llamaBlock(), Formats: formats,
			Step:   jit.Step{Rows: 1, Page: 64, Keys: 64, ActWindow: 32},
			Device: jit.Device{Kind: jit.DeviceCPU, Slots: 8, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
		return codes, err
	}
	codes, err := form(2)
	if err != nil {
		t.Fatalf("the block does not form with every count word: %v", err)
	}
	if !needsTwoCounts(codes) {
		t.Fatal("no kernel of the block passes two loop counts: the gate exercises nothing")
	}
	codes, err = form(1)
	if err == nil {
		for _, c := range codes {
			seen := map[int]bool{}
			for _, a := range c.Args {
				if !a.Length {
					continue
				}
				if seen[a.Field] {
					t.Fatalf("%s passes two loop counts in field %d", c.Name, a.Field)
				}
				seen[a.Field] = true
			}
		}
		t.Fatal("a block whose kernels need two count words formed with one")
	}
	if !strings.Contains(err.Error(), "count words") {
		t.Fatalf("refused for another reason: %v", err)
	}
}

func needsTwoCounts(codes []jit.Code) bool {
	for _, c := range codes {
		n := 0
		for _, a := range c.Args {
			if a.Length {
				n++
			}
		}
		if n >= 2 {
			return true
		}
	}
	return false
}

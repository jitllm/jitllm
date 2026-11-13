package composer_test

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// TestCountsBeyondTheWordsAreTheKernelsConstants forms a Llama-shaped block
// on the CPU with one count word fewer than its kernels' loops pass: it
// forms, no word holds two loop counts, and the block passes fewer lengths
// than with every word, the ones past the word its kernels' constants.
// TestCPUCallWordsRunOut runs such a form against the oracle.
func TestCountsBeyondTheWordsAreTheKernelsConstants(t *testing.T) {
	q4k, _ := jlm.TypeOf(quant.Q4_K)
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp, jlm.RoleFFNDown} {
		formats[r] = q4k
	}
	form := func(counts int) ([]composer.Code, error) {
		reg := composer.Default()
		p := reg.Profiles[composer.DeviceCPU]
		p.Passes = slices.Clone(p.Passes)
		for i, ps := range p.Passes {
			if a, ok := ps.(composer.Args); ok {
				a.Fields.Counts = a.Fields.Counts[:counts]
				p.Passes[i] = a
			}
		}
		reg.Profiles[composer.DeviceCPU] = p
		_, codes, err := reg.Form(composer.Request{Graph: llamaBlock(), Formats: formats,
			Step:   composer.Step{Rows: 1, Page: 64, Keys: 64, ActWindow: 32},
			Device: composer.Device{Kind: composer.DeviceCPU, Slots: 8, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}})
		return codes, err
	}
	codes, err := form(2)
	if err != nil {
		t.Fatalf("the block does not form with every count word: %v", err)
	}
	if !needsTwoCounts(codes) {
		t.Fatal("no kernel of the block passes two loop counts: the gate exercises nothing")
	}
	for _, c := range codes {
		n := 0
		for _, a := range c.Args {
			if a.Length {
				n++
			}
		}
		if n >= 2 {
			t.Logf("%s passes %d lengths over calls %v", c.Name, n, c.Calls)
		}
	}
	cut, err := form(1)
	if err != nil {
		t.Fatalf("the block does not form with one count word: %v", err)
	}
	for _, c := range cut {
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
	if passed(cut) >= passed(codes) {
		t.Fatalf("with one count word the block passes %d lengths, with two %d: no length became its kernel's constant", passed(cut), passed(codes))
	}
}

// passed is the lengths codes pass in call words.
func passed(codes []composer.Code) int {
	n := 0
	for _, c := range codes {
		for _, a := range c.Args {
			if a.Length {
				n++
			}
		}
	}
	return n
}

func needsTwoCounts(codes []composer.Code) bool {
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

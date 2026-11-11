package engine

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// parityConstraint allows at the round's token i only the ids of i's parity:
// a constraint whose allowed set moves with every token, so a speculation
// held to it must mask each draft and each verified row at its own depth and
// commit what it accepted. off is the violation: it allows every token.
type parityConstraint struct {
	n, pushed int
	buf       []float32
	off       bool
}

func (c *parityConstraint) ok(at int, tok int32) bool { return c.off || int(tok)%2 == at%2 }

func (c *parityConstraint) Mask(depth int, lg []float32) ([]float32, error) {
	c.buf = append(c.buf[:0], lg...)
	for i := range c.buf {
		if !c.ok(c.n+depth, int32(i)) {
			c.buf[i] = float32(math.Inf(-1))
		}
	}
	return c.buf, nil
}

func (c *parityConstraint) Push(tok int32) bool {
	if !c.ok(c.n+c.pushed, tok) {
		return false
	}
	c.pushed++
	return true
}

func (c *parityConstraint) Commit(k int) { c.n, c.pushed = c.n+k, 0 }

// TestSpecHoldsToAConstraint: a greedy speculation held to a constraint
// (WithSpecConstraint) draws what a greedy decode held to it draws, the
// prediction block's drafts and prompt lookup's alike, and keeps drafts
// (the rounds verified some); with the constraint off it leaves the allowed
// set.
func TestSpecHoldsToAConstraint(t *testing.T) {
	n := specTokens()
	for name, path := range specModels(t) {
		t.Run(name, func(t *testing.T) {
			m, openOpts := openSpecModel(t, path)
			defer m.Close()
			var drafted, accepted int64
			for _, text := range specPrompts() {
				prompt := m.Vocab.Encode(text, true)
				want := constrainedGreedy(t, m, prompt, n, openOpts...)
				for _, arm := range []struct {
					name string
					opts []SpecOption
				}{{"k3", []SpecOption{WithSpecDraft(3)}}, {"lookup", []SpecOption{WithSpecDrafter(SpecDraftLookup)}}} {
					con := &parityConstraint{}
					got, st := specGreedy(t, m, openOpts, prompt, n, nil, specFaultNone, oracleFor(name, want),
						append([]SpecOption{WithSpecConstraint(con)}, arm.opts...)...)
					for i, y := range got {
						if int(y)%2 != i%2 {
							t.Fatalf("%s %q: token %d is %d, outside the constraint", arm.name, text, i, y)
						}
					}
					specFlip(t, m, prompt, got, want, fmt.Sprintf("%s held to the constraint, prompt %q", arm.name, text), openOpts...)
					drafted, accepted = drafted+st.Drafted, accepted+st.Accepted
				}
				off := &parityConstraint{off: true}
				got, _ := specGreedy(t, m, openOpts, prompt, n, nil, specFaultNone, nil, WithSpecDraft(3), WithSpecConstraint(off))
				left := false
				for i, y := range got {
					left = left || int(y)%2 != i%2
				}
				if !left {
					t.Errorf("%q: with the constraint off every token kept the parity: the gate cannot see it", text)
				}
			}
			if drafted == 0 || accepted == 0 {
				t.Fatalf("%d drafted and %d accepted: the constrained speculation never ran", drafted, accepted)
			}
			t.Logf("%s: %d drafted, %d accepted, every token held to the constraint", name, drafted, accepted)
		})
	}
}

// constrainedGreedy is plainGreedy held to parityConstraint.
func constrainedGreedy(t *testing.T, m *model.Model, prompt []int32, n int, io ...Option) []int32 {
	t.Helper()
	st := testInstance(t, m, len(prompt)+n+8, io...)
	defer st.Close()
	lg, err := st.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	con := &parityConstraint{}
	out := make([]int32, 0, n)
	for len(out) < n {
		lm, err := con.Mask(0, lg)
		if err != nil {
			t.Fatal(err)
		}
		y := Greedy(lm)
		con.Commit(0)
		con.Push(y)
		con.Commit(1)
		out = append(out, y)
		if lg, err = st.Forward(y); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

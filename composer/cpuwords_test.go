package composer

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// TestCPUCallWordsRunOut forms the lone matvec on the CPU with fewer call
// words than its kernels pass -- one count word where a kernel passes two
// lengths, one coordinate word where its calls' space has two axes -- and
// holds each form to the oracle as the form with every word is: it forms,
// the word it lacks is gone from its calls (the length its kernel's
// constant, the axis a loop), and it computes the product.
func TestCPUCallWordsRunOut(t *testing.T) {
	cpuOn := hostArm()
	host := uint64(cpuOn.tg.Features().ISA())
	rng := rand.New(rand.NewSource(11))
	typ := quant.Q4_K
	q, _ := layout.QuantOf(typ)
	jt, _ := jlm.TypeOf(typ)
	counts, coords := 0, 0
	for _, sh := range []struct{ rows, k, tokens, slots int }{
		{64, 256, 1, 1}, {64, 256, 4, 1}, {64, 256, 1, 6}, {64, 256, 4, 6}, {1024, 2048, 1, 1}, {1024, 2048, 1, 6},
	} {
		name := fmt.Sprintf("%d rows by %d, %d tokens, %d workers", sh.rows, sh.k, sh.tokens, sh.slots)
		w := cpuTestWeights(typ, q, sh.rows, sh.k, rng)
		qs, d, sc, err := layout.PackWeights(q, w, sh.rows, sh.k)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float32, sh.tokens*sh.k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		ref := x
		if Default().Profiles[DeviceCPU].Int8 != nil {
			ref = int8Row(x, 32)
		}
		want := make([]float32, sh.tokens*sh.rows)
		for tk := range sh.tokens {
			if err := oracle.MatVec(want[tk*sh.rows:(tk+1)*sh.rows], typ, w, ref[tk*sh.k:(tk+1)*sh.k], sh.rows, sh.k); err != nil {
				t.Fatal(err)
			}
		}
		req := Request{
			Graph:   graph.MatVec(jlm.RoleOutput, sh.rows, sh.k),
			Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
			Step:    Step{Rows: sh.tokens, Page: 64, Keys: 1, ActWindow: 32},
			Device:  Device{Kind: DeviceCPU, Slots: sh.slots, Lanes: 8, ISA: host, ExactDiv: true},
		}
		run := func(arm string, nc, nx int) []Code {
			_, codes, err := callWords(nc, nx).Form(req)
			if err != nil {
				t.Fatalf("%s, %s: %v", name, arm, err)
			}
			got, err := runCPU(cpuOn, codes, x, sh.tokens*sh.rows, [3][]uint32{qs, d, sc})
			if err != nil {
				t.Fatalf("%s, %s: %v", name, arm, err)
			}
			if e := cpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-9) {
				t.Errorf("%s, %s: NMSE %.3g from the oracle", name, arm, e)
			}
			return codes
		}
		full := run("every word", 2, 2)
		if mostLengths(full) >= 2 {
			counts++
			cut := run("one count word", 1, 2)
			if mostLengths(cut) >= 2 || passedLengths(cut) >= passedLengths(full) {
				t.Errorf("%s: with one count word a kernel passes %d lengths and the form %d (%d with two): no length became its kernel's constant",
					name, mostLengths(cut), passedLengths(cut), passedLengths(full))
			}
		}
		if mostAxes(full) >= 2 {
			coords++
			if cut := run("one coordinate word", 2, 1); mostAxes(cut) >= 2 {
				t.Errorf("%s: with one coordinate word a kernel's calls' space keeps %d axes", name, mostAxes(cut))
			}
		}
	}
	if counts == 0 {
		t.Fatal("no shape's kernel passes two lengths: the count arm exercises nothing")
	}
	if coords == 0 {
		t.Fatal("no shape's calls' space has two axes: the coordinate arm exercises nothing")
	}
	t.Logf("%d shapes ran short of a count word, %d of a coordinate word", counts, coords)
}

// callWords is the CPU's profile as it ships, with nc count words and nx
// coordinate words in its main argument block.
func callWords(nc, nx int) *Registry {
	r := Default()
	p := r.Profiles[DeviceCPU]
	p.Passes = slices.Clone(p.Passes)
	for i, ps := range p.Passes {
		switch ps := ps.(type) {
		case Args:
			ps.Fields.Counts = ps.Fields.Counts[:nc]
			ps.Fields.Coords = ps.Fields.Coords[:nx]
			p.Passes[i] = ps
		case Serial:
			ps.Coords = nx
			p.Passes[i] = ps
		}
	}
	r.Profiles[DeviceCPU] = p
	return r
}

// mostLengths is the most lengths one of codes passes in call words.
func mostLengths(codes []Code) int {
	most := 0
	for _, c := range codes {
		n := 0
		for _, a := range c.Args {
			if a.Length && !a.Share {
				n++
			}
		}
		most = max(most, n)
	}
	return most
}

// passedLengths is the lengths codes pass in call words.
func passedLengths(codes []Code) int {
	n := 0
	for _, c := range codes {
		for _, a := range c.Args {
			if a.Length && !a.Share {
				n++
			}
		}
	}
	return n
}

// mostAxes is the most axes a calls' space of codes has.
func mostAxes(codes []Code) int {
	most := 0
	for _, c := range codes {
		most = max(most, len(c.Calls))
	}
	return most
}

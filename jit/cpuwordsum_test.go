package jit

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
)

// wordSumPrim is a sum of words a general register holds, over each of R
// rows of S read down a column (no vector loop: a word strided across rows
// is no lane's): the count of the words below bound, or with carried the
// largest word and, as a float, the first place it is at -- an integer and
// a float alive together.
func wordSumPrim(R, S int, bound uint32, carried bool) Prim {
	U := ir.U32
	words := Operand{Kind: OperandScratch, Name: "ws.words", Elem: U}
	out := Operand{Kind: OperandScratch, Name: "ws.out", Elem: U}
	var x exprs
	w := x.load(0, x.op(OpAdd, U, x.op(OpMul, U, x.coord(1), x.u(uint32(R))), x.coord(0)))
	p := Prim{Kind: PrimReduce, Shape: []int{R, S}, In: []Operand{words}, Out: out, Combine: OpAdd, Type: U,
		Facts: Facts{Extents: map[int]int{0: R * S}}}
	if !carried {
		x.sel(U, x.op(OpLess, ir.Pred, w, x.u(bound)), x.u(1), x.u(0))
		p.Expr = x.e
		return p
	}
	var c exprs
	c.un(OpCvtF32, ir.F32, c.coord(1))
	p.Expr, p.Combine, p.Carry = x.e, OpMax, c.e
	p.CarryOut = Operand{Kind: OperandScratch, Name: "ws.carried", Elem: ir.F32}
	return p
}

// runWordSum forms p on the CPU and runs it over words: the sums and the
// carried floats, each with a guard of poisoned words after it that the
// code must leave.
func runWordSum(p Prim, R int, words []uint32) ([]uint32, []float32, error) {
	const guard = 16
	dev := cpuDevice()
	reg := Default()
	k, err := generate(p)
	if err != nil {
		return nil, nil, err
	}
	ks, err := reg.Profiles[DeviceCPU].on(dev).run([]*Kernel{k}, dev)
	if err != nil {
		return nil, nil, err
	}
	pcs, err := reg.Lower(ks, dev)
	if err != nil {
		return nil, nil, err
	}
	var codes []Code
	for _, c := range pcs {
		codes = append(codes, c.Code)
	}
	poison := math.Float32bits(float32(math.NaN()))
	sums, carried := make([]uint32, R+guard), make([]uint32, R+guard)
	for i := range sums {
		sums[i], carried[i] = poison, poison
	}
	if _, err := runCPUBound(codes, []float32{0}, 1, [3][]uint32{}, nil, func(o Operand) []uint32 {
		switch o.Name {
		case "ws.words":
			return words
		case "ws.out":
			return sums
		case "ws.carried":
			return carried
		}
		return nil
	}); err != nil {
		return nil, nil, err
	}
	for i := R; i < R+guard; i++ {
		if sums[i] != poison || carried[i] != poison {
			return nil, nil, fmt.Errorf("the code wrote past the rows, at %d", i)
		}
	}
	return sums[:R], wordsF32(carried[:R]), nil
}

// TestComposedCPUWordSums executes on the CPU sums whose words a general
// register holds -- a count, and the largest word carrying its place as a
// float -- at ragged rows and lengths, held to the definition exactly. The
// register binding once gave every sum a vector register as well, over the
// general register it held, so the code counted in a register something
// else owned (the address the words are read from, which faulted); and a
// float chosen by a word's compare read the compare's general register as
// a vector mask. The violation: the bound one word lower, which moves the
// counts, must part.
func TestComposedCPUWordSums(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	ran, parted := 0, 0
	for _, R := range []int{1, 2, 3, 5, 9} {
		for _, S := range []int{1, 2, 7, 8, 9, 33, 100} {
			const bound = 500
			words := make([]uint32, R*S+16)
			for i := range words {
				words[i] = uint32(rng.Intn(4 * bound))
				if i%5 == 0 {
					words[i] = bound - 1
				}
			}
			count, top, place := make([]uint32, R), make([]uint32, R), make([]float32, R)
			for r := range R {
				for s := range S {
					w := words[s*R+r]
					if w < bound {
						count[r]++
					}
					if s == 0 || top[r] < w {
						top[r], place[r] = w, float32(s)
					}
				}
			}
			name := fmt.Sprintf("%d rows of %d", R, S)
			got, _, err := runWordSum(wordSumPrim(R, S, bound, false), R, words)
			if err != nil {
				t.Errorf("%s: count: %v", name, err)
				continue
			}
			for r := range R {
				if got[r] != count[r] {
					t.Errorf("%s: row %d counts %d, want %d", name, r, got[r], count[r])
				}
			}
			lower, _, err := runWordSum(wordSumPrim(R, S, bound-1, false), R, words)
			if err != nil {
				t.Errorf("%s: count below %d: %v", name, bound-1, err)
				continue
			}
			for r := range R {
				if lower[r] != count[r] {
					parted++
					break
				}
			}
			mx, at, err := runWordSum(wordSumPrim(R, S, 0, true), R, words)
			if err != nil {
				t.Errorf("%s: largest: %v", name, err)
				continue
			}
			for r := range R {
				if mx[r] != top[r] || math.Float32bits(at[r]) != math.Float32bits(place[r]) {
					t.Errorf("%s: row %d's largest %d at %v, want %d at %v", name, r, mx[r], at[r], top[r], place[r])
				}
			}
			ran++
		}
	}
	if ran != 35 {
		t.Fatalf("%d of 35 shapes ran", ran)
	}
	if parted == 0 {
		t.Fatal("no count moved with the bound: the gate cannot see a wrong count")
	}
}

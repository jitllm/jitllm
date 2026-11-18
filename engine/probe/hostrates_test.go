//go:build amd64 || arm64

package probe

import (
	"math"
	"runtime"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestComposedHostRatesMatchTheBuilt holds each class's composed rate on
// one thread and its delay, the mix's rate and a call's cost to the
// hand-built kernels they replaced, the two run in turn over rounds: each
// median composed/built within a band of 1, twice the built kernel's own
// spread over its middle rounds (middleSpread) and never under 10%, or
// where the two take
// different work, within the bounds that work's reason states (rateCase).
func TestComposedHostRatesMatchTheBuilt(t *testing.T) { ratesMatch(t, cpu.New()) }

// rateCase is one measurement taken both ways. Where lo is set the two
// take different work, why says how, and composed/built is held to
// [lo, hi]; else to the band around 1.
type rateCase struct {
	name            string
	built, composed func() (float64, error)
	lo, hi          float64
	why             string
}

// ratesMatch holds the probe of h to what TestComposedHostRatesMatchTheBuilt
// states.
func ratesMatch(t *testing.T, h *cpu.Target) {
	if err := hostCheck(h); err != nil {
		t.Fatal(err)
	}
	e := emitter{sse: h.Tier() == cpu.TierSSE}
	fm := newFormer(h)
	mapped := func(code []byte, err error) *cpu.Code {
		if err != nil {
			t.Fatal(err)
		}
		k, err := h.Map(code)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { k.Close() })
		return k
	}
	names := [cost.NumClasses]string{"float", "special", "int", "index", "lanes", "group", "mem"}
	lanes := e.formLanes()
	var cases []rateCase
	for _, cl := range []cost.Class{cost.ClassFloat, cost.ClassSpecial, cost.ClassInt, cost.ClassIndex} {
		rk, ck := mapped(e.rateKernel(cl)), mapped(e.chainKernel(cl))
		rate := rateCase{name: names[cl] + " rate",
			built:    func() (float64, error) { return issueRate(rk, 8, 1), nil },
			composed: func() (float64, error) { return stepRate(fm, cl, lanes, 1) }}
		delay := rateCase{name: names[cl] + " delay",
			built: func() (float64, error) { return 1 / issueRate(ck, 8, 1), nil },
			composed: func() (float64, error) {
				r, err := chainRate(fm, cl, lanes)
				return 1 / r, err
			}}
		switch {
		case cl == cost.ClassIndex:
			rate.lo, rate.hi = 0.9, 1.3
			if runtime.GOARCH == "arm64" {
				rate.lo, rate.hi = 1.2, 1.8
			}
			rate.why = "(the built adds twice a trip to each of four registers, so each trip waits on two adds in a row; the composed adds once to each of eight; measured 1.15 on a Xeon, 1.62 on an M4)"
		case runtime.GOARCH == "arm64" && (cl == cost.ClassSpecial || cl == cost.ClassInt):
			rate.lo, rate.hi = 0.4, 0.6
			rate.why = "(the composer's arm64 vector is two NEON registers, which the estimator counts as one instruction and the built as two; measured 0.50 on an M4)"
		}
		cases = append(cases, rate, delay)
	}
	mk := mapped(e.mixKernel())
	cases = append(cases, rateCase{name: "mix rate", lo: 0.5, hi: 0.85,
		why:      "(the composed mix's values are all words: index adds and reads, each read loading its table's address too, which the estimator does not count, where the built issues float work beside them; measured 0.64 on a Xeon at AVX2, 0.73 at SSE, 0.59 on an M4)",
		built:    func() (float64, error) { return issueRate(mk, 24, 1), nil },
		composed: func() (float64, error) { return mixAt(fm, 1) }})
	ek := mapped(e.emptyKernel())
	cases = append(cases, rateCase{name: "call", lo: 1, hi: 3.5,
		why:      "(the smallest form reads its argument words, its element and its constant and stores its element, where the built returns at once; measured 3.0 on a Xeon, a hand-built call of two loads and a store 1.7)",
		built:    func() (float64, error) { return builtCallCost(ek), nil },
		composed: func() (float64, error) { return callCost(fm) }})
	const rounds = 5
	built := make([][]float64, len(cases))
	ratios := make([][]float64, len(cases))
	for range rounds {
		for i, c := range cases {
			x, err := c.built()
			if err != nil {
				t.Fatal(err)
			}
			y, err := c.composed()
			if err != nil {
				t.Fatalf("%s: %v", c.name, err)
			}
			if !(x > 0) || !(y > 0) || math.IsInf(x, 0) || math.IsInf(y, 0) {
				t.Fatalf("%s: built %v, composed %v: not a measurement", c.name, x, y)
			}
			built[i], ratios[i] = append(built[i], x), append(ratios[i], y/x)
		}
	}
	for i, c := range cases {
		b, r := built[i], ratios[i]
		slices.Sort(b)
		slices.Sort(r)
		band := max(0.10, 2*(middleSpread(b)-1))
		m := r[len(r)/2]
		lo, hi := 1/(1+band), 1+band
		if c.lo > 0 {
			lo, hi = c.lo, c.hi
		}
		t.Logf("%-14s built %10.4g (spread %.3f), composed/built %.3f [%.3f, %.3f], held to [%.3f, %.3f] %s", c.name, b[len(b)/2], middleSpread(b), m, r[0], r[len(r)-1], lo, hi, c.why)
		if m < lo || m > hi {
			t.Errorf("%s: composed/built %.3f outside [%.3f, %.3f]", c.name, m, lo, hi)
		}
	}
}

// builtCallCost is the seconds a call of the hand-built empty kernel k
// costs.
func builtCallCost(k *cpu.Code) float64 {
	const calls = 1 << 20
	var block [8]uint64
	return parallel(1, func(int) {
		for range calls {
			k.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
		}
	}) / calls
}

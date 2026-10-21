//go:build amd64 || arm64

package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// argmaxGo is model.Greedy's scan, copied here so the gate compares against
// the engine's own rule rather than a re-derivation.
func argmaxGo(x []float32) int32 {
	best, bi := float32(math.Inf(-1)), int32(0)
	for i, v := range x {
		if v > best {
			best, bi = v, int32(i)
		}
	}
	return bi
}

// vocabSizes are the real ones plus the shapes around them: the
// kernel is only worth having if it serves the vocabularies that ship.
// The ragged ones matter as much as the round ones: the kernel takes its own
// tail.
var vocabSizes = []int{1, 3, 7, 8, 9, 15, 17, 63, 255, 256, 257, 32000, 32767,
	32768, 49152, 128255, 128256, 151936, 201088, 256000}

func TestArgmaxMatchesTheScan(t *testing.T) {
	ran := 0
	for _, n := range vocabSizes {
		r := rand.New(rand.NewSource(int64(n)))
		for _, shape := range []string{"random", "ascending", "descending", "flat", "negative"} {
			x := make([]float32, n)
			for i := range x {
				switch shape {
				case "ascending":
					x[i] = float32(i) * 1e-3
				case "descending":
					x[i] = float32(n-i) * 1e-3
				case "flat":
					x[i] = 1.25 // every element ties; the first must win
				case "negative":
					x[i] = -float32(r.Float64()) * 40
				default:
					x[i] = float32(r.NormFloat64()) * 8
				}
			}
			got, ok := Argmax32JIT(x)
			if !ok {
				t.Logf("n=%d: declined on tier %v", n, cpu.HostTier())
				continue
			}
			if want := argmaxGo(x); got != want {
				t.Fatalf("n=%d %s: kernel %d, scan %d (x[%d]=%v, x[%d]=%v)",
					n, shape, got, want, got, x[got], want, x[want])
			}
			ran++
		}
	}
	if ran == 0 && cpu.HostTier() == cpu.TierAVX2 {
		t.Fatal("the AVX2 tier declined every shape -- this gate proved nothing")
	}
	t.Logf("%d (vocabulary, shape) pairs agree with the scan on tier %v", ran, cpu.HostTier())
}

// TestArgmaxKeepsTheFirstOfATie is the property greedy decoding rests on,
// tested where a lane-parallel fold is most likely to get it wrong: the two
// equal maxima land in different vector lanes, in different 8-element groups,
// and at the very ends of the vector.
func TestArgmaxKeepsTheFirstOfATie(t *testing.T) {
	const n = 32000
	for _, pair := range [][2]int{{0, 1}, {0, n - 1}, {5, 13}, {7, 8}, {n - 9, n - 1}, {1000, 24000}} {
		x := make([]float32, n)
		for i := range x {
			x[i] = -1
		}
		x[pair[0]], x[pair[1]] = 9, 9
		got, ok := Argmax32JIT(x)
		if !ok {
			t.Skipf("declined on tier %v", cpu.HostTier())
		}
		if want := int32(pair[0]); got != want {
			t.Fatalf("maxima at %d and %d: kernel chose %d, and the lower index must win -- "+
				"a greedy chain that breaks ties differently from llama.cpp diverges "+
				"exactly where two logits are equal", pair[0], pair[1], got)
		}
	}
}

// TestArgmaxGateSeesAWrongIndex runs the comparison against a violation: a
// kernel that returned the last of a tie, or an index off by a lane, would
// pass a gate written only over random data, because random floats do not tie
// and the maximum rarely sits at a lane boundary.
func TestArgmaxGateSeesAWrongIndex(t *testing.T) {
	if _, ok := Argmax32JIT(make([]float32, 64)); !ok {
		t.Skipf("declined on tier %v", cpu.HostTier())
	}
	// The violation is in the oracle, which is the honest way round: if the
	// scan is mutated to keep the last of a tie, the two must disagree.
	lastOfTie := func(x []float32) int32 {
		best, bi := float32(math.Inf(-1)), int32(0)
		for i, v := range x {
			if v >= best {
				best, bi = v, int32(i)
			}
		}
		return bi
	}
	x := make([]float32, 256)
	for i := range x {
		x[i] = -1
	}
	x[3], x[200] = 5, 5
	got, _ := Argmax32JIT(x)
	if got == lastOfTie(x) {
		t.Fatal("the kernel agrees with a last-of-tie scan, so this gate cannot tell " +
			"the two rules apart and certifies nothing about the tie order")
	}
	if got != 3 {
		t.Fatalf("kernel chose %d, want 3", got)
	}
}

// TestArgmaxServesEveryLength: the kernel takes its own tail, so a decline at
// any positive length would be a fallback coming back.
func TestArgmaxServesEveryLength(t *testing.T) {
	if cpu.HostTier() == cpu.TierNone {
		t.Skip("no code generator on this host")
	}
	for _, n := range []int{1, 2, 7, 8, 9, 31999, 128255, 128256} {
		if _, ok := Argmax32JIT(make([]float32, n)); !ok {
			t.Fatalf("n=%d was declined on tier %v -- a length the kernel does not "+
				"serve is a length model.Greedy scans in Go", n, cpu.HostTier())
		}
	}
	// An empty vector has no argmax at all, which is a contract violation
	// rather than a shape: there is no index to return.
	if _, ok := Argmax32JIT(nil); ok {
		t.Fatal("an empty vector has no argmax and the kernel accepted it")
	}
}

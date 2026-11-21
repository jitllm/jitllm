//go:build amd64 || arm64

package probe

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestComposedHostChaseMatchesTheBuilt holds the composed chase to the
// hand-built one it replaced, the two run in turn over the same sets: each
// cache's half and memory's, the median round's composed/built within a
// band of 1, twice the built chase's own spread over its middle rounds
// (middleSpread) and never under 10%. The first cache's may read up to
// l1Slower of the built: the composed walk reads each word at the table's
// address plus its index times four, and Intel documents a load's use
// waiting 5 cycles in the first cache at such an address against 4 at a
// register's address alone, the built walk's (the optimization reference
// manual's load latency for a complex address); measured on a Xeon at 1.247.
// An M4 reads its first cache at 1.332 the same way, a load scaling a
// register's index (ldr w, [x, x, lsl #2]) against the built's at a
// register alone: 4 cycles against 3, measured, Apple documenting neither.
func TestComposedHostChaseMatchesTheBuilt(t *testing.T) { chaseMatches(t, cpu.New()) }

// middleSpread is the slowest of sorted rounds b over the fastest, the
// first and last left out: a round the clock ramped through widens no band.
func middleSpread(b []float64) float64 { return b[len(b)-2] / b[1] }

// l1Slower is the most the composed chase may wait in the first cache over
// the built: the cycle a scaled index adds, 5 against 4 on Intel and 4
// against 3 on an M4, and a band.
const l1Slower = 1.4

// chaseMatches holds the chase of h to what
// TestComposedHostChaseMatchesTheBuilt states.
func chaseMatches(t *testing.T, h *cpu.Target) {
	quiet(t)
	if err := hostCheck(h); err != nil {
		t.Fatal(err)
	}
	sizes, err := cacheSizes()
	if err != nil {
		t.Fatal(err)
	}
	const dram = 512 << 20
	fm, err := newFormer(h)
	if err != nil {
		t.Fatal(err)
	}
	w, err := newWalk(fm, dram)
	if err != nil {
		t.Fatal(err)
	}
	defer w.close()
	code, err := emitter{sse: h.Tier() == cpu.TierSSE}.chaseKernel()
	if err != nil {
		t.Fatal(err)
	}
	built, err := h.Map(code)
	if err != nil {
		t.Fatal(err)
	}
	defer built.Close()
	var sets []int64
	for _, c := range sizes {
		sets = append(sets, c.bytes/2)
	}
	sets = append(sets, dram)
	const rounds = 5
	for l, set := range sets {
		var b, ratios []float64
		for range rounds {
			x, err := builtChaseLatency(w.clk, built, set)
			if err != nil {
				t.Fatal(err)
			}
			y, err := chaseLatency(w, set)
			if err != nil {
				t.Fatal(err)
			}
			if !(x > 0) || !(y > 0) || math.IsInf(x, 0) || math.IsInf(y, 0) {
				t.Fatalf("%d bytes: built %v, composed %v seconds a read: not a measurement", set, x, y)
			}
			b, ratios = append(b, x), append(ratios, y/x)
		}
		slices.Sort(b)
		slices.Sort(ratios)
		band := max(0.10, 2*(middleSpread(b)-1))
		r := ratios[len(ratios)/2]
		t.Logf("%10d bytes: built %6.2f ns (spread %.3f), composed/built %.3f [%.3f, %.3f], band %.3f", set, b[len(b)/2]*1e9, middleSpread(b), r, ratios[0], ratios[len(ratios)-1], band)
		if l == 0 {
			if r < 1/(1+band) || r > l1Slower {
				t.Errorf("%d bytes: composed/built %.3f outside [%.3f, %.3f]", set, r, 1/(1+band), l1Slower)
			}
			continue
		}
		if math.Abs(math.Log(r)) > math.Log1p(band) {
			t.Errorf("%d bytes: composed/built %.3f past the band %.3f", set, r, band)
		}
	}
}

// builtChaseLatency is the chase as the probe took it before it was
// composed: the hand-built kernel k following pointers through a random
// cycle of set bytes' lines, one call of at least a million steps.
func builtChaseLatency(c clock, k *cpu.Code, set int64) (float64, error) {
	const line = 64
	lines := int(max(2, set/line))
	raw, free, err := hugeBuffer(lines * line)
	if err != nil {
		return 0, err
	}
	defer free()
	buf := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[0])), len(raw)/8)
	order := rand.Perm(lines)
	for i, at := range order {
		next := order[(i+1)%lines]
		buf[at*line/8] = uint64(uintptr(unsafe.Pointer(&buf[next*line/8])))
	}
	steps := max(1<<20, 4*lines)
	block := [2]uint64{uint64(uintptr(unsafe.Pointer(&buf[order[0]*line/8]))), uint64(steps)}
	secs := c.parallel(1, func(int) {
		k.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
	})
	return secs / float64(steps), nil
}

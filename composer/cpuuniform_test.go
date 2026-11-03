package composer

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// philoxRef is Philox2x32 over counter (c0, c1) keyed by k, rounds of them,
// as Random123 defines it: each round's 64-bit product of the multiplier and
// the first word, its high half xored with the key and the second word, its
// low half the new second word; the key bumped before every round past the
// first.
func philoxRef(c0, c1, k uint32, rounds int) (uint32, uint32) {
	for r := range rounds {
		if r > 0 {
			k += 0x9E3779B9
		}
		p := uint64(0xD256D193) * uint64(c0)
		c0, c1 = uint32(p>>32)^k^c1, uint32(p)
	}
	return c0, c1
}

// UniformRef is OpUniform's definition: each key's Philox2x32-10 under the
// counter (key, pos) keyed by seed, its first word's top 24 bits over 2^24.
func UniformRef(keys []uint32, seed, pos uint32, rounds int) []float32 {
	out := make([]float32, len(keys))
	for i, key := range keys {
		w, _ := philoxRef(key, pos, seed, rounds)
		out[i] = float32(w>>8) / (1 << 24)
	}
	return out
}

// UniformGraph is OpUniform over w keys: inputs the keys and the counter
// (seed, position), output the uniforms.
func UniformGraph(w int) (g *graph.Graph, keys, ctr, u graph.ValueID) {
	b := graph.New()
	keys, ctr = b.In("keys", w), b.In("counter", 2)
	b.G.Values[keys].Elem, b.G.Values[ctr].Elem = graph.ElemU32, graph.ElemU32
	u = b.Emit(graph.OpUniform, []graph.ValueID{keys, ctr}, []graph.Result{{Name: "u", Width: w}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(u)
	return b.G, keys, ctr, u
}

// TestPhiloxRefMatchesRandom123 holds the reference to Random123's
// known-answer vectors for philox2x32-10.
func TestPhiloxRefMatchesRandom123(t *testing.T) {
	for _, v := range [][5]uint32{
		{0, 0, 0, 0xff1dae59, 0x6cd10df2},
		{0xffffffff, 0xffffffff, 0xffffffff, 0x2c3f628b, 0xab4fd7ad},
		{0x243f6a88, 0x85a308d3, 0x13198a2e, 0xdd7ce038, 0xf62a4c12},
	} {
		if a, b := philoxRef(v[0], v[1], v[2], 10); a != v[3] || b != v[4] {
			t.Errorf("philox2x32-10 of counter %08x %08x key %08x is %08x %08x, want %08x %08x", v[0], v[1], v[2], a, b, v[3], v[4])
		}
	}
}

// TestComposedCPUUniform holds OpUniform on the CPU device to its
// definition, bit for bit, at ragged widths and several rows each under its
// own counter. The violation: nine rounds, which the comparison must tell
// apart.
func TestComposedCPUUniform(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	ran, nine := 0, 0
	for _, w := range []int{1, 3, 8, 40, 64, 129} {
		g, keys, ctr, u := UniformGraph(w)
		if err := graph.Verify(g, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		for _, R := range []int{1, 3} {
			_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: R, Page: 64, Keys: 1}, Device: cpuDevice()})
			if err != nil {
				t.Fatalf("%d keys at %d rows: %v", w, R, err)
			}
			ks, cs := make([]uint32, R*w), make([]uint32, 2*R)
			for i := range ks {
				ks[i] = rng.Uint32()
			}
			for i := range cs {
				cs[i] = rng.Uint32()
			}
			out := make([]uint32, R*w)
			vals := map[graph.ValueID][]uint32{keys: ks, ctr: cs, u: out}
			if err := runCPUGridValues(codes, g, vals, [3][]uint32{}, nil, func(graph.ValueID) int { return R }, nil); err != nil {
				t.Fatalf("%d keys at %d rows: %v", w, R, err)
			}
			for r := range R {
				want := UniformRef(ks[r*w:(r+1)*w], cs[2*r], cs[2*r+1], 10)
				got := make([]float32, w)
				for i := range got {
					got[i] = math.Float32frombits(vals[u][r*w+i])
				}
				if !slices.Equal(got, want) {
					t.Fatalf("%d keys, row %d of %d: %v, want %v", w, r, R, got[:min(4, w)], want[:min(4, w)])
				}
				for _, v := range got {
					if !(v >= 0 && v < 1) {
						t.Fatalf("%d keys, row %d: %v is not in [0, 1)", w, r, v)
					}
				}
				if !slices.Equal(got, UniformRef(ks[r*w:(r+1)*w], cs[2*r], cs[2*r+1], 9)) {
					nine++
				}
			}
			ran++
		}
	}
	if ran == 0 || nine == 0 {
		t.Fatalf("%d shapes, %d rows parted from nine rounds: the gate cannot see a round", ran, nine)
	}
	t.Logf("%d shapes held bit for bit; %d rows parted from nine rounds", ran, nine)
}

package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestMatVecSegmentsIsTheSeparateLaunches: one MatVecSegments launch writes,
// bit for bit, what each segment's own MatVec writes -- mixed formats (a q/k/v
// shape with a Q6_K v, and a Q8_0 one), split 1 and in-group splits, with and
// without biases, for one token and for a step of a few sequences carried in
// one thread (tier.groupQKV), whose live rows lead a scratch of 8. Every
// output is poisoned with NaN and carries a guard word.
func TestMatVecSegmentsIsTheSeparateLaunches(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	type seg struct {
		q    kernels.Quant
		rows int
	}
	for _, d := range devs {
		defer d.Close()
		for _, c := range []struct {
			segs []seg
			k    int
		}{
			{[]seg{{kernels.Q4_K, 4096}, {kernels.Q4_K, 1024}, {kernels.Q6_K, 1024}}, 4096},
			{[]seg{{kernels.Q8_0, 256}, {kernels.Q4_K, 64}, {kernels.Q8_0, 64}}, 512},
			// Qwen2-1.5B's: two KV heads, a k that is six Q6_K super-blocks.
			{[]seg{{kernels.Q4_K, 1536}, {kernels.Q4_K, 256}, {kernels.Q6_K, 256}}, 1536},
		} {
			for _, sp := range []struct {
				split int
				grp   bool
			}{{1, false}, {4, true}, {16, true}, {32, true}} {
				for _, bias := range []bool{false, true} {
					for _, ntok := range []int{1, 2, 4} {
						var shapes []kernels.MatVecShape
						name := d.API()
						for _, s := range c.segs {
							sh := kernels.MatVecShape{T: s.q, K: c.k, Rows: s.rows, Bias: bias}
							if ntok > 1 {
								sh.NTok, sh.Tok, sh.ActRows = ntok, ntok, 8
							}
							shapes = append(shapes, sh)
							name += fmt.Sprintf("/%v-%d", s.q, s.rows)
						}
						name += fmt.Sprintf("/s%d-g%v-bias%v-t%d", sp.split, sp.grp, bias, ntok)
						segCase(t, d, name, shapes, sp.split, sp.grp)
					}
				}
			}
		}
	}
}

func rawWeights(q kernels.Quant, rows, k int, rng *rand.Rand) []byte {
	nb := k / q.Elems()
	bb := q.BlockBytes()
	raw := make([]byte, rows*nb*bb)
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	f16 := func() uint16 { return uint16(0x2000 | rng.Intn(0x0C00)) }
	for blk := 0; blk < rows*nb; blk++ {
		b := raw[blk*bb:]
		switch q {
		case kernels.Q8_0, kernels.Q4_0, kernels.Q5_0:
			binary.LittleEndian.PutUint16(b, f16())
		case kernels.Q6_K, kernels.Q3_K, kernels.TQ2:
			binary.LittleEndian.PutUint16(b[bb-2:], f16())
		case kernels.TQ2F:
			binary.LittleEndian.PutUint32(b[bb-4:], math.Float32bits(0.5+rng.Float32()))
		case kernels.Q4_K, kernels.Q5_K, kernels.Q5_1:
			binary.LittleEndian.PutUint16(b, f16())
			binary.LittleEndian.PutUint16(b[2:], f16())
		default:
			// Random f16 scales are Inf or NaN one time in 32: a format with
			// no arm here would compare NaN against NaN and say nothing.
			if !kernels.IsFloat(q) && q != kernels.MXFP4 {
				panic("rawWeights: no scale planter for " + q.String())
			}
		}
	}
	return raw
}

func segCase(t *testing.T, d backend.Device, name string, shapes []kernels.MatVecShape, split int, grp bool) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(len(name))))
	k := shapes[0].K
	nt := max(shapes[0].NTok, 1)
	g := newGPU(t, d)
	defer g.free()
	x := make([]float32, max(nt, shapes[0].ActRows)*k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	av, as, asum, err := kernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}
	bA, bAX := g.up(u32bytes(av)), g.up(f32bytes(append(append([]float32{}, as...), asum...)))
	const guard = 0x5EED5EED
	poisoned := func(n int) backend.Buf {
		p := make([]float32, n+1)
		for i := range p {
			p[i] = float32(math.NaN())
		}
		b := f32bytes(p)
		binary.LittleEndian.PutUint32(b[n*4:], guard)
		return g.up(b)
	}
	type w struct{ qs, dw, sc, bias, sep, got backend.Buf }
	ws := make([]w, len(shapes))
	width := 128
	if grp {
		width = kernels.GroupSplitWidth(split)
	}
	for i, s := range shapes {
		qs, dw, scw, err := kernels.PackWeights(s.T, rawWeights(s.T, s.Rows, k, rng), s.Rows, k)
		if err != nil {
			t.Fatal(err)
		}
		ws[i] = w{qs: g.up(u32bytes(qs)), dw: g.up(u32bytes(dw)), sc: g.up(u32bytes(scw)),
			sep: poisoned(s.Rows * nt), got: poisoned(s.Rows * nt)}
		if s.Bias {
			bv := make([]float32, s.Rows)
			for j := range bv {
				bv[j] = float32(rng.NormFloat64())
			}
			ws[i].bias = g.up(f32bytes(bv))
		}
		one := s
		one.Split, one.GroupSplit = split, grp
		kk, err := kernels.MatVec(one)
		if err != nil {
			// Not a skip: that would end every case after this one.
			t.Logf("%s: not a shape (segment %d: %v)", name, i, err)
			return
		}
		kern, err := d.Compile(kk)
		if err != nil {
			t.Fatalf("%s: compile segment %d: %v", name, i, err)
		}
		bufs := []backend.Buf{ws[i].qs, ws[i].dw, ws[i].sc, bA, bAX, ws[i].sep}
		if s.Bias {
			bufs = append(bufs, ws[i].bias)
		}
		if err := kern.Launch((s.Rows*split+width-1)/width, width, bufs...); err != nil {
			t.Fatalf("%s: segment %d: %v", name, i, err)
		}
		kern.Close()
	}
	kk, err := kernels.MatVecSegments(shapes, split, grp)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := kk.Validate(); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatalf("%s: compile: %v", name, err)
	}
	defer kern.Close()
	blocks, group, err := kernels.SegmentsGrid(shapes, split, grp)
	if err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, n := range blocks {
		total += n
	}
	bufs := []backend.Buf{bA, bAX}
	for i, s := range shapes {
		bufs = append(bufs, ws[i].qs, ws[i].dw, ws[i].sc, ws[i].got)
		if s.Bias {
			bufs = append(bufs, ws[i].bias)
		}
	}
	if err := kern.Launch(total, group, bufs...); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	for i, s := range shapes {
		n := s.Rows * nt
		want, got := make([]byte, (n+1)*4), make([]byte, (n+1)*4)
		if ws[i].sep.Read(want) != nil || ws[i].got.Read(got) != nil {
			t.Fatalf("%s: read back", name)
		}
		bad := 0
		for j := 0; j <= n; j++ {
			a, b := binary.LittleEndian.Uint32(want[4*j:]), binary.LittleEndian.Uint32(got[4*j:])
			if j < n && math.IsNaN(float64(math.Float32frombits(a))) {
				t.Fatalf("%s: segment %d row %d of the separate launch is NaN -- a degenerate oracle", name, i, j)
			}
			if a != b {
				if bad++; bad <= 2 {
					t.Errorf("%s: segment %d word %d = %#x, want %#x", name, i, j, b, a)
				}
			}
		}
		if bad > 0 {
			t.Fatalf("%s: segment %d: %d of %d words differ", name, i, bad, n+1)
		}
	}
}

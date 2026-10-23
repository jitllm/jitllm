package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

func TestFlashAttention(t *testing.T) {
	gpuLock(t)
	ds := backend.Open()
	if len(ds) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range ds {
		defer d.Close()
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			skipNoFlashLanes(t, d)
			// 96 and 256 put three and eight accumulators in a lane, so a
			// V load group of 21 keys leaves a ragged last group, and 8 is
			// the 256-wide head's.
			for _, dim := range []int{7, 32, 64, 96, 128, 256} {
				for _, n := range []int{1, 31, 32, 33, 97} {
					t.Run(fmt.Sprintf("d%d/n%d", dim, n), func(t *testing.T) {
						s := kernels.FlashShape{Heads: 4, KVHeads: 2, Dim: dim, Rows: 3, KStride: 128, Scale: float32(1 / math.Sqrt(float64(dim)))}
						flashCheck(t, d, s, n)
						s.Splits = 4
						flashCheck(t, d, s, n)
					})
				}
			}
			for _, split := range []int{0, 4, 16} {
				t.Run(fmt.Sprintf("long/split%d", split), func(t *testing.T) {
					flashCheck(t, d, kernels.FlashShape{Heads: 4, KVHeads: 2, Dim: 64, Rows: 3, KStride: 4096, Scale: 0.125, Splits: split}, 4093)
				})
			}
			for _, s := range []kernels.FlashShape{
				{Heads: 4, KVHeads: 1, Dim: 64, Rows: 1, KStride: 128, Scale: .125, F16: true},
				{Heads: 4, KVHeads: 2, Dim: 64, Rows: 3, Scale: .125, Window: 17, Sink: true, Softcap: 2},
				{Heads: 2, KVHeads: 2, Dim: 32, Rows: 3, KStride: 128, Scale: .177, F16: true, Window: 1, Sink: true},
				{Heads: 4, KVHeads: 2, Dim: 256, Rows: 3, KStride: 128, Scale: .0625, F16: true, Window: 40, Sink: true},
				// Llama 4's aligned chunk: at n=73 the three rows sit at 71, 72
				// and 73 keys, so a chunk of 24 gives row 0 keys 48..70 (23 of
				// them) and row 2 key 72 ALONE, where a sliding window of 24
				// gives 24 each. Treating the chunk as a window fails 5 subtests.
				{Heads: 4, KVHeads: 2, Dim: 64, Rows: 3, KStride: 128, Scale: .125, Window: 24, Chunked: true},
			} {
				t.Run(fmt.Sprintf("extra/%+v", s), func(t *testing.T) { flashCheck(t, d, s, 73); s.Splits = 8; flashCheck(t, d, s, 73) })
			}
		})
	}
}

// TestFlashDecodeKV holds the KV-head decode kernel to FlashAttention's oracle
// on every shape that oracle covers with one query row: GQA and MQA, row-major
// and transposed K, f16 V, windows, aligned chunks, sinks and softcap, a
// ragged Dim, one and several key partitions, and a context longer than one
// partition holds (4093 keys at split 4).
func TestFlashDecodeKV(t *testing.T) {
	gpuLock(t)
	ds := backend.Open()
	if len(ds) == 0 {
		t.Skip("no GPU")
	}
	ran := 0
	for _, d := range ds {
		defer d.Close()
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			skipNoFlashLanes(t, d)
			for _, dim := range []int{7, 48, 64, 96, 128, 256} {
				for _, n := range []int{1, 31, 128, 129, 300} {
					for _, sh := range []struct{ heads, kv, stride int }{{4, 2, 512}, {8, 1, 0}, {6, 6, 512}} {
						s := kernels.FlashShape{Heads: sh.heads, KVHeads: sh.kv, Dim: dim, Rows: 1, KStride: sh.stride,
							Scale: float32(1 / math.Sqrt(float64(dim)))}
						for _, split := range []int{1, 3} {
							for _, grp := range []int{0, 1, 2} {
								if grp > 0 && (sh.heads/sh.kv)%grp != 0 {
									continue
								}
								// Warps moves the score pass's split between
								// dimension slices and key groups.
								for _, warps := range []int{0, 2, 4} {
									s.Splits, s.Group, s.Warps = split, grp, warps
									// Vector V loads where the backend lowers
									// them, beside the scalar form.
									s.VecV = d.API() != "spirv" && warps != 2
									t.Run(fmt.Sprintf("d%d/n%d/h%d-%d/split%d/g%d/w%d/vec%v", dim, n, sh.heads, sh.kv, split, grp, warps, s.VecV), func(t *testing.T) {
										flashCheckWith(t, d, s, n, true)
									})
									ran++
								}
							}
						}
					}
				}
			}
			t.Run("long/split4", func(t *testing.T) {
				flashCheckWith(t, d, kernels.FlashShape{Heads: 4, KVHeads: 2, Dim: 64, Rows: 1, KStride: 4096, Scale: 0.125, Splits: 4}, 4093, true)
			})
			for _, s := range []kernels.FlashShape{
				{Heads: 4, KVHeads: 1, Dim: 64, Rows: 1, KStride: 128, Scale: .125, F16: true},
				{Heads: 4, KVHeads: 2, Dim: 64, Rows: 1, Scale: .125, Window: 17, Sink: true, Softcap: 2},
				{Heads: 2, KVHeads: 2, Dim: 32, Rows: 1, KStride: 128, Scale: .177, F16: true, Window: 1, Sink: true},
				{Heads: 4, KVHeads: 2, Dim: 64, Rows: 1, KStride: 128, Scale: .125, Window: 24, Chunked: true},
			} {
				t.Run(fmt.Sprintf("extra/%+v", s), func(t *testing.T) {
					flashCheckWith(t, d, s, 73, true)
					s.Splits = 3
					flashCheckWith(t, d, s, 73, true)
				})
			}
		})
	}
	if ran == 0 {
		for _, d := range ds {
			if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); ok {
				t.Fatal("no shape ran")
			}
		}
		t.Skip("no device here guarantees the 32-lane subgroup the flash kernels need")
	}
}

func flashCheck(t *testing.T, d backend.Device, s kernels.FlashShape, n int) {
	t.Helper()
	flashCheckWith(t, d, s, n, false)
}

// flashCheckWith is flashCheck for either decode kernel: kv selects
// kernels.FlashDecodeKV (one workgroup per KV head and partition) in place of
// kernels.FlashAttention, against the same oracle and the same merge.
func flashCheckWith(t *testing.T, d backend.Device, s kernels.FlashShape, n int, kv bool) {
	t.Helper()
	if nmse, err := flashDecodeNMSE(t, d, s, n, kv); err != nil || nmse > 1e-10 {
		t.Fatalf("NMSE %.3g, %v", nmse, err)
	}
}

// flashDecodeNMSE is flashCheckWith's measurement: the NMSE against the oracle, or
// an error for an overwritten guard or a non-finite output.
func flashDecodeNMSE(t *testing.T, d backend.Device, s kernels.FlashShape, n int, kv bool) (float64, error) {
	t.Helper()
	build, groups, width := kernels.FlashAttention, s.Rows*s.Heads*max(1, s.Splits), 32
	if kv {
		// The kernel's contract: Splits*FlashKVChunk(s) covers every count.
		if c := kernels.FlashKVChunk(s); max(1, s.Splits)*c < n {
			s.Splits = (n + c - 1) / c
		}
		build, groups, width = kernels.FlashDecodeKV, kernels.FlashKVGroups(s), kernels.FlashKVWidth(s)
	}
	ker, err := build(s)
	if err != nil {
		t.Fatal(err)
	}
	c, err := d.Compile(ker)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	g := newGPU(t, d)
	defer g.free()
	rng := rand.New(rand.NewSource(71))
	q := make([]float32, s.Rows*s.Heads*s.Dim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	capacity := max(128, n, s.KStride)
	kdim := s.KVHeads * s.Dim
	k, v := make([]float32, capacity*kdim), make([]float32, capacity*kdim)
	for i := range k {
		k[i] = float32(math.NaN())
		v[i] = float32(math.NaN())
	}
	// Poison all unwritten KV, including the partial final tile.
	for p := 0; p < n; p++ {
		for x := 0; x < kdim; x++ {
			i := p*kdim + x
			if s.KStride > 0 {
				i = x*s.KStride + p
			}
			k[i] = float32(rng.NormFloat64())
			v[p*kdim+x] = float32(rng.NormFloat64())
		}
	}
	vb := f32bytes(v)
	if s.F16 {
		vb = make([]byte, len(v)*2)
		for i, x := range v {
			h := quant.EncodeHalf(x)
			binary.LittleEndian.PutUint16(vb[2*i:], h)
			v[i] = float32(quant.DecodeHalf(h))
		}
	}
	counts := []uint32{uint32(n)}
	for r := 0; r < s.Rows; r++ {
		counts = append(counts, uint32(max(1, n-s.Rows+1+r)))
	}
	sinks := make([]float32, s.Heads)
	for h := range sinks {
		sinks[h] = float32(h) - 1.5
		if h == 0 {
			sinks[h] = float32(math.Inf(-1))
		}
	}
	outInit := make([]float32, len(q)+32)
	for i := range outInit {
		outInit[i] = float32(math.NaN())
	}
	qo, ko, vo, no, oo := g.up(f32bytes(q)), g.up(f32bytes(k)), g.up(vb), g.up(u32bytes(counts)), g.up(f32bytes(outInit))
	dst := oo
	if s.Splits > 1 {
		parts := make([]float32, kernels.FlashPartialFloats(s))
		for i := range parts {
			parts[i] = float32(math.NaN())
		}
		dst = g.up(f32bytes(parts))
	}
	args := []backend.Buf{qo, ko, vo, no, dst}
	if s.Sink {
		args = append(args, g.up(f32bytes(sinks)))
	}
	if err := c.Launch(groups, width, args...); err != nil {
		t.Fatal(err)
	}
	if s.Splits > 1 {
		mk, e := kernels.FlashAttentionMerge(s)
		if e != nil {
			t.Fatal(e)
		}
		mc, e := d.Compile(mk)
		if e != nil {
			t.Fatal(e)
		}
		defer mc.Close()
		ma := []backend.Buf{dst, oo}
		if s.Sink {
			ma = append(ma, args[5])
		}
		if e = mc.Launch(s.Rows*s.Heads, 32, ma...); e != nil {
			t.Fatal(e)
		}
	}
	raw := make([]byte, len(outInit)*4)
	if err := oo.Read(raw); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(outInit))
	for i := range got {
		got[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	for i := len(q); i < len(got); i++ {
		if !math.IsNaN(float64(got[i])) {
			return 0, fmt.Errorf("output guard %d overwritten", i)
		}
	}
	var se, ss float64
	for r := 0; r < s.Rows; r++ {
		cnt := n
		if s.Rows > 1 {
			cnt = int(counts[r+1])
		}
		lo := 0
		if s.Window > 0 && s.Chunked {
			lo = (cnt - 1) / s.Window * s.Window
		} else if s.Window > 0 {
			lo = max(0, cnt-s.Window)
		}
		for h := 0; h < s.Heads; h++ {
			scores := make([]float32, cnt-lo)
			mx := float32(-math.MaxFloat32)
			kvh := h / (s.Heads / s.KVHeads)
			for p := lo; p < cnt; p++ {
				var score float32
				for x := 0; x < s.Dim; x++ {
					ki := p*kdim + kvh*s.Dim + x
					if s.KStride > 0 {
						ki = (kvh*s.Dim+x)*s.KStride + p
					}
					score += q[(r*s.Heads+h)*s.Dim+x] * k[ki]
				}
				score *= s.Scale
				if s.Softcap > 0 {
					score = s.Softcap * float32(math.Tanh(float64(score/s.Softcap)))
				}
				scores[p-lo] = score
				mx = max(mx, score)
			}
			if s.Sink {
				mx = max(mx, sinks[h])
			}
			var sum float32
			for i, x := range scores {
				scores[i] = float32(math.Exp(float64(x - mx)))
				sum += scores[i]
			}
			if s.Sink {
				sum += float32(math.Exp(float64(sinks[h] - mx)))
			}
			for x := 0; x < s.Dim; x++ {
				var want float32
				for p := lo; p < cnt; p++ {
					want += (scores[p-lo] / sum) * v[p*kdim+kvh*s.Dim+x]
				}
				actual := got[(r*s.Heads+h)*s.Dim+x]
				if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) {
					return math.Inf(1), fmt.Errorf("nonfinite row%d head%d dim%d", r, h, x)
				}
				delta := float64(actual - want)
				se += delta * delta
				ss += float64(want) * float64(want)
			}
		}
	}
	return se / max(ss, 1e-30), nil
}

// skipNoFlashLanes skips a device the tier never runs flash attention on: the
// flash kernels reduce across a 32-lane subgroup, and tier's flash and
// flash-KV selection (flash.go, pagedattn.go) take plain attention wherever the
// device does not guarantee one -- llvmpipe's 8-lane subgroups, say.
func skipNoFlashLanes(t *testing.T, d backend.Device) {
	t.Helper()
	if ok, why := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok {
		t.Skipf("%s: no guaranteed 32-lane subgroup (%s); the tier runs plain attention here, never flash", d.API(), why)
	}
}

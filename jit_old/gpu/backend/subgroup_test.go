package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestSubgroupCollectivesMatchTheReference runs every subgroup collective
// (ir/subgroup.go) on every device that guarantees a 32-lane subgroup, at
// each scalar type, and holds each result to a host evaluation of the same
// definition: the reductions over the thread's aligned 32 lanes, the broadcast
// from a uniform lane, the shuffle from a per-lane source masked to 31, the
// lane index and the width. The outputs start poisoned, so a result a target
// never wrote reads as the poison rather than as a plausible zero.
//
// The inputs differ in every lane and the shuffle sources include lanes past
// 31, so a reduction that skips a butterfly step, a collective that crosses
// into the other subgroup of the workgroup, or a source left unmasked each
// change at least one value.
func TestSubgroupCollectivesMatchTheReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const groups = 3
	const width = 2 * ir.SubgroupLanes
	n := groups * width
	ran := 0
	for _, d := range devs {
		defer d.Close()
		if ok, why := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok {
			t.Logf("%s: no 32-lane promise (%s); the collectives are not offered there", d.API(), why)
			continue
		}
		for _, ty := range []ir.Type{ir.F32, ir.I32, ir.U32} {
			t.Run(d.API()+"/"+ty.String(), func(t *testing.T) {
				kk, err := mainkernels.SubgroupProbe(ty)
				if err != nil {
					t.Fatal(err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					t.Fatalf("compile: %v", err)
				}
				defer kern.Close()
				rng := rand.New(rand.NewSource(int64(ty) + 1))
				in := make([]uint32, n)
				src := make([]uint32, n)
				for i := range in {
					switch ty {
					case ir.F32:
						// Small integers: every sum is exact, so the
						// reduction order cannot hide a missing lane.
						in[i] = math.Float32bits(float32(rng.Intn(2001) - 1000))
					case ir.I32:
						in[i] = uint32(int32(rng.Intn(2_000_001) - 1_000_000))
					default:
						in[i] = uint32(rng.Intn(3_000_000))
					}
					src[i] = uint32(rng.Intn(200)) // past 31 on purpose
				}
				out := poisonWords(n*mainkernels.SubgroupProbeOuts, 0x7FC0DEAD)
				meta := poisonWords(n*mainkernels.SubgroupProbeMeta, 0xDEADBEEF)
				bufs := []backend.Buf{}
				for _, w := range [][]uint32{in, src, out, meta} {
					b, err := d.Alloc(4 * len(w))
					if err != nil {
						t.Fatal(err)
					}
					defer b.Free()
					if err := b.Write(wordBytes(w)); err != nil {
						t.Fatal(err)
					}
					bufs = append(bufs, b)
				}
				if err := kern.Launch(groups, width, bufs...); err != nil {
					t.Fatalf("launch: %v", err)
				}
				got := make([]byte, 4*len(out))
				if err := bufs[2].Read(got); err != nil {
					t.Fatal(err)
				}
				gm := make([]byte, 4*len(meta))
				if err := bufs[3].Read(gm); err != nil {
					t.Fatal(err)
				}
				bad := 0
				for i := 0; i < n; i++ {
					base := i &^ (ir.SubgroupLanes - 1)
					sg := (i % width) / ir.SubgroupLanes
					want := subgroupWant(ty, in[base:base+ir.SubgroupLanes], (7*sg+3)&31, int(src[i]&31))
					for j, w := range want {
						g := binary.LittleEndian.Uint32(got[4*(i*mainkernels.SubgroupProbeOuts+j):])
						if g != w {
							if bad < 8 {
								t.Errorf("thread %d result %d (%s): got %#x want %#x",
									i, j, []string{"sum", "max", "min", "broadcast", "shuffle"}[j], g, w)
							}
							bad++
						}
					}
					lane := binary.LittleEndian.Uint32(gm[4*(i*mainkernels.SubgroupProbeMeta):])
					size := binary.LittleEndian.Uint32(gm[4*(i*mainkernels.SubgroupProbeMeta+1):])
					if lane != uint32(i%ir.SubgroupLanes) || size != ir.SubgroupLanes {
						if bad < 8 {
							t.Errorf("thread %d: lane %d width %d, want %d and %d", i, lane, size, i%32, 32)
						}
						bad++
					}
				}
				if bad > 0 {
					t.Fatalf("%d wrong results of %d", bad, n*(mainkernels.SubgroupProbeOuts+mainkernels.SubgroupProbeMeta))
				}
				ran++
			})
		}
	}
	if ran == 0 {
		t.Skip("no device here guarantees a 32-lane subgroup")
	}
}

// subgroupWant is one thread's five results, as bits, from its subgroup's 32
// inputs: sum, max, min, the value at lane from, the value at lane shuf.
func subgroupWant(ty ir.Type, lanes []uint32, from, shuf int) []uint32 {
	switch ty {
	case ir.F32:
		var s, mx, mn float32 = 0, float32(math.Inf(-1)), float32(math.Inf(1))
		for _, w := range lanes {
			f := math.Float32frombits(w)
			s += f
			mx = max(mx, f)
			mn = min(mn, f)
		}
		return []uint32{math.Float32bits(s), math.Float32bits(mx), math.Float32bits(mn), lanes[from], lanes[shuf]}
	case ir.I32:
		var s, mx, mn int32 = 0, math.MinInt32, math.MaxInt32
		for _, w := range lanes {
			v := int32(w)
			s += v
			mx = max(mx, v)
			mn = min(mn, v)
		}
		return []uint32{uint32(s), uint32(mx), uint32(mn), lanes[from], lanes[shuf]}
	}
	var s, mx, mn uint32 = 0, 0, math.MaxUint32
	for _, v := range lanes {
		s += v
		mx = max(mx, v)
		mn = min(mn, v)
	}
	return []uint32{s, mx, mn, lanes[from], lanes[shuf]}
}

func poisonWords(n int, v uint32) []uint32 {
	w := make([]uint32, n)
	for i := range w {
		w[i] = v
	}
	return w
}

func wordBytes(w []uint32) []byte {
	b := make([]byte, 4*len(w))
	for i, x := range w {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

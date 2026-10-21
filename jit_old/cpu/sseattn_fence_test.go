//go:build amd64 && linux

package cpu

import (
	"math"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// attnFenced returns n bytes that end exactly at a PROT_NONE page, so a
// kernel reading even one byte past them faults.
func attnFenced(t *testing.T, n int) []byte {
	t.Helper()
	ps := os.Getpagesize()
	pages := (n+ps-1)/ps + 1
	mem, err := syscall.Mmap(-1, 0, pages*ps, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	fence := (pages - 1) * ps
	if err := syscall.Mprotect(mem[fence:], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(mem) })
	return mem[fence-n : fence]
}

// TestAttnReadsNothingPastTheRowSSE puts the KV cache, both queries and both
// weight rows each against an unmapped page, the cache's last row ending
// exactly at the fence, and runs all seven SSE kernels at every hd%4 residue
// on both KV widths.
//
// The NaN guards elsewhere cannot see this: only lane 0 of a tail load reaches
// the arithmetic (MULSS, ADDSS), so an over-wide tail load pulls guard NaNs
// into lanes nothing reads. On a real cache it would fault only at a page
// boundary; here it faults every time.
//
// The fenced results must also equal the same call on ordinary memory, bit for
// bit, so the gate cannot pass by running nothing.
func TestAttnReadsNothingPastTheRowSSE(t *testing.T) {
	const npos = 3
	type kern struct {
		name   string
		emit   func(hd, stride int, fm KVFmt) ([]byte, error)
		outLen func(hd int) int
	}
	perPos := func(int) int { return npos }
	perDim := func(hd int) int { return hd }
	kerns := []kern{
		{"scores", EmitAttnScoresSSE, perPos}, {"scores2", EmitAttnScores2SSE, perPos},
		{"acc", EmitAttnAccSSE, perDim}, {"acc_into", EmitAttnAccIntoSSE, perDim},
		{"acc2", EmitAttnAcc2SSE, perDim}, {"acc2_into", EmitAttnAcc2IntoSSE, perDim},
		{"scores_t1", func(hd, stride int, fm KVFmt) ([]byte, error) {
			return EmitAttnScoresTiledSSE(hd, stride, hd, npos, 1, fm)
		}, perPos},
	}
	ran := 0
	for _, hd := range []int{1, 2, 3, 4, 5, 6, 7, 8, 12, 17, 63, 64, 66, 128} {
		for _, f16 := range []bool{false, true} {
			d := newAttnSSEData(hd, hd, npos, int64(hd), false)
			elem := 4
			src := unsafe.Pointer(&d.kv32[0])
			if f16 {
				elem, src = 2, unsafe.Pointer(&d.kv16[0])
			}
			kv := attnFenced(t, npos*hd*elem)
			copy(kv, unsafe.Slice((*byte)(src), len(kv)))
			f32 := func(v []float32, n int) *float32 {
				b := attnFenced(t, 4*n)
				copy(b, unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*n))
				return (*float32)(unsafe.Pointer(&b[0]))
			}
			q, q2, w, w2 := f32(d.q, hd), f32(d.q2, hd), f32(d.w, npos), f32(d.w2, npos)

			for _, k := range kerns {
				b, err := k.emit(hd, hd, KVOf(f16))
				c := sseAttnKernel(t, k.name+"_fence_hd"+itoa(hd)+map[bool]string{true: "_f16"}[f16], b, err)
				n := k.outLen(hd)
				call := func(fenced bool) ([]float32, []float32) {
					o0, o1 := sentinelRow(n), sentinelRow(n)
					for i := 0; i < n; i++ {
						o0[i], o1[i] = float32(i), float32(-i)
					}
					a := Args{Out: &o0[0], Out2: &o1[0], W: d.cache(f16, 0), Rows: npos,
						Q32: &d.q[0], Q2: &d.q2[0], AScale: &d.w[0], AScale2: &d.w2[0]}
					if fenced {
						a.W, a.Q32, a.Q2, a.AScale, a.AScale2 = &kv[0], q, q2, w, w2
					}
					c.Call(&a)
					return o0, o1
				}
				f0, f1 := call(true)
				p0, p1 := call(false)
				for i := 0; i < n; i++ {
					if math.Float32bits(f0[i]) != math.Float32bits(p0[i]) ||
						math.Float32bits(f1[i]) != math.Float32bits(p1[i]) {
						t.Fatalf("%s hd=%d f16=%v: element %d is (%v, %v) fenced and (%v, %v) on ordinary memory",
							k.name, hd, f16, i, f0[i], f1[i], p0[i], p1[i])
					}
				}
				ran++
			}
		}
	}
	t.Logf("%d kernel calls read nothing past a fenced cache, query or weight row", ran)
}

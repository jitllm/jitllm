//go:build linux && (amd64 || arm64)

package nn

import (
	"math"
	"syscall"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// interleaved copies b into fresh anonymous memory whose pages alternate over
// nodes, as jlm places a model's frames.
func interleaved(t *testing.T, b []byte, nodes []int) []byte {
	t.Helper()
	if len(b) == 0 {
		return b
	}
	m, mapped, err := kernels.MapOffHeap(len(b))
	if err != nil || !mapped {
		t.Skipf("no anonymous mapping here: %v", err)
	}
	t.Cleanup(func() { kernels.UnmapOffHeap(m) })
	mask := make([]uint64, 1+nodes[len(nodes)-1]/64)
	for _, n := range nodes {
		mask[n/64] |= 1 << uint(n%64)
	}
	const mpolInterleave, mpolMFMove = 3, 1 << 1
	if _, _, e := syscall.Syscall6(syscall.SYS_MBIND, uintptr(unsafe.Pointer(&m[0])), uintptr(len(m)),
		mpolInterleave, uintptr(unsafe.Pointer(&mask[0])), uintptr(len(mask)*64+1), mpolMFMove); e != 0 {
		t.Skipf("mbind refused: %v", e)
	}
	copy(m, b)
	return m[:len(b):len(b)]
}

// numaNodeRows is the rows of a packed word-line that one 4 KiB page holds, the
// block DoNodes places on one node. It is stated here, not read from the
// engine, so that a change to the layout fails this gate instead of moving it.
const numaNodeRows = 4096 / 4

// TestFusedMatVecRunsNodeAffine: on a host of two or more NUMA nodes, a
// node-affine pool takes the fused matvec of every packed format the host has a
// fused kernel for -- the ternary ones included -- through sched.DoNodes when
// the weight's rows are a whole number of rounds of one block per node, and
// gives the plain pool's bits. Any other row count runs the plain region, and
// a block past a round and a ragged count are the controls that the count of
// node-affine regions discriminates.
func TestFusedMatVecRunsNodeAffine(t *testing.T) {
	nodes := sched.NUMANodes()
	if len(nodes) < 2 {
		t.Skip("one NUMA node: nothing to place -- this gate proved nothing here")
	}
	// A host that refuses the policy skips here, once, not in every subtest.
	interleaved(t, make([]byte, 1), nodes)
	round := numaNodeRows * len(nodes)
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedFusedSupported(gt) {
			t.Logf("%s: no fused kernel on this host", gt)
			continue
		}
		grp, step := grpOf(gt), kStep(gt)
		for _, sh := range []struct {
			name string
			rows int
			want int64
		}{
			{"two rounds", 2 * round, 1},
			{"a block past a round", round + numaNodeRows, 0},
			{"ragged", round + grp, 0},
		} {
			t.Run(gt.String()+" "+sh.name, func(t *testing.T) {
				k := 2 * step
				pk := packOne(t, gt, sh.rows, k)
				pk.QS, pk.D, pk.SC = interleaved(t, pk.QS, nodes), interleaved(t, pk.D, nodes), interleaved(t, pk.SC, nodes)
				x := prevnniAct(k)
				run := func(numa bool) ([]float32, int64) {
					j := NewJIT(k, sh.rows, []quant.Type{gt}, WithTune(TuneOff), WithFusedKSlices(-1),
						WithSched(sched.WithNUMA(numa)))
					if j == nil {
						t.Fatal("no JIT")
					}
					defer j.Close()
					if j.packedFused[gt] == nil {
						t.Fatalf("PackedFusedSupported but the JIT built no fused kernel")
					}
					out := poison(sh.rows)
					if !j.MatVecPacked(out, gt, pk, x, sh.rows, k) {
						t.Fatal("MatVecPacked declined")
					}
					_, affine := j.NUMA()
					return out, affine
				}
				want, a0 := run(false)
				got, a1 := run(true)
				if a0 != 0 || a1 != sh.want {
					t.Fatalf("node-affine regions: %d without NUMA, %d with (want 0 and %d)", a0, a1, sh.want)
				}
				for r := range want {
					if math.Float32bits(got[r]) != math.Float32bits(want[r]) || math.IsNaN(float64(want[r])) {
						t.Fatalf("row %d: node-affine %v, plain %v", r, got[r], want[r])
					}
				}
				ran++
			})
		}
	}
	if ran == 0 {
		if cpu.PackedFusedSupported(quant.Q4_K) {
			t.Fatal("a fused kernel exists and no shape ran -- this gate proved nothing")
		}
		t.Skip("no fused packed kernel on this host, so there is no matvec to place")
	}
	t.Logf("%d (format, shape) pair(s) took the layout they should and gave the plain pool's bits", ran)
}

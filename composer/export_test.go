package composer

import "github.com/jitllm/jitllm/graph"

// What the external tests reach of the package's internals.

func EntryName(s string) string { return entryName(s) }

func GPULowering() Lowering { return gpuLowering{} }

func RopePosOracle(p uint32, marks []ropeMark) [4]uint32 { return ropePosOracle(p, marks) }

type RopeMark = ropeMark

func TextMark(at, base uint32) RopeMark { return textMark(at, base) }

func GridMark(at, base, h, w, stride uint32) RopeMark { return gridMark(at, base, h, w, stride) }

func XDMark(at, ends, h, w, idx uint32) RopeMark { return xdMark(at, ends, h, w, idx) }

// WeightLoads is the weight words a launch of req's form at its capacity
// loads: each live load of a weight operand a thread makes, times the trips
// of every loop it varies with, times the threads that make it.
func WeightLoads(r *Registry, req Request) (int, error) {
	ks, err := r.formKernels(req)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, k := range ks {
		threads := k.Elements()
		if k.Work {
			threads *= k.Group
		}
		live := k.Live()
		for i, nd := range k.Nodes {
			if !live[i] || nd.Op != OpLoad || k.Bufs[nd.Buf].Operand.Kind != OperandWeight {
				continue
			}
			trips := 1
			for l, lp := range k.Loops {
				if k.Varies(i, l) {
					trips *= lp.Trip
				}
			}
			n += threads * trips
		}
	}
	return n, nil
}

func Q8RowsCase(R, W int) []float32 { return q8RowsCase(R, W) }

func BlockFmts() []graph.BlockFmt { return blockFmts() }

func HalvesWithinAnUlp(a, b uint32) bool { return halvesWithinAnUlp(a, b) }

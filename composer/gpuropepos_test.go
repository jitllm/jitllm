package composer_test

import (
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPURopePos is TestComposedCPURopePos on every GPU: each row's
// four rotary positions from its sequence's marks, held to the oracle.
func TestComposedGPURopePos(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const R, M = 8, 4
	seqs := [][]composer.RopeMark{
		nil,
		{{5, 9, graph.MarkText}, {10, 12, graph.MarkGrid, 2, 2, 3, 0, 0, 3}, {22, 15, graph.MarkText}},
		{{4, 0, graph.MarkGridXD, 1, 2, 3, 1, 0}, {12, 0, graph.MarkGridXD, 1, 1, 2, 0, 1}},
	}
	rows := [][2]uint32{{3, 0}, {7, 1}, {11, 1}, {19, 1}, {25, 1}, {4, 2}, {7, 2}, {13, 2}}
	marks := make([]uint32, len(seqs)*(1+M*graph.MarkWords))
	off := make([]uint32, len(seqs))
	for s, ms := range seqs {
		o := s * (1 + M*graph.MarkWords)
		off[s] = uint32(o)
		marks[o] = uint32(len(ms))
		for i, m := range ms {
			copy(marks[o+1+i*graph.MarkWords:], m[:])
		}
	}
	desc := []uint32{R}
	for _, r := range rows {
		desc = append(desc, r[0], off[r[1]])
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			b := graph.New()
			y := b.Emit(graph.OpRopePos, nil, []graph.Result{{Name: "rope.pos", Width: 4, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
			b.Out(y)
			_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Step: composer.Step{Rows: R, Page: 16, Keys: 64, Seqs: len(seqs), Marks: M},
				Device: gpuDevice(d)})
			if err != nil {
				t.Fatal(err)
			}
			fr := &recRun{d: d, codes: codes, vals: map[graph.ValueID][]uint32{y: poisoned(R * 4)},
				tables: map[string][]uint32{"rope.rows": desc, "rope.marks": marks}, wts: map[jlm.Role][]uint32{}, bufs: map[string]backend.Buf{}}
			for _, c := range codes {
				k, err := d.Compile(c.GPU)
				if err != nil {
					fr.close()
					t.Fatalf("compiling %s: %v", c.Name, err)
				}
				fr.kerns = append(fr.kerns, k)
			}
			defer fr.close()
			if err := fr.launch(); err != nil {
				t.Fatal(err)
			}
			got := fr.vals[y]
			for i, r := range rows {
				want := composer.RopePosOracle(r[0], seqs[r[1]])
				for a := range 4 {
					if got[4*i+a] != want[a] {
						t.Errorf("row %d (position %d of sequence %d) axis %d: %d, want %d", i, r[0], r[1], a, got[4*i+a], want[a])
					}
				}
			}
		})
	}
}

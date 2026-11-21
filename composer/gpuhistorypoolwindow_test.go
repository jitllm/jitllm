package composer_test

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUHistoryPoolHoldsNoWindow: a history pool's form on every
// GPU writes no value wider than its output, a row's width: the
// exponentials its sum and its weighted sum read run in both reductions, and
// the segments are read where they are rather than copied side by side. The
// pool once wrote two values of every row's window and a copy of the
// segments.
func TestComposedGPUHistoryPoolHoldsNoWindow(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	const rows = 256
	for _, d := range devs {
		for _, sh := range []struct {
			name  string
			p, w  int
			segs  [][4]uint64
			width int
		}{
			{"one", 16, 64, [][4]uint64{{0, 0, 64, 0}}, 128},
			{"overlap", 4, 32, [][4]uint64{{1, 0, 64, 0}, {0, 32, 96, 32}}, 128},
		} {
			t.Run(d.API()+"/"+sh.name, func(t *testing.T) {
				g := composer.HistoryPoolGraph(sh.p, sh.w, sh.width, sh.segs, jlm.RoleCompAPE)
				_, codes, err := composer.Default().Form(composer.Request{Graph: g,
					History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}},
					Step:    composer.Step{Rows: rows, Page: 64, Keys: 512, Seqs: 1}, Device: gpuDevice(d)})
				if err != nil {
					t.Fatal(err)
				}
				limit, n, total := int64(rows*sh.w*4), 0, int64(0)
				for _, c := range codes {
					for _, b := range c.Binds {
						if !b.Writes || b.Operand.Kind != composer.OperandValue && b.Operand.Kind != composer.OperandScratch {
							continue
						}
						n++
						total += b.Bytes
						if b.Bytes > limit {
							t.Errorf("%s writes %s, %d bytes: wider than the pool's %d", c.Name, b.Operand.Name, b.Bytes, limit)
						}
					}
				}
				if n == 0 {
					t.Fatalf("%d codes write no value", len(codes))
				}
				t.Logf("%d codes, %d values written, %d bytes, each at most %d", len(codes), n, total, limit)
			})
		}
	}
}

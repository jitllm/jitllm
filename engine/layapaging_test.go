package engine

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestLayaPagesWithTheSameAnswer holds Laya to the paging principle: under a
// budget of one page, which evicts and reads pages again, every
// row it answers is bit for bit the row it answers with every page resident.
// A budget that read no more pages than the resident run evicted nothing and
// proves nothing.
func TestLayaPagesWithTheSameAnswer(t *testing.T) {
	for _, name := range auditDecision() {
		t.Run(filepath.Base(name), func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(name))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(name))
			}
			path := jlmOf(t, src)
			ids := []int32{1, 5, 9, 13, 17, 21, 25, 29}
			run := func(budget bool) ([][]float32, int64, int) {
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				pages := m.Container.NPages()
				var opts []Option
				if budget {
					opts = append(opts, WithPageBudget(m.PageSize()))
				}
				if m.IsEmbedding() {
					e, err := NewEmbedder(m, opts...)
					if err != nil {
						t.Fatal(err)
					}
					defer e.Close()
					var rows [][]float32
					for i := range 3 {
						v, err := e.Embed(ids[:4+i])
						if err != nil {
							t.Fatal(err)
						}
						rows = append(rows, append([]float32(nil), v...))
					}
					_, reads, _ := m.PageStats(e.st.fr)
					return rows, reads, pages
				}
				in := testInstance(t, m, len(ids)+1, opts...)
				defer in.Close()
				rows := teacherForced(t, in, ids)
				_, reads, _ := m.PageStats(in.fr)
				return rows, reads, pages
			}
			whole, once, _ := run(false)
			paged, reads, pages := run(true)
			if reads <= once {
				t.Fatalf("a budget of one of the %d block pages read %d, the resident run %d: nothing was evicted, so nothing was tested",
					pages, reads, once)
			}
			for k := range whole {
				for i := range whole[k] {
					if math.Float32bits(whole[k][i]) != math.Float32bits(paged[k][i]) {
						t.Fatalf("row %d element %d: %v resident, %v paged", k, i, whole[k][i], paged[k][i])
					}
				}
			}
			t.Logf("%d rows alike under one of %d block pages, %d page reads against %d resident", len(whole), pages, reads, once)
		})
	}
}

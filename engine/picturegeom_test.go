package engine

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// TestPictureGeometryIsTheFits holds every census tower's stated geometry
// (jlm.PictureGeometry, docs/design/picture-statement.md) to the squares
// the tower's own fit gives (TowerGraphs.Square), which the towers' gates
// hold to their processors: the count and every square's resize size,
// window, grid and markers, over pictures from one pixel to past the
// largest a tower reads, every aspect past each threshold the fits branch
// on. A geometry rewritten -- a tie kept rather than taken -- is its
// violation.
func TestPictureGeometryIsTheFits(t *testing.T) {
	sides := []int{1, 2, 3, 13, 14, 15, 27, 28, 29, 31, 32, 33, 55, 56, 64, 100, 223, 224, 225, 300, 333, 336, 447,
		448, 449, 500, 512, 640, 896, 1000, 1023, 1024, 1025, 1280, 1500, 2000, 2048, 3000, 4096}
	ran := 0
	for _, src := range auditModels(t) {
		if src.hf || src.mmproj == "" || src.size > 3<<30 {
			continue
		}
		t.Run(src.name, func(t *testing.T) {
			path, err := convertPair(src.path, src.mmproj)
			if err != nil {
				t.Fatalf("convert: %v", err)
			}
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			v := m.Container.Vision()
			if m.Tower == nil || v == nil || v.Tower == nil {
				t.Skip("no picture tower")
			}
			g := v.Tower.Geometry
			if g == nil {
				t.Fatal("the container states no picture geometry")
			}
			ran++
			bad := 0
			for _, w := range sides {
				for _, h := range sides {
					_, n := m.Tower.Square(w, h, 0)
					for i := range n {
						want, _ := m.Tower.Square(w, h, i)
						want.Class = 0
						got, gn, err := model.GeometrySquare(g, w, h, i)
						if err != nil || gn != n || got != want {
							if bad++; bad <= 8 {
								t.Errorf("%dx%d square %d of %d: geometry %+v of %d (%v), the fit %+v", w, h, i, n, got, gn, err, want)
							}
						}
					}
				}
			}
			if bad > 0 {
				t.Errorf("%d squares part from the fit", bad)
			}
			// The violation: the resize's sides swapped, or, where every
			// square is square, one square more.
			viol := *g
			viol.Square[jlm.SqOw], viol.Square[jlm.SqOh] = g.Square[jlm.SqOh], g.Square[jlm.SqOw]
			if !partsFromFit(m, &viol, sides) {
				terms := append([]jlm.Term(nil), g.Count.Terms...)
				last := jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(len(terms) - 1)}
				viol = *g
				viol.Count = jlm.Expr{Terms: append(terms, jlm.Term{Fn: jlm.FnAdd, Args: [3]jlm.Arg{last, jlm.ExactNum(1)}})}
				if !partsFromFit(m, &viol, sides) {
					t.Error("VIOLATION: a geometry rewritten gives the fit's squares")
				}
			}
		})
	}
	if ran == 0 {
		t.Fatal("no tower's geometry was held to its fit -- this gate proved nothing")
	}
	t.Logf("%d towers' geometry held to their fits", ran)
}

// partsFromFit reports that g gives some picture of sides a square, or a
// count, the tower's fit does not.
func partsFromFit(m *model.Model, g *jlm.PictureGeometry, sides []int) bool {
	for _, w := range sides {
		for _, h := range sides {
			_, n := m.Tower.Square(w, h, 0)
			for i := range n {
				want, _ := m.Tower.Square(w, h, i)
				want.Class = 0
				got, gn, err := model.GeometrySquare(g, w, h, i)
				if err != nil || gn != n || got != want {
					return true
				}
			}
		}
	}
	return false
}

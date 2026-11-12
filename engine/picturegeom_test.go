package engine

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// TestPictureGeometryHoldsItsSweep holds every census tower's squares, as
// its stated geometry gives them (jlm.PictureGeometry,
// docs/design/picture-statement.md), over pictures from one pixel to past
// the largest a tower reads -- every aspect past each threshold the
// processors branch on -- to the sums the towers' own fits gave over the
// same sweep (testdata/picturegeom.sums). The fits were held to the
// processors by each tower's gates, and to the geometry square for square
// before they were deleted: a sum is that record. Each square's count,
// class, resize size, window, grid and markers are summed, and a video's
// frames read as pictures likewise. A geometry rewritten -- its resize's
// sides swapped -- is the violation.
func TestPictureGeometryHoldsItsSweep(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "picturegeom.sums"))
	if err != nil {
		t.Fatal(err)
	}
	sums := map[string]string{}
	for line := range strings.Lines(string(raw)) {
		if f := strings.Fields(line); len(f) == 2 {
			sums[f[0]] = f[1]
		}
	}
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
			if m.Tower == nil {
				t.Skip("no picture tower")
			}
			want, ok := sums[src.name]
			if !ok {
				// The sum printed is for review against a gate that holds the
				// tower's pixels to its processor, never to record unread.
				got, err := sweepSum(m.Tower, false)
				t.Fatalf("no sum is recorded for %s (its geometry sums to %s, %v)", src.name, got, err)
			}
			ran++
			if got, err := sweepSum(m.Tower, false); err != nil || got != want {
				t.Errorf("the geometry's squares sum to %s over the sweep (%v), the fits' to %s", got, err, want)
			}
			if v := m.Tower.Video; v != nil && v.AsPictures() {
				if got, err := sweepSum(m.Tower, true); err != nil || got != sums[src.name+"/frames"] {
					t.Errorf("the video's frames sum to %s over the sweep (%v), the fits' to %s", got, err, sums[src.name+"/frames"])
				}
			}
			// The violation: the resize's sides swapped, or, where every
			// square is square, one square more.
			g := m.Tower.Geometry
			viol := *g
			viol.Square[jlm.SqOw], viol.Square[jlm.SqOh] = g.Square[jlm.SqOh], g.Square[jlm.SqOw]
			m.Tower.Geometry = &viol
			if got, err := sweepSum(m.Tower, false); err == nil && got == want {
				terms := append([]jlm.Term(nil), g.Count.Terms...)
				last := jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(len(terms) - 1)}
				viol = *g
				viol.Count = jlm.Expr{Terms: append(terms, jlm.Term{Fn: jlm.FnAdd, Args: [3]jlm.Arg{last, jlm.ExactNum(1)}})}
				if got, err := sweepSum(m.Tower, false); err == nil && got == want {
					t.Error("VIOLATION: a geometry rewritten sums as the fits did")
				}
			}
			m.Tower.Geometry = g
		})
	}
	if ran == 0 {
		t.Fatal("no tower's geometry was held to its sum -- this gate proved nothing")
	}
	t.Logf("%d towers' geometry held to their fits' sums", ran)
}

// sweepSum is tg's squares over the sweep, summed: each picture's count and
// every square's fields, or, for frames, each frame's one square.
func sweepSum(tg *model.TowerGraphs, frames bool) (string, error) {
	sides := []int{1, 2, 3, 13, 14, 15, 27, 28, 29, 31, 32, 33, 55, 56, 64, 100, 223, 224, 225, 300, 333, 336, 447,
		448, 449, 500, 512, 640, 896, 1000, 1023, 1024, 1025, 1280, 1500, 2000, 2048, 3000, 4096}
	h := sha256.New()
	var ps model.PictureSquares
	for _, w := range sides {
		for _, ht := range sides {
			if frames {
				sq, err := tg.FrameSquare(w, ht)
				if err != nil {
					return "", fmt.Errorf("a %dx%d frame: %w", w, ht, err)
				}
				putSquare(h, sq)
				continue
			}
			n, err := ps.Of(tg, w, ht)
			if err != nil {
				return "", fmt.Errorf("a %dx%d picture: %w", w, ht, err)
			}
			binary.Write(h, binary.LittleEndian, int64(n))
			for i := range n {
				sq, err := ps.At(i)
				if err != nil {
					return "", fmt.Errorf("a %dx%d picture's square %d: %w", w, ht, i, err)
				}
				putSquare(h, sq)
			}
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

// squareOf is square i of a w x h picture and how many the picture has
// (TowerGraphs.Square), the gate failed where the geometry cannot say.
func squareOf(t *testing.T, tg *model.TowerGraphs, w, h, i int) (model.Square, int) {
	t.Helper()
	sq, n, err := tg.Square(w, h, i)
	if err != nil {
		t.Fatalf("square %d of a %dx%d picture: %v", i, w, h, err)
	}
	return sq, n
}

func putSquare(h hash.Hash, sq model.Square) {
	for _, v := range []int{sq.Class, sq.Ow, sq.Oh, sq.X0, sq.Y0, sq.Pw, sq.Ph, sq.Gw, sq.Gh, sq.Mark, sq.MarkN, sq.Trail, sq.TrailN} {
		binary.Write(h, binary.LittleEndian, int64(v))
	}
}

// restateStep rewrites m's tower step of kind k by edit and has the
// converter state the picture's geometry again from the steps
// (convert.GeometryOf): a golden made under the processor's other bounds,
// or a gate's violation. The container's own steps are left as they are,
// and the tower's classes as they were formed.
func restateStep(t *testing.T, m *model.Model, k jlm.PreKind, edit func(s *jlm.PreStep)) {
	t.Helper()
	if !restateIfStated(t, m, k, edit) {
		t.Fatalf("the tower states no step of kind %d", k)
	}
}

// restateIfStated is restateStep's restatement, false and m left as it
// was where the tower states no step of kind k.
func restateIfStated(t *testing.T, m *model.Model, k jlm.PreKind, edit func(s *jlm.PreStep)) bool {
	t.Helper()
	v := m.Container.Vision()
	tw := *v.Tower
	tw.Pre = slices.Clone(tw.Pre)
	found := false
	for i := range tw.Pre {
		if s := &tw.Pre[i]; s.Kind == k {
			s.Ints, s.Floats = slices.Clone(s.Ints), slices.Clone(s.Floats)
			edit(s)
			found = true
		}
	}
	if !found {
		return false
	}
	g, err := convert.GeometryOf(&tw, v)
	if err != nil {
		t.Fatal(err)
	}
	m.Tower.Geometry = g
	return true
}

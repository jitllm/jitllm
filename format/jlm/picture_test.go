package jlm

import (
	"math"
	"reflect"
	"testing"
)

// testGeometry is a geometry no processor has: a value binding, a choice of
// a grid by two costs and a tie, the count its tiles and a square's fields
// of them.
func testGeometry() *PictureGeometry {
	var half, el, c0, c1, tie, cnt ExprBuilder
	half.Fn(FnMul, Operand(PicW), ExactNum(0.1))
	el.Fn(FnLess, Operand(PicBound+1), Operand(PicBound))
	c0.Fn(FnAbs, c0.Fn(FnSub, c0.Fn(FnDiv, Operand(PicW), Operand(PicH)), c0.Fn(FnDiv, Operand(PicBound+1), Operand(PicBound+2))))
	c1.Fn(FnMul, Operand(PicBound+1), Operand(PicBound+2))
	tie.Fn(FnSub, Operand(PicW), Num(300))
	cnt.Fn(FnMul, Operand(PicBound+1), Operand(PicBound+2))
	g := &PictureGeometry{
		Bind: []PictureBind{
			{Value: Expr{Terms: half.T}},
			{Choice: &PictureChoice{Cols: 2, Rows: []float64{1, 1, 1, 2, 2, 1, 2, 2}, Eligible: Expr{Terms: el.T},
				Cost: []Expr{{Terms: c0.T}, {Terms: c1.T}}, Tie: Expr{Terms: tie.T}}},
		},
		Count:  Expr{Terms: cnt.T},
		Bounds: PictureBounds{Unit: 14, MaxPix: 448 * 448, MaxPatches: 1024},
	}
	for f := range g.Square {
		var e ExprBuilder
		e.Fn(FnAdd, Operand(uint32(f%PicBound)), Num(float32(f)))
		g.Square[f] = Expr{Terms: e.T}
	}
	return g
}

// TestGeometryRoundTrips writes a tower's geometry into the vision
// section's tail, behind a video and with none, and reads the same record
// back; a section whose tail holds bytes past its records is refused.
func TestGeometryRoundTrips(t *testing.T) {
	for _, video := range []bool{false, true} {
		v := &Vision{NLayer: 12, NEmbd: 768, NHead: 12, NFFN: 3072, ImageSize: 512, PatchSize: 16,
			ProjDim: 576, Scale: 4, Eps: 1e-6, Projector: ProjIdefics3, Tower: sampleTower(t)}
		v.Tower.Geometry = testGeometry()
		if video {
			v.Tower.Video = testPictureVideo()
		}
		got, err := decodeVision(encodeVision(v))
		if err != nil {
			t.Fatalf("video %v: %v", video, err)
		}
		if !reflect.DeepEqual(got.Tower.Geometry, v.Tower.Geometry) || !reflect.DeepEqual(got.Tower.Video, v.Tower.Video) {
			t.Fatalf("video %v: the geometry read back as\n%+v\nnot\n%+v", video, got.Tower.Geometry, v.Tower.Geometry)
		}
		b := append(encodeVision(v), 1, 2, 3, 4)
		if _, err := decodeVision(b); err == nil {
			t.Fatalf("video %v: four bytes past the last record were read", video)
		}
	}
}

// TestGeometryRefusesWhatItCannotRead holds Valid to each field it bounds,
// and the kernel's expressions to their binary32 constants.
func TestGeometryRefusesWhatItCannotRead(t *testing.T) {
	if err := testGeometry().Valid(); err != nil {
		t.Fatal(err)
	}
	for name, breaks := range map[string]func(g *PictureGeometry){
		"a value and a choice":   func(g *PictureGeometry) { g.Bind[1].Value = g.Bind[0].Value },
		"a choice of no columns": func(g *PictureGeometry) { g.Bind[1].Choice.Cols = 0 },
		"a table cut mid-row":    func(g *PictureGeometry) { g.Bind[1].Choice.Rows = g.Bind[1].Choice.Rows[:3] },
		"a table holding a NaN":  func(g *PictureGeometry) { g.Bind[1].Choice.Rows[2] = math.NaN() },
		"five costs": func(g *PictureGeometry) {
			c := g.Bind[1].Choice
			c.Cost = append(c.Cost, c.Cost[0], c.Cost[0], c.Cost[0])
		},
		"a cost of an empty expression": func(g *PictureGeometry) { g.Bind[1].Choice.Cost[0] = Expr{} },
		"a binding reading a later one": func(g *PictureGeometry) { g.Bind[0].Value.Terms[0].Args[0] = Operand(PicBound) },
		"a count past the operands":     func(g *PictureGeometry) { g.Count.Terms[0].Args[0] = Operand(PicBound + 3) },
		"a field of a stored vector":    func(g *PictureGeometry) { g.Square[0].Terms[0].Args[0] = Arg{Kind: ArgStored, Role: RoleVPosEmbd} },
		"a field reduced":               func(g *PictureGeometry) { g.Square[0].Terms[0].Reduce = true },
		"no unit":                       func(g *PictureGeometry) { g.Bounds.Unit = 0 },
		"an area under a unit":          func(g *PictureGeometry) { g.Bounds.MaxPix = 1 },
		"no patches":                    func(g *PictureGeometry) { g.Bounds.MaxPatches = 0 },
	} {
		g := testGeometry()
		breaks(g)
		if g.Valid() == nil {
			t.Errorf("%s: Valid accepted it", name)
		}
	}
	var e ExprBuilder
	e.Fn(FnMul, Operand(0), ExactNum(0.1))
	x := Expr{Terms: e.T}
	if x.Valid() == nil {
		t.Error("a kernel's expression holds an exact constant")
	}
	if err := x.ValidOver(1); err != nil {
		t.Errorf("a host expression's exact constant was refused: %v", err)
	}
}

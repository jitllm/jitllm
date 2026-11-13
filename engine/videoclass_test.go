package engine

import "testing"

// TestEveryVideoPairHasAClassThatHoldsIt plans two-frame videos, from a few
// pixels to past the frames' resize's most and from square to a long strip,
// through each tower that reads a video in pairs, and holds each pair's
// square to a class whose entry holds it and whose rows hold its patches,
// within the rows the tower reserves. One tower's two frames reach five
// times its pictures' pixels: the classes of the pictures alone put such a
// pair in their largest, which does not hold it, and the gate fails on them.
func TestEveryVideoPairHasAClassThatHoldsIt(t *testing.T) {
	past := 0
	for _, name := range []string{"synth-qwen2vl", "synth-qwen25vl", "synth-glm4v"} {
		t.Run(name, func(t *testing.T) {
			m, _, _ := openQVLTower(t, name)
			defer m.Close()
			tg := m.Tower
			if tg.Video == nil || tg.Video.AsPictures() {
				t.Fatal("the tower reads no video in pairs: the gate proves nothing of it")
			}
			pic := int(tg.Geometry.Bounds.MaxPix)
			for _, wh := range [][2]int{{56, 56}, {640, 480}, {1920, 1080}, {4096, 4096}, {9000, 7000}, {20000, 600}, {300, 16000}} {
				plan, err := tg.VideoPlan(2, wh[0], wh[1], 0)
				if err != nil {
					t.Fatalf("%dx%d: %v", wh[0], wh[1], err)
				}
				sq := plan.Square
				cl := tg.Videos[sq.Class]
				rows := sq.Gw*sq.Gh + tg.Lead + tg.Tail
				t.Logf("%dx%d: frames %dx%d in class %d of %d (%d rows of %d, the tower %d)",
					wh[0], wh[1], sq.Ow, sq.Oh, sq.Class, len(tg.Videos), rows, cl.Rows, tg.Rows)
				if !cl.Pic.Holds(sq.Ow, sq.Oh) || cl.Rows < rows || tg.Rows < cl.Rows {
					t.Fatalf("%dx%d: frames resized to %dx%d (%d rows) in class %d, which holds %dx%d in %d of %d pixels and %d rows, the tower %d",
						wh[0], wh[1], sq.Ow, sq.Oh, rows, sq.Class, cl.Pic.PicW, cl.Pic.PicH, cl.Pic.Area(), sq.Ow*sq.Oh, cl.Rows, tg.Rows)
				}
				if sq.Ow*sq.Oh > pic {
					past++
				}
			}
		})
	}
	if past == 0 {
		t.Fatal("no pair's frames reached past their tower's pictures: the gate saw no pair the pictures' classes cannot hold")
	}
}

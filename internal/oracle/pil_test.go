package oracle

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

// pilCase is one of scripts/pilgold.py's resizes and Pillow's answer.
type pilCase struct {
	Src          string `json:"src"`
	W, H, OW, OH int
	Filter       string `json:"filter"`
	SHA256       string `json:"sha256"`
	Bytes        []int  `json:"bytes"`
}

// pilSource is a case's source picture as RGB bytes.
func pilSource(t *testing.T, c pilCase) []byte {
	t.Helper()
	if c.Src == "pattern" {
		return PILPattern(c.W, c.H)
	}
	f, err := os.Open(filepath.Join("..", "..", "engine", "testdata", c.Src))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	out := make([]byte, 0, b.Dx()*b.Dy()*3)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, a := img.At(x, y).RGBA()
			if a != 0xffff {
				t.Fatalf("%s is translucent at (%d, %d): its premultiplied colour is not the straight RGB Pillow reads", c.Src, x, y)
			}
			out = append(out, byte(r>>8), byte(g>>8), byte(bl>>8))
		}
	}
	return out
}

// TestPILResizeMatchesPillow holds PILResize to the bytes Pillow's resize
// wrote for every case scripts/pilgold.py records: bilinear and bicubic, a
// picture shrunk, grown and of another aspect, up to 4096 wide, and the test
// picture at Gemma 3's and CLIP's sizes. A filter of the other kind is the
// violation.
func TestPILResizeMatchesPillow(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "pil", "resample.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Pillow string    `json:"pillow"`
		Cases  []pilCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("no cases")
	}
	for _, c := range g.Cases {
		src := pilSource(t, c)
		got := PILResize(src, c.W, c.H, c.OW, c.OH, c.Filter == "bicubic")
		sum := sha256.Sum256(got)
		if h := hex.EncodeToString(sum[:]); h != c.SHA256 {
			for i, b := range c.Bytes {
				if int(got[i]) != b {
					t.Fatalf("%s %dx%d to %dx%d %s: byte %d is %d, Pillow's %d", c.Src, c.W, c.H, c.OW, c.OH, c.Filter, i, got[i], b)
				}
			}
			t.Fatalf("%s %dx%d to %dx%d %s: sha256 %s, Pillow's %s", c.Src, c.W, c.H, c.OW, c.OH, c.Filter, h, c.SHA256)
		}
		other := sha256.Sum256(PILResize(src, c.W, c.H, c.OW, c.OH, c.Filter != "bicubic"))
		if hex.EncodeToString(other[:]) == c.SHA256 && (c.W != c.OW || c.H != c.OH) {
			t.Fatalf("%s %dx%d to %dx%d: the other filter writes the same bytes", c.Src, c.W, c.H, c.OW, c.OH)
		}
		t.Logf("%s %dx%d to %dx%d %s: Pillow %s's bytes", c.Src, c.W, c.H, c.OW, c.OH, c.Filter, g.Pillow)
	}
}

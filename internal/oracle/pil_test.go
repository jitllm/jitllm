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

// pilCase is one of scripts/pilgold.py's resizes and its library's answer:
// Pillow's, or torchvision's where Lib says.
type pilCase struct {
	Src          string `json:"src"`
	W, H, OW, OH int
	Filter       string `json:"filter"`
	Lib          string `json:"lib"`
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

// TestPILResizeMatchesPillow holds PILResizeBy to the bytes Pillow's resize
// wrote, and TorchResizeBy to torchvision's, for every case
// scripts/pilgold.py records: bilinear, bicubic and Lanczos, a picture
// shrunk, grown and of another aspect, up to 4096 wide, the test picture at
// Gemma 3's, CLIP's and SmolVLM's sizes, and Idefics3's three resizes one
// after another as transformers' torchvision processor runs them. A filter
// of another kind is the violation, and for torchvision's cases the other
// library's precision too.
func TestPILResizeMatchesPillow(t *testing.T) {
	filters := map[string]PILFilter{"bilinear": PILBilinear, "bicubic": PILBicubic, "lanczos": PILLanczos}
	// other is the filter each case's violation resizes by instead.
	other := map[PILFilter]PILFilter{PILBilinear: PILBicubic, PILBicubic: PILBilinear, PILLanczos: PILBicubic}
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "pil", "resample.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Pillow      string    `json:"pillow"`
		Torchvision string    `json:"torchvision"`
		Cases       []pilCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("no cases")
	}
	resizers := map[string]func([]byte, int, int, int, int, PILFilter) []byte{"pillow": PILResizeBy, "torchvision": TorchResizeBy}
	ran := map[string]int{}
	for _, c := range g.Cases {
		f, ok := filters[c.Filter]
		resize, known := resizers[c.Lib]
		if !ok || !known {
			t.Fatalf("%s %dx%d to %dx%d: filter %q of %q", c.Src, c.W, c.H, c.OW, c.OH, c.Filter, c.Lib)
		}
		ran[c.Lib+" "+c.Filter]++
		src := pilSource(t, c)
		got := resize(src, c.W, c.H, c.OW, c.OH, f)
		sum := sha256.Sum256(got)
		if h := hex.EncodeToString(sum[:]); h != c.SHA256 {
			for i, b := range c.Bytes {
				if int(got[i]) != b {
					t.Fatalf("%s %dx%d to %dx%d %s %s: byte %d is %d, the library's %d", c.Src, c.W, c.H, c.OW, c.OH, c.Lib, c.Filter, i, got[i], b)
				}
			}
			t.Fatalf("%s %dx%d to %dx%d %s %s: sha256 %s, the library's %s", c.Src, c.W, c.H, c.OW, c.OH, c.Lib, c.Filter, h, c.SHA256)
		}
		v := sha256.Sum256(resize(src, c.W, c.H, c.OW, c.OH, other[f]))
		if hex.EncodeToString(v[:]) == c.SHA256 && (c.W != c.OW || c.H != c.OH) {
			t.Fatalf("%s %dx%d to %dx%d: another filter writes the same bytes", c.Src, c.W, c.H, c.OW, c.OH)
		}
		if c.Lib == "torchvision" {
			p := sha256.Sum256(PILResizeBy(src, c.W, c.H, c.OW, c.OH, f))
			if hex.EncodeToString(p[:]) == c.SHA256 {
				t.Fatalf("%s %dx%d to %dx%d %s: Pillow's precision writes torchvision's bytes", c.Src, c.W, c.H, c.OW, c.OH, c.Filter)
			}
		}
		t.Logf("%s %dx%d to %dx%d %s %s: the library's bytes (Pillow %s, torchvision %s)", c.Src, c.W, c.H, c.OW, c.OH, c.Lib, c.Filter, g.Pillow, g.Torchvision)
	}
	for lib := range resizers {
		for name := range filters {
			if ran[lib+" "+name] == 0 {
				t.Errorf("no case resizes by %s's %s", lib, name)
			}
		}
	}
}

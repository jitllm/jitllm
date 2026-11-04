package model

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestStatedTowerComposesOnTheCPU opens SmolVLM-256M's container, whose
// tower its converter states (jlm.Tower), and holds what Open lowered it to:
// an entry for each picture format that reads the picture through
// graph.OpPicture and writes the tower's patch rows, one graph a block of
// the tower, and a head writing the picture's span of rows -- every one
// verified against the tensors it binds and composed for the CPU device
// with no decline. A record whose head binds a tensor the container does not
// hold is refused at Open, naming it.
func TestStatedTowerComposesOnTheCPU(t *testing.T) {
	path := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf")
	mm := testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
	if !sourcePresent(path, mm) {
		t.Skipf("MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	src := jlmOfPair(t, path, mm)
	m, err := Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tg := m.tower
	v := m.container.Vision()
	switch {
	case tg == nil:
		t.Fatal("Open lowered no tower")
	case len(tg.blocks) != int(v.NLayer):
		t.Fatalf("%d block graphs for a tower of %d", len(tg.blocks), v.NLayer)
	case tg.rows != int(v.ImageSize/v.PatchSize)*int(v.ImageSize/v.PatchSize):
		t.Fatalf("the entry writes %d rows for %d patches of %d", tg.rows, v.ImageSize, v.PatchSize)
	case tg.out != tg.rows/int(v.Scale*v.Scale):
		t.Fatalf("the head writes %d rows, want %d shuffled by %d", tg.out, tg.rows, v.Scale)
	}
	s := m.NewInstance(64)
	defer s.Close()
	dev := s.composedCPU()
	reg := composer.Default()
	form := func(name string, g *graph.Graph, rows int) {
		t.Helper()
		_, _, err := reg.Form(composer.Request{Graph: g, Formats: m.towerFormats(g),
			Step: composer.Step{Rows: rows, Page: 64, Keys: rows, ActWindow: m.actWindow}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%s does not compose for the CPU: %v", name, err)
		}
	}
	for px, g := range tg.entry {
		if !slices.ContainsFunc(g.Ops, func(o graph.Op) bool { return o.Kind == graph.OpPicture && o.Attr.Pixels == px }) {
			t.Fatalf("the entry for format %d reads no picture of that format", px)
		}
		form("the entry", g, tg.rows)
	}
	for i, g := range tg.blocks {
		form(fmt.Sprintf("block %d", i), g, tg.rows)
	}
	form("the head", tg.head, tg.rows)
	t.Logf("%d entries, %d blocks over %d rows, a head of %d rows: verified and composed", len(tg.entry), len(tg.blocks), tg.rows, tg.out)

	// The violation: the head's matrix a tensor the container does not hold.
	saved := v.Tower.Head.Ops[2].Bind[0]
	v.Tower.Head.Ops[2].Bind[0] = jlm.RoleVProj2
	_, err = m.buildTowerGraphs()
	v.Tower.Head.Ops[2].Bind[0] = saved
	if err == nil || !strings.Contains(err.Error(), "does not hold") {
		t.Fatalf("a head binding a tensor the container lacks lowered: %v", err)
	}
}

// TestStatedTowerBlocksArePagedBlocks: the stated tower's blocks are blocks
// of the model's space after its text blocks, each paged in from its own page
// of the container's vision array with its matrices bound to the frame it
// landed in. Under a budget of one block, paging a second block evicts the
// first, and paging the first back binds its matrices again. The text half of
// the container answers a prompt as the text model alone does, bit for bit.
func TestStatedTowerBlocksArePagedBlocks(t *testing.T) {
	path := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf")
	mm := testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
	if !sourcePresent(path, mm) {
		t.Skipf("MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	src := jlmOfPair(t, path, mm)
	probe, err := Open(src)
	if err != nil {
		t.Fatal(err)
	}
	page := probe.container.PageBytes(probe.towerBase)
	probe.Close()
	m, err := Open(src, WithPageBudget(page))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	v := m.container.Vision()
	if m.towerBase != m.Cfg.NLayer+m.Cfg.NMTP || m.Blocks() != m.towerBase+int(v.NLayer) {
		t.Fatalf("the tower's %d blocks start at %d of %d, want after the %d text blocks", v.NLayer, m.towerBase, m.Blocks(), m.Cfg.NLayer)
	}
	bound := func(li int) error {
		l := &m.layers[li]
		if len(l.mats) == 0 {
			return fmt.Errorf("block %d holds no matrix", li)
		}
		for _, mt := range l.mats {
			qs, _, _ := m.container.Span(mt.t.e)
			if !sameSpan(mt.t.data, qs) || len(qs) == 0 {
				return fmt.Errorf("block %d's %v is not bound to its page", li, mt.role)
			}
		}
		return nil
	}
	a, b := m.towerBase, m.towerBase+1
	for _, li := range []int{a, b, a} {
		if err := m.pageIn(li); err != nil {
			t.Fatal(err)
		}
		if err := bound(li); err != nil {
			t.Fatal(err)
		}
		other := a + b - li
		if m.container.Resident(other) {
			t.Fatalf("under a budget of one block, block %d stayed resident after block %d paged in", other, li)
		}
	}

	text, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer text.Close()
	ids := text.Vocab.Encode("The capital of France is", true)
	ts := text.NewInstance(64)
	defer ts.Close()
	want, err := ts.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	full, err := Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer full.Close()
	fs := full.NewInstance(64)
	defer fs.Close()
	got, err := fs.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("logit %d is %v beside the tower and %v alone", i, got[i], want[i])
		}
	}
	t.Logf("%d tower blocks from %d, paged and rebound under one page; %d logits as the text model's", v.NLayer, m.towerBase, len(want))
}

// openSmolVLM is SmolVLM-256M's container and the picture the gates show it.
func openSmolVLM(t *testing.T) (*Model, image.Image) {
	t.Helper()
	path := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf")
	mm := testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
	if !sourcePresent(path, mm) {
		t.Skipf("MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	m, err := Open(jlmOfPair(t, path, mm))
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		m.Close()
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		m.Close()
		t.Fatal(err)
	}
	return m, img
}

// TestStatedTowerEncodesAPicture runs a picture through SmolVLM's stated
// tower on the CPU device: 64 rows of the language model's width, finite and
// not one row repeated, the same rows bit for bit when the picture is encoded
// again, other rows for another picture, and a warm encode allocates nothing.
func TestStatedTowerEncodesAPicture(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	s := m.NewInstance(256)
	defer s.Close()
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	v := m.container.Vision()
	if len(rows) != m.tower.out*int(v.ProjDim) || m.tower.out != 64 {
		t.Fatalf("%d floats in %d rows, want 64 rows of %d", len(rows), m.tower.out, v.ProjDim)
	}
	d := int(v.ProjDim)
	same := true
	for i, x := range rows {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatalf("value %d (row %d) is %v", i, i/d, x)
		}
		same = same && x == rows[i%d]
	}
	if same {
		t.Fatal("every row is the first: the picture's rows do not differ")
	}
	first := slices.Clone(rows)
	again, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	for i := range first {
		if math.Float32bits(again[i]) != math.Float32bits(first[i]) {
			t.Fatalf("value %d differs between two encodes of one picture: %v, %v", i, first[i], again[i])
		}
	}
	grey := image.NewGray(image.Rect(0, 0, 384, 384))
	for i := range grey.Pix {
		grey.Pix[i] = 128
	}
	other, err := s.EncodePicture(grey)
	if err != nil {
		t.Fatal(err)
	}
	diff := 0.0
	for i := range first {
		diff = max(diff, math.Abs(float64(other[i]-first[i])))
	}
	if diff < 1e-2 {
		t.Fatalf("a grey picture encodes to within %g of the coloured one", diff)
	}
	if n := testing.AllocsPerRun(3, func() {
		if _, err := s.EncodePicture(img); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Fatalf("a warm encode allocates %v times", n)
	}
	t.Logf("64 rows of %d, deterministic, %g from a grey picture's, no allocation warm", d, diff)
}

// TestStatedTowerSeesThePicture asks SmolVLM to describe a picture through
// its stated tower and the CPU device: the coloured quadrants and yellow disc
// are described with one of their colours, and a blank grey picture with
// none of them -- a tower wired to nothing still answers fluently, so the
// blank is the violation. The prompt is SmolVLM's own template's, its
// picture's markers the container's (jlm.TowerSpan), laid out by ChatSpans.
func TestStatedTowerSeesThePicture(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	seesThePicture(t, m, img)
}

// seesThePicture asks m to describe img and a blank grey picture in one
// sentence, each through ChatSpans and a fresh instance: the picture's
// description names one of its colours -- the quadrants' red, green and
// blue, the disc's yellow -- and the blank's names none of them.
func seesThePicture(t *testing.T, m *Model, img image.Image) {
	t.Helper()
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	grey := image.NewGray(image.Rect(0, 0, 384, 384))
	for i := range grey.Pix {
		grey.Pix[i] = 128
	}
	got, none := describePicture(t, m, img, 32), describePicture(t, m, grey, 32)
	t.Logf("picture %q", got)
	t.Logf("blank   %q", none)
	if !namesAColour(got) {
		t.Errorf("the description names none of the picture's colours: %q", got)
	}
	if namesAColour(none) {
		t.Errorf("a blank picture has the picture's colours too (%q): the picture is not what answered", none)
	}
}

// namesAColour reports whether a description names one of the test
// picture's colours.
func namesAColour(s string) bool {
	for _, c := range []string{"red", "green", "blue", "yellow"} {
		if strings.Contains(s, c) {
			return true
		}
	}
	return false
}

// describePicture is m's greedy description of pic in one sentence, n tokens
// at most, lower-cased.
func describePicture(t *testing.T, m *Model, pic image.Image, n int) string {
	t.Helper()
	pspan, err := m.PictureSpan(pic)
	if err != nil {
		t.Fatal(err)
	}
	spans, err := m.ChatSpans([]ChatMessage{{Role: "user", Content: "Describe this image in one sentence.", Images: 1}},
		[]Span{pspan}, true)
	if err != nil {
		t.Fatal(err)
	}
	s := m.NewInstance(SpanPositions(spans, m.Cfg.NEmbd) + n + 8)
	defer s.Close()
	lg, err := s.PrefillMixed(spans...)
	if err != nil {
		t.Fatal(err)
	}
	var out []int32
	for range n {
		tok := int32(argmax(lg))
		if m.Vocab.IsEOG(tok) {
			break
		}
		out = append(out, tok)
		if lg, err = s.Forward(tok); err != nil {
			t.Fatal(err)
		}
	}
	return strings.ToLower(m.Vocab.Decode(out))
}

// TestGemma3TowerSeesThePicture runs Gemma 3 4B's stated tower -- Pillow's
// bilinear resize, SigLIP's 27 blocks over 4096 patches, the pooled
// projector -- on the CPU device, the picture's 256 rows attending both ways
// in the text model: its colours are named, and a blank picture's are not.
func TestGemma3TowerSeesThePicture(t *testing.T) {
	text := testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf")
	mm := testmodels.Path("vlm/mmproj-gemma-3-4b-it-f16.gguf")
	if !sourcePresent(text, mm) {
		t.Skipf("MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.SeesPictures() || m.tower.out != 256 || len(m.tower.blocks) != 27 {
		t.Fatalf("the tower writes %d rows over %d blocks, want 256 over 27", m.tower.out, len(m.tower.blocks))
	}
	f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	seesThePicture(t, m, img)
}

// flatSpans lays spans out as one id sequence with -1 where a picture goes,
// so a whole prompt is one comparison.
func flatSpans(spans []Span) []int32 {
	var out []int32
	for _, sp := range spans {
		if sp.Embd != nil {
			out = append(out, -1)
			continue
		}
		out = append(out, sp.Tokens...)
	}
	return out
}

// TestChatSpansFollowTheTemplate pins the picture's marker sequence, which no
// numerical gate sees: with every marker wrong the picture still arrives,
// unannounced. The markers are mtmd's (tools/mtmd/mtmd.cpp at b10825), which
// the converter states in the container:
//
//	tok_ov_img_start = {"\n\n", "<fake_token_around_image>", "<global-img>"}
//	tok_ov_img_end   = {"<fake_token_around_image>"}
//
// and the turn is SmolVLM's own template's, which departs from mtmd by one
// token on purpose: it writes "User:" before an image-led turn, where mtmd
// emits "User: ". The placeholder ("<image>") is read off the template.
func TestChatSpansFollowTheTemplate(t *testing.T) {
	m, _ := openSmolVLM(t)
	defer m.Close()
	rows := make([]float32, m.tower.out*m.Cfg.NEmbd)
	one := []ChatMessage{{Role: "user", Content: "Describe the image.", Images: 1}}
	spans, err := m.ChatSpans(one, []Span{{Embd: rows}}, true)
	if err != nil {
		t.Fatal(err)
	}
	//                <|im_start|> User  :   \n\n  <fake…> <global-img> IMG <fake…>
	want := []int32{1, 11126, 42, 1116, 49189, 49152, -1, 49189,
		// Describe  the  image  .  <end_of_utterance>  \n  Ass  istant  :
		37964, 260, 2443, 30, 49279, 198, 9519, 9531, 42}
	if got := flatSpans(spans); !slices.Equal(got, want) {
		t.Fatalf("spans\n  got  %v\n  want %v", got, want)
	}
	for _, sp := range spans {
		if sp.Embd != nil && &sp.Embd[0] != &rows[0] {
			t.Error("the picture's span copied its rows instead of referring to them")
		}
	}
	if got, want := SpanPositions(spans, m.Cfg.NEmbd), len(want)-1+m.tower.out; got != want {
		t.Errorf("SpanPositions %d, want %d", got, want)
	}
	// The refusals, each of which would otherwise put a picture on the wrong
	// turn or none at all.
	if _, err := m.ChatSpans(one, nil, true); err == nil {
		t.Error("one image and no picture was accepted")
	}
	lit := []ChatMessage{{Role: "user", Content: "what is <image>?", Images: 1}}
	if _, err := m.ChatSpans(lit, []Span{{Embd: rows}}, true); err == nil {
		t.Error("user text holding the placeholder shifted the picture count and was accepted")
	}
	if _, err := m.ChatSpans(one, []Span{{Embd: rows[:len(rows)-1]}}, true); err == nil {
		t.Error("a picture of a ragged row was accepted")
	}
	if _, err := m.ChatIDs(one, true); err == nil {
		t.Error("ChatIDs tokenized an image placeholder as text")
	}
	if _, err := m.ChatPrompt(one, true); err == nil {
		t.Error("ChatPrompt rendered an image part as text")
	}
}

// nrgbaOf is img's pixels as an NRGBA picture: YCbCr through
// image/color.YCbCrToRGB of its planes, a palette's colours straight.
func nrgbaOf(img image.Image) *image.NRGBA {
	b := img.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	yc, _ := img.(*image.YCbCr)
	if a, ok := img.(*image.NYCbCrA); ok {
		yc = &a.YCbCr
	}
	for y := range b.Dy() {
		for x := range b.Dx() {
			X, Y := b.Min.X+x, b.Min.Y+y
			var c color.NRGBA
			if yc != nil {
				i := yc.COffset(X, Y)
				c.R, c.G, c.B = color.YCbCrToRGB(yc.Y[yc.YOffset(X, Y)], yc.Cb[i], yc.Cr[i])
			} else {
				c = color.NRGBAModel.Convert(img.At(X, Y)).(color.NRGBA)
			}
			c.A = 255
			out.SetNRGBA(x, y, c)
		}
	}
	return out
}

// TestPictureFormatsEncodeAsTheirPixels encodes a picture in each sample
// format a decoder leaves -- a JPEG's YCbCr at 4:2:0, a sub-picture of one at
// an odd corner, YCbCr at 4:4:4, YCbCr with alpha, a paletted one -- and its
// pixels as an NRGBA picture (nrgbaOf), and holds the two to the same rows
// bit for bit: the planes, their offsets and the chroma's phase are laid out
// as the picture op reads them. A warm encode of a copied layout allocates
// nothing.
func TestPictureFormatsEncodeAsTheirPixels(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	s := m.NewInstance(256)
	defer s.Close()
	jpg := func(sub bool) image.Image {
		var b bytes.Buffer
		if err := jpeg.Encode(&b, img, &jpeg.Options{Quality: 90}); err != nil {
			t.Fatal(err)
		}
		d, err := jpeg.Decode(&b)
		if err != nil {
			t.Fatal(err)
		}
		if sub {
			r := d.Bounds()
			d = d.(*image.YCbCr).SubImage(image.Rect(r.Min.X+3, r.Min.Y+5, r.Max.X-2, r.Max.Y-1))
		}
		return d
	}
	full := image.NewYCbCr(img.Bounds(), image.YCbCrSubsampleRatio444)
	draw.Draw(&ycbcrDst{full}, img.Bounds(), img, image.Point{}, draw.Src)
	withA := &image.NYCbCrA{YCbCr: *full, A: make([]byte, len(full.Y)), AStride: full.YStride}
	pal := image.NewPaletted(img.Bounds(), palette.WebSafe)
	draw.Draw(pal, img.Bounds(), img, image.Point{}, draw.Src)
	for _, c := range []struct {
		name string
		img  image.Image
	}{{"jpeg 4:2:0", jpg(false)}, {"jpeg 4:2:0 at an odd corner", jpg(true)},
		{"ycbcr 4:4:4", full}, {"ycbcr with alpha", withA}, {"paletted", pal}} {
		got, err := s.EncodePicture(c.img)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got = slices.Clone(got)
		want, err := s.EncodePicture(nrgbaOf(c.img))
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: value %d is %v, and %v from its pixels as NRGBA", c.name, i, got[i], want[i])
			}
		}
		if n := testing.AllocsPerRun(2, func() {
			if _, err := s.EncodePicture(c.img); err != nil {
				t.Fatal(err)
			}
		}); n != 0 {
			t.Fatalf("%s: a warm encode allocates %v times", c.name, n)
		}
		t.Logf("%s (%T): %d values as its pixels' NRGBA, no allocation warm", c.name, c.img, len(want))
	}
}

// ycbcrDst draws into a 4:4:4 YCbCr picture through image/color's own
// RGB-to-YCbCr conversion.
type ycbcrDst struct{ *image.YCbCr }

func (d *ycbcrDst) Set(x, y int, c color.Color) {
	r, g, b, _ := c.RGBA()
	yy, cb, cr := color.RGBToYCbCr(uint8(r>>8), uint8(g>>8), uint8(b>>8))
	d.Y[d.YOffset(x, y)], d.Cb[d.COffset(x, y)], d.Cr[d.COffset(x, y)] = yy, cb, cr
}

// TestStatedTowerRunsOnTheCard places an instance of SmolVLM on each device,
// which offers its tower run every tower block: all 12 are placed there, the
// picture's rows sit within the reduction-order band of the CPU device's
// (an NMSE of 1e-3 over the 64 rows the head writes), and the description
// still names the yellow disc where a blank grey picture's does not. A half
// seam -- the tower's first 6 blocks on the card, the rest on the CPU device
// through the prefill -- sits in the band too.
func TestStatedTowerRunsOnTheCard(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	host := m.NewInstance(256)
	want, err := host.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	want = slices.Clone(want)
	host.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, dev, err)
			}
			defer g.Close()
			s := m.NewInstance(256)
			defer s.Close()
			s.SetDeviceFallback(false)
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			ts := s.tower.s
			if n := ts.GPULayers(); n != len(m.tower.blocks) {
				t.Fatalf("%d of the tower's %d blocks placed on %s: %s", n, len(m.tower.blocks), dev, g.Err())
			}
			half := m.NewInstance(256)
			defer half.Close()
			half.SetDeviceFallback(false)
			if err := half.tower.s.SetDeviceLayers(g, len(m.tower.blocks)/2); err != nil {
				t.Fatal(err)
			}
			if n := half.tower.s.GPULayers(); n != len(m.tower.blocks)/2 {
				t.Fatalf("%d of the tower's %d blocks placed on %s for a half seam", n, len(m.tower.blocks), dev)
			}
			hgot, err := half.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			var hn, hd float64
			for i := range want {
				d := float64(hgot[i] - want[i])
				hn, hd = hn+d*d, hd+float64(want[i])*float64(want[i])
			}
			if e := hn / hd; math.IsNaN(e) || e > 1e-3 {
				t.Fatalf("the picture's rows over a half seam on %s sit at NMSE %.3e from the CPU device's", dev, e)
			}
			t.Logf("%s: a half seam (%d tower blocks on the card) at NMSE %.3e", dev, len(m.tower.blocks)/2, hn/hd)
			got, err := s.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			var num, den float64
			for i := range want {
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
					t.Fatalf("value %d is %v", i, got[i])
				}
				d := float64(got[i] - want[i])
				num, den = num+d*d, den+float64(want[i])*float64(want[i])
			}
			if e := num / den; e > 1e-3 {
				t.Fatalf("the picture's rows on %s sit at NMSE %.3e from the CPU device's", dev, e)
			}
			ask := func(pic image.Image) string {
				t.Helper()
				rows, err := s.EncodePicture(pic)
				if err != nil {
					t.Fatal(err)
				}
				spans, err := m.ChatSpans([]ChatMessage{{Role: "user", Content: "Describe this image in one sentence.",
					Images: 1}}, []Span{{Embd: rows}}, true)
				if err != nil {
					t.Fatal(err)
				}
				s.Reset()
				lg, err := s.PrefillMixed(spans...)
				if err != nil {
					t.Fatal(err)
				}
				var out []int32
				for range 32 {
					tok := int32(argmax(lg))
					if m.Vocab.IsEOG(tok) {
						break
					}
					out = append(out, tok)
					if lg, err = s.Forward(tok); err != nil {
						t.Fatal(err)
					}
				}
				return strings.ToLower(m.Vocab.Decode(out))
			}
			grey := image.NewGray(image.Rect(0, 0, 384, 384))
			for i := range grey.Pix {
				grey.Pix[i] = 128
			}
			pic, none := ask(img), ask(grey)
			t.Logf("%s: %d tower blocks placed; NMSE %.3e from the CPU device's rows", dev, ts.GPULayers(), num/den)
			t.Logf("picture %q", pic)
			t.Logf("blank   %q", none)
			if !strings.Contains(pic, "yellow") {
				t.Errorf("the description does not name the yellow disc: %q", pic)
			}
			if strings.Contains(none, "yellow") {
				t.Errorf("a blank picture is yellow too (%q): the picture is not what answered", none)
			}
		})
	}
}

// TestLlavaTowerSeesThePicture runs llava-phi-3-mini's stated tower --
// Pillow's bicubic resize of the shorter side and its centre, CLIP's 23
// blocks over a class row and 576 patches, the two-layer projector -- on the
// CPU device, the picture's 576 rows where Phi-3's string template carries
// "<image>": its colours are named, and a blank picture's are not.
func TestLlavaTowerSeesThePicture(t *testing.T) {
	text := testmodels.Path("llava-phi-3-mini-int4.gguf")
	mm := testmodels.Path("llava-phi-3-mini-mmproj-f16.gguf")
	if !sourcePresent(text, mm) {
		t.Skipf("MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.SeesPictures() || m.tower.out != 576 || m.tower.rows != 577 || len(m.tower.blocks) != 23 || m.tower.fit.short != 336 {
		t.Fatalf("the tower writes %d rows from %d over %d blocks, the shorter side to %d; want 576 from 577 over 23, 336",
			m.tower.out, m.tower.rows, len(m.tower.blocks), m.tower.fit.short)
	}
	f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	seesThePicture(t, m, img)
}

// TestTileGridFollowsTheReference holds tileGrid to InternVL's
// find_closest_aspect_ratio (transformers' get_optimal_tiled_canvas), each
// grid worked by hand from the reference's rule over 448-pixel tiles, 1 to
// 12 of them: the closest aspect, a tie to the larger grid only while the
// picture covers more than half of it.
func TestTileGridFollowsTheReference(t *testing.T) {
	for _, c := range []struct{ w, h, cols, rows int }{
		{384, 384, 1, 1},  // (2,2) ties at 0 and is more than twice the picture
		{800, 600, 4, 3},  // 4:3 exactly, after (2,2) took the tie with (1,1)
		{1000, 500, 2, 1}, // (4,2) ties at 0 and is more than twice the picture
		{2000, 1000, 4, 2},
		{448, 896, 1, 2},
		{640, 480, 4, 3},
		{100, 1000, 1, 10}, // 1:10 exactly, before (1,11) and (1,12)
	} {
		if cols, rows := tileGrid(c.w, c.h, 448, 1, 12); cols != c.cols || rows != c.rows {
			t.Errorf("%dx%d: a grid of %dx%d, want %dx%d", c.w, c.h, cols, rows, c.cols, c.rows)
		}
	}
	f := pictureFit{minTiles: 1, maxTiles: 12, thumb: true}
	for _, c := range []struct{ w, h, n int }{{384, 384, 1}, {1000, 500, 3}, {800, 600, 13}} {
		if _, _, n := f.tiles(c.w, c.h, 448); n != c.n {
			t.Errorf("%dx%d: %d squares, want %d (a thumbnail after a grid of more than one)", c.w, c.h, n, c.n)
		}
	}
	if ow, oh, x0, y0 := f.window(1000, 500, 448, 448, 1); ow != 896 || oh != 448 || x0 != 448 || y0 != 0 {
		t.Errorf("the second tile of 1000x500 is (%d, %d) of %dx%d, want (448, 0) of 896x448", x0, y0, ow, oh)
	}
	if ow, oh, x0, y0 := f.window(1000, 500, 448, 448, 2); ow != 448 || oh != 448 || x0 != 0 || y0 != 0 {
		t.Errorf("1000x500's thumbnail is (%d, %d) of %dx%d, want the whole picture at 448x448", x0, y0, ow, oh)
	}
}

// TestInternVLTowerSeesThePicture runs InternVL3-1B's stated tower --
// Pillow's bicubic to 448-pixel tiles, InternViT's 24 blocks over a class
// row and 1024 patches, the shuffled projector -- on the CPU device: a
// square picture is one tile of 256 rows, and the same picture beside an
// as wide grey field is two tiles and a thumbnail, 768 rows; each is
// described naming its colours, and a blank picture is not.
func TestInternVLTowerSeesThePicture(t *testing.T) {
	text := testmodels.Path("vlm/InternVL3-1B-Instruct-Q8_0.gguf")
	mm := testmodels.Path("vlm/mmproj-InternVL3-1B-Instruct-Q8_0.gguf")
	if !sourcePresent(text, mm) {
		t.Skipf("MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.SeesPictures() || m.tower.out != 256 || m.tower.rows != 1025 || len(m.tower.blocks) != 24 || m.tower.fit.maxTiles != 12 {
		t.Fatalf("the tower writes %d rows from %d over %d blocks, %d tiles at most; want 256 from 1025 over 24, 12",
			m.tower.out, m.tower.rows, len(m.tower.blocks), m.tower.fit.maxTiles)
	}
	f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	seesThePicture(t, m, img)
	b := img.Bounds()
	wide := image.NewRGBA(image.Rect(0, 0, 2*b.Dx(), b.Dy()))
	draw.Draw(wide, wide.Bounds(), image.NewUniform(color.Gray{128}), image.Point{}, draw.Src)
	draw.Draw(wide, b, img, b.Min, draw.Src)
	sp, err := m.PictureSpan(wide)
	if err != nil {
		t.Fatal(err)
	}
	if sp.PicRows != 3*256 {
		t.Fatalf("a %dx%d picture takes %d rows, want two tiles and a thumbnail's 768", wide.Bounds().Dx(), b.Dy(), sp.PicRows)
	}
	got := describePicture(t, m, wide, 32)
	t.Logf("wide    %q", got)
	if !namesAColour(got) {
		t.Errorf("the wide picture's description names none of its colours: %q", got)
	}
}

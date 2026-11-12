package engine

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

	"github.com/jitllm/jitllm/model"

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
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	src := jlmOfPair(t, path, mm)
	m, err := model.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tg := m.Tower
	v := m.Container.Vision()
	switch {
	case tg == nil:
		t.Fatal("Open lowered no tower")
	case len(tg.Blocks) != int(v.NLayer):
		t.Fatalf("%d block graphs for a tower of %d", len(tg.Blocks), v.NLayer)
	case len(tg.Classes) != 1 || tg.Rows != int(v.ImageSize/v.PatchSize)*int(v.ImageSize/v.PatchSize):
		t.Fatalf("the entry writes %d rows in %d classes for %d patches of %d", tg.Rows, len(tg.Classes), v.ImageSize, v.PatchSize)
	case tg.Out-tg.Marks != tg.Rows/int(v.Scale*v.Scale):
		t.Fatalf("the head writes %d rows beside %d marker rows, want %d shuffled by %d", tg.Out-tg.Marks, tg.Marks, tg.Rows, v.Scale)
	case tg.Marks == 0:
		t.Fatal("the head writes no marker rows: Idefics3's fake image and tile markers are the tower's rows")
	}
	s := testInstance(t, m, 64)
	defer s.Close()
	dev := execOf(s).Composer().Device()
	reg := composer.Default()
	form := func(name string, g *graph.Graph, rows int) {
		t.Helper()
		_, _, err := reg.Form(composer.Request{Graph: g, Formats: m.TowerFormats(g),
			Step: composer.Step{Rows: rows, Page: 64, Keys: rows, ActWindow: m.ActWindow}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("%s does not compose for the CPU: %v", name, err)
		}
	}
	for px, g := range tg.Classes[0].Entry {
		if !slices.ContainsFunc(g.Ops, func(o graph.Op) bool { return o.Kind == graph.OpPicture && o.Attr.Pixels == px }) {
			t.Fatalf("the entry for format %d reads no picture of that format", px)
		}
		form("the entry", g, tg.Rows)
	}
	for i, g := range tg.Blocks {
		form(fmt.Sprintf("block %d", i), g, tg.Rows)
	}
	form("the head", tg.Head, tg.Rows)
	t.Logf("%d entries, %d blocks over %d rows, a head of %d rows: verified and composed", len(tg.Classes[0].Entry), len(tg.Blocks), tg.Rows, tg.Out)

	// The violation: the head's matrix a tensor the container does not hold.
	saved := v.Tower.Head.Ops[2].Bind[0]
	v.Tower.Head.Ops[2].Bind[0] = jlm.RoleVProj2
	_, err = m.BuildTowerGraphs()
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
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	src := jlmOfPair(t, path, mm)
	probe, err := model.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	page := probe.Container.PageBytes(probe.TowerBase)
	probe.Close()
	m, err := model.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	v := m.Container.Vision()
	if m.TowerBase != m.Cfg.NLayer+m.Cfg.NMTP || m.Blocks() != m.TowerBase+int(v.NLayer) {
		t.Fatalf("the tower's %d blocks start at %d of %d, want after the %d text blocks", v.NLayer, m.TowerBase, m.Blocks(), m.Cfg.NLayer)
	}
	fr := testFrames(t, m, jlm.Budget(page))
	bound := func(li int) error {
		l := &m.Layers[li]
		if len(l.Mats) == 0 {
			return fmt.Errorf("block %d holds no matrix", li)
		}
		for _, mt := range l.Mats {
			qs, _, _ := fr.Span(mt.T.E)
			if !model.SameSpan(mt.T.Data, qs) || len(qs) == 0 {
				return fmt.Errorf("block %d's %v is not bound to its page", li, mt.Role)
			}
		}
		return nil
	}
	a, b := m.TowerBase, m.TowerBase+1
	for _, li := range []int{a, b, a} {
		if err := m.PageIn(fr, li, &m.Layers[li]); err != nil {
			t.Fatal(err)
		}
		if err := bound(li); err != nil {
			t.Fatal(err)
		}
		other := a + b - li
		if fr.Resident(other) {
			t.Fatalf("under a budget of one block, block %d stayed resident after block %d paged in", other, li)
		}
	}

	text, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer text.Close()
	ids := text.Vocab.Encode("The capital of France is", true)
	ts := testInstance(t, text, 64)
	defer ts.Close()
	want, err := ts.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	full, err := model.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer full.Close()
	fs := testInstance(t, full, 64)
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
	t.Logf("%d tower blocks from %d, paged and rebound under one page; %d logits as the text model's", v.NLayer, m.TowerBase, len(want))
}

// openSmolVLM is SmolVLM-256M's container and the picture the gates show it.
func openSmolVLM(t *testing.T) (*model.Model, image.Image) {
	t.Helper()
	path := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf")
	mm := testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
	if !sourcePresent(path, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	m, err := model.Open(jlmOfPair(t, path, mm))
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
// tower on the CPU device: a 384-pixel square cut as Idefics3's processor
// cuts it, 16 tiles and the global view of 64 rows each with their markers
// -- 1127 rows of the language model's width -- finite and not one row
// repeated, the same rows bit for bit when the picture is encoded again,
// other rows for another picture, and a warm encode allocates nothing.
func TestStatedTowerEncodesAPicture(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	s := testInstance(t, m, 256)
	defer s.Close()
	rows, err := s.EncodePicture(img)
	if err != nil {
		t.Fatal(err)
	}
	v := m.Container.Vision()
	b := img.Bounds()
	n, err := m.Tower.PicRows(b.Dx(), b.Dy())
	// 16 tiles of a marker pair and 64 rows, a newline after three rows'
	// last, the global view's three markers, 64 rows and one.
	const want = 16*(2+64) + 3 + 3 + 64 + 1
	if err != nil || n != want || len(rows) != n*int(v.ProjDim) {
		t.Fatalf("%d floats, %d rows (%v), want %d rows of %d", len(rows), n, err, want, v.ProjDim)
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
	t.Logf("%d rows of %d, deterministic, %g from a grey picture's, no allocation warm", n, d, diff)
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
func seesThePicture(t *testing.T, m *model.Model, img image.Image, io ...Option) {
	t.Helper()
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	grey := image.NewGray(image.Rect(0, 0, 384, 384))
	for i := range grey.Pix {
		grey.Pix[i] = 128
	}
	got, none := describePicture(t, m, img, 32, io...), describePicture(t, m, grey, 32, io...)
	t.Logf("picture %q", got)
	t.Logf("blank   %q", none)
	// A model may make a colour up for a blank picture (Qwen2-VL-2B's "a
	// person wearing a red shirt"); the picture's description names more of
	// its colours than the blank's does, one at least (SmolVLM-256M names
	// the disc alone).
	if n := colours(got); n < 1 || n <= colours(none) {
		t.Errorf("the description names %d of the picture's colours, the blank's %d: %q", n, colours(none), got)
	}
}

// colours is how many of the test picture's colours a description names.
func colours(s string) int {
	n := 0
	for _, c := range []string{"red", "green", "blue", "yellow"} {
		if strings.Contains(s, c) {
			n++
		}
	}
	return n
}

// describePicture is m's greedy description of pic in one sentence, n tokens
// at most, lower-cased.
func describePicture(t *testing.T, m *model.Model, pic image.Image, n int, io ...Option) string {
	t.Helper()
	pspan, err := m.PictureSpan(pic)
	if err != nil {
		t.Fatal(err)
	}
	spans, err := m.ChatSpans([]model.ChatMessage{{Role: "user", Content: "Describe this image in one sentence.", Images: 1}},
		[]model.Span{pspan}, true)
	if err != nil {
		t.Fatal(err)
	}
	s := testInstance(t, m, model.SpanPositions(spans, m.Cfg.NEmbd)+n+8, io...)
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
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := model.Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.SeesPictures() || m.Tower.Out != 256 || len(m.Tower.Blocks) != 27 {
		t.Fatalf("the tower writes %d rows over %d blocks, want 256 over 27", m.Tower.Out, len(m.Tower.Blocks))
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
func flatSpans(spans []model.Span) []int32 {
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

// TestChatSpansFollowTheTemplate pins the turn around a picture, which no
// numerical gate sees: SmolVLM's own template writes "User:" before an
// image-led turn and the picture where its placeholder ("<image>", read off
// the template) stands. The picture's markers are its own rows, the head's
// (TOpMarkRow), so its span is one run between the text's tokens.
func TestChatSpansFollowTheTemplate(t *testing.T) {
	m, _ := openSmolVLM(t)
	defer m.Close()
	rows := make([]float32, m.Tower.Out*m.Cfg.NEmbd)
	one := []model.ChatMessage{{Role: "user", Content: "Describe the image.", Images: 1}}
	spans, err := m.ChatSpans(one, []model.Span{{Embd: rows}}, true)
	if err != nil {
		t.Fatal(err)
	}
	//                <|im_start|> User  :  IMG
	want := []int32{1, 11126, 42, -1,
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
	if got, want := model.SpanPositions(spans, m.Cfg.NEmbd), len(want)-1+m.Tower.Out; got != want {
		t.Errorf("SpanPositions %d, want %d", got, want)
	}
	// The refusals, each of which would otherwise put a picture on the wrong
	// turn or none at all.
	if _, err := m.ChatSpans(one, nil, true); err == nil {
		t.Error("one image and no picture was accepted")
	}
	lit := []model.ChatMessage{{Role: "user", Content: "what is <image>?", Images: 1}}
	if _, err := m.ChatSpans(lit, []model.Span{{Embd: rows}}, true); err == nil {
		t.Error("user text holding the placeholder shifted the picture count and was accepted")
	}
	if _, err := m.ChatSpans(one, []model.Span{{Embd: rows[:len(rows)-1]}}, true); err == nil {
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
	cm, _ := img.(*image.CMYK)
	for y := range b.Dy() {
		for x := range b.Dx() {
			X, Y := b.Min.X+x, b.Min.Y+y
			var c color.NRGBA
			if yc != nil {
				i := yc.COffset(X, Y)
				c.R, c.G, c.B = color.YCbCrToRGB(yc.Y[yc.YOffset(X, Y)], yc.Cb[i], yc.Cr[i])
			} else if cm != nil {
				// Pillow's convert("RGB"), which image/color's CMYKToRGB
				// rounds otherwise.
				p := cm.Pix[cm.PixOffset(X, Y):]
				nk := 255 - uint32(p[3])
				ink := func(v uint8) uint8 {
					t := uint32(v)*nk + 128
					return uint8(nk - ((t>>8)+t)>>8)
				}
				c.R, c.G, c.B = ink(p[0]), ink(p[1]), ink(p[2])
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
// an odd corner, YCbCr at 4:4:4, YCbCr with alpha, a paletted one, 16-bit
// RGBA, NRGBA and gray, whose low bytes are noise the op must not read, and
// CMYK with a key that varies, made colours as Pillow does -- and its
// pixels as an NRGBA picture (nrgbaOf), and holds the two to the same rows
// bit for bit: the planes, their offsets and the chroma's phase are laid out
// as the picture op reads them. A warm encode of a copied layout allocates
// nothing.
func TestPictureFormatsEncodeAsTheirPixels(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	s := testInstance(t, m, 256)
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
	b := img.Bounds()
	rgba64, nrgba64, gray16, cmyk := image.NewRGBA64(b), image.NewNRGBA64(b), image.NewGray16(b), image.NewCMYK(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA)
			wide := func(v uint8) uint16 { return uint16(v)<<8 | uint16(v^0x5a) }
			rgba64.SetRGBA64(x, y, color.RGBA64{wide(c.R), wide(c.G), wide(c.B), 0xffff})
			nrgba64.SetNRGBA64(x, y, color.NRGBA64{wide(c.R), wide(c.G), wide(c.B), 0xffff})
			gray16.SetGray16(x, y, color.Gray16{wide(c.G)})
			cmyk.SetCMYK(x, y, color.CMYK{255 - c.R, 255 - c.G, 255 - c.B, uint8(x*7+y*3) % 96})
		}
	}
	for _, c := range []struct {
		name string
		img  image.Image
	}{{"jpeg 4:2:0", jpg(false)}, {"jpeg 4:2:0 at an odd corner", jpg(true)},
		{"ycbcr 4:4:4", full}, {"ycbcr with alpha", withA}, {"paletted", pal},
		{"rgba 16-bit", rgba64}, {"nrgba 16-bit", nrgba64}, {"gray 16-bit", gray16}, {"cmyk", cmyk}} {
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

// TestANarrowTowerRunsOnTheCard encodes a picture through synth-janus's
// tower, whose largest grid (16 rows) is narrower than a composed history
// page, with every tower block on each card: the rows sit within the
// reduction-order band of the CPU device's. A pair allocated for its 17
// positions while its parts wrote keys at a page's stride put them 7e-2 to
// 1.9e-1 away and corrupted the process's heap.
func TestANarrowTowerRunsOnTheCard(t *testing.T) {
	path, mm := testmodels.Path("synth-janus.gguf"), testmodels.Path("mmproj-synth-janus.gguf")
	if !sourcePresent(path, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	m, err := model.Open(jlmOfPair(t, path, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Tower.Rows >= 64 {
		t.Fatalf("the tower's largest grid is %d rows: no narrower than a page, this gate proves nothing", m.Tower.Rows)
	}
	rgb := make([]byte, 3*64*64)
	for i := range rgb {
		rgb[i] = byte(i*37 + i/97*11)
	}
	pic := rgbPicture(rgb, 64, 64)
	host := testInstance(t, m, 128)
	want, err := host.EncodePicture(pic)
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
			s := testInstance(t, m, 128)
			defer s.Close()
			s.SetDeviceFallback(false)
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if n := s.tower.s.GPULayers(); n != len(m.Tower.Blocks) {
				t.Fatalf("%d of the tower's %d blocks placed on %s: %s", n, len(m.Tower.Blocks), dev, g.Err())
			}
			for range 3 {
				got, err := s.EncodePicture(pic)
				if err != nil {
					t.Fatal(err)
				}
				if n, w := nmseOf(got, want); math.IsNaN(n) || n > 1e-3 {
					t.Fatalf("the picture's rows on %s sit at NMSE %.3e from the CPU device's (max|d| %.3g)", dev, n, w)
				} else {
					t.Logf("%s: the %d-row tower's rows at NMSE %.3e from the CPU device's", dev, m.Tower.Rows, n)
				}
			}
		})
	}
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
	host := testInstance(t, m, 256)
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
			s := testInstance(t, m, 2048)
			defer s.Close()
			s.SetDeviceFallback(false)
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			ts := s.tower.s
			if n := ts.GPULayers(); n != len(m.Tower.Blocks) {
				t.Fatalf("%d of the tower's %d blocks placed on %s: %s", n, len(m.Tower.Blocks), dev, g.Err())
			}
			half := testInstance(t, m, 256)
			defer half.Close()
			half.SetDeviceFallback(false)
			if err := half.tower.s.SetDeviceLayers(g, len(m.Tower.Blocks)/2); err != nil {
				t.Fatal(err)
			}
			if n := half.tower.s.GPULayers(); n != len(m.Tower.Blocks)/2 {
				t.Fatalf("%d of the tower's %d blocks placed on %s for a half seam", n, len(m.Tower.Blocks), dev)
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
			t.Logf("%s: a half seam (%d tower blocks on the card) at NMSE %.3e", dev, len(m.Tower.Blocks)/2, hn/hd)
			got, err := s.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			if st := g.Stats(); st.UnformedAtStep != 0 {
				t.Fatalf("%d of the tower's parts formed at a step on %s, not at the placement", st.UnformedAtStep, dev)
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
				spans, err := m.ChatSpans([]model.ChatMessage{{Role: "user", Content: "Describe this image in one sentence.",
					Images: 1}}, []model.Span{{Embd: rows}}, true)
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
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := model.Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.SeesPictures() || m.Tower.Out != 576 || m.Tower.Rows != 577 || len(m.Tower.Blocks) != 23 {
		t.Fatalf("the tower writes %d rows from %d over %d blocks; want 576 from 577 over 23",
			m.Tower.Out, m.Tower.Rows, len(m.Tower.Blocks))
	}
	if sq, _ := squareOf(t, m.Tower, 1000, 500, 0); sq.Oh != 336 || sq.Pw != 336 || sq.Ph != 336 {
		t.Fatalf("a 1000x500 picture is resized to %dx%d and cropped to %dx%d; want its shorter side 336 and a 336 crop",
			sq.Ow, sq.Oh, sq.Pw, sq.Ph)
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
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := model.Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.SeesPictures() || m.Tower.Out != 256 || m.Tower.Rows != 1025 || len(m.Tower.Blocks) != 24 {
		t.Fatalf("the tower writes %d rows from %d over %d blocks; want 256 from 1025 over 24",
			m.Tower.Out, m.Tower.Rows, len(m.Tower.Blocks))
	}
	if _, n := squareOf(t, m.Tower, 13*448, 448, 0); n != 13 {
		t.Fatalf("a picture 13 tiles wide is %d squares; want 12 tiles at most and the thumbnail", n)
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
	if colours(got) < 1 {
		t.Errorf("the wide picture's description names %d of its colours: %q", colours(got), got)
	}
}

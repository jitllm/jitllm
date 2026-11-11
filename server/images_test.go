package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// The pictures of a chat request, without a model: what each shim accepts,
// what it refuses and how it names the part, and that what it accepts reaches
// the generate. The real-model gates are in images_model_test.go.

func testPNG(t testing.TB, w, h int, c color.Color) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func readTestdata(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// hugePNG is a PNG whose header claims w x h and which carries no pixel data:
// a decoder that sizes its raster before checking the header would allocate
// w*h*4 bytes for it.
func hugePNG(w, h uint32) []byte {
	var b bytes.Buffer
	b.WriteString("\x89PNG\r\n\x1a\n")
	chunk := func(typ string, data []byte) {
		binary.Write(&b, binary.BigEndian, uint32(len(data)))
		c := crc32.NewIEEE()
		c.Write([]byte(typ))
		c.Write(data)
		b.WriteString(typ)
		b.Write(data)
		binary.Write(&b, binary.BigEndian, c.Sum32())
	}
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:], w)
	binary.BigEndian.PutUint32(ihdr[4:], h)
	ihdr[8], ihdr[9] = 8, 6 // 8-bit RGBA
	chunk("IHDR", ihdr)
	chunk("IEND", nil)
	return b.Bytes()
}

func dataURL(mt string, b []byte) string {
	return "data:" + mt + ";base64," + base64.StdEncoding.EncodeToString(b)
}

func TestImagePolicyDecodesEachFormat(t *testing.T) {
	p := ImagePolicy{}.withDefaults()
	var gifb, jpg bytes.Buffer
	if err := gif.Encode(&gifb, image.NewPaletted(image.Rect(0, 0, 2, 1), color.Palette{color.Black, color.White}), nil); err != nil {
		t.Fatal(err)
	}
	if err := jpeg.Encode(&jpg, image.NewGray(image.Rect(0, 0, 9, 4)), nil); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		mt   string
		b    []byte
		w, h int
	}{
		{"image/png", testPNG(t, 5, 3, color.White), 5, 3},
		{"image/webp", readTestdata(t, "blue-purple-pink.lossy.webp"), 0, 0},
		{"image/gif", gifb.Bytes(), 2, 1},
		{"image/jpg", jpg.Bytes(), 9, 4},
	}
	for _, c := range cases {
		img, err := p.decodeDataURL(dataURL(c.mt, c.b))
		if err != nil {
			t.Fatalf("%s: %v", c.mt, err)
		}
		if c.w > 0 && (img.Bounds().Dx() != c.w || img.Bounds().Dy() != c.h) {
			t.Fatalf("%s decoded to %v, want %dx%d", c.mt, img.Bounds(), c.w, c.h)
		}
	}
}

func TestImagePolicyRefusals(t *testing.T) {
	p := ImagePolicy{}.withDefaults()
	small := testPNG(t, 4, 4, color.Black)
	cases := []struct {
		name, url, want string
		p               ImagePolicy
	}{
		{"not base64", "data:image/png;base64,@@@@", "not valid base64", p},
		{"unsupported type", dataURL("image/bmp", small), `"image/bmp"`, p},
		{"not base64 encoded", "data:image/png," + string(small), "not base64", p},
		{"no comma", "data:image/png;base64", "no comma", p},
		{"declared type differs", dataURL("image/jpeg", small), "declared image/jpeg and its bytes are image/png", p},
		{"undecodable", dataURL("image/png", []byte("not a picture at all")), "is not a PNG", p},
		{"truncated", dataURL("image/png", small[:len(small)-20]), "does not decode", p},
		{"over the pixel limit, from the header", dataURL("image/png", hugePNG(60000, 60000)), "over the limit of", p},
		{"over the byte limit", dataURL("image/png", small), "over the limit of 10", ImagePolicy{MaxBytes: 10}.withDefaults()},
		{"over a pixel limit", dataURL("image/png", small), "16 pixels, over the limit of 15", ImagePolicy{MaxPixels: 15}.withDefaults()},
		{"remote, fetching off", "https://example.com/cat.png", "does not fetch remote images", p},
		{"another scheme", "file:///etc/passwd", "neither a data: URL nor an http(s) URL", p},
	}
	for _, c := range cases {
		_, err := c.p.imageURL(context.Background(), c.url)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want one containing %q", c.name, err, c.want)
		}
	}
}

// TestOpenAIImagePartsReachTheGenerate: each image_url part becomes a decoded
// picture of the chat, counted on its message, in the client's order.
func TestOpenAIImagePartsReachTheGenerate(t *testing.T) {
	f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
	s := serve(t, f)
	a, b := dataURL("image/png", testPNG(t, 3, 1, color.White)), dataURL("image/png", testPNG(t, 7, 1, color.White))
	body := fmt.Sprintf(`{"model":"m-1","messages":[
		{"role":"user","content":[{"type":"text","text":"one"},{"type":"image_url","image_url":{"url":%q}}]},
		{"role":"assistant","content":"x"},
		{"role":"user","content":[{"type":"image_url","image_url":{"url":%q,"detail":"high"}},{"type":"text","text":"two"}]}]}`, a, b)
	resp := post(t, s, "/v1/chat/completions", body)
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	checkTwoPictures(t, f.last)
}

func checkTwoPictures(t *testing.T, o GenerateOptions) {
	t.Helper()
	c := o.Prompt.Chat
	if c == nil || len(c.Images) != 2 {
		t.Fatalf("the generate carried %+v, want two pictures", c)
	}
	if c.Images[0].Bounds().Dx() != 3 || c.Images[1].Bounds().Dx() != 7 {
		t.Fatalf("the pictures are out of order: %v %v", c.Images[0].Bounds(), c.Images[1].Bounds())
	}
	if c.Messages[0].Images != 1 || c.Messages[1].Images != 0 || c.Messages[2].Images != 1 {
		t.Fatalf("per-message counts %d %d %d", c.Messages[0].Images, c.Messages[1].Images, c.Messages[2].Images)
	}
}

func TestAnthropicImageBlocksReachTheGenerate(t *testing.T) {
	f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
	s := serve(t, f)
	a := base64.StdEncoding.EncodeToString(testPNG(t, 3, 1, color.White))
	b := base64.StdEncoding.EncodeToString(testPNG(t, 7, 1, color.White))
	body := fmt.Sprintf(`{"model":"m-1","max_tokens":4,"messages":[
		{"role":"user","content":[{"type":"text","text":"one"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q}}]},
		{"role":"assistant","content":"x"},
		{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q}},{"type":"text","text":"two"}]}]}`, a, b)
	resp := post(t, s, "/v1/messages", body)
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	checkTwoPictures(t, f.last)
}

// TestImageRefusalsNameThePart: every refusal is a 400 naming where in the
// request the picture was, and nothing reaches the generate.
func TestImageRefusalsNameThePart(t *testing.T) {
	bad := base64.StdEncoding.EncodeToString([]byte("nope"))
	png64 := base64.StdEncoding.EncodeToString(testPNG(t, 2, 2, color.White))
	cases := []struct{ path, body, want string }{
		{"/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":[{"type":"text","text":"a"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,` + bad + `"}}]}]}`,
			"messages[0].content[1].image_url is not a PNG"},
		{"/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}`,
			"messages[0].content[0].image_url names a picture by a remote URL"},
		{"/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":[{"type":"image_url"}]}]}`,
			"messages[0].content[0] is an image_url part with no image_url.url"},
		{"/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":[{"type":"input_audio"}]}]}`,
			`messages[0].content[0] is a "input_audio" part`},
		{"/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":[
			{"type":"file","file":{"file_data":"data:application/pdf;base64,JVBERi0=","filename":"a.pdf"}}]}]}`,
			"messages[0].content[0] is a application/pdf file, and this server reads no PDF"},
		{"/v1/messages", `{"model":"m-1","max_tokens":4,"messages":[{"role":"user","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/tiff","data":"` + png64 + `"}}]}]}`,
			`messages[0].content[0].source has media_type "image/tiff"`},
		{"/v1/messages", `{"model":"m-1","max_tokens":4,"messages":[{"role":"user","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"` + png64 + `"}}]}]}`,
			"messages[0].content[0].source is declared image/jpeg and its bytes are image/png"},
		{"/v1/messages", `{"model":"m-1","max_tokens":4,"messages":[{"role":"user","content":[
			{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}`,
			"does not fetch remote images"},
		{"/v1/messages", `{"model":"m-1","max_tokens":4,"messages":[{"role":"user","content":[{"type":"image"}]}]}`,
			"messages[0].content[0].source is missing"},
		{"/v1/messages", `{"model":"m-1","max_tokens":4,"messages":[{"role":"user","content":[{"type":"document"}]}]}`,
			"messages[0].content[0].source is missing"},
		{"/v1/messages", `{"model":"m-1","max_tokens":4,"messages":[{"role":"user","content":[
			{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}`,
			"messages[0].content[0].source is a application/pdf document, and this server reads no PDF"},
		{"/v1/messages", `{"model":"m-1","max_tokens":4,"messages":[{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{}}]}]}]}`,
			"messages[0].content[0].content[0] is an image block"},
	}
	for _, c := range cases {
		f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
		s := serve(t, f)
		resp := post(t, s, c.path, c.body)
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 400 || !strings.Contains(string(raw), jsonEscape(c.want)) {
			t.Errorf("%s: %d %s, want 400 containing %q", c.path, resp.StatusCode, raw, c.want)
		}
		if f.last.Prompt.Kind != PromptNone {
			t.Errorf("%s: a refused request reached the generate", c.path)
		}
	}
}

// TestADocumentBlockIsTheTurnsText: an Anthropic document block of text
// -- a text source, content blocks, base64 text/plain -- reaches the chat
// turn as text after the turn's own, its title and context first.
func TestADocumentBlockIsTheTurnsText(t *testing.T) {
	for src, want := range map[string]string{
		`{"type":"text","media_type":"text/plain","data":"the body"}`:       "Read this.\n\nNotes\n\nfrom a meeting\n\nthe body",
		`{"type":"content","content":[{"type":"text","text":"the body"}]}`:  "Read this.\n\nNotes\n\nfrom a meeting\n\nthe body",
		`{"type":"base64","media_type":"text/plain","data":"dGhlIGJvZHk="}`: "Read this.\n\nNotes\n\nfrom a meeting\n\nthe body",
	} {
		f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
		s := serve(t, f)
		resp := post(t, s, "/v1/messages", `{"model":"m-1","max_tokens":4,"messages":[{"role":"user","content":[`+
			`{"type":"text","text":"Read this."},{"type":"document","title":"Notes","context":"from a meeting","source":`+src+`}]}]}`)
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", src, resp.StatusCode, raw)
		}
		ch := f.last.Prompt.Chat
		if ch == nil || len(ch.Messages) == 0 || ch.Messages[len(ch.Messages)-1].Content != want {
			t.Fatalf("%s: the turn reads %+v, want %q", src, ch, want)
		}
	}
}

// TestAFileContentPartIsItsText: an OpenAI file part of text -- chat's file
// object, the responses API's fields on the part, text/plain, markdown or
// JSON, and a file the Files API holds, named by id -- reaches the chat as its
// file name and its text, after the turn's text. A stored PDF is refused.
func TestAFileContentPartIsItsText(t *testing.T) {
	f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
	s := serve(t, filesBackend{f, t.TempDir()})
	code, up := uploadBatchFile(t, s.URL, "user_data", "the stored body")
	_, pdf := uploadBatchFile(t, s.URL, "user_data", "%PDF-1.4 x")
	if code != 200 {
		t.Fatalf("the upload: %d", code)
	}
	for part, want := range map[string]string{
		`{"type":"file","file":{"file_data":"data:text/plain;base64,dGhlIGJvZHk=","filename":"notes.txt"}}`:         "Read this.\n\nnotes.txt\n\nthe body",
		`{"type":"input_file","file_data":"data:text/plain;base64,dGhlIGJvZHk=","filename":"notes.txt"}`:            "Read this.\n\nnotes.txt\n\nthe body",
		`{"type":"file","file":{"file_data":"data:application/json;base64,eyJhIjoxfQ=="}}`:                          "Read this.\n\n{\"a\":1}",
		`{"type":"file","file":{"file_data":"data:text/markdown;charset=utf-8;base64,IyBIaQ==","filename":"a.md"}}`: "Read this.\n\na.md\n\n# Hi",
		`{"type":"file","file":{"file_id":"` + up.ID + `"}}`:                                                        "Read this.\n\n" + up.Filename + "\n\nthe stored body",
	} {
		resp := post(t, s, "/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":[`+
			`{"type":"text","text":"Read this."},`+part+`]}]}`)
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", part, resp.StatusCode, raw)
		}
		ch := f.last.Prompt.Chat
		if ch == nil || len(ch.Messages) == 0 || ch.Messages[len(ch.Messages)-1].Content != want {
			t.Fatalf("%s: the turn reads %+v, want %q", part, ch, want)
		}
	}
	for part, want := range map[string]string{
		`{"type":"file","file":{"file_id":"` + pdf.ID + `"}}`: "which is not text, and this server reads no PDF",
		`{"type":"file","file":{"file_id":"file-none"}}`:      `names file \"file-none\": no such file`,
	} {
		resp := post(t, s, "/v1/chat/completions", `{"model":"m-1","messages":[{"role":"user","content":[`+part+`]}]}`)
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 400 || !strings.Contains(string(raw), want) {
			t.Fatalf("%s: %d %s, want 400 containing %q", part, resp.StatusCode, raw, want)
		}
	}
}

// filesBackend is a fake backend with a Files API store.
type filesBackend struct {
	*fakeBackend
	dir string
}

func (f filesBackend) BatchDir() string { return f.dir }

func jsonEscape(s string) string {
	b, _ := json.Marshal(s)
	return string(b[1 : len(b)-1])
}

func TestImageCountIsBounded(t *testing.T) {
	f := &fakeBackend{tokens: []string{"ok"}, reason: FinishEOS}
	s := httptest.NewServer(CompatHandlerWith(f, ImagePolicy{MaxImages: 1}))
	defer s.Close()
	u := dataURL("image/png", testPNG(t, 2, 2, color.White))
	part := fmt.Sprintf(`{"type":"image_url","image_url":{"url":%q}}`, u)
	resp := post(t, &httptest.Server{URL: s.URL}, "/v1/chat/completions",
		`{"model":"m-1","messages":[{"role":"user","content":[`+part+`,`+part+`]}]}`)
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 400 || !strings.Contains(string(raw), "messages[0].content[1].image_url is picture 2; a request carries at most 1") {
		t.Fatalf("%d %s", resp.StatusCode, raw)
	}
}

// The parsers take untrusted bytes; RULE 9 runs them with an exec budget:
//
//	../scripts/cap 4G -- go test . -run=XXX -fuzz=FuzzImageDataURL -fuzztime=100000x -parallel=2
func FuzzImageDataURL(f *testing.F) {
	f.Add(dataURL("image/png", testPNG(f, 2, 2, color.White)))
	f.Add(dataURL("image/webp", readTestdata(f, "blue-purple-pink.lossy.webp")))
	f.Add(dataURL("image/png", hugePNG(1<<30, 1<<30)))
	f.Add("data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAACAkQBADs=")
	f.Add("data:;base64,")
	p := ImagePolicy{MaxBytes: 1 << 16, MaxPixels: 1 << 16}.withDefaults()
	f.Fuzz(func(t *testing.T, u string) {
		img, err := p.imageURL(context.Background(), u)
		if err == nil && int64(img.Bounds().Dx())*int64(img.Bounds().Dy()) > p.MaxPixels {
			t.Fatalf("decoded %v past the pixel limit", img.Bounds())
		}
	})
}

func FuzzImageBase64(f *testing.F) {
	f.Add(testPNG(f, 2, 2, color.White), "image/png")
	f.Add(readTestdata(f, "blue-purple-pink.lossy.webp"), "image/webp")
	f.Add(hugePNG(1<<20, 1<<20), "image/png")
	p := ImagePolicy{MaxBytes: 1 << 16, MaxPixels: 1 << 16}.withDefaults()
	f.Fuzz(func(t *testing.T, b []byte, mt string) {
		img, err := p.anImage(context.Background(), &anImageSource{Type: "base64", MediaType: mt,
			Data: base64.StdEncoding.EncodeToString(b)})
		if err == nil && int64(img.Bounds().Dx())*int64(img.Bounds().Dy()) > p.MaxPixels {
			t.Fatalf("decoded %v past the pixel limit", img.Bounds())
		}
	})
}

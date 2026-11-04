package model

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
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
// are described with the disc's colour, and a blank grey picture is not
// called yellow -- a tower wired to nothing still answers fluently, so the
// blank is the violation. The prompt is SmolVLM's own template's, its
// picture's markers the container's (jlm.TowerSpan).
func TestStatedTowerSeesThePicture(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	span := m.container.Vision().Tower.Span
	ask := func(pic image.Image) string {
		t.Helper()
		s := m.NewInstance(256)
		defer s.Close()
		rows, err := s.EncodePicture(pic)
		if err != nil {
			t.Fatal(err)
		}
		pre := m.Vocab.EncodeSpecial("<|im_start|>User:", false)
		post := m.Vocab.EncodeSpecial("Describe this image in one sentence.<end_of_utterance>\nAssistant:", false)
		lg, err := s.PrefillMixed(Span{Tokens: pre}, Span{Tokens: span.Before}, Span{Embd: rows},
			Span{Tokens: span.After}, Span{Tokens: post})
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
	got := ask(img)
	grey := image.NewGray(image.Rect(0, 0, 384, 384))
	for i := range grey.Pix {
		grey.Pix[i] = 128
	}
	none := ask(grey)
	t.Logf("picture %q", got)
	t.Logf("blank   %q", none)
	if !strings.Contains(got, "yellow") {
		t.Errorf("the description does not name the yellow disc: %q", got)
	}
	if strings.Contains(none, "yellow") {
		t.Errorf("a blank picture is yellow too (%q): the picture is not what answered", none)
	}
}

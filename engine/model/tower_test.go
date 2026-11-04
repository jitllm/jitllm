package model

import (
	"fmt"
	"math"
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

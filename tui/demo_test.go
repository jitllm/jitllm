package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"connectrpc.com/connect"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/jitllm/jitllm/server/client"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// updateGolden is the -update flag TestMain registers.
func updateGolden() bool { return flag.Lookup("update").Value.String() == "true" }

// fixturePlacement is eight blocks of a model loaded on two cards: blocks 0-3
// on cuda:0, 4-5 on vulkan:1, 6-7 on the host. Each block holds 4 KiB a
// position, and block 7 is windowed to 64 positions.
func fixturePlacement() *v1.Placement {
	where := []string{"cuda:0", "cuda:0", "cuda:0", "cuda:0", "vulkan:1", "vulkan:1", "", ""}
	p := &v1.Placement{SessionId: "s-1"}
	for i, dev := range where {
		b := &v1.BlockPlacement{
			Index:              int32(i),
			Location:           v1.BlockLocation_BLOCK_LOCATION_CPU,
			HistoryPerPosition: &v1.ByteSize{Bytes: 4096},
		}
		if dev != "" {
			b.Location, b.DeviceId = v1.BlockLocation_BLOCK_LOCATION_DEVICE, dev
		}
		if i == 7 {
			b.HistoryWindow = 64
		}
		p.Blocks = append(p.Blocks, b)
	}
	return p
}

func fixtureView() mapView {
	blocks := blocksOf(fixturePlacement())
	return mapView{Cols: columnsOf([]string{"cuda:0", "vulkan:1"}, blocks), Blocks: blocks, Pos: 100, MaxSeq: 512,
		Slides: map[int]slide{}}
}

// golden compares got, with its colours stripped, to testdata/name. -update
// rewrites it; a golden is only blessed after reading what it holds.
func golden(t *testing.T, name, got string) {
	t.Helper()
	got = stripANSI(got) + "\n"
	path := filepath.Join("testdata", name)
	if updateGolden() {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(want) != got {
		t.Errorf("%s differs from what the map renders:\n--- want\n%s--- got\n%s", name, want, got)
	}
}

// TestMapDrawsEachBlockInItsColumn: three columns, every block in the one the
// placement names, each with a bar of its history at the live position.
func TestMapDrawsEachBlockInItsColumn(t *testing.T) {
	v := fixtureView()
	out := v.render(72, 14)
	golden(t, "demo_map.golden", out)

	// The golden is a picture; these are the facts it must hold.
	for id, want := range map[string][]int{hostID: {6, 7}, "cuda:0": {0, 1, 2, 3}, "vulkan:1": {4, 5}} {
		body := stripANSI(v.column(id, 24, 14))
		for i := range 8 {
			if has := strings.Contains(body, fmt.Sprintf("%02d ", i)); has != slices.Contains(want, i) {
				t.Errorf("%s: block %d present=%v, want %v\n%s", columnLabel(id), i, has, !has, body)
			}
		}
	}
}

// TestBarsGrowWithTheHistoryAndStopAtAWindow: a block's history is its state
// plus a cost a position, capped at its window; the bar and its size follow.
func TestBarsGrowWithTheHistoryAndStopAtAWindow(t *testing.T) {
	b := mapBlock{PerPos: 4096, State: 100}
	if got := b.history(10); got != 100+10*4096 {
		t.Errorf("history at 10 = %d", got)
	}
	w := mapBlock{PerPos: 4096, Window: 64}
	if w.history(1000) != 64*4096 || w.history(1000) != w.history(64) {
		t.Errorf("a windowed block kept growing: %d at 1000, %d at 64", w.history(1000), w.history(64))
	}
	low, high := fixtureView(), fixtureView()
	low.Pos, high.Pos = 10, 400
	if a, b := stripANSI(low.render(72, 14)), stripANSI(high.render(72, 14)); a == b {
		t.Error("the map is the same at position 10 and at 400")
	}
}

// TestMapSlidesBlocksToTheirNewColumn: after a move the blocks wait as ghosts
// where they were, then land highlighted in the new column in a wave, then sit
// plain. The golden is the wave's middle.
func TestMapSlidesBlocksToTheirNewColumn(t *testing.T) {
	v := fixtureView()
	// Blocks 0-3 moved from cuda:0 to vulkan:1 at frame 0.
	for i := range 4 {
		v.Blocks[i].Dev = "vulkan:1"
		v.Slides[i] = slide{From: "cuda:0", Start: 0, Rank: i}
	}
	v.Frame = 0 // blocks 0 and 1 have landed (two leave a frame), 2 and 3 have not
	mid := v.render(72, 14)
	golden(t, "demo_map_sliding.golden", mid)
	text := stripANSI(mid)
	if !strings.Contains(text, "┄") {
		t.Error("no block is shown waiting at its source")
	}
	if got := strings.Count(text, "◂"); got != 2 {
		t.Errorf("%d blocks drawn as just landed, want 2 (blocks 0 and 1)", got)
	}
	v.Frame = 40
	if done := stripANSI(v.render(72, 14)); strings.Contains(done, "┄") || strings.Contains(done, "◂") {
		t.Errorf("a finished slide still shows its ghost or flash:\n%s", done)
	}
}

func TestPlansPickTheNextDevice(t *testing.T) {
	v := fixtureView()
	// Blocks 0-5 are on devices: m sends them from cuda:0 to the next, vulkan:1.
	if r, err := planAll(v.Cols, v.Blocks, 6); err != nil || r != (moveReq{0, 5, "vulkan:1", false}) {
		t.Errorf("m = %+v, %v", r, err)
	}
	// h brings the device set home.
	if r, err := planHome(v.Blocks); err != nil || r != (moveReq{0, 5, hostID, true}) {
		t.Errorf("h = %+v, %v", r, err)
	}
	// 8 blocks are nine groups of one: group 5 is block 4, on vulkan:1, whose
	// next device wraps to cuda:0.
	if r, err := planGroup(5, v.Cols, v.Blocks); err != nil || r != (moveReq{4, 4, "cuda:0", false}) {
		t.Errorf("5 = %+v, %v", r, err)
	}
	if _, err := planGroup(9, v.Cols, v.Blocks); err == nil {
		t.Error("group 9 of an 8-block model was planned")
	}
	// Everything home: m takes the blocks up to the first device.
	for i := range v.Blocks {
		v.Blocks[i].Dev = hostID
	}
	if r, err := planAll(v.Cols, v.Blocks, 6); err != nil || r != (moveReq{0, 5, "cuda:0", false}) {
		t.Errorf("m from home = %+v, %v", r, err)
	}
	if r, _ := planAll(v.Cols, v.Blocks, 0); r != (moveReq{0, 7, "cuda:0", false}) {
		t.Errorf("m from home with no device set to return to = %+v", r)
	}
	if _, err := planHome(v.Blocks); err == nil {
		t.Error("h on a model already home was planned")
	}
	// One device: m has no other card, so it toggles home and back.
	one := columnsOf([]string{"cuda:0"}, nil)
	up := []mapBlock{{Index: 0, Dev: "cuda:0"}, {Index: 1, Dev: "cuda:0"}}
	if r, _ := planAll(one, up, 2); r != (moveReq{0, 1, hostID, true}) {
		t.Errorf("m on one device = %+v", r)
	}
}

// fakePlacement is a jitllmd's placement and session services that record
// what the demo asked them to do.
type fakePlacement struct {
	jitllmv1connect.UnimplementedPlacementServiceHandler
	jitllmv1connect.UnimplementedSessionServiceHandler
	mu        sync.Mutex
	relocates []*v1.RelocateBlocksRequest
}

func (f *fakePlacement) GetPlacement(context.Context, *connect.Request[v1.GetPlacementRequest]) (*connect.Response[v1.GetPlacementResponse], error) {
	return connect.NewResponse(&v1.GetPlacementResponse{Placement: fixturePlacement()}), nil
}

func (f *fakePlacement) RelocateBlocks(_ context.Context, r *connect.Request[v1.RelocateBlocksRequest]) (*connect.Response[v1.RelocateBlocksResponse], error) {
	f.mu.Lock()
	f.relocates = append(f.relocates, r.Msg)
	f.mu.Unlock()
	return connect.NewResponse(&v1.RelocateBlocksResponse{Placement: fixturePlacement(), BlocksMoved: 6}), nil
}

func (f *fakePlacement) GetSession(context.Context, *connect.Request[v1.GetSessionRequest]) (*connect.Response[v1.GetSessionResponse], error) {
	return connect.NewResponse(&v1.GetSessionResponse{Session: &v1.Session{SessionId: "s-1"}}), nil
}

// TestKeysAskForTheMoves: each key sends the RelocateBlocks call it names, and
// nothing else sends one.
func TestKeysAskForTheMoves(t *testing.T) {
	f := &fakePlacement{}
	mux := http.NewServeMux()
	mux.Handle(jitllmv1connect.NewPlacementServiceHandler(f))
	mux.Handle(jitllmv1connect.NewSessionServiceHandler(f))
	srv := httptest.NewServer(mux)
	defer srv.Close()
	blocks := blocksOf(fixturePlacement())
	d := &demo{c: client.Dial(srv.URL), sid: "s-1", blocks: blocks, upBlocks: 6, maxSeq: 512, slides: map[int]slide{},
		cols: columnsOf([]string{"cuda:0", "vulkan:1"}, blocks), events: make(chan tea.Msg, 4)}

	press := func(k string) *moveMsg {
		t.Helper()
		var cmd tea.Cmd
		if len(k) == 1 {
			_, cmd = d.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)})
		} else {
			_, cmd = d.Update(tea.KeyMsg{Type: tea.KeyEsc})
		}
		if cmd == nil {
			return nil
		}
		m, ok := cmd().(moveMsg)
		if !ok {
			return nil
		}
		d.Update(m)
		return &m
	}

	for _, c := range []struct {
		key   string
		first int32
		last  int32
		to    string
	}{
		{"m", 0, 5, "vulkan:1"},
		{"h", 0, 5, "cpu"},
		{"5", 4, 4, "cuda:0"},
	} {
		if press(c.key) == nil {
			t.Fatalf("%q sent no move", c.key)
		}
		got := f.relocates[len(f.relocates)-1]
		if got.GetSessionId() != "s-1" || got.GetFirstBlock() != c.first || got.GetLastBlock() != c.last ||
			got.GetToDeviceId() != c.to {
			t.Errorf("%q asked for %+v", c.key, got)
		}
	}
	n := len(f.relocates)
	for _, k := range []string{"x", "9", "z"} {
		if press(k) != nil {
			t.Errorf("%q sent a move", k)
		}
	}
	if len(f.relocates) != n {
		t.Errorf("%d moves sent by keys that name none", len(f.relocates)-n)
	}
	if d.moves != 3 {
		t.Errorf("%d moves counted, want 3", d.moves)
	}
	// While one is in flight a second key is refused, with its reason.
	d.pending = &moveReq{}
	if c := d.key(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("m")}); c != nil || !strings.Contains(d.note, "in flight") {
		t.Errorf("a second move was started while one ran: note %q", d.note)
	}
}

// TestBannerTellsWhatTheServiceReported: every figure on the banner is a field
// of the response, and a move the service refused is shown as refused.
func TestBannerTellsWhatTheServiceReported(t *testing.T) {
	resp := &v1.RelocateBlocksResponse{
		BlocksMoved: 16, History: &v1.ByteSize{Bytes: 3 << 20}, Copied: &v1.ByteSize{Bytes: 3 << 20},
		Path: v1.RelocationPath_RELOCATION_PATH_STAGED, PauseMicros: 3200, StagedBlocks: 16, PrepareMillis: 640,
	}
	got := stripANSI(banner(&moveInfo{From: "cuda:0", To: "vulkan:1", First: 0, Last: 15, AtToken: 80, Resp: resp},
		nil, nil, 140, 6))
	for _, want := range []string{"cuda:0 → vulkan:1", "blocks 0-15 (16 moved)", "history moved 3.0 MiB", "paused 3.2 ms",
		"at token 80", "through a host buffer", "copied 3.0 MiB", "prompt rows recomputed 0", "projection moved no",
		"16 block(s) prepared in 640 ms"} {
		if !strings.Contains(got, want) {
			t.Errorf("the banner lacks %q:\n%s", want, got)
		}
	}
	resp.HeadMoved = true
	if got := stripANSI(banner(&moveInfo{Resp: resp}, nil, nil, 140, 6)); !strings.Contains(got, "projection moved yes") {
		t.Errorf("a moved projection is not reported:\n%s", got)
	}
	refused := stripANSI(banner(&moveInfo{Err: connect.NewError(connect.CodeFailedPrecondition, context.Canceled)}, nil, nil, 140, 6))
	if !strings.Contains(refused, "refused") {
		t.Errorf("a refused move reads %q", refused)
	}
}

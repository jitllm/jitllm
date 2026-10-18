package server

import (
	"context"
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestAHostLoadedModelRelocatesToADeviceAndBack: a model loaded on the host
// alone has no tier, and a session of it still moves its blocks to a device
// when asked (the device joins the model's tier on first need), carries the
// conversation on from there, and brings them home again, each reply the
// token-for-token continuation an unmoved host session gives.
//
// VIOLATION SIGNATURE. Put back the host-only refusal in RelocateBlocks and
// the first move fails with "loaded host-only"; skip bindSession and the move
// answers 0 blocks moved; drop the history on the way and a reply parts from
// the control's.
func TestAHostLoadedModelRelocatesToADeviceAndBack(t *testing.T) {
	devs := testdevices.Take(t, 1, "cuda:0")
	dev := devs[0]
	e := New(Config{Version: "test", NoMemCache: true})
	t.Cleanup(e.Close)
	// The device gates' small model when it is on the box, else a Qwen3
	// container the model directory holds.
	name := deviceModel
	if _, err := os.Stat(testmodels.Path(name)); err != nil {
		name = "Qwen3-0.6B-Q8_0.jlm"
	}
	lm, err := e.LoadModel(LoadOptions{Path: modelPath(t, name), ModelID: "host", DeviceIDs: []string{"cpu"}})
	if err != nil {
		t.Fatal(err)
	}
	if lm.device() != nil {
		t.Fatalf("a model loaded on cpu holds a tier")
	}
	c := serveEngine(t, e)
	ctx := context.Background()
	turn := func(id, prompt string) []int32 {
		t.Helper()
		r, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: id, Prompt: text(prompt), MaxTokens: 6}))
		if err != nil {
			t.Fatal(err)
		}
		return r.Msg.GetTokenIds()
	}
	open := func(id string) {
		t.Helper()
		if _, err := e.CreateSession(SessionOptions{ModelID: "host", SessionID: id, MaxSeq: 128}); err != nil {
			t.Fatal(err)
		}
	}
	devBlocks := func(id string) int {
		t.Helper()
		pl, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: id}))
		if err != nil {
			t.Fatal(err)
		}
		return int(pl.Msg.GetPlacement().GetDeviceBlockCount())
	}

	// The control stays on the host for every turn.
	open("ctl")
	ctl := [][]int32{turn("ctl", story), turn("ctl", " and then"), turn("ctl", " until")}
	if err := e.CloseSession("ctl"); err != nil {
		t.Fatal(err)
	}

	open("moved")
	nb := int32(lm.m.Cfg.NLayer)
	if got := turn("moved", story); !slices.Equal(got, ctl[0]) {
		t.Fatalf("two unmoved host sessions part: %v and %v", got, ctl[0])
	}
	r, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{
		SessionId: "moved", FirstBlock: 0, LastBlock: nb - 1, ToDeviceId: dev,
	}))
	if err != nil {
		t.Fatalf("a host-loaded model refused a move to %s: %v", dev, err)
	}
	if lm.device() == nil || !slices.Contains(lm.devIDs(), dev) || len(lm.hwIDs) != 1 {
		t.Fatalf("the model holds no tier with %s after the move: %v", dev, lm.devIDs())
	}
	if r.Msg.GetBlocksMoved() == 0 || devBlocks("moved") == 0 {
		t.Skipf("NO DEVICE ROOM: %d blocks moved, %d on the device -- this gate proved nothing", r.Msg.GetBlocksMoved(), devBlocks("moved"))
	}
	if got := turn("moved", " and then"); !slices.Equal(got, ctl[1]) {
		t.Fatalf("after the move to %s the reply parts from the host control: %v and %v", dev, got, ctl[1])
	}
	if _, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{
		SessionId: "moved", FirstBlock: 0, LastBlock: int32(devBlocks("moved")) - 1, ToHost: true,
	})); err != nil {
		t.Fatalf("bringing the blocks home: %v", err)
	}
	if n := devBlocks("moved"); n != 0 {
		t.Fatalf("%d blocks still on the device after bringing them home", n)
	}
	if got := turn("moved", " until"); !slices.Equal(got, ctl[2]) {
		t.Fatalf("after coming home the reply parts from the host control: %v and %v", got, ctl[2])
	}
}

// TestADeviceOutsideTheTierJoinsIt: a model loaded on one device moves a
// session's blocks to another the tier was not opened with, through a tier
// opened over both (Engine.attachDevice); the blocks are on the new device
// afterwards and the conversation goes on.
//
// VIOLATION SIGNATURE. Put back the "not one of them" refusal and the move
// fails naming the tier's devices.
func TestADeviceOutsideTheTierJoinsIt(t *testing.T) {
	devs := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	if len(devs) != 2 {
		t.Fatalf("two devices wanted, got %v", devs)
	}
	name := deviceModel
	if _, err := os.Stat(testmodels.Path(name)); err != nil {
		name = "Qwen3-0.6B-Q8_0.jlm"
	}
	e := New(Config{Version: "test", NoMemCache: true})
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: modelPath(t, name), ModelID: "one", DeviceIDs: devs[:1]})
	if err != nil {
		t.Skipf("NO DEVICE: loading onto %s failed (%v)", devs[0], err)
	}
	c := serveEngine(t, e)
	ctx := context.Background()
	tierBefore := lm.gpuTier()
	if _, err := e.CreateSession(SessionOptions{ModelID: "one", SessionID: "s", MaxSeq: 128}); err != nil {
		t.Fatal(err)
	}
	turn := func(prompt string) int {
		t.Helper()
		r, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: "s", Prompt: text(prompt), MaxTokens: 6}))
		if err != nil {
			t.Fatal(err)
		}
		return len(r.Msg.GetTokenIds())
	}
	turn(story)
	nb := int32(lm.m.Cfg.NLayer)
	r, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{SessionId: "s", FirstBlock: 0, LastBlock: nb - 1, ToDeviceId: devs[1]}))
	if err != nil {
		t.Fatalf("a device outside the tier was refused: %v", err)
	}
	if !slices.Contains(lm.devIDs(), devs[1]) {
		t.Fatalf("the tier did not take %s: %v", devs[1], lm.devIDs())
	}
	// One tier took the device: the weights are not on a card twice.
	if lm.gpuTier() != tierBefore {
		t.Fatalf("the model's tier was replaced, not extended")
	}
	on := 0
	for _, b := range r.Msg.GetPlacement().GetBlocks() {
		if b.GetDeviceId() == devs[1] {
			on++
		}
	}
	if on == 0 {
		t.Skipf("NO DEVICE ROOM: no block landed on %s", devs[1])
	}
	if turn(" and then") == 0 {
		t.Fatalf("no reply after the move to %s", devs[1])
	}
}

// TestAMiddleRangeMovesUpAndASubRangeComesHome: blocks of a host-loaded
// model's session go to a device as a range in the middle of the stack, with
// the blocks either side left at home, and a sub-range of them comes home
// again leaving a hole between two device runs; each reply is the unmoved
// host control's tokens.
//
// VIOLATION SIGNATURE. Put back "only a range that EXTENDS the set" and the
// first move is refused; send the range to the head and the placement holds
// no host block below the range.
func TestAMiddleRangeMovesUpAndASubRangeComesHome(t *testing.T) {
	dev := testdevices.Take(t, 1, "cuda:0")[0]
	name := deviceModel
	if _, err := os.Stat(testmodels.Path(name)); err != nil {
		name = "Qwen3-0.6B-Q8_0.jlm"
	}
	e := New(Config{Version: "test", NoMemCache: true})
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: modelPath(t, name), ModelID: "host", DeviceIDs: []string{"cpu"}})
	if err != nil {
		t.Fatal(err)
	}
	c := serveEngine(t, e)
	ctx := context.Background()
	turn := func(id, prompt string) []int32 {
		t.Helper()
		r, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: id, Prompt: text(prompt), MaxTokens: 6}))
		if err != nil {
			t.Fatal(err)
		}
		return r.Msg.GetTokenIds()
	}
	open := func(id string) {
		t.Helper()
		if _, err := e.CreateSession(SessionOptions{ModelID: "host", SessionID: id, MaxSeq: 128}); err != nil {
			t.Fatal(err)
		}
	}
	where := func(id string) []string {
		t.Helper()
		pl, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: id}))
		if err != nil {
			t.Fatal(err)
		}
		var on []string
		for _, b := range pl.Msg.GetPlacement().GetBlocks() {
			if b.GetLocation() == v1.BlockLocation_BLOCK_LOCATION_DEVICE {
				on = append(on, "D")
			} else {
				on = append(on, "H")
			}
		}
		return on
	}
	move := func(first, last int, toHost bool) {
		t.Helper()
		r := &v1.RelocateBlocksRequest{SessionId: "moved", FirstBlock: int32(first), LastBlock: int32(last), ToHost: toHost}
		if !toHost {
			r.ToDeviceId = dev
		}
		resp, err := c.placement.RelocateBlocks(ctx, req(r))
		if err != nil {
			t.Fatalf("moving %d..%d (toHost %v): %v", first, last, toHost, err)
		}
		if len(resp.Msg.GetDeclined()) > 0 {
			t.Skipf("NO DEVICE ROOM: %v", resp.Msg.GetDeclined()[0].GetReason())
		}
	}

	open("ctl")
	ctl := [][]int32{turn("ctl", story), turn("ctl", " and then"), turn("ctl", " until")}
	if err := e.CloseSession("ctl"); err != nil {
		t.Fatal(err)
	}
	open("moved")
	nb := lm.m.Cfg.NLayer
	if nb < 6 {
		t.Skipf("model has %d blocks", nb)
	}
	if got := turn("moved", story); !slices.Equal(got, ctl[0]) {
		t.Fatalf("two unmoved host sessions part: %v and %v", got, ctl[0])
	}
	lo, hi := 1, nb-2 // a middle range, blocks lo..hi
	move(lo, hi, false)
	got := where("moved")
	for li, w := range got {
		want := "H"
		if li >= lo && li <= hi {
			want = "D"
		}
		if w != want {
			t.Fatalf("after placing %d..%d the stack is %v: block %d is %s, want %s", lo, hi, got, li, w, want)
		}
	}
	if g := turn("moved", " and then"); !slices.Equal(g, ctl[1]) {
		t.Fatalf("with blocks %d..%d on %s the reply parts from the host control: %v and %v", lo, hi, dev, g, ctl[1])
	}
	mid := (lo + hi) / 2
	move(mid, mid+1, true)
	got = where("moved")
	if got[mid] != "H" || got[mid+1] != "H" || got[lo] != "D" || got[hi] != "D" {
		t.Fatalf("a sub-range home left %v, want a hole at %d..%d between device runs", got, mid, mid+1)
	}
	if g := turn("moved", " until"); !slices.Equal(g, ctl[2]) {
		t.Fatalf("with a hole at %d..%d the reply parts from the host control: %v and %v", mid, mid+1, g, ctl[2])
	}
}

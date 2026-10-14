package server

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestBlocksMoveBetweenDevicesThroughThePlacementService: a session loaded on
// two devices moves every block from one to the other mid-conversation
// through RelocateBlocks, its history with it, and the conversation carries
// on: the placement names the target for every block, the response counts
// the history it moved and the path it took, and the next reply is an
// unmoved control's token for token (both devices run the same kernels) or
// within the band two backends agree in (JITLLM_MOVE_DEVICES, default the
// laptop's one card under CUDA and Vulkan).
func TestBlocksMoveBetweenDevicesThroughThePlacementService(t *testing.T) {
	spec := os.Getenv("JITLLM_MOVE_DEVICES")
	if spec == "" {
		spec = "cuda:0,vulkan:1"
	}
	devs := strings.Split(spec, ",")
	if len(devs) != 2 {
		t.Fatalf("JITLLM_MOVE_DEVICES=%q: want two devices", spec)
	}
	path := modelPath(t, deviceModel)
	// No memory cache: the control runs first, and the moved session would
	// restore the control's prompt from it -- a restore these gates do not
	// test, and one that on a V100 answers a second session differently from
	// the first with no move at all (origin/main, cuda:0 alone).
	e := New(Config{Version: "test", NoMemCache: true})
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: devs}); err != nil {
		t.Skipf("NO DEVICE: loading onto -devices %s failed (%v) -- this gate proved nothing", spec, err)
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
	// The control runs first and closes: a block another session is using
	// stays where it is (the weights are the model's), so the two cannot
	// share the tier while one of them moves.
	open := func(id string) {
		t.Helper()
		if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: id, MaxSeq: 128}); err != nil {
			t.Fatal(err)
		}
		pl, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: id}))
		if err != nil {
			t.Fatal(err)
		}
		for _, b := range pl.Msg.GetPlacement().GetBlocks() {
			if b.GetLocation() != v1.BlockLocation_BLOCK_LOCATION_DEVICE {
				t.Skipf("NO DEVICE ROOM: block %d of %s is on the host", b.GetIndex(), id)
			}
		}
	}
	open("ctl")
	ctl1, ctl2 := turn("ctl", story), turn("ctl", " and then")
	if err := e.CloseSession("ctl"); err != nil {
		t.Fatal(err)
	}
	open("moved")
	nb := int32(6)
	pl, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: "moved"}))
	if err != nil {
		t.Fatal(err)
	}
	from := pl.Msg.GetPlacement().GetBlocks()[0].GetDeviceId()
	to := devs[0]
	if to == from {
		to = devs[1]
	}
	if got := turn("moved", story); !slices.Equal(got, ctl1) {
		t.Fatalf("two unmoved sessions part before any move: %v and %v", got, ctl1)
	}
	r, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{
		SessionId: "moved", FirstBlock: 0, LastBlock: nb - 1, ToDeviceId: to,
	}))
	if err != nil {
		t.Fatal(err)
	}
	m := r.Msg
	t.Logf("%s -> %s: %d block(s), %s of history, path %s, head moved %v, %d ms",
		from, to, m.GetBlocksMoved(), m.GetHistory().GetHuman(), m.GetPath(), m.GetHeadMoved(), m.GetMillis())
	if m.GetBlocksMoved() != nb || m.GetHistory().GetBytes() == 0 ||
		m.GetPath() == v1.RelocationPath_RELOCATION_PATH_UNSPECIFIED || len(m.GetDeclined()) != 0 {
		t.Fatalf("the move did not run: %+v", m)
	}
	for _, blk := range m.GetPlacement().GetBlocks() {
		if blk.GetDeviceId() != to {
			t.Fatalf("block %d is on %q after a move to %s", blk.GetIndex(), blk.GetDeviceId(), to)
		}
	}
	a := turn("moved", " and then")
	sameKernels := strings.SplitN(from, ":", 2)[0] == strings.SplitN(to, ":", 2)[0]
	if sameKernels && !slices.Equal(a, ctl2) {
		t.Fatalf("after the move the reply is %v, unmoved %v", a, ctl2)
	}
	if len(a) == 0 {
		t.Fatal("the moved session generated nothing")
	}
}

// TestAMoveRunsPastThePool: a model loaded warm with the default session
// pool keeps the warm-up's State in the pool, reset and holding no sequence
// but still on every block it was placed on. A move between devices runs
// anyway: the tier moves a block only when no other session is on it, and
// the pooled State kept jitllmd's relocation demo from moving anything
// ("another session is using it") until -session-pool -1.
func TestAMoveRunsPastThePool(t *testing.T) {
	spec := os.Getenv("JITLLM_MOVE_DEVICES")
	if spec == "" {
		spec = "cuda:0,vulkan:1"
	}
	devs := strings.Split(spec, ",")
	if len(devs) != 2 {
		t.Fatalf("JITLLM_MOVE_DEVICES=%q: want two devices", spec)
	}
	path := modelPath(t, deviceModel)
	e := New(Config{Version: "test", NoMemCache: true})
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: devs, Warm: true})
	if err != nil {
		t.Skipf("NO DEVICE: loading onto -devices %s failed (%v) -- this gate proved nothing", spec, err)
	}
	if n := e.MemCacheStats(lm).Pooled; n == 0 {
		t.Fatalf("the warm load pooled no State: the gate would move past nothing")
	}
	c := serveEngine(t, e)
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: "moved", MaxSeq: 128}); err != nil {
		t.Fatal(err)
	}
	pl, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: "moved"}))
	if err != nil {
		t.Fatal(err)
	}
	blocks := pl.Msg.GetPlacement().GetBlocks()
	for _, b := range blocks {
		if b.GetLocation() != v1.BlockLocation_BLOCK_LOCATION_DEVICE {
			t.Skipf("NO DEVICE ROOM: block %d is on the host", b.GetIndex())
		}
	}
	from := blocks[0].GetDeviceId()
	to := devs[0]
	if to == from {
		to = devs[1]
	}
	if _, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: "moved", Prompt: text(story), MaxTokens: 4})); err != nil {
		t.Fatal(err)
	}
	nb := int32(len(blocks))
	r, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{
		SessionId: "moved", FirstBlock: 0, LastBlock: nb - 1, ToDeviceId: to,
	}))
	if err != nil {
		t.Fatalf("%s -> %s with the pool on: %v", from, to, err)
	}
	if m := r.Msg; m.GetBlocksMoved() != nb || len(m.GetDeclined()) != 0 {
		t.Fatalf("the move did not run: %+v", m)
	}
	got, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: "moved", Prompt: text(" and then"), MaxTokens: 4}))
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Msg.GetTokenIds()) == 0 {
		t.Fatal("the moved session generated nothing")
	}
	// A model_id request after the move is served beside the moved session.
	if _, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{ModelId: "dev", Prompt: text(story), MaxTokens: 2})); err != nil {
		t.Fatal(err)
	}
}

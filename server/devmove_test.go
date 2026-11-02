package server

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestBlocksMoveBetweenDevicesThroughThePlacementService: a session loaded on
// two devices moves every block from one to the other mid-conversation
// through RelocateBlocks, its history with it, and the conversation carries
// on: the placement names the target for every block, the response counts
// the history it moved and the path it took, and the next reply is an
// unmoved control's token for token (both devices run the same kernels) or
// within the band two backends agree in (JITLLM_TEST_DEVICES, default the
// laptop's one card under CUDA and Vulkan).
func TestBlocksMoveBetweenDevicesThroughThePlacementService(t *testing.T) {
	devs := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	spec := strings.Join(devs, ",")
	if len(devs) != 2 {
		t.Fatalf("two devices wanted, got %q", spec)
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

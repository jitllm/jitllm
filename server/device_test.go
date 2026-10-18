package server

import (
	"context"
	"strings"
	"testing"
	"time"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The placement surface with a real tier under it: the only way to reach the
// seam moves, RelocateBlocks' arms and TuneSeam, which a host-only model
// refuses before it gets to them.

// deviceModel is a container every block of which a device can run:
// stories260K's FFN width (172) is not a multiple of the device quantizer's
// 32-element block, so a card declines all five of its blocks.
const deviceModel = "stories15M-q8_0.jlm"

// deviceEngine loads deviceModel onto the first GPU, or skips naming why.
func deviceEngine(t *testing.T) (*Engine, clients) {
	t.Helper()
	path := modelPath(t, deviceModel)
	e := New(Config{Probe: oneCardProbe, Version: "test"})
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: []string{"gpu:0"}}); err != nil {
		t.Skipf("NO DEVICE: loading onto -devices gpu:0 failed (%v) -- this gate proved nothing", err)
	}
	return e, serveEngine(t, e)
}

// TestTheSeamMovesThroughThePlacementService: a session's device set can be
// trimmed from its tail and extended again, by count or by range, and every
// move the seam cannot express -- a hole, a device the tier does not hold --
// is refused with its reason. With RelocateBlocks' tail check removed the
// hole is accepted and the placement silently differs from the request.
func TestTheSeamMovesThroughThePlacementService(t *testing.T) {
	e, c := deviceEngine(t)
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: "d", MaxSeq: 64}); err != nil {
		t.Fatal(err)
	}
	placement := func() *v1.Placement {
		t.Helper()
		r, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: "d"}))
		if err != nil {
			t.Fatal(err)
		}
		return r.Msg.GetPlacement()
	}
	full := placement()
	n := full.GetDeviceBlockCount()
	if n < 2 {
		t.Skipf("NO DEVICE ROOM: the card took %d of 6 blocks (declines %v), and the seam moves below need two",
			n, full.GetDeclines())
	}
	for i, b := range full.GetBlocks()[:n] {
		if b.GetLocation() != v1.BlockLocation_BLOCK_LOCATION_DEVICE || b.GetDeviceId() != "gpu:0" || b.GetHostResident() {
			t.Fatalf("placed block %d reads %+v", i, b)
		}
	}

	// The tail of the device set comes home.
	home, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{
		SessionId: "d", FirstBlock: n - 1, LastBlock: n - 1, ToHost: true,
	}))
	if err != nil {
		t.Fatal(err)
	}
	if home.Msg.GetBlocksMoved() != 1 || home.Msg.GetPlacement().GetDeviceBlockCount() != n-1 {
		t.Fatalf("bringing block %d home moved %d and left %d on the card", n-1,
			home.Msg.GetBlocksMoved(), home.Msg.GetPlacement().GetDeviceBlockCount())
	}
	// And goes back out, as a range that extends the set.
	out, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{
		SessionId: "d", FirstBlock: n - 1, LastBlock: n - 1, ToDeviceId: "gpu:0",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if out.Msg.GetBlocksMoved() != 1 || out.Msg.GetPlacement().GetDeviceBlockCount() != n || len(out.Msg.GetDeclined()) != 0 {
		t.Fatalf("placing block %d again: %+v", n-1, out.Msg)
	}

	type refusal struct {
		what string
		m    *v1.RelocateBlocksRequest
		want connect.Code
		says string
	}
	refusals := []refusal{
		{"a device the server does not have", &v1.RelocateBlocksRequest{SessionId: "d", FirstBlock: 0, LastBlock: 0, ToDeviceId: "cuda:7"},
			connect.CodeFailedPrecondition, "tier:"},
	}
	for _, tc := range refusals {
		_, err := c.placement.RelocateBlocks(ctx, req(tc.m))
		if ce := wantCode(t, tc.what, err, tc.want); !strings.Contains(ce.Message(), tc.says) {
			t.Errorf("%s: %q does not say %q", tc.what, ce.Message(), tc.says)
		}
	}
	if got := placement().GetDeviceBlockCount(); got != n {
		t.Fatalf("refused moves changed the placement: %d on the card, want %d", got, n)
	}

	// SetPlacement by count, then everything home, then as many as fit.
	two := int32(2)
	sp, err := c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "d", DeviceIds: []string{"gpu:0"}, MaxDeviceBlocks: &two}))
	if err != nil {
		t.Fatal(err)
	}
	if got := sp.Msg.GetPlacement().GetDeviceBlockCount(); got != 2 {
		t.Fatalf("SetPlacement(max 2) left %d on the card", got)
	}
	if sp, err = c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "d"})); err != nil {
		t.Fatal(err)
	}
	if got := sp.Msg.GetPlacement().GetDeviceBlockCount(); got != 0 {
		t.Fatalf("SetPlacement with no devices left %d on the card", got)
	}
	if sp, err = c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "d", DeviceIds: []string{"gpu:0"}})); err != nil {
		t.Fatal(err)
	}
	if got := sp.Msg.GetPlacement().GetDeviceBlockCount(); got != n {
		t.Fatalf("SetPlacement with no cap placed %d, the first placement took %d", got, n)
	}
	_, err = c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "d", DeviceIds: []string{"cuda:7"}}))
	wantCode(t, "SetPlacement onto a device the tier was not opened with", err, connect.CodeFailedPrecondition)

	// The session still generates on whatever the moves left it.
	r, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: "d", Prompt: text(story), MaxTokens: 3}))
	if err != nil {
		t.Fatal(err)
	}
	if s := r.Msg.GetStarted(); s.GetDeviceBlocks() != n || s.GetHostBlocks() != 6-n || s.GetDeviceIds()[0] != "gpu:0" {
		t.Fatalf("the generate reports %d device / %d host blocks on %v", s.GetDeviceBlocks(), s.GetHostBlocks(), s.GetDeviceIds())
	}
	ls, err := c.session.ListSessions(ctx, req(&v1.ListSessionsRequest{DeviceId: "gpu:0"}))
	if err != nil || len(ls.Msg.GetSessions()) != 1 {
		t.Fatalf("a session on the card is not listed under it: %v %v", ls, err)
	}

	// TuneSeam arms the tuner and says why it reports nothing more until the
	// session generates.
	tctx, cancel := context.WithCancel(ctx)
	ts, err := c.placement.TuneSeam(tctx, req(&v1.TuneSeamRequest{SessionId: "d"}))
	if err != nil {
		t.Fatal(err)
	}
	if !ts.Receive() {
		t.Fatalf("TuneSeam sent nothing: %v", ts.Err())
	}
	if !strings.Contains(ts.Msg().GetNote(), "armed") || ts.Msg().GetDone() {
		t.Fatalf("TuneSeam's first frame %+v", ts.Msg())
	}
	cancel()
	for ts.Receive() {
	}
	ts.Close()
}

// TestTuneSeamReadsTheSessionsSnapshot: TuneSeam's ticker reports what the
// tuner settled on while the session generates. model.State's tuner is
// written by Forward with no lock, so the report must come from the snapshot
// the session publishes under its own lock, never from the State.
//
// VIOLATION SIGNATURE. Read sess.st.SeamTuned() in TuneSeam's loop again and
// this fails under -race with "WARNING: DATA RACE" between seamTuner.observe
// (or finish) and State.SeamTuned.
//
// Batching is off: the tuner advances once per Forward, and a session that
// decodes as a row of the step loop never reaches it. The model's whole
// placement, head included, is on the card, which is the Forward path that
// once returned before stepping the tuner: with those steps removed this
// fails with "the tuner did not settle".
func TestTuneSeamReadsTheSessionsSnapshot(t *testing.T) {
	e := New(Config{Probe: oneCardProbe, Version: "test", MaxBatchRows: 1})
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: modelPath(t, deviceModel), ModelID: "dev", DeviceIDs: []string{"gpu:0"}}); err != nil {
		t.Skipf("NO DEVICE: loading onto -devices gpu:0 failed (%v) -- this gate proved nothing", err)
	}
	c := serveEngine(t, e)
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: "d", MaxSeq: 256}); err != nil {
		t.Fatal(err)
	}
	tctx, cancel := context.WithCancel(ctx)
	defer cancel()
	ts, err := c.placement.TuneSeam(tctx, req(&v1.TuneSeamRequest{SessionId: "d"}))
	if err != nil {
		t.Fatal(err)
	}
	if !ts.Receive() {
		t.Fatalf("TuneSeam sent nothing: %v", ts.Err())
	}
	done := make(chan bool, 1)
	go func() {
		settled := false
		for ts.Receive() {
			settled = ts.Msg().GetDone()
		}
		done <- settled
	}()
	// The default schedule decides in about 450 tokens with its warm-ups, so
	// eight generates to the model's 128-position context, each from a reset
	// session, take the tuner through its runs and its decision.
	for i := 0; i < 8; i++ {
		if _, err := c.session.ResetSession(ctx, req(&v1.ResetSessionRequest{SessionId: "d"})); err != nil {
			t.Fatal(err)
		}
		r, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: "d", Prompt: text(story), MaxTokens: 128, IgnoreEos: true}))
		if err != nil {
			t.Fatal(err)
		}
		if got := r.Msg.GetFinished().GetCompletionTokens(); got < 100 {
			t.Fatalf("generate %d decoded %d tokens", i, got)
		}
	}
	// One more tick sees the snapshot the last generate published.
	time.Sleep(time.Second)
	cancel()
	if !<-done {
		t.Fatalf("the tuner did not settle in 1000 tokens -- the ticker read nothing that moved, " +
			"so this gate proved nothing")
	}
	ts.Close()
}

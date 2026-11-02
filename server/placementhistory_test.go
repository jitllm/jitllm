package server

import (
	"context"
	"testing"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/engine/model"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestPlacementReportsEachBlocksHistoryCost: the placement carries, per block,
// what its history costs (a fixed state plus bytes a kept position), so a
// client can draw the history a move carries at any position. The costs,
// priced at the session's position, must add up to what the instance itself
// counts as history -- a field that was zero, or one that disagreed with the
// move's own arithmetic, would draw a bar the move never carried.
func TestPlacementReportsEachBlocksHistoryCost(t *testing.T) {
	e, _, c := hostBatchEngine(t, deviceModel, Config{})
	if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: "h", MaxSeq: 128}); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err := c.inference.Complete(ctx, connect.NewRequest(&v1.GenerateRequest{
		SessionId: "h", Prompt: text("Once upon a time"), MaxTokens: 6})); err != nil {
		t.Fatal(err)
	}
	pl, err := c.placement.GetPlacement(ctx, connect.NewRequest(&v1.GetPlacementRequest{SessionId: "h"}))
	if err != nil {
		t.Fatal(err)
	}
	s, err := e.Session("h")
	if err != nil {
		t.Fatal(err)
	}
	var pos int
	var want uint64
	s.Inspect(func(st *model.Instance) { pos, want = st.Pos(), st.HistoryBytes() })
	var got uint64
	for _, b := range pl.Msg.GetPlacement().GetBlocks() {
		n := pos
		if w := int(b.GetHistoryWindow()); w > 0 {
			n = min(n, w)
		}
		got += b.GetStateBytes().GetBytes() + uint64(n)*b.GetHistoryPerPosition().GetBytes()
	}
	if want == 0 || got != want {
		t.Fatalf("the blocks' costs at position %d add up to %d bytes, the State holds %d", pos, got, want)
	}
}

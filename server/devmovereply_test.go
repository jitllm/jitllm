package server

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestAMoveLandsBetweenTwoTokensOfAReply: a RelocateBlocks asked while the
// session is streaming a reply is prepared beside the reply's tokens and
// committed between two of them, not after the reply: the call returns while
// tokens are still arriving, the target prepared every block ahead, the
// history went device to device rather than home, and the reply is the
// unmoved control's token for token where both devices run the same kernels.
//
// The model is one whose reply outlasts its own move's preparation
// (JITLLM_MOVE_REPLY_MODEL, Llama-3.2-1B by default): stories15M finishes a
// reply before its target is ready, and a move that lands after the reply
// proves nothing about a token boundary.
func TestAMoveLandsBetweenTwoTokensOfAReply(t *testing.T) {
	spec := os.Getenv("JITLLM_MOVE_DEVICES")
	if spec == "" {
		spec = "cuda:0,vulkan:1"
	}
	devs := strings.Split(spec, ",")
	name := os.Getenv("JITLLM_MOVE_REPLY_MODEL")
	if name == "" {
		name = "Llama-3.2-1B-Instruct-Q4_K_M.jlm"
	}
	path := modelPath(t, name)
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
	const n = 400
	reply := func(id string, move func()) ([]int32, int) {
		t.Helper()
		if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: id, MaxSeq: 512}); err != nil {
			t.Fatal(err)
		}
		// Every block on the first device to start from: a model that fits is
		// spread over both, and a target already holding blocks is prepared
		// at the commit rather than ahead.
		pl, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: id}))
		if err != nil {
			t.Fatal(err)
		}
		last := int32(len(pl.Msg.GetPlacement().GetBlocks()) - 1)
		setup, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{
			SessionId: id, FirstBlock: 0, LastBlock: last, ToDeviceId: devs[0]}))
		if err != nil {
			t.Fatal(err)
		}
		// A block the first device refused stays where it was, and the reply
		// says so: the move below would then start from two devices and
		// prepare nothing ahead on the one already holding blocks.
		if d := setup.Msg.GetDeclined(); len(d) > 0 {
			t.Skipf("NO DEVICE ROOM: %s refused %d of %d blocks (block %d: %s) -- this gate proved nothing",
				devs[0], len(d), last+1, d[0].GetBlockIndex(), d[0].GetReason())
		}
		st, err := c.inference.Generate(ctx, req(&v1.GenerateRequest{SessionId: id, Prompt: text(story),
			MaxTokens: n, IgnoreEos: true}))
		if err != nil {
			t.Fatal(err)
		}
		defer st.Close()
		var ids []int32
		landed := make(chan struct{})
		after := -1
		for st.Receive() {
			if tok, ok := st.Msg().GetEvent().(*v1.GenerateResponse_Token); ok && tok.Token.GetTokenId() >= 0 {
				ids = append(ids, tok.Token.GetTokenId())
				if len(ids) == 8 && move != nil {
					go func() { move(); close(landed) }()
				}
			}
			select {
			case <-landed:
				if after < 0 {
					after = len(ids)
				}
			default:
			}
		}
		if err := st.Err(); err != nil {
			t.Fatal(err)
		}
		if move != nil {
			<-landed
		}
		return ids, after
	}
	ctl, _ := reply("ctl", nil)
	if err := e.CloseSession("ctl"); err != nil {
		t.Fatal(err)
	}
	var m *v1.RelocateBlocksResponse
	var from, to string
	var nb int32
	got, after := reply("moved", func() {
		pl, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: "moved"}))
		if err != nil {
			t.Error(err)
			return
		}
		from, to = devs[0], devs[1]
		nb = int32(len(pl.Msg.GetPlacement().GetBlocks()))
		for _, b := range pl.Msg.GetPlacement().GetBlocks() {
			if b.GetDeviceId() != from {
				t.Errorf("block %d is on %q before the move, not %s", b.GetIndex(), b.GetDeviceId(), from)
			}
		}
		r, err := c.placement.RelocateBlocks(ctx, req(&v1.RelocateBlocksRequest{
			SessionId: "moved", FirstBlock: 0, LastBlock: nb - 1, ToDeviceId: to}))
		if err != nil {
			t.Error(err)
			return
		}
		m = r.Msg
	})
	if m == nil {
		t.Fatal("the move did not return")
	}
	t.Logf("%s -> %s during a reply: landed by token %d of %d, %d block(s) prepared ahead in %d ms, "+
		"paused %d us, %s via %s", from, to, after, n, m.GetStagedBlocks(), m.GetPrepareMillis(),
		m.GetPauseMicros(), m.GetCopied().GetHuman(), m.GetPath())
	switch {
	case after < 0 || after >= n:
		t.Fatalf("the move returned after the reply (token %d of %d): it waited for the reply, not a token", after, n)
	case m.GetBlocksMoved() != nb || m.GetStagedBlocks() != nb:
		t.Fatalf("moved %d and prepared %d of %d blocks ahead", m.GetBlocksMoved(), m.GetStagedBlocks(), nb)
	case m.GetPath() == v1.RelocationPath_RELOCATION_PATH_HOST || m.GetCopied().GetBytes() == 0:
		t.Fatalf("the history went %v, %d bytes copied: the direct path did not run", m.GetPath(), m.GetCopied().GetBytes())
	}
	for _, blk := range m.GetPlacement().GetBlocks() {
		if blk.GetDeviceId() != to {
			t.Fatalf("block %d is on %q after a move to %s", blk.GetIndex(), blk.GetDeviceId(), to)
		}
	}
	sameKernels := strings.SplitN(from, ":", 2)[0] == strings.SplitN(to, ":", 2)[0]
	if len(got) != len(ctl) || sameKernels && !slices.Equal(got, ctl) {
		t.Fatalf("the moved reply (%d tokens) is not the unmoved one (%d)", len(got), len(ctl))
	}
}

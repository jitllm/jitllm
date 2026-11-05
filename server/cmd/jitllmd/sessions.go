package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/nn"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// cmdSessions is SessionService. Placement is per session and `run -model`
// makes an ephemeral one, so a named session is what `place` needs to
// address.
func cmdSessions(ctx context.Context, c cli, args []string) error {
	fs := flag.NewFlagSet("jitllmd sessions", flag.ContinueOnError)
	addr := addrFlag(fs)
	modelID := fs.String("model", "", "filter the list, or the model a -new session runs")

	create := fs.Bool("new", false, "create a session against -model")
	id := fs.String("id", "", "a stable id for -new (default: generated)")
	maxSeq := fs.Int("max-seq", 0, "KV capacity in positions for -new; 0 takes the model's context length")
	devices := fs.String("devices", "", "devices for -new, comma-separated; empty inherits the model's default")
	gpuLayers := fs.Int("gpu-layers", -1, "at most this many blocks on a device for -new; -1 is as many as fit")
	grow := fs.Bool("gpu-grow", false, "adopt one block per token while serving, as memory frees")
	keepOff := fs.Bool("keep-off-host", false, "for -new: never hand a block to the host when the context outgrows the device")
	promptCache := fs.Bool("prompt-cache", false, "for -new: keep prompt prefixes in the server's -kv-cache store")
	cacheKey := fs.String("cache-key", "", "for -new: what the prompt store may share between sessions (needs -prompt-cache)")

	closeID := fs.String("close", "", "a session id to close")
	resetID := fs.String("reset", "", "a session id to reset to position 0")
	queue := fs.String("queue", "", "report the queue on a device id")

	if err := parse(c, fs, args); err != nil {
		return err
	}
	cn, err := target(ctx, c, fs, *addr)
	if err != nil {
		return err
	}

	switch {
	case *create:
		if *modelID == "" {
			return usagef("sessions -new needs -model")
		}
		req := &v1.CreateSessionRequest{
			ModelId:              *modelID,
			SessionId:            *id,
			MaxSeq:               int32(*maxSeq),
			DeviceIds:            csv(*devices),
			RelocateWhileServing: *grow,
			KeepOffHost:          *keepOff,
			PromptCache:          *promptCache,
			CacheKey:             *cacheKey,
		}
		if wasSet(fs, "gpu-layers") {
			v := int32(*gpuLayers)
			req.MaxDeviceBlocks = &v
		}
		resp, err := cn.Session.CreateSession(ctx, connect.NewRequest(req))
		if err != nil {
			return clientError("creating a session", cn.base, err)
		}
		printSession(c.out, resp.Msg.GetSession(), resp.Msg.GetBudget())
		return nil

	case *closeID != "":
		if _, err := cn.Session.CloseSession(ctx, connect.NewRequest(&v1.CloseSessionRequest{
			SessionId: *closeID,
		})); err != nil {
			return clientError("closing "+*closeID, cn.base, err)
		}
		fmt.Fprintf(c.out, "closed %s\n", *closeID)
		return nil

	case *resetID != "":
		resp, err := cn.Session.ResetSession(ctx, connect.NewRequest(&v1.ResetSessionRequest{
			SessionId: *resetID,
		}))
		if err != nil {
			return clientError("resetting "+*resetID, cn.base, err)
		}
		printSession(c.out, resp.Msg.GetSession(), nil)
		return nil

	case *queue != "":
		resp, err := cn.Session.GetDeviceQueue(ctx, connect.NewRequest(&v1.GetDeviceQueueRequest{
			DeviceId: *queue,
		}))
		if err != nil {
			return clientError("reading the queue on "+*queue, cn.base, err)
		}
		m := resp.Msg
		fmt.Fprintf(c.out, "device %s   %s   %d waiting, running=%v\n",
			m.GetDeviceId(), execName(m.GetExecution()), m.GetWaiting(), m.GetRunning())
		for i, s := range m.GetSessionIds() {
			fmt.Fprintf(c.out, "  %d. %s\n", i, s)
		}
		// The note says why the device serialises sessions.
		if n := m.GetNote(); n != "" {
			fmt.Fprintf(c.out, "  %s\n", n)
		}
		return nil
	}

	if want := strings.Join(fs.Args(), " "); want != "" {
		resp, err := cn.Session.GetSession(ctx, connect.NewRequest(&v1.GetSessionRequest{
			SessionId: want,
		}))
		if err != nil {
			return clientError("describing "+want, cn.base, err)
		}
		printSession(c.out, resp.Msg.GetSession(), resp.Msg.GetBudget())
		return nil
	}

	resp, err := cn.Session.ListSessions(ctx, connect.NewRequest(&v1.ListSessionsRequest{
		ModelId: *modelID,
	}))
	if err != nil {
		return clientError("listing sessions", cn.base, err)
	}
	ss := resp.Msg.GetSessions()
	fmt.Fprintf(c.out, "%d session(s)\n", len(ss))
	for _, s := range ss {
		where := nn.CPU
		if ids := s.GetDeviceIds(); len(ids) > 0 {
			where = strings.Join(ids, ",")
		}
		fmt.Fprintf(c.out, "  %-20s %-20s pos %-6d %2d dev / %2d host block(s)  kv %-11s [%s]\n",
			s.GetSessionId(), s.GetModelId(), s.GetPosition(),
			s.GetDeviceBlocks(), s.GetHostBlocks(), bytesOf(s.GetKvBytes()), where)
	}
	return nil
}

func printSession(w io.Writer, s *v1.Session, b *v1.SessionBudget) {
	if s == nil {
		return
	}
	where := nn.CPU
	if ids := s.GetDeviceIds(); len(ids) > 0 {
		where = strings.Join(ids, ",")
	}
	fmt.Fprintf(w, "\nsession %s   model %s\n", s.GetSessionId(), s.GetModelId())
	fmt.Fprintf(w, "  position   %d of %d max_seq\n", s.GetPosition(), s.GetMaxSeq())
	fmt.Fprintf(w, "  blocks     %d on %s, %d on cpu\n", s.GetDeviceBlocks(), where, s.GetHostBlocks())
	fmt.Fprintf(w, "  kv         %s\n", bytesOf(s.GetKvBytes()))
	state := "idle"
	if s.GetGenerating() {
		state = "generating"
	}
	fmt.Fprintf(w, "  state      %s, %s, queue position %d\n",
		state, execName(s.GetExecution()), s.GetQueuePosition())
	fmt.Fprintf(w, "  tokens     %d generated, %d prefilled\n",
		s.GetTokensGenerated(), s.GetTokensPrefilled())
	fmt.Fprintf(w, "  last used  %s\n",
		time.UnixMilli(s.GetLastUsedUnixMillis()).Format(time.RFC3339))

	if b == nil {
		return
	}
	fmt.Fprintf(w, "  kv budget  %s per session, %s in use across %d open\n",
		bytesOf(b.GetKvBytesPerSession()), bytesOf(b.GetKvBytesInUse()), b.GetSessionsOpen())
	if !b.GetSessionsPerDeviceMeasured() {
		// Print the unmeasured note so kv_bytes_* is not read as an admission
		// limit.
		fmt.Fprintf(w, "  ★ %s\n", b.GetUnmeasuredNote())
	}
}

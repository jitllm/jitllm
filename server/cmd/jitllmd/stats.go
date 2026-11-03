package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// cmdStats is TelemetryService: what the engine is doing right now. Every
// number is a counter the engine keeps (pool regions, expert dispatch,
// page-ins, bytes read), never a timing either side took.
func cmdStats(ctx context.Context, c cli, args []string) error {
	fs := flag.NewFlagSet("jitllmd stats", flag.ContinueOnError)
	addr := addrFlag(fs)
	modelID := fs.String("model", "", "only this model")
	sessionID := fs.String("session", "", "only this session")
	watch := fs.Bool("watch", false, "stream a snapshot on an interval instead of taking one")
	interval := fs.Int("interval", 1000, "milliseconds between snapshots, with -watch")
	if err := parse(c, fs, args); err != nil {
		return err
	}
	cn, err := target(ctx, c, fs, *addr)
	if err != nil {
		return err
	}

	info, err := cn.Telemetry.GetServerInfo(ctx, connect.NewRequest(&v1.GetServerInfoRequest{}))
	if err != nil {
		return clientError("reading server info", cn.base, err)
	}
	printServerInfo(c.out, info.Msg.GetInfo())

	if !*watch {
		resp, err := cn.Telemetry.GetStats(ctx, connect.NewRequest(&v1.GetStatsRequest{
			ModelId: *modelID, SessionId: *sessionID,
		}))
		if err != nil {
			return clientError("reading stats", cn.base, err)
		}
		printStats(c.out, resp.Msg)
		return nil
	}

	// -watch is a server-pushed stream, printed as it arrives. The server
	// clamps the interval to a floor.
	st, err := cn.Telemetry.WatchStats(ctx, connect.NewRequest(&v1.WatchStatsRequest{
		ModelId: *modelID, SessionId: *sessionID, IntervalMillis: int32(*interval),
	}))
	if err != nil {
		// A cancel that lands while the stream is opening is still the caller's
		// ^C, not a failure to open.
		if ctx.Err() != nil {
			return nil
		}
		return clientError("watching stats", cn.base, err)
	}
	defer st.Close()
	for st.Receive() {
		var b strings.Builder
		// Milliseconds, because the interval floor is 250 ms: at second
		// resolution a stalled watch would look like a working one.
		fmt.Fprintf(&b, "\n---- %s\n", time.UnixMilli(st.Msg().GetUnixMillis()).Format("15:04:05.000"))
		printStats(&b, st.Msg())
		if err := emit(c.out, b.String()); err != nil {
			return err
		}
	}
	// WatchStats streams until cancelled, so ^C is the normal way out and
	// exits 0.
	if err := st.Err(); err != nil && ctx.Err() == nil {
		return clientError("watching stats", cn.base, err)
	}
	return nil
}

func printServerInfo(w io.Writer, i *v1.ServerInfo) {
	if i == nil {
		return
	}
	backends := strings.Join(i.GetAvailableBackends(), ", ")
	if backends == "" {
		backends = "none"
	}
	fmt.Fprintf(w, "jitllmd %s   up %s   %d model(s), %d session(s)   backends %s\n",
		i.GetVersion(), (time.Duration(i.GetUptimeMillis()) * time.Millisecond).Round(time.Second),
		i.GetLoadedModels(), i.GetOpenSessions(), backends)
	fmt.Fprintf(w, "build    %s %s/%s   api %d\n", i.GetRevision(), i.GetOs(), i.GetArch(), i.GetApiVersion())
	fmt.Fprintf(w, "models   %s\n", i.GetModelDirectory())
}

func printStats(w io.Writer, s *v1.GetStatsResponse) {
	for _, m := range s.GetModels() {
		fmt.Fprintf(w, "\nmodel %s   %d session(s)   %d tok generated, %d prefilled\n",
			m.GetModelId(), m.GetSessions(), m.GetTokensGenerated(), m.GetTokensPrefilled())
		printResidency(w, "  ", m.GetResidency())
		printBatch(w, "  ", m.GetBatch())
	}
	for _, ss := range s.GetSessions() {
		state := "idle"
		if ss.GetGenerating() {
			state = "generating"
		}
		if q := ss.GetQueuePosition(); q > 0 {
			state = fmt.Sprintf("waiting, %d ahead", q)
		}
		fmt.Fprintf(w, "\nsession %s   model %s   pos %d   %s\n",
			ss.GetSessionId(), ss.GetModelId(), ss.GetPosition(), state)
		fmt.Fprintf(w, "  tokens   %d generated, %d prefilled\n",
			ss.GetTokensGenerated(), ss.GetTokensPrefilled())
		if p := ss.GetPool(); p != nil {
			// The parallel/inline split matters more than the total: a
			// parallel region ends at a barrier the other cores wait through.
			fmt.Fprintf(w, "  pool     %d parallel / %d inline region(s) over %d worker(s)\n",
				p.GetParallelRegions(), p.GetInlineRegions(), p.GetWorkers())
		}
	}
}

// printBatch is the model's step loop. Joint rows and solo rows are both
// printed: a row stepped alone is a lone request or a session the device could
// not take with the others, which is a fact about the placement, not a fault.
func printBatch(w io.Writer, pad string, b *v1.BatchStats) {
	if b == nil {
		return
	}
	fmt.Fprintf(w, "%sbatch    %d live / %d waiting (width %d)   %d step(s), %d joint\n",
		pad, b.GetLiveRows(), b.GetWaiting(), b.GetWidth(), b.GetSteps(), b.GetJointSteps())
	fmt.Fprintf(w, "%s         rows %d joint / %d solo, at most %d row(s) of %d session(s) a step\n",
		pad, b.GetJointRows(), b.GetSoloRows(), b.GetMaxRowsPerStep(), b.GetMaxSessionsPerStep())
	fmt.Fprintf(w, "%s         prompt %d chunk(s), %d step(s) beside sessions decoding\n",
		pad, b.GetPromptChunks(), b.GetPromptSteps())
	fmt.Fprintf(w, "%s         %d admitted, %d ms waiting in all\n",
		pad, b.GetAdmissions(), b.GetAdmissionWaitMillis())
	if n := b.GetJointRefusals(); n > 0 {
		fmt.Fprintf(w, "%s         %d joint step(s) refused, last: %s\n", pad, n, b.GetLastRefusal())
	}
	fmt.Fprintf(w, "%s         %d step(s) one session after another by choice\n", pad, b.GetSeparateSteps())
	for _, c := range b.GetChoices() {
		how := "joint"
		if !c.GetJoint() {
			how = "one after another"
		}
		if !c.GetSettled() {
			how = "probing"
		}
		fmt.Fprintf(w, "%s         %d row(s): %s (joint %.2f ms, separate %.2f ms a step, %d probe(s))\n",
			pad, c.GetRows(), how, c.GetJointStepMillis(), c.GetSeparateStepMillis(), c.GetProbes())
	}
}

func printResidency(w io.Writer, pad string, r *v1.Residency) {
	if r == nil {
		return
	}
	// fits is printed as a boolean because one page short of fitting is a
	// cliff: one eviction and one page read every token.
	verdict := "PAGES -- one eviction and one re-read per token"
	if r.GetFits() {
		verdict = "FITS -- nothing is ever evicted, so a token reads nothing"
	}
	fmt.Fprintf(w, "%sresident %s of %s budget   %d of %d block(s) in %d frame(s)   %s\n",
		pad, bytesOf(r.GetResident()), bytesOf(r.GetBudget()),
		r.GetResidentBlocks(), r.GetBlocks(), r.GetFrames(), verdict)
	fmt.Fprintf(w, "%spager    %d page-in(s), %d eviction(s), %s read in %d request(s)\n",
		pad, r.GetPageIns(), r.GetEvictions(), bytesOf(r.GetBytesRead()), r.GetReadRequests())
}

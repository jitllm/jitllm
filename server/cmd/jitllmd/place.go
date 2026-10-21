package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// cmdPlace is PlacementService, whole: the seam, relocation, seam tuning and
// the pager, forcible from outside the process.
//
// Placement is per session (SetDeviceLayers is a method on model.State), so
// -session is the usual argument; a model's default does not move a running
// session. -model addresses the pager, which is per model.
func cmdPlace(ctx context.Context, c cli, args []string) error {
	fs := flag.NewFlagSet("jitllmd place", flag.ContinueOnError)
	addr := addrFlag(fs)
	sessionID := fs.String("session", "", "the session whose blocks to report or move")
	modelID := fs.String("model", "", "the model whose pager residency to report or re-budget")

	devices := fs.String("devices", "", "devices to offer blocks to, comma-separated; empty brings everything home")
	gpuLayers := fs.Int("gpu-layers", -1, "at most this many blocks on a device; -1 is as many as fit")
	head := fs.Bool("head-on-device", false, "place the output projection on a device too")

	move := fs.String("move", "", "a block range to relocate, FIRST:LAST or a single index")
	to := fs.String("to", "", "where -move goes: `host`, or a device id -- another of the session's devices moves the blocks between them, history and all")
	relocate := fs.Bool("relocate", false, "adopt one block per token while serving, as memory frees")

	tuneSeam := fs.Bool("tune-seam", false, "measure whether fewer device blocks are faster, and relocate")
	warmup := fs.Int("warmup", 0, "tokens to discard before timing, with -tune-seam; 0 takes the engine's schedule")
	perRun := fs.Int("tokens", 0, "tokens to time per run, with -tune-seam")
	rounds := fs.Int("rounds", 0, "interleaved rounds, with -tune-seam")

	maxmem := fs.String("maxmem", "", "a new page budget for -model, e.g. 8G")

	if err := parse(c, fs, args); err != nil {
		return err
	}
	if *sessionID == "" && *modelID == "" {
		return usagef("place: give -session (the block seam) or -model (the pager budget). " +
			"Placement is per session because that is where the engine puts it; the pager is per " +
			"model because a page is a whole block of one container")
	}
	if *sessionID != "" && *modelID != "" {
		return usagef("place: -session and -model address different halves of this service; " +
			"give one")
	}
	cn, err := target(ctx, c, fs, *addr)
	if err != nil {
		return err
	}

	if *modelID != "" {
		return pagerBudget(ctx, c, cn, *modelID, *maxmem)
	}

	switch {
	case *move != "":
		return relocateBlocks(ctx, c, cn, *sessionID, *move, *to)
	case wasSet(fs, "relocate"):
		resp, err := cn.Placement.SetRelocation(ctx, connect.NewRequest(&v1.SetRelocationRequest{
			SessionId: *sessionID, Enabled: *relocate,
		}))
		if err != nil {
			return clientError("setting relocation", cn.base, err)
		}
		fmt.Fprintf(c.out, "relocation while serving: %v\n", resp.Msg.GetEnabled())
		return nil
	case *tuneSeam:
		return tuneTheSeam(ctx, c, cn, *sessionID, int32(*warmup), int32(*perRun), int32(*rounds))
	case wasSet(fs, "devices") || wasSet(fs, "gpu-layers") || wasSet(fs, "head-on-device"):
		req := &v1.SetPlacementRequest{SessionId: *sessionID, DeviceIds: csv(*devices)}
		if wasSet(fs, "gpu-layers") {
			v := int32(*gpuLayers)
			req.MaxDeviceBlocks = &v
		}
		if wasSet(fs, "head-on-device") {
			v := *head
			req.HeadOnDevice = &v
		}
		resp, err := cn.Placement.SetPlacement(ctx, connect.NewRequest(req))
		if err != nil {
			return clientError("moving the seam", cn.base, err)
		}
		fmt.Fprintf(c.out, "moved the seam in %s\n",
			(time.Duration(resp.Msg.GetMillis()) * time.Millisecond).Round(time.Millisecond))
		printPlacement(c.out, resp.Msg.GetPlacement())
		return nil
	}

	resp, err := cn.Placement.GetPlacement(ctx, connect.NewRequest(&v1.GetPlacementRequest{
		SessionId: *sessionID,
	}))
	if err != nil {
		return clientError("reading placement", cn.base, err)
	}
	printPlacement(c.out, resp.Msg.GetPlacement())
	return nil
}

func relocateBlocks(ctx context.Context, c cli, cn *conn, session, spec, to string) error {
	first, last, err := blockRange(spec)
	if err != nil {
		return err
	}
	req := &v1.RelocateBlocksRequest{
		SessionId: session, FirstBlock: first, LastBlock: last,
	}
	switch {
	case to == "" || to == "host" || to == "cpu":
		req.ToHost = true
	default:
		req.ToDeviceId = to
	}
	resp, err := cn.Placement.RelocateBlocks(ctx, connect.NewRequest(req))
	if err != nil {
		// A hole in the seam, a device the session was not loaded with and a
		// host-only model are declined by name; print the server's reason.
		return clientError("relocating blocks", cn.base, err)
	}
	m := resp.Msg
	fmt.Fprintf(c.out, "moved %d block(s) in %s\n", m.GetBlocksMoved(),
		(time.Duration(m.GetMillis()) * time.Millisecond).Round(time.Millisecond))
	// Between devices the history travelled with the blocks; say how much
	// and by which path, so a pause can be read against the bytes.
	if p := m.GetPath(); p != v1.RelocationPath_RELOCATION_PATH_UNSPECIFIED {
		line := fmt.Sprintf("  history %s via %s, the sequence waited %s", m.GetHistory().GetHuman(),
			relocationPath(p), (time.Duration(m.GetPauseMicros()) * time.Microsecond).Round(10*time.Microsecond))
		if n := m.GetStagedBlocks(); n > 0 {
			line += fmt.Sprintf("; %d block(s) prepared beside its tokens in %s", n,
				(time.Duration(m.GetPrepareMillis()) * time.Millisecond).Round(time.Millisecond))
		}
		if m.GetHeadMoved() {
			line += "; the output projection followed"
		}
		fmt.Fprintln(c.out, line)
	}
	// A refused block is reported, not an error: the rest of the range may
	// still have moved.
	for _, d := range m.GetDeclined() {
		fmt.Fprintf(c.out, "  block %d declined by %s: %s%s\n",
			d.GetBlockIndex(), d.GetDeviceId(), d.GetReason(), oomNote(d.GetOutOfMemory()))
	}
	printPlacement(c.out, m.GetPlacement())
	return nil
}

// relocationPath names a RelocationPath for a person.
func relocationPath(p v1.RelocationPath) string {
	switch p {
	case v1.RelocationPath_RELOCATION_PATH_HOST:
		return "the host"
	case v1.RelocationPath_RELOCATION_PATH_STAGED:
		return "a host buffer"
	case v1.RelocationPath_RELOCATION_PATH_PEER:
		return "peer access, card to card"
	}
	return strings.ToLower(strings.TrimPrefix(p.String(), "RELOCATION_PATH_"))
}

func tuneTheSeam(ctx context.Context, c cli, cn *conn, session string, warmup, perRun, rounds int32) error {
	st, err := cn.Placement.TuneSeam(ctx, connect.NewRequest(&v1.TuneSeamRequest{
		SessionId: session, WarmupTokens: warmup, TokensPerRun: perRun, Rounds: rounds,
	}))
	if err != nil {
		return clientError("tuning the seam", cn.base, err)
	}
	defer st.Close()
	// Rounds are printed as they land, since TuneSeam measures over real
	// tokens and takes time. Each ratio carries its IQR.
	var last *v1.Placement
	for st.Receive() {
		m := st.Msg()
		if p := m.GetPlacement(); p != nil {
			last = p
		}
		verdict := "rejected"
		if m.GetAccepted() {
			verdict = "ACCEPTED"
		}
		line := fmt.Sprintf("round %2d  %3d device block(s)  ratio %.4f  IQR/median %.3f  %s",
			m.GetRound(), m.GetCandidateDeviceBlocks(), m.GetMedianRatio(),
			m.GetIqrOverMedian(), verdict)
		if n := m.GetNote(); n != "" {
			line += "  -- " + n
		}
		if err := emit(c.out, line+"\n"); err != nil {
			return err
		}
		if m.GetDone() {
			break
		}
	}
	if err := st.Err(); err != nil {
		return clientError("tuning the seam", cn.base, err)
	}
	printPlacement(c.out, last)
	return nil
}

func pagerBudget(ctx context.Context, c cli, cn *conn, modelID, maxmem string) error {
	if maxmem == "" {
		resp, err := cn.Placement.GetResidency(ctx, connect.NewRequest(&v1.GetResidencyRequest{
			ModelId: modelID,
		}))
		if err != nil {
			return clientError("reading residency", cn.base, err)
		}
		fmt.Fprintf(c.out, "model %s\n", modelID)
		printResidency(c.out, "  ", resp.Msg.GetResidency())
		return nil
	}
	b, err := tier.ParseBytes(maxmem)
	if err != nil {
		return usagef("-maxmem: %v", err)
	}
	resp, err := cn.Placement.SetPageBudget(ctx, connect.NewRequest(&v1.SetPageBudgetRequest{
		ModelId: modelID, BudgetBytes: b,
	}))
	if err != nil {
		return clientError("setting the page budget", cn.base, err)
	}
	fmt.Fprintf(c.out, "model %s\n", modelID)
	printResidency(c.out, "  ", resp.Msg.GetResidency())
	if !resp.Msg.GetFits() {
		// A budget one page short does not error; it runs far slower, since
		// every token evicts a block and re-reads it.
		fmt.Fprintf(c.out, "  ★ this budget does NOT hold every block. Being one page short is a "+
			"cliff, not a gradient: the same model measured 10 tok/s one page short and 18 when "+
			"it fit\n")
	}
	return nil
}

// blockRange parses FIRST:LAST, or a single index.
func blockRange(spec string) (int32, int32, error) {
	parse1 := func(s string) (int32, error) {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			return 0, usagef("-move %q: %q is not a block index", spec, s)
		}
		return int32(n), nil
	}
	i := strings.IndexByte(spec, ':')
	if i < 0 {
		n, err := parse1(spec)
		return n, n, err
	}
	first, err := parse1(spec[:i])
	if err != nil {
		return 0, 0, err
	}
	last, err := parse1(spec[i+1:])
	if err != nil {
		return 0, 0, err
	}
	if last < first {
		return 0, 0, usagef("-move %q: the range runs backwards", spec)
	}
	return first, last, nil
}

// printPlacement prints the block map as runs of (location, device, kind), so
// the seam reads in a few lines and the block kind stays visible.
func printPlacement(w io.Writer, p *v1.Placement) {
	if p == nil {
		return
	}
	fmt.Fprintf(w, "\nsession %s   model %s\n", p.GetSessionId(), p.GetModelId())
	blocks := p.GetBlocks()

	type run struct {
		first, last int32
		where, kind string
		bytes       string
		resident    int
		total       int
	}
	var runs []run
	for _, b := range blocks {
		where := "host"
		if b.GetLocation() == v1.BlockLocation_BLOCK_LOCATION_DEVICE {
			where = b.GetDeviceId()
			if where == "" {
				where = "device"
			}
		}
		kind := kindName(b.GetKind())
		res := 0
		if b.GetHostResident() {
			res = 1
		}
		if n := len(runs); n > 0 && runs[n-1].where == where && runs[n-1].kind == kind &&
			runs[n-1].last == b.GetIndex()-1 {
			runs[n-1].last = b.GetIndex()
			runs[n-1].resident += res
			runs[n-1].total++
			continue
		}
		runs = append(runs, run{
			first: b.GetIndex(), last: b.GetIndex(), where: where, kind: kind,
			bytes: bytesOf(b.GetBytes()), resident: res, total: 1,
		})
	}
	for _, r := range runs {
		span := fmt.Sprintf("%d", r.first)
		if r.last != r.first {
			span = fmt.Sprintf("%d..%d", r.first, r.last)
		}
		fmt.Fprintf(w, "  blocks %-9s %-10s %-10s %s each   %d of %d page(s) resident on the host\n",
			span, r.where, r.kind, r.bytes, r.resident, r.total)
	}
	head := "host"
	if p.GetHeadOnDevice() {
		head = "device"
	}
	fmt.Fprintf(w, "  %d device / %d host block(s), output projection on the %s\n",
		p.GetDeviceBlockCount(), p.GetHostBlockCount(), head)
	fmt.Fprintf(w, "  counters: %d demotion(s), %d relocation(s), %d reclaim(s), relocating=%v\n",
		p.GetDemotions(), p.GetRelocations(), p.GetReclaims(), p.GetRelocating())

	// Declines are printed by name: the only one that should exist is "does
	// not fit", and anything else is a kernel that is owed.
	if d := p.GetDeclines(); len(d) > 0 {
		fmt.Fprintf(w, "  %d decline(s):\n", len(d))
		for _, x := range d {
			fmt.Fprintf(w, "    block %d on %s: %s%s\n",
				x.GetBlockIndex(), x.GetDeviceId(), x.GetReason(), oomNote(x.GetOutOfMemory()))
		}
	}
}

func oomNote(oom bool) string {
	if oom {
		return "   (capacity, not capability)"
	}
	return "   (a CAPABILITY the device lacks -- a kernel that is owed, not a memory problem)"
}

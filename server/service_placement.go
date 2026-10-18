package server

import (
	"context"
	"errors"
	"fmt"
	"time"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// PlacementService implements jitllm.v1.PlacementService: the engine's block
// placement, seam, relocation, seam tuning and page budget, reachable from
// outside the process.
type PlacementService struct{ E *Engine }

func (s *PlacementService) GetPlacement(ctx context.Context, req *connect.Request[v1.GetPlacementRequest]) (*connect.Response[v1.GetPlacementResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.GetPlacementResponse{Placement: s.E.pbPlacement(sess)}), nil
}

// SetPlacement moves the seam. It takes the session's lock, so it waits for
// a generate to finish rather than mutating the State mid-Forward.
// reach makes every device named part of the session's model's tier and puts
// the session's State on it: how a model loaded host-only, or without the
// device asked for, is moved there later (Engine.attachDevice).
func (s *PlacementService) reach(sess *Session, ids ...string) error {
	for _, id := range ids {
		if err := s.E.attachDevice(sess.lm, id); err != nil {
			// A device that does not exist, or that will not open, is a
			// precondition the caller can change, not a server fault.
			var de *DeviceError
			if errors.As(err, &de) {
				return connect.NewError(connect.CodeFailedPrecondition, err)
			}
			return connectErr(err)
		}
	}
	if len(ids) > 0 {
		if err := s.E.bindSession(sess); err != nil {
			return connectErr(err)
		}
	}
	return nil
}

func (s *PlacementService) SetPlacement(ctx context.Context, req *connect.Request[v1.SetPlacementRequest]) (*connect.Response[v1.SetPlacementResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	m := req.Msg
	want := normaliseDeviceIDs(m.DeviceIds)
	if err := s.reach(sess, want...); err != nil {
		return nil, err
	}

	start := time.Now()
	sess.mu.Lock()
	if len(want) == 0 {
		// Everything home.
		sess.st.SetGPULayers(0)
	} else {
		max := -1
		if m.MaxDeviceBlocks != nil {
			max = int(*m.MaxDeviceBlocks)
		}
		if max < 0 {
			if err := sess.st.SetDeviceLayers(sess.lm.device(), -1); err != nil {
				sess.mu.Unlock()
				return nil, connect.NewError(connect.CodeFailedPrecondition, err)
			}
		} else {
			sess.st.SetGPULayers(max)
		}
	}
	if m.HeadOnDevice != nil {
		sess.st.SetHeadOnDevice(*m.HeadOnDevice)
	}
	sess.refresh()
	sess.mu.Unlock()

	return connect.NewResponse(&v1.SetPlacementResponse{
		Placement: s.E.pbPlacement(sess),
		Millis:    millis(time.Since(start)),
	}), nil
}

// RelocateBlocks moves a contiguous block range while serving: between the
// host and the devices, and between the devices of the session's tier.
//
// A range that is wholly on devices and names a device moves there, each
// block's history with it (Engine.MoveBlocks). Otherwise the engine's seam is
// a count (SetGPULayers(n): the first n blocks are on the device), so a range
// that extends or trims the device set is honoured. These are refused with
// their reason:
//
//	a hole in the middle   the seam is a count; there is no "blocks 3..7 on
//	                       the card and 8..12 at home" to set.
//
// A device the model's tier was not opened with, and a model loaded host-only,
// are not refused: the device joins the model's tier (Engine.attachDevice).
func (s *PlacementService) RelocateBlocks(ctx context.Context, req *connect.Request[v1.RelocateBlocksRequest]) (*connect.Response[v1.RelocateBlocksResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	m := req.Msg
	nLayer := sess.lm.m.Cfg.NLayer
	first, last := int(m.FirstBlock), int(m.LastBlock)
	if first < 0 || last < first || last >= nLayer {
		return nil, invalid("relocate: block range %d..%d is outside 0..%d", first, last, nLayer-1)
	}
	if !m.ToHost && m.ToDeviceId == "" {
		return nil, invalid("relocate: set to_host, or name a to_device_id")
	}
	if m.ToDeviceId != "" {
		if err := s.reach(sess, m.ToDeviceId); err != nil {
			return nil, err
		}
	}
	if m.ToDeviceId != "" && sess.onDevices(first, last) {
		return s.moveBetweenDevices(ctx, sess, first, last, m.ToDeviceId)
	}

	start := time.Now()
	sess.mu.Lock()
	var moved int
	var declined []*v1.PlacementDecline
	if m.ToHost {
		n, err := sess.st.ReleaseBlocks(first, last+1)
		moved = n
		if err != nil {
			declined = append(declined, &v1.PlacementDecline{BlockIndex: int32(first), Reason: err.Error()})
		}
	} else {
		n, err := sess.st.PlaceBlocks(first, last+1)
		moved = n
		if err != nil {
			declined = append(declined, &v1.PlacementDecline{BlockIndex: int32(first), DeviceId: m.ToDeviceId, Reason: err.Error()})
		} else if want := last + 1 - first; n < want && !s.allOnDevices(sess, first, last) {
			declined = append(declined, &v1.PlacementDecline{
				BlockIndex:  int32(first),
				DeviceId:    m.ToDeviceId,
				Reason:      fmt.Sprintf("the device took %d of the %d block(s) asked for", n, want),
				OutOfMemory: true,
			})
		}
	}
	sess.refresh()
	sess.mu.Unlock()

	// The tier chose a card for what it placed; a named card gets the blocks
	// that landed elsewhere, moved there with their history.
	if !m.ToHost && moved > 0 && m.ToDeviceId != "" {
		if lo, hi, ok := s.elsewhere(sess, first, last, m.ToDeviceId); ok {
			if _, err := s.moveBetweenDevices(ctx, sess, lo, hi, m.ToDeviceId); err != nil {
				declined = append(declined, &v1.PlacementDecline{BlockIndex: int32(lo), DeviceId: m.ToDeviceId, Reason: err.Error()})
			}
		}
	}

	return connect.NewResponse(&v1.RelocateBlocksResponse{
		Placement:   s.E.pbPlacement(sess),
		BlocksMoved: int32(moved),
		Declined:    declined,
		Millis:      millis(time.Since(start)),
	}), nil
}

// allOnDevices reports whether blocks first..last are all on a device now.
func (s *PlacementService) allOnDevices(sess *Session, first, last int) bool {
	return sess.onDevices(first, last)
}

// elsewhere is the hull of the blocks in first..last that sit on a device
// other than want, when that hull is wholly on devices.
func (s *PlacementService) elsewhere(sess *Session, first, last int, want string) (lo, hi int, ok bool) {
	on := sess.placement().on
	lo, hi = -1, -1
	for li := first; li <= last && li < len(on); li++ {
		if on[li] == "" || s.E.gateKey(on[li]) == s.E.gateKey(want) {
			continue
		}
		if lo < 0 {
			lo = li
		}
		hi = li
	}
	if lo < 0 || !sess.onDevices(lo, hi) {
		return 0, 0, false
	}
	return lo, hi, true
}

func (s *PlacementService) SetRelocation(ctx context.Context, req *connect.Request[v1.SetRelocationRequest]) (*connect.Response[v1.SetRelocationResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	sess.mu.Lock()
	sess.st.SetRelocate(req.Msg.Enabled)
	sess.refresh()
	sess.mu.Unlock()
	return connect.NewResponse(&v1.SetRelocationResponse{Enabled: req.Msg.Enabled}), nil
}

// TuneSeam arms the engine's own measured re-placement and reports what it
// settles on. seamtune measures over runs of real tokens, so it only
// advances while the session is generating; this call arms it and reports its
// state without decoding throwaway tokens.
func (s *PlacementService) TuneSeam(ctx context.Context, req *connect.Request[v1.TuneSeamRequest], st *connect.ServerStream[v1.TuneSeamResponse]) error {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return connectErr(err)
	}
	if sess.lm.device() == nil {
		return connect.NewError(connect.CodeFailedPrecondition, fmt.Errorf(
			"session %q is host-only: there is no seam to tune", sess.id))
	}
	sess.mu.Lock()
	sess.st.SetSeamTuning(true)
	m := req.Msg
	sess.st.SetSeamSchedule(int(m.WarmupTokens), int(m.TokensPerRun), int(m.Rounds))
	sess.refresh()
	sess.mu.Unlock()

	if err := st.Send(&v1.TuneSeamResponse{
		Round: 0,
		Note: "the tuner is armed. It measures over RUNS OF TOKENS and has none of its own, " +
			"so it advances only while this session is generating; keep this stream open and " +
			"send generate requests on the session.",
	}); err != nil {
		return err
	}

	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	round := int32(0)
	lastBlocks := -1
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
			// The snapshot, not the State: Forward writes the tuner with no
			// lock, so it is read as of the session's last refresh.
			blocks, settled := int(sess.snapSeam.Load()), sess.snapSeamSettled.Load()
			if blocks == lastBlocks && !settled {
				continue
			}
			lastBlocks = blocks
			round++
			resp := &v1.TuneSeamResponse{
				Round:                 round,
				CandidateDeviceBlocks: int32(blocks),
				Accepted:              settled,
				Done:                  settled,
				Placement:             s.E.pbPlacement(sess),
			}
			if settled {
				resp.Note = "settled"
			}
			if err := st.Send(resp); err != nil {
				return err
			}
			if settled {
				return nil
			}
		}
	}
}

func (s *PlacementService) GetResidency(ctx context.Context, req *connect.Request[v1.GetResidencyRequest]) (*connect.Response[v1.GetResidencyResponse], error) {
	lm, err := s.E.Model(req.Msg.ModelId)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.GetResidencyResponse{Residency: pbResidency(lm)}), nil
}

// SetPageBudget retargets the pager. The response carries fits because a
// budget one page short is a cliff, not a gradient (see pbResidency).
func (s *PlacementService) SetPageBudget(ctx context.Context, req *connect.Request[v1.SetPageBudgetRequest]) (*connect.Response[v1.SetPageBudgetResponse], error) {
	lm, err := s.E.Model(req.Msg.ModelId)
	if err != nil {
		return nil, connectErr(err)
	}
	// The request names a total and pins it: the other models divide what is
	// left (Engine.Pin), and every session created afterwards re-divides it
	// (Engine.applyPageBudget). What the pager gets is that total less the
	// never-paged dense region and less what a tier on the host's own memory
	// holds (Model.SetPageBudget); Residency.budget reports what was actually
	// applied, so the response is never a repeat of the request. Zero releases
	// the pin back to the engine's division of the host budget.
	if err := s.E.Pin(lm.id, req.Msg.BudgetBytes); err != nil {
		return nil, connectErr(err)
	}
	res := pbResidency(lm)
	return connect.NewResponse(&v1.SetPageBudgetResponse{Residency: res, Fits: res.Fits}), nil
}

func sameSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for _, x := range a {
		if !hasString(b, x) {
			return false
		}
	}
	return true
}

// moveBetweenDevices is RelocateBlocks for a range already on the session's
// devices.
func (s *PlacementService) moveBetweenDevices(ctx context.Context, sess *Session, first, last int, to string) (*connect.Response[v1.RelocateBlocksResponse], error) {
	start := time.Now()
	mv, err := s.E.MoveBlocks(ctx, sess, first, last, to)
	if err != nil {
		return nil, connectErr(err)
	}
	var declined []*v1.PlacementDecline
	for _, li := range mv.Declined {
		declined = append(declined, &v1.PlacementDecline{
			BlockIndex: int32(li), DeviceId: to,
			Reason: "the device refused the block; it stays where it was, with its history",
		})
	}
	return connect.NewResponse(&v1.RelocateBlocksResponse{
		Placement:     s.E.pbPlacement(sess),
		BlocksMoved:   int32(len(mv.Moved)),
		Declined:      declined,
		Millis:        millis(time.Since(start)),
		Path:          pbRelocationPath(mv.Path),
		History:       pbBytes(mv.KVBytes + mv.RecBytes),
		HeadMoved:     mv.Head,
		Copied:        pbBytes(mv.Copied),
		PauseMicros:   mv.Pause.Microseconds(),
		StagedBlocks:  int32(mv.Staged),
		PrepareMillis: mv.Prepare.Milliseconds(),
	}), nil
}

// pbRelocationPath is model.BlockMove.Path on the wire.
func pbRelocationPath(p string) v1.RelocationPath {
	switch p {
	case "host":
		return v1.RelocationPath_RELOCATION_PATH_HOST
	case "staged":
		return v1.RelocationPath_RELOCATION_PATH_STAGED
	case "peer":
		return v1.RelocationPath_RELOCATION_PATH_PEER
	}
	return v1.RelocationPath_RELOCATION_PATH_UNSPECIFIED
}

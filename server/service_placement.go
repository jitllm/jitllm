package server

import (
	"context"
	"errors"
	"fmt"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/nn"
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
// a generate to finish rather than mutating the instance mid-Forward.
// reach makes every device named part of the session's model's tier and puts
// the session's instance on it: how a model loaded host-only, or without the
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
	// The caller placed this session: the orchestrator leaves it alone.
	sess.pinned.Store(true)

	// The placement is the instance's: every session leasing it moves with it.
	start := time.Now()
	sess.mu.Lock()
	var perr error
	ierr := sess.in().Place(func(st *engine.Instance) {
		if len(want) == 0 {
			// Everything on the CPU.
			st.SetGPULayers(0)
		} else {
			max := -1
			if m.MaxDeviceBlocks != nil {
				max = int(*m.MaxDeviceBlocks)
			}
			if max < 0 {
				if perr = st.SetDeviceLayers(sess.lm.device(), -1); perr != nil {
					return
				}
			} else {
				st.SetGPULayers(max)
			}
		}
		if m.HeadOnDevice != nil {
			st.SetHeadOnDevice(*m.HeadOnDevice)
		}
	})
	if perr == nil {
		perr = ierr
	}
	if perr != nil {
		sess.mu.Unlock()
		return nil, connect.NewError(connect.CodeFailedPrecondition, perr)
	}
	sess.refresh()
	sess.mu.Unlock()

	return connect.NewResponse(&v1.SetPlacementResponse{
		Placement: s.E.pbPlacement(sess),
		Millis:    millis(time.Since(start)),
	}), nil
}

// RelocateBlocks moves a contiguous block range while serving, to any device
// of the session's tier or to the CPU ("cpu"), each block's history with it
// (engine.Instance.MoveBlocks).
//
// A range wholly on devices that names a card moves there through
// Engine.MoveBlocks, so a generating session keeps generating while the
// target prepares. A card the model's tier was not opened with, and a model
// loaded host-only, are not refused: the card joins the model's tier
// (Engine.attachDevice).
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
	to := m.ToDeviceId
	if to == "" {
		return nil, invalid("relocate: name a to_device_id (%q for the CPU)", nn.CPU)
	}
	cpu := nn.DeviceName(to) == nn.CPU
	if !cpu {
		if err := s.reach(sess, to); err != nil {
			return nil, err
		}
	}
	// The caller placed this session: the orchestrator leaves it alone.
	sess.pinned.Store(true)
	if !cpu && sess.onDevices(first, last) {
		return s.moveBetweenDevices(ctx, sess, first, last, to)
	}

	start := time.Now()
	sess.mu.Lock()
	var mv engine.BlockMove
	if perr := sess.in().Place(func(st *engine.Instance) { mv, err = st.MoveBlocks(first, last+1, to) }); perr != nil {
		err = perr
	}
	sess.refresh()
	sess.mu.Unlock()
	dev := to
	if cpu {
		dev = ""
	}
	var declined []*v1.PlacementDecline
	if err != nil {
		declined = append(declined, &v1.PlacementDecline{BlockIndex: int32(first), DeviceId: dev, Reason: err.Error()})
	}
	for _, li := range mv.Declined {
		declined = append(declined, &v1.PlacementDecline{
			BlockIndex:  int32(li),
			DeviceId:    dev,
			Reason:      "the device refused the block; it stays where it was, with its history",
			OutOfMemory: true,
		})
	}
	return connect.NewResponse(&v1.RelocateBlocksResponse{
		Placement:   s.E.pbPlacement(sess),
		BlocksMoved: int32(len(mv.Moved)),
		Declined:    declined,
		Millis:      millis(time.Since(start)),
		Path:        pbRelocationPath(mv.Path),
		History:     pbBytes(mv.KVBytes + mv.RecBytes),
		HeadMoved:   mv.Head,
		Copied:      pbBytes(mv.Copied),
		PauseMicros: mv.Pause.Microseconds(),
	}), nil
}

func (s *PlacementService) SetRelocation(ctx context.Context, req *connect.Request[v1.SetRelocationRequest]) (*connect.Response[v1.SetRelocationResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	sess.mu.Lock()
	sess.in().With(func(st *engine.Instance) { st.SetRelocate(req.Msg.Enabled) })
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
	m := req.Msg
	sess.in().With(func(st *engine.Instance) {
		st.SetSeamTuning(true)
		st.SetSeamSchedule(int(m.WarmupTokens), int(m.TokensPerRun), int(m.Rounds))
	})
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
			// The snapshot, not the instance: Forward writes the tuner with no
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
	// never-paged dense region, the model's instance's page quota, less what
	// a tier on the host's own memory holds (Instance.SetPageBudget);
	// Residency.budget reports what was actually
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

// pbRelocationPath is engine.BlockMove.Path on the wire.
func pbRelocationPath(p string) v1.RelocationPath {
	switch p {
	case "host":
		return v1.RelocationPath_RELOCATION_PATH_CPU
	case "staged":
		return v1.RelocationPath_RELOCATION_PATH_STAGED
	case "peer":
		return v1.RelocationPath_RELOCATION_PATH_PEER
	}
	return v1.RelocationPath_RELOCATION_PATH_UNSPECIFIED
}

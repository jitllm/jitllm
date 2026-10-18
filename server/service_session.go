package server

import (
	"context"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// SessionService implements jitllm.v1.SessionService.
type SessionService struct{ E *Engine }

// CreateSession opens a session on a loaded model and returns it with the
// budget it was placed under.
func (s *SessionService) CreateSession(ctx context.Context, req *connect.Request[v1.CreateSessionRequest]) (*connect.Response[v1.CreateSessionResponse], error) {
	m := req.Msg
	o := SessionOptions{
		ModelID:   m.ModelId,
		SessionID: m.SessionId,
		MaxSeq:    int(m.MaxSeq),
		DeviceIDs: m.DeviceIds,
		Relocate:  m.RelocateWhileServing,

		KeepOffHost: m.KeepOffHost,
		PromptCache: m.PromptCache,
		CacheKey:    m.CacheKey,
	}
	if m.MaxDeviceBlocks != nil {
		o.MaxDeviceBlocks = int(*m.MaxDeviceBlocks)
	}
	if sp := pbSampling(m.Sampling); sp != nil {
		o.Sampling = *sp
	}
	if sp := pbSpeculation(m.Speculation); sp != nil {
		o.Speculation = *sp
	}
	sess, err := s.E.CreateSession(o)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.CreateSessionResponse{
		Session: s.E.pbSession(sess),
		Budget:  s.E.pbBudget(sess.lm, sess.snapKV.Load()),
	}), nil
}

// GetSession describes one open session and its budget.
func (s *SessionService) GetSession(ctx context.Context, req *connect.Request[v1.GetSessionRequest]) (*connect.Response[v1.GetSessionResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.GetSessionResponse{
		Session: s.E.pbSession(sess),
		Budget:  s.E.pbBudget(sess.lm, sess.snapKV.Load()),
	}), nil
}

// ListSessions lists the open sessions, filtered by model and by device when
// the request names them.
func (s *SessionService) ListSessions(ctx context.Context, req *connect.Request[v1.ListSessionsRequest]) (*connect.Response[v1.ListSessionsResponse], error) {
	out := &v1.ListSessionsResponse{}
	// A session is on a device when its model's spec names it as written, or
	// when its tier opened the physical device that id resolves to (`cuda:1`
	// lists a session of a model loaded with `cuda`).
	dev, key := req.Msg.DeviceId, ""
	if dev != "" {
		key = s.E.gateKey(dev)
	}
	for _, sess := range s.E.Sessions() {
		if id := req.Msg.ModelId; id != "" && sess.modelID != id {
			continue
		}
		if dev != "" && !hasString(sess.lm.devIDs(), dev) && !hasString(sess.lm.gateIDs(), key) {
			continue
		}
		out.Sessions = append(out.Sessions, s.E.pbSession(sess))
	}
	return connect.NewResponse(out), nil
}

// CloseSession closes a session (Engine.CloseSession).
func (s *SessionService) CloseSession(ctx context.Context, req *connect.Request[v1.CloseSessionRequest]) (*connect.Response[v1.CloseSessionResponse], error) {
	if err := s.E.CloseSession(req.Msg.SessionId); err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.CloseSessionResponse{}), nil
}

// ResetSession empties a session's sequence, waiting for a generate in flight
// on it to finish first.
func (s *SessionService) ResetSession(ctx context.Context, req *connect.Request[v1.ResetSessionRequest]) (*connect.Response[v1.ResetSessionResponse], error) {
	sess, err := s.E.Session(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	// Reset mutates the State, so it waits for any generate in flight rather
	// than racing it.
	sess.mu.Lock()
	sess.st.Reset()
	sess.refresh()
	sess.mu.Unlock()
	return connect.NewResponse(&v1.ResetSessionResponse{Session: s.E.pbSession(sess)}), nil
}

// GetDeviceQueue reports the sessions running on a device. Nothing queues for
// a device as a whole; requests wait only for a row of a full step loop.
func (s *SessionService) GetDeviceQueue(ctx context.Context, req *connect.Request[v1.GetDeviceQueueRequest]) (*connect.Response[v1.GetDeviceQueueResponse], error) {
	id := req.Msg.DeviceId
	if id == "" {
		id = HostGateID
	}
	queue, running, waiting := s.E.deviceQueue(s.E.gateKey(id))
	return connect.NewResponse(&v1.GetDeviceQueueResponse{
		DeviceId:   id,
		Execution:  pbExecution(ExecutionParallel),
		SessionIds: queue,
		Running:    running > 0,
		Waiting:    int32(max(0, waiting)),
		Note:       executionNote,
	}), nil
}

func hasString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

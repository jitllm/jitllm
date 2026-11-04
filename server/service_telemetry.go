package server

import (
	"context"
	"encoding/json"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/capabilities"
	"github.com/jitllm/jitllm/internal/localserver"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TelemetryService implements jitllm.v1.TelemetryService. Every number here
// is a counter the engine already keeps, never a timing this server took:
// counters are facts about what ran, not perf claims.
type TelemetryService struct{ E *Engine }

func (s *TelemetryService) GetStats(ctx context.Context, req *connect.Request[v1.GetStatsRequest]) (*connect.Response[v1.GetStatsResponse], error) {
	return connect.NewResponse(s.snapshot(req.Msg.ModelId, req.Msg.SessionId)), nil
}

func (s *TelemetryService) WatchStats(ctx context.Context, req *connect.Request[v1.WatchStatsRequest], st *connect.ServerStream[v1.GetStatsResponse]) error {
	iv := time.Duration(req.Msg.IntervalMillis) * time.Millisecond
	if iv < 250*time.Millisecond {
		iv = 250 * time.Millisecond
	}
	if err := st.Send(s.snapshot(req.Msg.ModelId, req.Msg.SessionId)); err != nil {
		return err
	}
	tick := time.NewTicker(iv)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
			if err := st.Send(s.snapshot(req.Msg.ModelId, req.Msg.SessionId)); err != nil {
				return err
			}
		}
	}
}

func (s *TelemetryService) snapshot(modelID, sessionID string) *v1.GetStatsResponse {
	out := &v1.GetStatsResponse{UnixMillis: time.Now().UnixMilli()}
	for _, lm := range s.E.Models() {
		if modelID != "" && lm.id != modelID {
			continue
		}
		out.Models = append(out.Models, &v1.ModelStats{
			ModelId:         lm.id,
			Residency:       pbResidency(lm),
			Sessions:        int32(s.E.sessionCount(lm)),
			TokensGenerated: lm.tokensGenerated.Load(),
			TokensPrefilled: lm.tokensPrefilled.Load(),
			Batch:           lm.loop.pb(),
		})
	}
	for _, sess := range s.E.Sessions() {
		if sessionID != "" && sess.id != sessionID {
			continue
		}
		if modelID != "" && sess.modelID != modelID {
			continue
		}
		ss := &v1.SessionStats{
			SessionId:       sess.id,
			ModelId:         sess.modelID,
			TokensGenerated: sess.generated.Load(),
			TokensPrefilled: sess.prefilled.Load(),
			Position:        sess.snapPos.Load(),
			Generating:      sess.running.Load(),
			QueuePosition:   sess.queuePos.Load(),
		}
		// The pool's counters are written by the decode loop, so
		// they are read only when the session is idle; a busy session
		// reports zeros rather than race.
		if !sess.running.Load() && sess.mu.TryLock() {
			var par, ser int64
			var workers int
			sess.in().With(func(st *model.Instance) {
				par, ser = st.Pool().Regions()
				workers = st.Workers()
			})
			ss.Pool = &v1.PoolStats{
				ParallelRegions: par,
				InlineRegions:   ser,
				Workers:         int32(workers),
			}
			sess.mu.Unlock()
		}
		out.Sessions = append(out.Sessions, ss)
	}
	if devs, err := s.E.Devices(false); err == nil {
		for _, d := range devs {
			out.Devices = append(out.Devices, s.E.pbDevice(d))
		}
	}
	host := s.E.Host()
	out.Host = pbHost(host)
	out.Orchestrator = s.E.pbOrchestrator()
	return out
}

// GetServerInfo is what a client checks before it relies on this server: the
// build, the control plane's compatibility version, the platform, the models
// folder, the devices and the capability manifest.
func (s *TelemetryService) GetServerInfo(ctx context.Context, req *connect.Request[v1.GetServerInfoRequest]) (*connect.Response[v1.GetServerInfoResponse], error) {
	id := capabilities.Identity(s.E.cfg.Version)
	info := &v1.ServerInfo{
		Version:           s.E.cfg.Version,
		Revision:          id.Revision,
		Modified:          id.Modified,
		ApiVersion:        localserver.APIVersion,
		Os:                id.OS,
		Arch:              id.Arch,
		ModelDirectory:    s.E.ModelDir(),
		StartedUnixMillis: s.E.started.UnixMilli(),
		UptimeMillis:      millis(time.Since(s.E.started)),
		LoadedModels:      int32(len(s.E.Models())),
		OpenSessions:      int32(len(s.E.Sessions())),
		Features:          Features(),
	}
	if devs, err := s.E.Devices(false); err == nil {
		for _, d := range devs {
			if d.Available {
				info.AvailableBackends = append(info.AvailableBackends, d.ID)
			}
			info.Devices = append(info.Devices, s.E.pbDevice(d))
		}
	}
	m, err := capabilities.WithAPI(s.E.cfg.Version, API())
	if err != nil {
		return nil, connectErr(err)
	}
	m.ServerAPI = localserver.APIVersion
	b, err := json.Marshal(m)
	if err != nil {
		return nil, connectErr(err)
	}
	info.CapabilitiesJson = string(b)
	return connect.NewResponse(&v1.GetServerInfoResponse{Info: info}), nil
}

// Features names what this server does beyond the manifest, for a client to
// switch its own features on by. A name is added, never renamed.
func Features() []string {
	return []string{"server_info", "model_aliases", "model_enable", "on_demand_load",
		// The native generate path carries what the OpenAI routes do
		// (docs/server.md, "The native API and the OpenAI routes").
		"chat_tools", "response_format", "logprobs", "choices", "image_url", "stream_usage",
		"model_capabilities"}
}

package server

import (
	"fmt"
	"strings"
	"time"

	"github.com/jitllm/jitllm/engine/model"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// Conversions between the engine's Go types and the generated protobuf ones.
// They live in one file so a field added to a proto message has one place to
// be filled in, rather than a handler each.

func humanBytes(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

func pbBytes(b uint64) *v1.ByteSize {
	return &v1.ByteSize{Bytes: b, Human: humanBytes(b)}
}

func pbBackend(name string) v1.Backend {
	switch name {
	case "cpu":
		return v1.Backend_BACKEND_CPU
	case "cuda":
		return v1.Backend_BACKEND_CUDA
	case "vulkan":
		return v1.Backend_BACKEND_VULKAN
	case "metal":
		return v1.Backend_BACKEND_METAL
	case "hip":
		return v1.Backend_BACKEND_HIP
	}
	return v1.Backend_BACKEND_UNSPECIFIED
}

func pbKind(k DeviceKind) v1.DeviceKind {
	switch k {
	case KindHost:
		return v1.DeviceKind_DEVICE_KIND_HOST
	case KindDiscrete:
		return v1.DeviceKind_DEVICE_KIND_DISCRETE
	case KindIntegrated:
		return v1.DeviceKind_DEVICE_KIND_INTEGRATED
	}
	return v1.DeviceKind_DEVICE_KIND_UNSPECIFIED
}

func pbExecution(m ExecutionMode) v1.ExecutionMode {
	switch m {
	case ExecutionParallel:
		return v1.ExecutionMode_EXECUTION_MODE_PARALLEL
	case ExecutionSerialised:
		return v1.ExecutionMode_EXECUTION_MODE_SERIALISED
	}
	return v1.ExecutionMode_EXECUTION_MODE_UNSPECIFIED
}

func pbFinish(r FinishReason) v1.FinishReason {
	switch r {
	case FinishStop:
		return v1.FinishReason_FINISH_REASON_STOP
	case FinishMaxTokens:
		return v1.FinishReason_FINISH_REASON_MAX_TOKENS
	case FinishEOS:
		return v1.FinishReason_FINISH_REASON_EOS
	case FinishCancelled:
		return v1.FinishReason_FINISH_REASON_CANCELLED
	case FinishError:
		return v1.FinishReason_FINISH_REASON_ERROR
	}
	return v1.FinishReason_FINISH_REASON_UNSPECIFIED
}

func (e *Engine) pbDevice(d DeviceInfo) *v1.Device {
	key := e.infoGateKey(d)
	_, running, waiting := e.deviceQueue(key)
	attached := 0
	e.mu.RLock()
	for _, s := range e.sessions {
		if hasString(s.lm.gateIDs(), key) {
			attached++
		}
	}
	e.mu.RUnlock()

	return &v1.Device{
		Ref: &v1.DeviceRef{
			Id:      d.ID,
			Backend: pbBackend(d.Backend),
			Index:   int32(d.Index),
		},
		Name:                   d.Name,
		Kind:                   pbKind(d.Kind),
		TotalMemory:            pbBytes(d.TotalMemory),
		FreeMemory:             pbBytes(d.FreeMemory),
		CountsTowardHostBudget: d.CountsTowardHostBudget,
		Execution:              pbExecution(ExecutionParallel),
		ExecutionNote:          executionNote,
		AttachedSessions:       int32(attached),
		RunningSessions:        int32(running),
		QueuedRequests:         int32(max(0, waiting)),
		Available:              d.Available,
		UnavailableReason:      d.Why,
		PhysicalId:             d.PhysicalID,
		SameDeviceAs:           d.SameDeviceAs,
	}
}

func pbHost(h HostInfo) *v1.HostMemory {
	return &v1.HostMemory{
		WeightBudget:     pbBytes(h.WeightBudget),
		Total:            pbBytes(h.Total),
		Available:        pbBytes(h.Available),
		DecodeCores:      int32(h.DecodeCores),
		PerformanceCores: int32(h.PCores),
		EfficiencyCores:  int32(h.ECores),
		SmtSiblings:      int32(h.SMTSiblings),
	}
}

func (e *Engine) pbModelInfo(lm *LoadedModel) *v1.ModelInfo {
	c := lm.m.Cfg
	attn, lin := c.NLayer, 0
	if c.Hybrid() {
		attn, lin = 0, 0
		for i := 0; i < c.NLayer; i++ {
			if c.LayerKind(i).Recurrent() {
				lin++
			} else {
				attn++
			}
		}
	}
	var names []string
	if lm.m.HasChatTemplate() {
		// The container may carry several templates; the default is the first
		// and is reachable by the empty name.
		if _, ok := lm.m.ChatTemplate(""); ok {
			names = append(names, "default")
		}
	}
	visBlocks := 0
	if t := lm.m.Tower(); t != nil {
		visBlocks = t.Cfg.NLayer
	}

	pooling := ""
	if lm.m.IsEmbedding() {
		pooling = lm.m.Pooling().String()
	}

	e.mu.RLock()
	sessions := len(lm.sessions)
	e.mu.RUnlock()

	return &v1.ModelInfo{
		ModelId:           lm.id,
		Path:              lm.path,
		Name:              lm.name,
		Architecture:      c.Arch,
		BlockCount:        int32(c.NLayer),
		EmbeddingDim:      int32(c.NEmbd),
		ContextLength:     int32(c.NCtx),
		VocabSize:         int32(c.NVocab),
		ExpertCount:       int32(c.NExpert),
		ExpertsUsed:       int32(c.NExpertUsed),
		AttentionBlocks:   int32(attn),
		LinearBlocks:      int32(lin),
		HasVisionTower:    lm.m.Tower() != nil,
		VisionBlockCount:  int32(visBlocks),
		ChatCapable:       lm.m.ChatCapable(),
		ChatTemplateNames: names,
		PageSize:          pbBytes(lm.m.PageSize()),
		WeightBytes:       pbBytes(lm.m.WeightBytes()),
		SessionCount:      int32(sessions),
		DefaultDeviceIds:  lm.devIDs(),
		Pooling:           pooling,
		Encoder:           lm.m.IsEncoder(),
		Aliases:           e.aliasesOf(lm.path),
		DefaultMaxSeq:     int32(e.defaultMaxSeq(lm)),
		Capabilities: modelCapabilities(lm.m.ChatCapable(), lm.m.Tower() != nil,
			lm.m.IsEmbedding(), lm.m.IsEncoder()),
	}
}

// pbSession reads only the published snapshot and the atomics: model.State is
// one sequence written by Forward with no lock, so a stats handler must never
// touch it while a generate is in flight. See the snap* fields on session.
func (e *Engine) pbSession(s *Session) *v1.Session {
	devBlocks := s.snapDevBlocks.Load()
	return &v1.Session{
		SessionId:          s.id,
		ModelId:            s.modelID,
		MaxSeq:             int32(s.maxSeq),
		Position:           s.snapPos.Load(),
		DeviceIds:          s.lm.devIDs(),
		DeviceBlocks:       devBlocks,
		HostBlocks:         int32(s.lm.m.Cfg.NLayer) - devBlocks,
		KvBytes:            pbBytes(s.snapKV.Load()),
		Execution:          pbExecution(sessionMode(s)),
		QueuePosition:      s.queuePos.Load(),
		Generating:         s.running.Load(),
		CreatedUnixMillis:  s.created.UnixMilli(),
		LastUsedUnixMillis: s.lastUsed.Load(),
		TokensGenerated:    s.generated.Load(),
		TokensPrefilled:    s.prefilled.Load(),
	}
}

// pbBudget answers "how many sessions fit" as far as it is known. KV is
// charged per session and the device ledger cannot express it, so sessions
// per device is unmeasured and kv_bytes is reported as an observation, never
// an admission limit.
func (e *Engine) pbBudget(lm *LoadedModel, per uint64) *v1.SessionBudget {
	e.mu.RLock()
	n := len(lm.sessions)
	var inUse uint64
	for _, s := range lm.sessions {
		inUse += s.snapKV.Load()
	}
	e.mu.RUnlock()
	return &v1.SessionBudget{
		KvBytesPerSession: pbBytes(per),
		KvBytesInUse:      pbBytes(inUse),
		SessionsOpen:      int32(n),
		// Always false until someone measures it; see the comment above.
		SessionsPerDeviceMeasured: false,
		UnmeasuredNote: "how many sessions a device holds at a given MaxSeq has not been " +
			"measured: KV is charged per session and the device ledger cannot express it. " +
			"Treat kv_bytes_per_session as an observation, never as an admission limit.",
	}
}

func pbBlockKind(c *model.Config, i int) v1.BlockKind {
	if c.Hybrid() && c.LayerKind(i).Recurrent() {
		return v1.BlockKind_BLOCK_KIND_LINEAR
	}
	return v1.BlockKind_BLOCK_KIND_ATTENTION
}

func (e *Engine) pbPlacement(s *Session) *v1.Placement {
	lm := s.lm
	c := lm.m.Cfg
	snap := s.placement()
	onDev := map[int]bool{}
	for _, run := range snap.runs {
		for i := run[0]; i < run[1]; i++ {
			onDev[i] = true
		}
	}
	// The tier names the device each block is on (model.State.DeviceOf);
	// with one device the id is the one the model was loaded with, which
	// is also what an unnamed device reports.
	single := ""
	if len(lm.devIDs()) == 1 {
		single = lm.devIDs()[0]
	}

	blocks := make([]*v1.BlockPlacement, 0, c.NLayer)
	devCount := 0
	for i := 0; i < c.NLayer; i++ {
		bp := &v1.BlockPlacement{
			Index:        int32(i),
			Kind:         pbBlockKind(c, i),
			Location:     v1.BlockLocation_BLOCK_LOCATION_HOST,
			Bytes:        pbBytes(lm.m.PageBytes(i)),
			HostResident: true,
		}
		if i < len(snap.hist) {
			h := snap.hist[i]
			bp.HistoryPerPosition, bp.HistoryWindow, bp.StateBytes = pbBytes(h.perPos), int32(h.window), pbBytes(h.state)
		}
		if onDev[i] {
			bp.Location = v1.BlockLocation_BLOCK_LOCATION_DEVICE
			bp.DeviceId = single
			if i < len(snap.on) && snap.on[i] != "" {
				bp.DeviceId = snap.on[i]
			}
			bp.HostResident = false
			devCount++
		}
		blocks = append(blocks, bp)
	}

	var declines []*v1.PlacementDecline
	for _, d := range snap.declines {
		declines = append(declines, &v1.PlacementDecline{
			DeviceId: single,
			Reason:   fmt.Sprintf("%d block(s): %s", d.Blocks, d.Why),
			// The one decline that should exist is "does not fit"; anything
			// else is a kernel that is owed, and the flag keeps that visible.
			OutOfMemory: isCapacityDecline(d.Why),
		})
	}

	return &v1.Placement{
		SessionId:        s.id,
		ModelId:          lm.id,
		Blocks:           blocks,
		Declines:         declines,
		HostBlockCount:   int32(c.NLayer - devCount),
		DeviceBlockCount: int32(devCount),
		HeadOnDevice:     snap.headOnDev,
		Demotions:        int64(snap.demotions),
		Relocations:      int64(snap.relocations),
		Reclaims:         int64(snap.reclaims),
		Relocating:       snap.relocating,
	}
}

func isCapacityDecline(why string) bool {
	for _, s := range []string{"budget", "memory", "does not fit", "no room", "out of"} {
		if strings.Contains(strings.ToLower(why), s) {
			return true
		}
	}
	return false
}

func pbResidency(lm *LoadedModel) *v1.Residency {
	m := lm.m
	frames, in, out := m.PageStats()
	resident := m.HostResidentBlocks()
	blocks := m.Cfg.NLayer
	if t := m.Tower(); t != nil {
		blocks += t.Cfg.NLayer
	}
	budget := m.PageBudget()
	// fits is a boolean because the step is a cliff: one block short of
	// resident is one eviction and one page read every token. It is the
	// budget's answer, not a count of frames: a model that fits has none
	// resident until its first token, and one under a tiny budget has none
	// resident right after the re-budget evicted them.
	fits := m.PagesFit()
	return &v1.Residency{
		ModelId:        lm.id,
		Resident:       pbBytes(m.HostBytes()),
		Budget:         pbBytes(budget),
		Frames:         int32(frames),
		Blocks:         int32(blocks),
		ResidentBlocks: int32(resident),
		Fits:           fits,
		PageIns:        in,
		Evictions:      out,
		BytesRead:      pbBytes(m.BytesRead()),
		ReadRequests:   m.PagerReads(),
	}
}

// pbSpeculation is the wire's speculation, nil when the request carries none.
func pbSpeculation(p *v1.SpeculationParams) *Speculation {
	if p == nil {
		return nil
	}
	return &Speculation{Enabled: p.Enabled, Draft: max(int(p.DraftTokens), 0)}
}

func pbSampling(p *v1.SamplingParams) *model.Sampler {
	if p == nil {
		return nil
	}
	s := &model.Sampler{}
	if p.Temperature != nil {
		s.Temp = float64(*p.Temperature)
	}
	if p.TopP != nil {
		s.TopP = float64(*p.TopP)
	}
	if p.TopK != nil {
		// -1 disables top-k, as the OpenAI surface spells it.
		s.TopK = max(int(*p.TopK), 0)
	}
	if p.MinP != nil {
		s.MinP = float64(*p.MinP)
	}
	if p.Seed != nil {
		s.Seed = int64(*p.Seed)
	}
	if p.RepeatPenalty != nil {
		s.RepeatPen = float64(*p.RepeatPenalty)
	}
	if p.RepeatLastN != nil {
		s.RepeatLastN = int(*p.RepeatLastN)
	}
	return s
}

func millis(d time.Duration) int64 { return d.Milliseconds() }

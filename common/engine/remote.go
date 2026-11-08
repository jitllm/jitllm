package engine

import (
	"context"
	"errors"
	"fmt"
	"mime"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/server/client"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// Remote is the engine as a client of a jitllmd: the second implementation of
// what the screens drive, over the service's control plane (server/client,
// the generated ConnectRPC stubs). The model list, the memory map, block
// placement and chat publish into the same State the embedded Engine writes,
// so the screens cannot tell the two apart except by the badge.
//
// A service that stops answering is a problem on screen with a Reconnect
// button, never a silent switch to an engine of the app's own: the app stays
// connected to the address it was given until the person chooses otherwise.
//
// What a service cannot do from here: the memory priority between models is
// the service's own policy (SetPriority says so), and a picture is sent as
// its bytes.
type Remote struct {
	sh Front
	st State
	c  *client.Conn

	onWhere func(Where)

	cmds chan func()
	done chan struct{}
	once sync.Once
	quit chan struct{}

	busyJobs atomic.Int32

	mu    sync.Mutex
	where Where
	// stop cancels the generate in flight; nil when none runs.
	stop context.CancelFunc
	// closed is set by Close; post sends nothing after it.
	closed bool

	// Worker goroutine only.
	ids     map[string]string // path -> the service's model id
	active  string            // the chat's model path
	session string            // the chat's session on it
	window  int
}

// heartbeat is how often a connected app asks the service whether it is still
// there, so a lost service shows before the next click finds it.
const heartbeat = 2 * time.Second

// NewRemote starts a client of c, whose Info is nil when the service did not
// answer: it then starts lost, showing the problem, and recovers on its own
// when the service comes up. onWhere is told every change of what the badge
// says. Close stops it; the service and its models stay.
func NewRemote(sh Front, st State, c *client.Conn, addr string, onWhere func(Where)) *Remote {
	r := &Remote{sh: sh, st: st, c: c, onWhere: onWhere,
		cmds: make(chan func(), queueDepth), done: make(chan struct{}), quit: make(chan struct{}),
		ids:   map[string]string{},
		where: Where{Service: true, Addr: addr, Version: c.Info.GetVersion()}}
	go r.run()
	r.publishWhere()
	if c.Info == nil {
		r.lost(errors.New("it did not answer when the app started"))
	} else {
		r.post("connect", r.refresh)
	}
	go r.watch()
	return r
}

// Where is the service and whether it answers, for the badge.
func (r *Remote) Where() Where {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.where
}

func (r *Remote) publishWhere() {
	if r.onWhere != nil {
		r.onWhere(r.Where())
	}
}

func (r *Remote) run() {
	defer close(r.done)
	for fn := range r.cmds {
		fn()
	}
}

// post queues work for the worker, as Engine.post does.
func (r *Remote) post(what string, fn func()) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return false
	}
	select {
	case r.cmds <- fn:
		return true
	default:
		r.sh.Post(func() { r.sh.SetStatus(what + ": busy, try again in a moment") })
		return false
	}
}

func (r *Remote) busyBegin() { r.busyJobs.Add(1); r.st.Busy.Set(true) }
func (r *Remote) busyEnd() {
	r.busyJobs.Add(-1)
	r.sh.Post(func() { r.st.Busy.Set(r.busyJobs.Load() > 0) })
}

func (r *Remote) status(f string, a ...any) {
	msg := fmt.Sprintf(f, a...)
	r.sh.Post(func() { r.sh.SetStatus(msg) })
}

// ---------------------------------------------------------------- the link

// watch asks the service for its ServerInfo every heartbeat: a failure shows
// the loss, and an answer after one clears it.
func (r *Remote) watch() {
	t := time.NewTicker(heartbeat)
	defer t.Stop()
	for {
		select {
		case <-r.quit:
			return
		case <-t.C:
			r.ping()
		}
	}
}

// ping is one heartbeat.
func (r *Remote) ping() bool {
	ctx, cancel := context.WithTimeout(context.Background(), heartbeat)
	defer cancel()
	resp, err := r.c.Telemetry.GetServerInfo(ctx, connect.NewRequest(&v1.GetServerInfoRequest{}))
	if err == nil {
		// A service that speaks another API version is refused, not used.
		err = localserver.Compatible(int(resp.Msg.GetInfo().GetApiVersion()))
	}
	if err != nil {
		r.lost(err)
		return false
	}
	r.mu.Lock()
	was := r.where.Lost
	r.where.Lost = false
	r.where.Version = resp.Msg.GetInfo().GetVersion()
	r.mu.Unlock()
	if was {
		r.publishWhere()
		r.sh.Post(func() { r.st.Problem.Set(session.Problem{}) })
		r.status("connected to jitllmd at %s again", r.where.Addr)
		r.post("reconnect", r.refresh)
	}
	return true
}

// Reconnect asks the service again now, rather than at the next heartbeat.
func (r *Remote) Reconnect() { go r.ping() }

// lost shows the service as not answering. It is idempotent: the heartbeat
// calls it every beat while the service is down.
func (r *Remote) lost(err error) {
	r.mu.Lock()
	was := r.where.Lost
	r.where.Lost = true
	addr := r.where.Addr
	r.mu.Unlock()
	if was {
		return
	}
	r.publishWhere()
	r.sh.Report(session.Problem{
		Title: "jitllmd at " + addr + " is not answering",
		Detail: errText(err) + ". The app stays connected to it and starts no engine of its own; " +
			"it reconnects when the service is back, or choose Embedded in Settings.",
		Action: "Reconnect", Do: r.Reconnect,
	})
}

// isLost is an error that says the service is gone rather than that it
// refused what was asked.
func isLost(err error) bool {
	switch connect.CodeOf(err) {
	case connect.CodeUnavailable, connect.CodeDeadlineExceeded:
		return true
	case connect.CodeUnknown:
		return !errors.Is(err, context.Canceled)
	}
	return false
}

func errText(err error) string {
	var ce *connect.Error
	if errors.As(err, &ce) {
		return ce.Message()
	}
	return err.Error()
}

// fail reports an RPC error: a lost service as the loss, anything else as
// what went wrong with path.
func (r *Remote) fail(what, path string, err error) {
	if isLost(err) {
		r.lost(err)
		return
	}
	title := what
	if path != "" {
		title += " " + short(path)
	}
	r.sh.Report(session.Problem{Title: title + " failed on the service", Detail: errText(err)})
}

func bg() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 30*time.Second)
}

// ---------------------------------------------------------------- models

// refresh lists the service's loaded models and republishes the chat's
// model, after a connect or a reconnect.
func (r *Remote) refresh() {
	r.publishModels()
	if r.session != "" {
		r.publishSession()
	}
}

// publishModels mirrors the service's loaded models into the store. A model
// another client loaded is listed too: it is one engine.
func (r *Remote) publishModels() {
	ctx, cancel := bg()
	defer cancel()
	resp, err := r.c.Model.ListModels(ctx, connect.NewRequest(&v1.ListModelsRequest{LoadedOnly: true}))
	if err != nil {
		r.fail("listing the models", "", err)
		return
	}
	out := make([]session.LoadedModel, 0, len(resp.Msg.GetLoaded()))
	for i, m := range resp.Msg.GetLoaded() {
		r.ids[m.GetPath()] = m.GetModelId()
		lm := session.LoadedModel{Path: m.GetPath(), Name: m.GetName(), Arch: m.GetArchitecture(),
			Blocks: int(m.GetBlockCount()), Colour: i, Active: m.GetPath() == r.active}
		if lm.Name == "" {
			lm.Name = short(m.GetPath())
		}
		if res, err := r.c.Placement.GetResidency(ctx, connect.NewRequest(&v1.GetResidencyRequest{ModelId: m.GetModelId()})); err == nil {
			lm.Grant = res.Msg.GetResidency().GetBudget().GetBytes()
		}
		out = append(out, lm)
	}
	active := r.active
	r.sh.Post(func() {
		r.st.Models.Set(out)
		r.st.Active.Set(active)
	})
}

// Load is screen.Engine.Load: the service opens path (a file, or a name it
// knows) and the chat moves to it. A model the service already holds is a
// switch, not a second copy.
func (r *Remote) Load(path string) {
	if path == "" {
		return
	}
	if _, ok := session.LoadingOf(r.st.Loads, path); ok {
		return
	}
	r.busyBegin()
	r.status("asking jitllmd to open %s", short(path))
	session.QueueLoad(r.st.Loads, session.Loading{Path: path, Stage: "Waiting"})
	if !r.post("load", func() {
		defer r.busyEnd()
		defer r.sh.Post(func() { session.EndLoad(r.st.Loads, path) })
		r.sh.Post(func() { session.SetLoadStage(r.st.Loads, path, "Opening on the service") })
		if _, ok := r.ids[path]; !ok {
			spec := r.st.DeviceSpec.Get()
			req := &v1.LoadModelRequest{Path: path}
			if spec != "" && spec != "auto" {
				req.DeviceIds = strings.Split(spec, ",")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			resp, err := r.c.Model.LoadModel(ctx, connect.NewRequest(req))
			cancel()
			if err != nil {
				r.fail("opening", path, err)
				return
			}
			// The service may name it by the file it resolved a name to.
			got := resp.Msg.GetModel().GetPath()
			r.ids[path] = resp.Msg.GetModel().GetModelId()
			if got != "" && got != path {
				r.ids[got] = r.ids[path]
			}
		}
		r.activate(path)
	}) {
		r.busyEnd()
		session.EndLoad(r.st.Loads, path)
	}
}

// Use is screen.Engine.Use.
func (r *Remote) Use(path string) {
	r.post("use", func() {
		if _, ok := r.ids[path]; !ok {
			r.status("%s is not open on the service", short(path))
			return
		}
		if r.active != path {
			r.activate(path)
		}
	})
}

// activate gives path the chat's session, closing the previous one. The
// history is re-rendered every turn, so nothing carries across.
func (r *Remote) activate(path string) {
	r.closeSession()
	id := r.ids[path]
	ctx, cancel := bg()
	defer cancel()
	mi, err := r.c.Model.GetModel(ctx, connect.NewRequest(&v1.GetModelRequest{ModelId: id}))
	if err != nil {
		r.fail("reading", path, err)
		return
	}
	info := mi.Msg.GetModel()
	window := maxSeqFor(int(info.GetContextLength()), r.st.MaxSeq.Get())
	cs := &v1.CreateSessionRequest{ModelId: id, MaxSeq: int32(window), PromptCache: r.st.KVCache.Get()}
	resp, err := r.c.Session.CreateSession(ctx, connect.NewRequest(cs))
	cacheOff := ""
	if err != nil && cs.PromptCache && connect.CodeOf(err) == connect.CodeInvalidArgument {
		// The prompt cache is the service's: one started without a store
		// (jitllmd serve -kv-cache DIR) has none to give.
		cs.PromptCache = false
		cacheOff = " (prompt cache off: the service keeps none)"
		resp, err = r.c.Session.CreateSession(ctx, connect.NewRequest(cs))
	}
	if err != nil {
		r.fail("starting a session on", path, err)
		r.active = ""
		r.publishModels()
		return
	}
	r.active, r.session, r.window = path, resp.Msg.GetSession().GetSessionId(), window
	r.publishModels()
	chatOK, vision := info.GetChatCapable(), info.GetHasVisionTower()
	r.sh.Post(func() {
		if !chatOK {
			r.st.Chat.Set(false)
		}
		r.st.ChatCapable.Set(chatOK)
		r.st.Vision.Set(vision)
		r.st.ModelPath.Set(path)
		r.st.Loaded.Set(true)
		r.st.Problem.Set(session.Problem{})
	})
	r.publishSession()
	r.status("loaded %s on jitllmd at %s%s", short(path), r.Where().Addr, cacheOff)
}

// closeSession closes the chat's session; its model stays loaded on the
// service.
func (r *Remote) closeSession() {
	if r.session == "" {
		return
	}
	ctx, cancel := bg()
	defer cancel()
	r.c.Session.CloseSession(ctx, connect.NewRequest(&v1.CloseSessionRequest{SessionId: r.session}))
	r.session = ""
}

// publishSession reads where the chat's model is: its blocks, its pages and
// the devices, and the header line.
func (r *Remote) publishSession() {
	ctx, cancel := bg()
	defer cancel()
	pl, err := r.c.Placement.GetPlacement(ctx, connect.NewRequest(&v1.GetPlacementRequest{SessionId: r.session}))
	if err != nil {
		r.fail("reading the placement of", r.active, err)
		return
	}
	p := pl.Msg.GetPlacement()
	var devs []*v1.Device
	if d, err := r.c.Device.ListDevices(ctx, connect.NewRequest(&v1.ListDevicesRequest{})); err == nil {
		devs = d.Msg.GetDevices()
	}
	blocks, alloc := placementView(p, devs)
	if s, err := r.c.Session.GetSession(ctx, connect.NewRequest(&v1.GetSessionRequest{SessionId: r.session})); err == nil {
		alloc.Pos, alloc.MaxSeq = int(s.Msg.GetSession().GetPosition()), int(s.Msg.GetSession().GetMaxSeq())
	}
	var pager session.PagerStat
	if res, err := r.c.Placement.GetResidency(ctx, connect.NewRequest(&v1.GetResidencyRequest{ModelId: p.GetModelId()})); err == nil {
		rs := res.Msg.GetResidency()
		pager = session.PagerStat{Frames: int(rs.GetFrames()), Resident: rs.GetResident().GetBytes(),
			PageIns: rs.GetPageIns(), PageOuts: rs.GetEvictions(), BytesRead: rs.GetBytesRead().GetBytes(),
			Reads: rs.GetReadRequests()}
		alloc.HostUsed, alloc.HostBudget = rs.GetResident().GetBytes(), rs.GetBudget().GetBytes()
	}
	where := "on the host"
	if n := p.GetDeviceBlockCount(); n > 0 {
		where = fmt.Sprintf("%d of %d blocks on a device", n, len(p.GetBlocks()))
	}
	summary := fmt.Sprintf("%s · %s · jitllmd at %s", short(r.active), where, r.Where().Addr)
	report := "jitllmd at " + r.Where().Addr
	r.sh.Post(func() {
		r.st.BlockMap.Set(blocks)
		r.st.Alloc.Set(alloc)
		r.st.Pager.Set(pager)
		r.st.ModelSummary.Set(summary)
		r.st.DeviceReport.Set(report)
	})
}

// placementView turns the service's placement into the memory map's input:
// one session.Placement byte a block, and the allocation by device.
func placementView(p *v1.Placement, devs []*v1.Device) ([]byte, session.Allocation) {
	ord := map[string]int{}
	var uses []session.DeviceUse
	blocks := make([]byte, len(p.GetBlocks()))
	alloc := session.Allocation{NBlocks: len(p.GetBlocks()),
		CPUBlocks: int(p.GetCpuBlockCount()), DeviceBlocks: int(p.GetDeviceBlockCount())}
	for i, b := range p.GetBlocks() {
		pl := session.PlaceCPU
		if b.GetLocation() == v1.BlockLocation_BLOCK_LOCATION_DEVICE {
			k, ok := ord[b.GetDeviceId()]
			if !ok {
				k = len(uses)
				ord[b.GetDeviceId()] = k
				u := session.DeviceUse{Name: b.GetDeviceId(), Short: b.GetDeviceId(), Pool: b.GetDeviceId()}
				for _, d := range devs {
					if d.GetRef().GetId() == b.GetDeviceId() {
						u.Name = d.GetName()
						u.Limit = d.GetTotalMemory().GetBytes()
						u.Used = u.Limit - d.GetFreeMemory().GetBytes()
						u.Shared = d.GetCountsTowardHostBudget()
					}
				}
				uses = append(uses, u)
			}
			uses[k].Blocks++
			uses[k].Weights += b.GetBytes().GetBytes()
			pl = session.OnDevice(k)
		}
		if b.GetCpuResident() {
			pl |= session.PlaceResident
		}
		if i < len(blocks) {
			blocks[i] = byte(pl)
		}
		if alloc.PageBytes == 0 {
			alloc.PageBytes = b.GetBytes().GetBytes()
		}
	}
	alloc.Devices = uses
	return blocks, alloc
}

// Unload closes one model on the service. Forced: the service's other
// sessions on it close with it, as the embedded engine's API sessions do.
func (r *Remote) Unload(path string) {
	r.post("unload", func() {
		id, ok := r.ids[path]
		if !ok {
			return
		}
		if r.active == path {
			r.closeSession()
			r.active = ""
		}
		ctx, cancel := bg()
		defer cancel()
		if _, err := r.c.Model.UnloadModel(ctx, connect.NewRequest(&v1.UnloadModelRequest{ModelId: id, Force: true})); err != nil {
			r.fail("closing", path, err)
			return
		}
		delete(r.ids, path)
		r.publishModels()
		if r.active == "" {
			r.sh.Post(func() {
				r.st.Loaded.Set(false)
				r.st.ModelPath.Set("")
				r.st.ModelSummary.Set("")
				r.st.Vision.Set(false)
				r.st.BlockMap.Set(nil)
				r.st.Alloc.Set(session.Allocation{})
			})
		}
		r.status("closed %s on the service", short(path))
	})
}

// Pin caps one model's page budget on the service, or releases it at 0.
func (r *Remote) Pin(path string, bytes uint64) {
	r.post("pin", func() {
		id, ok := r.ids[path]
		if !ok {
			return
		}
		ctx, cancel := bg()
		defer cancel()
		if _, err := r.c.Placement.SetPageBudget(ctx, connect.NewRequest(&v1.SetPageBudgetRequest{ModelId: id, BudgetBytes: bytes})); err != nil {
			r.fail("capping", path, err)
			return
		}
		r.publishModels()
	})
}

// SetPriority is the one control a service does not take from a client: how
// memory is divided between its models is its own policy.
func (r *Remote) SetPriority(bool) {
	r.status("memory priority is the service's own setting; it is not available from a connected app")
}

// Relocate asks the service to hold n of the chat's blocks on its devices.
func (r *Remote) Relocate(n int) {
	r.post("relocate", func() {
		if r.session == "" {
			r.status("no model is loaded")
			return
		}
		ctx, cancel := bg()
		defer cancel()
		k := int32(n)
		resp, err := r.c.Placement.SetPlacement(ctx, connect.NewRequest(&v1.SetPlacementRequest{SessionId: r.session, MaxDeviceBlocks: &k}))
		if err != nil {
			r.fail("moving the blocks of", r.active, err)
			return
		}
		p := resp.Msg.GetPlacement()
		r.publishSession()
		r.status("%d of %d block(s) on a device", p.GetDeviceBlockCount(), len(p.GetBlocks()))
	})
}

// Run is screen.Engine.Run: a long job of the app's own (a conversion) on
// the worker, so it does not overlap a reply.
func (r *Remote) Run(job func()) bool { return r.post("job", job) }

// Stop cancels the reply in flight.
func (r *Remote) Stop() {
	r.mu.Lock()
	stop := r.stop
	r.mu.Unlock()
	if stop != nil {
		stop()
	}
}

// Close stops the client. The service and its models stay; the chat's
// session is closed.
func (r *Remote) Close() {
	r.once.Do(func() {
		close(r.quit)
		r.Stop()
		// After closed is set no post sends, so the channel can close.
		r.mu.Lock()
		r.closed = true
		r.mu.Unlock()
		r.cmds <- r.closeSession
		close(r.cmds)
		<-r.done
	})
}

// ---------------------------------------------------------------- chat

// Send is screen.Engine.Send.
func (r *Remote) Send(req session.ChatRequest) {
	if !r.st.Loaded.Get() {
		r.sh.Alert("No model loaded", "Open a model on the Models tab first.")
		return
	}
	r.busyBegin()
	r.st.Streaming.Set(true)
	if !r.post("send", func() { defer r.busyEnd(); r.generate(req) }) {
		r.busyEnd()
		r.st.Streaming.Set(false)
	}
}

// promptOf is the request's prompt as the service takes it: the messages,
// pictures as their bytes, or the raw text.
func promptOf(req session.ChatRequest) (*v1.PromptInput, error) {
	if !req.Chat && !hasImages(req) {
		return &v1.PromptInput{Input: &v1.PromptInput_Text{Text: req.Prompt}}, nil
	}
	msgs, err := chatOf(req, true)
	if err != nil {
		return nil, err
	}
	return &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{Messages: msgs, AddGenerationPrompt: true}}}, nil
}

func chatOf(req session.ChatRequest, images bool) ([]*v1.ChatMessage, error) {
	out := make([]*v1.ChatMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		pm := &v1.ChatMessage{Role: m.Role, Content: m.Content}
		if images {
			for _, p := range m.Images {
				b, err := os.ReadFile(p)
				if err != nil {
					return nil, err
				}
				pm.Images = append(pm.Images, &v1.ChatImage{Data: b, MediaType: mime.TypeByExtension(filepath.Ext(p))})
			}
		}
		out = append(out, pm)
	}
	return out, nil
}

// opensThinking asks the service to render the chat and reports whether it
// ends inside a reasoning block, as Engine.templateOpensThinking does.
func (r *Remote) opensThinking(req session.ChatRequest) bool {
	if !req.Chat {
		return false
	}
	msgs, err := chatOf(req, false)
	if err != nil {
		return false
	}
	ctx, cancel := bg()
	defer cancel()
	resp, err := r.c.Model.ApplyChatTemplate(ctx, connect.NewRequest(&v1.ApplyChatTemplateRequest{
		ModelId: r.ids[r.active], Messages: msgs, AddGenerationPrompt: true}))
	if err != nil {
		return false
	}
	return strings.HasSuffix(strings.TrimRight(resp.Msg.GetPrompt(), " \t\r\n"), thinkOpen)
}

func samplingOf(s session.Sampling) *v1.SamplingParams {
	temp, topP, minP, pen := s.Temp, s.TopP, s.MinP, s.RepeatPen
	topK, lastN := int32(s.TopK), int32(s.RepeatLastN)
	p := &v1.SamplingParams{Temperature: &temp, TopP: &topP, TopK: &topK, MinP: &minP,
		RepeatPenalty: &pen, RepeatLastN: &lastN}
	if s.Seed != 0 {
		seed := uint64(s.Seed)
		p.Seed = &seed
	}
	return p
}

func (r *Remote) generate(req session.ChatRequest) {
	defer r.sh.Post(func() { r.st.Streaming.Set(false) })
	if r.session == "" {
		return
	}
	prompt, err := promptOf(req)
	if err != nil {
		r.sh.Report(session.Problem{Title: "The prompt could not be sent", Detail: err.Error()})
		return
	}
	preOpened := r.opensThinking(req)
	max := req.MaxTokens
	if max <= 0 {
		max = session.DefaultMaxTokens
	}
	ctx, cancel := context.WithCancel(context.Background())
	r.mu.Lock()
	r.stop = cancel
	r.mu.Unlock()
	defer func() {
		r.mu.Lock()
		r.stop = nil
		r.mu.Unlock()
		cancel()
	}()

	stream, err := r.c.Inference.Generate(ctx, connect.NewRequest(&v1.GenerateRequest{
		SessionId: r.session, Prompt: prompt, MaxTokens: int32(max), Sampling: samplingOf(req.Sampling)}))
	if err != nil {
		r.fail("the reply from", r.active, err)
		return
	}
	defer stream.Close()

	var text strings.Builder
	var fin *v1.GenerateFinished
	var openedOnce, foldedOnce bool
	n := 0
	var t1 time.Time
	for stream.Receive() {
		switch ev := stream.Msg().GetEvent().(type) {
		case *v1.GenerateResponse_Started:
			st := ev.Started
			if ms := st.GetPrefillMillis(); ms > 0 {
				rate := float64(st.GetPromptTokens()) / (float64(ms) / 1000)
				r.sh.Post(func() { r.st.PromptTokS.Set(rate) })
			}
			t1 = time.Now()
		case *v1.GenerateResponse_Finished:
			fin = ev.Finished
		case *v1.GenerateResponse_Token:
			text.WriteString(ev.Token.GetText())
			n++
			partialThink, partial := splitThinking(holdPartialTag(display(text.String())), preOpened)
			live := float64(n) / time.Since(t1).Seconds()
			openNow := !openedOnce && partialThink != "" && partial == ""
			foldNow := !foldedOnce && partial != ""
			openedOnce = openedOnce || openNow
			foldedOnce = foldedOnce || foldNow
			reply := req.Reply
			r.sh.Post(func() {
				r.st.Stream.Set(partial)
				r.st.StreamThink.Set(partialThink)
				r.st.DecodeTokS.Set(live)
				if openNow {
					r.sh.SetThinkOpen(reply, true)
				}
				if foldNow && !r.st.ShowThinking.Get() {
					r.sh.SetThinkOpen(reply, false)
				}
				r.st.Revision.Set(r.st.Revision.Get() + 1)
			})
		}
	}
	err = stream.Err()
	stop := "end of text"
	switch {
	case ctx.Err() != nil:
		stop = "stopped"
		cctx, ccancel := bg()
		r.c.Inference.Cancel(cctx, connect.NewRequest(&v1.CancelRequest{SessionId: r.session}))
		ccancel()
	case err != nil:
		r.fail("the reply from", r.active, err)
		return
	case fin != nil && fin.GetReason() == v1.FinishReason_FINISH_REASON_MAX_TOKENS && n < max:
		stop = fmt.Sprintf("context full at %d positions", r.window)
	case fin != nil && fin.GetReason() == v1.FinishReason_FINISH_REASON_MAX_TOKENS:
		stop = fmt.Sprintf("hit the %d-token cap", max)
	}
	rate := fin.GetDecodeTokensPerSecond()
	perTok := float64(fin.GetBytesPerToken())
	gbs := rate * perTok / (1 << 30)
	think, answer := splitThinking(display(text.String()), preOpened)
	name, colour := short(r.active), 0
	for i, m := range r.st.Models.Get() {
		if m.Path == r.active {
			colour = i
		}
	}
	reply := req.Reply
	r.publishSession()
	r.sh.Post(func() {
		r.st.Stream.Set(answer)
		r.st.StreamThink.Set(think)
		r.st.DecodeTokS.Set(rate)
		r.st.BytesPerTok.Set(perTok)
		r.st.GBs.Set(gbs)
		r.st.Problem.Set(session.Problem{})
		r.sh.SetTurn(reply, session.Turn{Role: session.RoleAssistant, Text: answer, Think: think,
			Tokens: n, TokPerSec: rate, Model: name, Colour: colour})
		r.st.Revision.Set(r.st.Revision.Get() + 1)
		r.sh.SetStatus(fmt.Sprintf("%d token(s) at %.2f tok/s — %s", n, rate, stop))
	})
}

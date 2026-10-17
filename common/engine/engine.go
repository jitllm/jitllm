// Package engine owns every jitllm object the app touches, on one goroutine.
//
// The UI goroutine never calls jitllm: gogpu.App.Run holds the UI thread, so a
// Prefill on it would freeze the window, and a terminal UI's update loop would
// stall the same way. Every entry point here queues and returns, as the
// window's screen.Engine requires.
//
// Models and sessions live in a server.Engine (Server): the app's chat is one
// session on it, and the API, when a front end serves it, is the same engine
// seen over HTTP -- one budget, one pager per model, one set of gates.
//
// The engine knows no front end. It reports through a [Front] and the [State]
// values the front end hands it; the window and the terminal each implement
// both.
//
// It is one worker rather than a goroutine per job because the objects are not
// safe to share: model.State holds a sequence and a conversion beside a decode
// would contend for the same pool and pager. A single command channel makes "a
// long job cannot overlap a decode" true by construction.
package engine

import (
	"context"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/jitllm/jitllm/common/crash"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/server"
)

// Engine is the app's single owner of a loaded model.
type Engine struct {
	// loadingPath is the model the worker is opening or switching to, for
	// stage. Worker goroutine only.
	loadingPath string

	sh Front
	st State

	cmds chan func()
	done chan struct{}
	once sync.Once

	// cancel is read by the decode loop and set by Stop. Atomic because Stop
	// runs on the UI goroutine and the loop runs on the worker -- the one
	// piece of state that genuinely crosses.
	cancel atomic.Bool

	// busyJobs is how many jobs the app has to wait for are queued or
	// running; Store.Busy is published from it. See busyEnd.
	busyJobs atomic.Int32

	// srv holds every model and session; its methods are safe from any
	// goroutine, and an API serving it runs beside this worker.
	srv *server.Engine

	// Everything below is touched only on the worker goroutine.
	//
	// models is every model the app opened and active the one its chat runs
	// on. The scalars below (m, sess, gpu, path, sessMax) belong to the active
	// entry.
	models []*entry
	active *entry

	m *model.Model
	// sess is the chat's session on the active model.
	sess *server.Session
	// gpu is the active model's tier, nil on the host: only *tier.GPU can
	// answer "how much of each card is this model using".
	gpu  *tier.GPU
	path string
	// sessMax is the window this session was built for, so the decode loop can
	// tell a context that filled from a cap that was reached.
	sessMax int
	// lastPromptTok is the last prompt's length, read by the reuse probe.
	lastPromptTok int
	// pictures is each picture file's preprocessed pixels and name, by the
	// file's identity, so a picture in every turn's history is decoded once.
	// The rows are the model's image cache's; dropped with the model.
	pictures map[fileID]*model.Picture
}

// fileID is a picture file's identity: its path, size and modification time.
type fileID struct {
	path  string
	size  int64
	mtime int64
}

// queueDepth is small on purpose: the commands are user actions, and a queue
// deep enough to hide a stall is a queue that hides a stall.
const queueDepth = 16

// New starts the worker. Close stops it.
func New(sh Front, st State) *Engine {
	e := &Engine{sh: sh, st: st, cmds: make(chan func(), queueDepth), done: make(chan struct{}),
		srv: server.New(server.Config{Version: "app"})}
	go e.run()
	return e
}

// Server is the engine the app's models and chat live in, for a front end to
// serve as the API (server.Engine.Handler).
func (e *Engine) Server() *server.Engine { return e.srv }

func (e *Engine) run() {
	defer close(e.done)
	for fn := range e.cmds {
		e.do(fn)
	}
}

// do runs one command. A panic in it is reported (package crash) and ends the
// command, not the worker: the window stays up to show the report, and a
// later command -- Close above all -- still runs.
//
// What a command counts it uncounts in a defer (busyEnd, the load list,
// Streaming), and defers run on a panic, so those stay consistent by
// construction. What they cannot vouch for is the chat's session: a panic
// mid-step leaves its KV history and position wherever the step stopped. So
// the session is closed and nothing is active; the models stay open, and
// choosing one builds a fresh session.
func (e *Engine) do(fn func()) {
	defer func() {
		if r := recover(); r != nil {
			crash.Handle("engine worker", r, debug.Stack())
			e.afterPanic()
		}
	}()
	fn()
}

// afterPanic drops the chat's session after a command panicked. A second
// panic while closing it is reported too: the worker must survive both.
func (e *Engine) afterPanic() {
	defer func() {
		if r := recover(); r != nil {
			crash.Handle("engine worker, closing the session after a panic", r, debug.Stack())
			e.sess, e.active = nil, nil
			e.gpu, e.m, e.path, e.sessMax = nil, nil, "", 0
		}
	}()
	e.loadingPath = ""
	e.releaseSession()
	e.active = nil
	e.unpublish()
	e.sh.Post(func() {
		e.st.Streaming.Set(false)
		e.st.Busy.Set(e.busyJobs.Load() > 0)
	})
	e.status("the engine stopped on an error; choose a model to carry on")
}

// post queues work for the worker. It reports whether the queue took it: a
// full queue means the worker is busy with something the user can see, and
// dropping silently is how a button becomes a button that sometimes does
// nothing.
func (e *Engine) post(what string, fn func()) bool {
	select {
	case e.cmds <- fn:
		return true
	default:
		e.sh.Post(func() { e.sh.SetStatus(what + ": busy, try again in a moment") })
		return false
	}
}

// busyBegin counts a job the app must wait for, at the moment it is queued.
func (e *Engine) busyBegin() {
	e.busyJobs.Add(1)
	e.st.Busy.Set(true)
}

// busyEnd uncounts one and publishes what is left.
//
// The count is read when the post runs, not when the job ends: per-job
// Busy=false posts let a queued switch or Send run with Busy overwritten, which
// left Clear enabled under a live decode.
func (e *Engine) busyEnd() {
	e.busyJobs.Add(-1)
	e.sh.Post(func() { e.st.Busy.Set(e.busyJobs.Load() > 0) })
}

// Run is screen.Engine.Run: a long job that must not overlap a decode. It
// reports whether the queue took it.
func (e *Engine) Run(job func()) bool { return e.post("job", job) }

// Close stops the worker and releases the model and the device.
func (e *Engine) Close() {
	e.once.Do(func() {
		e.cancel.Store(true)
		// Queued, so it runs after whatever is in flight rather than freeing a
		// model another command is still holding.
		e.cmds <- func() { e.release() }
		close(e.cmds)
		<-e.done
	})
}

// release tears down the current model and device. Worker goroutine only.
func (e *Engine) release() {
	e.releaseSession()
	// Every model in the engine, the API's included: the app is the process.
	e.srv.Close()
	e.models, e.active = nil, nil
}

// releaseSession closes the chat's session and leaves its model loaded, with
// its tier: a backgrounded model keeps its blocks where they are, and API
// sessions on it are untouched.
func (e *Engine) releaseSession() {
	if e.sess != nil {
		e.srv.CloseSession(e.sess.ID())
		e.sess = nil
	}
	e.gpu, e.m, e.path, e.sessMax = nil, nil, "", 0
}

// inspect runs f on the chat session's State between turns.
func (e *Engine) inspect(f func(st *model.State)) {
	if e.sess != nil {
		e.sess.Inspect(f)
	}
}

// touch says a turn's content changed without the turn count moving.
func (e *Engine) touch() { e.st.Revision.Set(e.st.Revision.Get() + 1) }

// status writes the status bar from a worker.
func (e *Engine) status(f string, a ...any) {
	msg := fmt.Sprintf(f, a...)
	e.sh.Post(func() { e.sh.SetStatus(msg) })
}

// report shows an engine error on the session screen, explained, with the
// fix as a button where there is one.
func (e *Engine) report(stage, path string, err error) {
	title, detail, f := explain(stage, path, err)
	p := session.Problem{Title: title, Detail: detail}
	switch f {
	case fixConvert:
		p.Action, p.Do = "Convert\u2026", func() { e.sh.OfferConvert(path) }
	case fixReconvert:
		p.Action, p.Do = "Reconvert", func() { e.sh.OfferReconvert(path) }
	case fixDevices:
		p.Action, p.Do = "Open Settings", e.sh.GoSettings
	}
	e.sh.Report(p)
}

// Load is screen.Engine.Load. It queues and returns; a model already waiting
// to load is not queued twice.
func (e *Engine) Load(path string) {
	if path == "" {
		return
	}
	if _, ok := session.LoadingOf(e.st.Loads, path); ok {
		return
	}
	e.busyBegin()
	e.status("opening %s", short(path))
	session.QueueLoad(e.st.Loads, session.Loading{Path: path, Stage: "Waiting"})
	if !e.post("load", func() {
		defer e.busyEnd()
		defer e.loading(path)()
		e.stage("Reading the model")
		e.load(path)
	}) {
		e.busyEnd()
		session.EndLoad(e.st.Loads, path)
	}
}

// loading marks path as the load the worker is on, for stage, and returns
// what ends it -- whether it opened or failed: the error, if any, is already
// on its way as a Problem.
func (e *Engine) loading(path string) func() {
	e.loadingPath = path
	return func() {
		e.loadingPath = ""
		e.sh.Post(func() { session.EndLoad(e.st.Loads, path) })
	}
}

// stage says what the load in progress is doing now. It does nothing when no
// load is running: activate also runs on a switch.
func (e *Engine) stage(s string) {
	path := e.loadingPath
	if path == "" {
		return
	}
	e.sh.Post(func() { session.SetLoadStage(e.st.Loads, path, s) })
}

// load runs on the worker. It adds a model to the registry and makes it
// active. Re-loading a model already open is a switch, not a second copy: two
// entries over one file would be two pagers over the same bytes.
func (e *Engine) load(path string) {
	// The device choice is the model's for as long as it is loaded: its tier
	// holds its blocks while another model is in use. Loading an open model
	// again under another choice re-opens it there; under the same one it is a
	// switch.
	spec := e.st.DeviceSpec.Get()
	if spec == "" {
		spec = "auto"
	}
	placement := strings.TrimSpace(e.st.Placement.Get())
	if en := e.find(path); en != nil {
		if en.spec == spec && en.placement == placement {
			if err := e.activate(en); err != nil {
				e.report(stageDevice, path, err)
			}
			return
		}
		e.unload(en)
	}

	lm, err := e.srv.LoadModel(server.LoadOptions{
		Path: path, DeviceIDs: strings.Split(spec, ","),
		// Strict: a block named for a device that cannot take it is an error
		// the person reads, not a quiet placement somewhere else.
		Placement: placement, PlacementStrict: placement != "",
	})
	if err != nil {
		// A GGUF or a stale container is the commonest wrong file; report turns
		// both into the button that fixes them.
		stage := stageOpen
		var de *server.DeviceError
		if errors.As(err, &de) {
			stage = stageDevice
		}
		e.report(stage, path, err)
		if e.active == nil {
			e.unpublish()
		}
		return
	}
	en := &entry{path: path, lm: lm, m: lm.Model(), spec: spec, placement: placement}
	e.models = append(e.models, en)

	if err := e.activate(en); err != nil {
		e.report(stageDevice, path, err)
	}
}

// activate gives one loaded model the chat's session, closing the one the
// previous model had. Its tier was opened at load and stays with it.
//
// The history is not carried across. In chat mode every turn renders the whole
// conversation through the model's own template, so the next message
// re-prefills it correctly; the KV namespace carries the model's size and
// mtime, so the new model cannot read the old one's pages.
func (e *Engine) activate(en *entry) error {
	if en == nil || en.m == nil {
		return errNoModel
	}
	e.releaseSession()
	e.active = en
	// The model in use is the one priority gives the memory to.
	e.srv.Favor(en.lm.ID())
	m, path := en.m, en.path

	window := maxSeq(m, e.st.MaxSeq.Get())
	// A context that outgrows the card hands blocks to the host one at a time
	// -- only where the device choice lets the host run them, as the CLI's
	// -relocate does. A choice of devices alone keeps the model off the CPU, so
	// there the token takes the device's refusal instead.
	relocate, _ := tier.AdmitsHost(en.spec)
	o := server.SessionOptions{ModelID: en.lm.ID(), MaxSeq: window, KeepOffHost: !relocate}
	cacheOff := "turned off in Settings"
	if e.st.KVCache.Get() {
		o.KVStore, o.CacheKey, cacheOff = kvCache(path)
	}
	if en.lm.GPU() != nil {
		e.stage("Placing layers on the GPU")
	}
	s, err := e.srv.CreateSession(o)
	if err != nil {
		// The previous session is already closed, so the screen must stop showing
		// it. The model stays loaded and listed, ready once the device is fixed.
		e.active = nil
		e.unpublish()
		return err
	}

	e.m, e.sess, e.gpu, e.path, e.sessMax = m, s, en.lm.GPU(), path, window
	e.publishModels()

	devName := "cpu"
	if e.gpu != nil {
		devName = en.spec
		if ds := e.gpu.DevStats(); len(ds) > 0 {
			devName = fmt.Sprintf("%s (%d device(s))", en.spec, len(ds))
		}
	}

	// A base model cannot chat, so the header says "completion only" and Chat
	// is switched off at load rather than failing every turn.
	summary := e.headerNow().String()
	chatOK := m.ChatCapable()
	vision := m.Tower() != nil
	// A different model is a different preprocessor: every picture is stale.
	e.pictures = nil
	var blocks []byte
	var alloc session.Allocation
	e.inspect(func(st *model.State) {
		blocks = placementOf(m, st, e.gpu)
		alloc = allocStat(m, st, e.gpu, m.Cfg.NLayer)
	})
	pager := pagerStat(m)

	e.sh.Post(func() {
		if !chatOK {
			e.st.Chat.Set(false)
		}
		e.st.ChatCapable.Set(chatOK)
		e.st.Vision.Set(vision)
		e.st.ModelPath.Set(path)
		e.st.ModelSummary.Set(summary)
		e.st.Loaded.Set(true)
		e.st.DeviceReport.Set(devName)
		e.st.BlockMap.Set(blocks)
		e.st.Pager.Set(pager)
		e.st.Alloc.Set(alloc)
		// A model that loaded answers whatever problem the banner was showing.
		e.st.Problem.Set(session.Problem{})
		loaded := "loaded " + short(path)
		if cacheOff != "" {
			loaded += " (prompt cache off: " + cacheOff + ")"
		}
		e.sh.SetStatus(loaded)
	})
	return nil
}

// unpublish stops the screen showing a session when none is active, and lists
// what is still open.
func (e *Engine) unpublish() {
	e.publishModels()
	e.sh.Post(func() {
		e.st.Loaded.Set(false)
		e.st.ModelPath.Set("")
		e.st.ModelSummary.Set("")
		e.st.Vision.Set(false)
		e.st.BlockMap.Set(nil)
		e.st.Alloc.Set(session.Allocation{})
	})
}

// maxSeq is the context window a session allocates: what the caller asked for,
// bounded by what the model was trained with. The model's context is a ceiling
// (State.ContextClamped reports a clamp); asking for less is legitimate, since
// the KV cache is allocated up front.
func maxSeq(m *model.Model, want int) int { return maxSeqFor(m.Cfg.NCtx, want) }

// maxSeqFor is the arithmetic, separated so it can be gated without a model.
func maxSeqFor(modelCtx, want int) int {
	n := modelCtx
	if n <= 0 {
		n = defaultMaxSeq
	}
	if want > 0 && want < n {
		n = want
	}
	if n > defaultMaxSeq && want <= 0 {
		n = defaultMaxSeq
	}
	return n
}

// defaultMaxSeq is the window when nothing asks for one. It is a cap on the
// engine's own default rather than on the user's request: a model trained to
// 32k still opens at 4k unless asked, because the cache is allocated up front.
const defaultMaxSeq = 4096

// samplerFor turns the app's settings into the engine's sampler.
//
// Temp == 0 is greedy and is passed straight through: model.Sampler
// short-circuits to argmax, and every correctness comparison in the engine
// (llama.cpp goldens, CPU vs device, verify) is a comparison of argmax.
func samplerFor(s session.Sampling) *model.Sampler {
	return &model.Sampler{
		Temp:        float64(s.Temp),
		TopK:        s.TopK,
		TopP:        float64(s.TopP),
		MinP:        float64(s.MinP),
		RepeatPen:   float64(s.RepeatPen),
		RepeatLastN: s.RepeatLastN,
		Seed:        s.Seed,
	}
}

func short(p string) string {
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		return p[i+1:]
	}
	return p
}

// blockMap is the memory map's input: one byte a block, saying which device
// runs it and whether its page is in host memory (see session.Placement).
//
// It is built from the runs, not a count, because placement is a set and not a
// prefix: a declined block leaves a hole, and the map exists to show it. devOf
// is the device ordinal of a block the runs place; resident is the host pager's.
func blockMap(nLayer int, runs [][2]int, devOf func(li int) int, resident func(li int) bool) []byte {
	if nLayer <= 0 {
		return nil
	}
	b := make([]byte, nLayer)
	for _, r := range runs {
		for i := max(r[0], 0); i < r[1] && i < nLayer; i++ {
			b[i] = byte(session.OnDevice(devOf(i)))
		}
	}
	for i := range b {
		if resident(i) {
			b[i] |= byte(session.PlaceResident)
		}
	}
	return b
}

// placementOf is blockMap for a live session. A block the tier cannot name a
// device for is drawn on device 0 rather than dropped from the device count.
func placementOf(m *model.Model, st *model.State, g *tier.GPU) []byte {
	ord := map[string]int{}
	if g != nil {
		for i, b := range g.Budgets() {
			ord[b.Name] = i
		}
	}
	devOf := func(li int) int {
		if g != nil {
			if name, ok := g.DeviceOf(li); ok {
				return ord[name]
			}
		}
		return 0
	}
	// The session's runs name the text blocks it placed. The blocks after them
	// -- prediction blocks and the vision tower's -- are blocks of the same
	// space, placed by whichever State runs them, so the tier says where each is.
	runs := st.DeviceBlocks()
	if g != nil {
		for li := m.Cfg.NLayer; li < m.Blocks(); li++ {
			if _, ok := g.DeviceOf(li); ok {
				runs = append(runs, [2]int{li, li + 1})
			}
		}
	}
	return blockMap(m.Blocks(), runs, devOf, m.BlockResident)
}

// allocStat is where the model is: bytes per device and bytes on the host.
//
// Nothing here sums across devices. A pool is per heap: one card reached
// through CUDA and Vulkan is one pool, and unified-memory devices share the
// host pool, so two rows can report the same bytes. Each row carries its pool
// name and a Shared flag instead.
//
// Budgets(), DevStats() and Placed() are parallel arrays indexed by device
// ordinal, fastest first.
func allocStat(m *model.Model, st *model.State, g *tier.GPU, nBlocks int) session.Allocation {
	a := session.Allocation{
		HostUsed:   m.HostBytes(),
		HostBudget: m.PageBudget(),
		Dense:      m.DenseBytes(),
		NBlocks:    nBlocks,
		PageBytes:  m.PageSize(),
	}
	if st != nil {
		a.DeviceBlocks = st.GPULayers()
		a.HostBlocks = a.NBlocks - a.DeviceBlocks
		a.HostKV = st.KVBytes()
		a.Pos, a.MaxSeq = st.Pos(), st.MaxSeq()
	}
	if g == nil {
		a.HostBlocks = a.NBlocks
		return a
	}

	// A pool is found by name: a Budget names its pool and there is no index
	// between the two lists.
	used := map[string][2]uint64{}
	for _, p := range g.Pools() {
		used[p.Name()] = [2]uint64{p.Used(), p.Limit()}
	}

	budgets, stats, placed := g.Budgets(), g.DevStats(), g.Placed()
	for i, b := range budgets {
		d := session.DeviceUse{Name: b.Device, Short: b.Name, Pool: b.Pool, Shared: b.Host, Limit: b.Limit, Weights: b.Used}
		if u, ok := used[b.Pool]; ok {
			d.Used = u[0]
			if u[1] > 0 {
				d.Limit = u[1]
			}
		}
		if i < len(placed) {
			d.Blocks = placed[i]
		}
		if i < len(stats) {
			d.KV = stats[i].KVBytes
			d.KVPool = stats[i].KVPoolBytes
		}
		a.Devices = append(a.Devices, d)
	}
	return a
}

func pagerStat(m *model.Model) session.PagerStat {
	fr, in, out := m.PageStats()
	// Resident is the resident bytes: Frames is what jlm holds, Resident is
	// what that costs.
	return session.PagerStat{
		Frames: fr, PageIns: in, PageOuts: out,
		Resident:  m.HostBytes(),
		BytesRead: m.BytesRead(), Reads: m.PagerReads(),
		ChunkBytes: m.ChunkBytes(),
	}
}

// Relocate moves the seam: it asks the device to hold n blocks and publishes
// what it actually took.
//
// It is queued, not called: model.State has no mutex and nothing may overlap a
// decode, so the command channel puts this between turns. It must not copy
// Stop's shortcut, which bypasses the queue on purpose.
//
// The published count is what SetGPULayers returned, not n: growing is best
// effort and may take fewer blocks, and a shrink whose history migration fails
// leaves the seam unchanged.
func (e *Engine) Relocate(n int) {
	e.post("relocate", func() {
		if e.m == nil || e.sess == nil {
			e.status("no model is loaded")
			return
		}
		if e.gpu == nil {
			e.status("this model is on the host: load it with a device to move blocks")
			return
		}
		var was int
		e.inspect(func(st *model.State) { was = st.GPULayers() })
		got := e.srv.SetDeviceBlocks(e.sess, n)
		nb := e.m.Cfg.NLayer
		var blocks []byte
		var alloc session.Allocation
		e.inspect(func(st *model.State) {
			blocks = placementOf(e.m, st, e.gpu)
			alloc = allocStat(e.m, st, e.gpu, nb)
		})
		summary := e.headerNow().String()

		msg := fmt.Sprintf("%d of %d block(s) on a device", got, nb)
		switch {
		case got == was && n != was:
			msg = fmt.Sprintf("the seam did not move: still %d of %d", got, nb)
		case got < n:
			msg = fmt.Sprintf("%d of %d block(s) placed, %d asked for", got, nb, n)
		}
		e.sh.Post(func() {
			e.st.BlockMap.Set(blocks)
			e.st.Alloc.Set(alloc)
			e.st.ModelSummary.Set(summary)
			e.st.Status.Set(msg)
		})
	})
}

// Stop cancels the generation in flight and keeps the model loaded.
func (e *Engine) Stop() { e.cancel.Store(true) }

// Send is screen.Engine.Send. It queues and returns.
func (e *Engine) Send(r session.ChatRequest) {
	if !e.st.Loaded.Get() {
		e.sh.Alert("No model loaded", "Open a model on the Models tab first.")
		return
	}
	e.cancel.Store(false)
	e.busyBegin()
	e.st.Streaming.Set(true)
	if !e.post("send", func() { defer e.busyEnd(); e.generate(r) }) {
		e.busyEnd()
		e.st.Streaming.Set(false)
	}
}

func (e *Engine) generate(r session.ChatRequest) {
	defer e.sh.Post(func() { e.st.Streaming.Set(false) })
	if e.m == nil || e.sess == nil {
		return
	}

	// Whether the template already opened a thinking block, asked of the
	// rendered prompt. It is rendered only to be inspected; the ids still come
	// from ChatIDs.
	preOpened := e.templateOpensThinking(r)

	// The thinking disclosure opens itself once and folds itself once; between
	// and after those edges it is the user's. The flags are one-shot so a latch
	// does not overwrite a click on every token.
	var openedOnce, foldedOnce bool

	// A turn with a picture in its history is spans, not ids: a picture is rows
	// the prefill encodes (model.Picture), named by what the tower reads, so the
	// prefix cache restores past it as it does past text and a restored picture
	// runs no tower block.
	var ids []int32
	var spans []model.Span
	var err error
	if hasImages(r) {
		spans, err = e.imageSpans(r)
	} else {
		ids, err = e.promptIDs(r)
	}
	if err != nil {
		e.report(stagePrompt, e.path, err)
		return
	}
	if len(ids) == 0 && len(spans) == 0 {
		e.status("nothing to send")
		return
	}

	ph := e.m.NewPhaseMeter()
	ph.Mark("load")
	var moved0 int
	e.inspect(func(st *model.State) { moved0 = st.Relocations() })
	_, _, outs0 := e.m.PageStats()

	// The session resets before every turn (server.Engine.Generate): a chat
	// prompt is the whole history re-rendered, so prefilling onto a session
	// still holding the last turn would show the model the conversation twice.
	// With a prompt store attached the prefill resumes from the longest prefix
	// it holds, so re-prefilling the history is affordable.
	p := server.Prompt{Kind: server.PromptIDs, IDs: ids}
	if spans != nil {
		p = server.Prompt{Kind: server.PromptSpans, Spans: spans}
	}
	max := r.MaxTokens
	if max <= 0 {
		max = session.DefaultMaxTokens
	}
	sm := samplerFor(r.Sampling)
	out := make([]int32, 0, max)
	var t1 time.Time
	var fin *server.Finished
	n := 0

	err = e.srv.Generate(context.Background(), server.GenerateOptions{
		SessionID: e.sess.ID(), Prompt: p, MaxTokens: max, Sampling: sm,
	}, func(ev server.Event) error {
		switch ev.Kind {
		case server.EventStarted:
			ph.Mark("prefill")
			e.lastPromptTok = ev.Started.PromptTokens
			rate := float64(ev.Started.PromptTokens) / ev.Started.Prefill.Seconds()
			e.sh.Post(func() { e.st.PromptTokS.Set(rate) })
			t1 = time.Now()
		case server.EventFinished:
			fin = ev.Finished
			return nil
		}
		if ev.Kind != server.EventToken || ev.Token.ID < 0 {
			return nil
		}
		if e.cancel.Load() {
			return errStopped
		}
		out = append(out, ev.Token.ID)
		n++
		// Every token is published. RequestRedraw is coalesced and the post queue
		// drops on overflow, so no throttle is needed here. The real cost is one
		// Decode of the whole id slice per token (quadratic, like cmd/jitllm); the fix
		// for that would be an incremental detokenizer, not a timer.
		raw := display(e.m.Vocab.DecodeChat(out))
		partialThink, partial := splitThinking(holdPartialTag(raw), preOpened)
		// The live rate is published with each token so the readout is not "--"
		// during the generation.
		live := float64(n) / time.Since(t1).Seconds()
		// The two edges are latched on the worker so the posted closure carries a
		// decision rather than re-deriving one from racing state.
		openNow := !openedOnce && partialThink != "" && partial == ""
		foldNow := !foldedOnce && partial != ""
		if openNow {
			openedOnce = true
		}
		if foldNow {
			foldedOnce = true
		}
		e.sh.Post(func() {
			e.st.Stream.Set(partial)
			e.st.StreamThink.Set(partialThink)
			e.st.DecodeTokS.Set(live)
			// Open the reasoning while it is the only output, so a long thinking
			// phase does not look like a frozen window.
			if openNow {
				e.sh.SetThinkOpen(r.Reply, true)
			}
			// Fold it once the answer starts. Once only: a click after this sticks.
			if foldNow && !e.st.ShowThinking.Get() {
				e.sh.SetThinkOpen(r.Reply, false)
			}
			// Touch every token: core/listview caches a row's measured height, so a
			// bubble laid out while empty would keep height zero and the reply would
			// paint over its neighbour. InvalidateData re-measures.
			e.touch()
		})
		return nil
	})

	// Why a reply ended is part of the reply: a cap, a full context and Stop
	// otherwise all look like a finished answer.
	stop := "end of text"
	switch {
	case errors.Is(err, errStopped):
		stop = "stopped"
	case err != nil:
		e.report(stageReply, e.path, err)
		return
	case fin.Reason == server.FinishMaxTokens && n < max:
		stop = fmt.Sprintf("context full at %d positions", e.sessMax)
	case fin.Reason == server.FinishMaxTokens:
		stop = fmt.Sprintf("hit the %d-token cap", max)
	}
	if t1.IsZero() {
		t1 = time.Now()
	}
	var reused, prompted int
	var fails int64
	var moved int
	e.inspect(func(st *model.State) {
		reused, fails, moved = st.KVRestored(), st.KVStoreFailures(), st.Relocations()-moved0
	})
	prompted = e.lastPromptTok
	// Printed even when zero: a working cache and a silently dead one look the
	// same on the first turn otherwise.
	kvNote := ""
	if prompted > 0 {
		kvNote = fmt.Sprintf("  |  kv %d/%d reused (%.0f%%)",
			reused, prompted, 100*float64(reused)/float64(prompted))
	}
	if fails > 0 {
		kvNote += fmt.Sprintf("  %d STORE WRITE FAILURE(S)", fails)
	}

	decode := time.Since(t1)
	dec := ph.Mark("decode")
	rawText := display(e.m.Vocab.DecodeChat(out))
	think, text := splitThinking(rawText, preOpened)
	rate := 0.0
	if decode > 0 {
		rate = float64(n) / decode.Seconds()
	}
	perTok := float64(dec.PerToken(n))
	gbs := dec.Rate() / (1 << 30)
	pager := pagerStat(e.m)
	pager.TurnOuts = pager.PageOuts - outs0
	var alloc session.Allocation
	var blocks []byte
	e.inspect(func(st *model.State) {
		alloc = allocStat(e.m, st, e.gpu, e.m.Cfg.NLayer)
		blocks = placementOf(e.m, st, e.gpu)
	})
	summary := e.headerNow().String()
	modelName, modelColour := e.activeName()
	reply := r.Reply
	// A block moved to the host runs at host speed from here on, so say so.
	if moved > 0 {
		kvNote += fmt.Sprintf("  |  %d layer(s) moved to the CPU to fit the context", moved)
	}

	e.sh.Post(func() {
		// The final text goes into Stream as well as the turn, and both halves are
		// republished from the complete text: the last partial split may differ from
		// the finished one (holdPartialTag trims up to seven characters). SetTurn
		// does not publish a new count, so Touch is what makes the reply visible.
		e.st.Stream.Set(text)
		e.st.StreamThink.Set(think)
		e.st.DecodeTokS.Set(rate)
		e.st.BytesPerTok.Set(perTok)
		e.st.GBs.Set(gbs)
		e.st.Pager.Set(pager)
		e.st.Alloc.Set(alloc)
		e.st.BlockMap.Set(blocks)
		e.st.ModelSummary.Set(summary)
		e.st.Problem.Set(session.Problem{}) // a turn that finished answers the last one's error
		e.sh.SetTurn(reply, session.Turn{
			Role: session.RoleAssistant, Text: text, Think: think, Tokens: n, TokPerSec: rate,
			Model: modelName, Colour: modelColour,
		})
		e.touch()
		e.sh.SetStatus(fmt.Sprintf("%d token(s) at %.2f tok/s — %s%s",
			n, rate, stop, kvNote))
	})
}

// promptIDs builds the token ids for one request.
//
// The chat path goes through ChatIDs, not ChatPrompt+Encode: a rendered
// template is a complete prompt with its own specials and must be encoded with
// addSpecial=false.
func (e *Engine) promptIDs(r session.ChatRequest) ([]int32, error) {
	if e.m.Vocab == nil {
		return nil, fmt.Errorf("this container has no tokenizer: %v", e.m.TokErr)
	}
	if !r.Chat {
		return e.m.Vocab.Encode(r.Prompt, true), nil
	}
	return e.m.ChatIDs(chatMessages(r), true)
}

// chatMessages is the request's conversation in the model's terms.
func chatMessages(r session.ChatRequest) []model.ChatMessage {
	msgs := make([]model.ChatMessage, 0, len(r.Messages))
	for _, m := range r.Messages {
		msgs = append(msgs, model.ChatMessage{Role: m.Role, Content: m.Content, Images: len(m.Images)})
	}
	return msgs
}

// hasImages reports whether any turn of the request carries a picture.
func hasImages(r session.ChatRequest) bool {
	for _, m := range r.Messages {
		if len(m.Images) > 0 {
			return true
		}
	}
	return false
}

// imageSpans lays every picture of the request into the conversation as a
// picture span, which the prefill encodes in the session's own vision segment
// (model.Picture): its JIT and pool, its device, the model's image cache. A
// picture is preprocessed once per file -- pictures maps a file's identity to
// its pixels and name, never to rows.
func (e *Engine) imageSpans(r session.ChatRequest) ([]model.Span, error) {
	if e.m.Tower() == nil {
		return nil, fmt.Errorf("the loaded model has no vision tower, so it cannot take a picture")
	}
	if e.pictures == nil {
		e.pictures = map[fileID]*model.Picture{}
	}
	var imgs []model.Image
	for _, m := range r.Messages {
		for _, p := range m.Images {
			fi, err := os.Stat(p)
			if err != nil {
				return nil, err
			}
			k := fileID{p, fi.Size(), fi.ModTime().UnixNano()}
			pc, ok := e.pictures[k]
			if !ok {
				if pc, err = e.picture(p); err != nil {
					return nil, fmt.Errorf("%s: %w", filepath.Base(p), err)
				}
				e.pictures[k] = pc
			}
			imgs = append(imgs, model.Image{Picture: pc})
		}
	}
	return e.m.ChatSpansImages(chatMessages(r), imgs, true)
}

// picture decodes and preprocesses one picture on the session's vision
// segment.
func (e *Engine) picture(path string) (*model.Picture, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	var pc *model.Picture
	e.inspect(func(st *model.State) { pc, err = st.Picture(img) })
	return pc, err
}

// display is the decoded text made safe to put on screen.
//
// It decodes the whole id slice rather than concatenating per-token pieces,
// because Vocab.Text returns raw pieces (SentencePiece "▁" markers). A
// partial must not split a rune, so an incomplete multi-byte tail is held
// back. It does nothing else: reasoning is split by the caller, and
// TestThePipelineKeepsTheReasoning guards that pairing.
func display(s string) string {
	for s != "" && !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	return s
}

// SplitThinking separates a model's reasoning block from its answer.
//
// The answer is the reply and the reasoning is shown in a collapsed section,
// so the two are separated. An unclosed block is all thinking and no answer,
// which is the true state while a reasoning model is still working. The tags
// are matched literally as text.
func SplitThinking(s string) (think, answer string) { return splitThinking(s, false) }

// SplitThinkingOpen is SplitThinking for a template that already opened the
// block, so the output begins inside it.
//
// The engine knows this from the rendered prompt (templateOpensThinking); the
// parser cannot, since reasoning without a tag is indistinguishable from an
// answer.
func SplitThinkingOpen(s string) (think, answer string) { return splitThinking(s, true) }

func splitThinking(s string, inBlock bool) (think, answer string) {
	var tb, ab strings.Builder
	if inBlock {
		i := strings.Index(s, thinkClose)
		if i < 0 {
			// Still inside: all of it is reasoning and there is no answer yet.
			return strings.TrimSpace(s), ""
		}
		tb.WriteString(s[:i])
		s = s[i+len(thinkClose):]
	}

	// A closing tag before any opening one means the template opened the block.
	// This is the fallback for a caller that does not know; the engine uses
	// SplitThinkingOpen.
	if i := strings.Index(s, thinkClose); i >= 0 && !inBlock {
		if j := strings.Index(s, thinkOpen); j < 0 || j > i {
			tb.WriteString(s[:i])
			s = s[i+len(thinkClose):]
		}
	}

	for {
		i := strings.Index(s, thinkOpen)
		if i < 0 {
			ab.WriteString(s)
			break
		}
		ab.WriteString(s[:i])
		rest := s[i+len(thinkOpen):]
		j := strings.Index(rest, thinkClose)
		if j < 0 {
			// Still inside the block: everything after the tag is reasoning.
			if tb.Len() > 0 {
				tb.WriteString("\n")
			}
			tb.WriteString(rest)
			s = ""
			break
		}
		// A separator, because two blocks are two thoughts: concatenated they
		// read as one sentence running into the next.
		if tb.Len() > 0 {
			tb.WriteString("\n")
		}
		tb.WriteString(rest[:j])
		s = rest[j+len(thinkClose):]
	}
	return strings.TrimSpace(tb.String()), strings.TrimSpace(ab.String())
}

// holdPartialTag trims a trailing fragment of a tag that has not fully
// arrived.
//
// Otherwise every block flickers "<", "<t", "<th" on its way in.
func holdPartialTag(s string) string {
	for _, tag := range []string{thinkClose, thinkOpen} {
		for n := len(tag) - 1; n > 0; n-- {
			if strings.HasSuffix(s, tag[:n]) {
				return s[:len(s)-n]
			}
		}
	}
	return s
}

const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// kvCache is the prefix cache in session.KVCacheDir and the key its pages
// are named under, or why there is none. That is not an error: the session
// still runs and re-prefills, but the load says so.
//
// The namespace carries size and mtime, not just the basename: many containers
// are named model.jlm, and the geometry hash cannot separate two fine-tunes of
// one architecture.
func kvCache(path string) (model.KVStore, string, string) {
	dir := session.KVCacheDir()
	if dir == "" {
		return nil, "", "this system has no cache folder"
	}
	// Bounded, because nothing in a KV cache expires.
	store, err := model.NewFileStoreLimit(dir, kvCacheMax)
	if err != nil {
		return nil, "", err.Error()
	}
	ns := "jlm/" + filepath.Base(path)
	if fi, err := os.Stat(path); err == nil {
		ns = fmt.Sprintf("jlm/%s/%d-%d", filepath.Base(path), fi.Size(), fi.ModTime().UnixNano())
	}
	return store, ns, ""
}

// kvCacheMax bounds the prefix cache on disk. 8 GiB is cmd/jitllm's default
// and is ~64 prompts of a thousand positions on a 1B model.
const kvCacheMax = 8 << 30

// kvStatus reports the reuse for one turn.
func kvStatus(reused, prompted int, fails int64) string {
	if prompted <= 0 {
		return ""
	}
	msg := fmt.Sprintf("kv cache %d of %d prompt position(s) reused (%.0f%%)",
		reused, prompted, 100*float64(reused)/float64(prompted))
	if fails > 0 {
		msg += fmt.Sprintf("  %d STORE WRITE FAILURE(S)", fails)
	}
	return msg
}

// lastReused and lastFailures expose the prefix cache's own counters for the
// turn just finished. They read the session directly and so must be called
// when no generation is in flight, which a test after pump() is.
func (e *Engine) lastReused() (n int) {
	e.inspect(func(st *model.State) { n = st.KVRestored() })
	return n
}

func (e *Engine) lastFailures() (n int64) {
	e.inspect(func(st *model.State) { n = st.KVStoreFailures() })
	return n
}

// errStopped ends a generate the person stopped.
var errStopped = errors.New("stopped")

// lastPrompt is how many tokens the last prompt was, for the reuse probe.
func (e *Engine) lastPrompt() int { return e.lastPromptTok }

// templateOpensThinking reports whether the rendered chat prompt ends inside a
// reasoning block, which is how qwen3's template is written.
func (e *Engine) templateOpensThinking(r session.ChatRequest) bool {
	if !r.Chat || e.m == nil {
		return false
	}
	rendered, err := e.m.ChatPrompt(chatMessages(r), true)
	if err != nil {
		return false
	}
	return strings.HasSuffix(strings.TrimRight(rendered, " \t\r\n"), thinkOpen)
}

// Package mock stands in for everything the screens reach outside the widget
// tree -- the engine, the hardware and the model files -- so the whole app runs
// with no model and no GPU: in the shots, in the stage tests and under
// `jitllm-ui -mock`.
//
// The mock engine publishes exactly as engine.Engine does: every result is a
// Store write posted to the UI goroutine through Shell.Post, after the call
// that caused it has returned. A screen therefore meets the mock in the same
// order it meets the engine -- built empty, then filled -- which is the order
// that hides layout bugs from a fixture set up front.
package mock

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/hardware"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/screen"
	"github.com/jitllm/jitllm/ui/widgets"
)

// Model is one fixture: a catalog entry and what the engine publishes once it
// is loaded.
type Model struct {
	Entry   catalog.Entry
	Summary string
	Device  string
	Blocks  []byte
	Alloc   app.Allocation
	Pager   app.PagerStat
	Vision  bool
	// Reply is what every generation streams, a word at a time; Think is the
	// reasoning streamed before it.
	Reply, Think string
}

const (
	llamaPage = 39_600_000
	qwenPage  = 392_660_000
	gpuName   = "0:NVIDIA GeForce RTX 3050 Ti Laptop GPU [cuda]"
)

// Llama is Llama-3.2-1B, all 16 blocks on one card: a model that fits.
func Llama() Model {
	b := make([]byte, 16)
	for i := range b {
		b[i] = byte(widgets.OnDevice(0) | widgets.PlaceResident)
	}
	return Model{
		Entry: catalog.Entry{Path: "/mock/Llama-3.2-1B-Instruct-Q4_K_M.jlm", Name: "Llama-3.2-1B-Instruct-Q4_K_M.jlm",
			Kind: catalog.KindContainer, Size: 816 << 20, Probed: true, Quant: "Q4_K_M", NLayer: 16, NEmbd: 2048,
			Version: catalog.CurrentVersion, MTime: time.Unix(1758300000, 0)},
		Summary: "Llama-3.2-1B-Instruct-Q4_K_M · chat · context 4,096 · all 16 layers on the NVIDIA GeForce RTX 3050 Ti Laptop GPU",
		Device:  "auto (1 device(s))",
		Blocks:  b,
		Alloc: app.Allocation{
			NBlocks: 16, DeviceBlocks: 16,
			HostUsed: 16 * llamaPage, HostBudget: 8 << 30, Dense: 182_100_000,
			PageBytes: llamaPage, MaxSeq: 4096,
			Devices: []app.DeviceUse{{Name: gpuName, Short: "cuda:0", Blocks: 16,
				Used: 1_610_612_736, Limit: 2_147_483_648, Pool: "sm_86",
				Weights: 16 * llamaPage, KVPool: 134_217_728}},
		},
		Pager: app.PagerStat{Frames: 16, Resident: 16 * llamaPage, PageIns: 16, BytesRead: 598 << 20, Reads: 17, ChunkBytes: 16 << 10},
		Reply: "Paris is the capital of France. It sits on the Seine, and with about two million " +
			"people in the city proper it is the country's largest city.",
	}
}

// Qwen is Qwen3-30B-A3B over two cards and a host one budget short: seams,
// paging and a shared-memory device in one fixture.
func Qwen() Model {
	const n = 48
	b := make([]byte, n)
	for i := range b {
		switch {
		case i < 4:
			b[i] = byte(widgets.OnDevice(0))
		case i < 13:
			b[i] = byte(widgets.OnDevice(1))
		case i == 13 || (i > 20 && i < 24):
			b[i] = byte(widgets.PlaceCPU)
		default:
			b[i] = byte(widgets.PlaceCPU | widgets.PlaceResident)
		}
	}
	b[30] = byte(widgets.OnDevice(0))
	return Model{
		Entry: catalog.Entry{Path: "/mock/Qwen3-30B-A3B-Q4_K_M.jlm", Name: "Qwen3-30B-A3B-Q4_K_M.jlm",
			Kind: catalog.KindContainer, Size: 18 << 30, Probed: true, Quant: "Q4_K_M", NLayer: n, NEmbd: 2048,
			Version: catalog.CurrentVersion, MTime: time.Unix(1758310000, 0)},
		Summary: "Qwen3-30B-A3B-Q4_K_M · chat · context 32,768 · 14 of 48 layers on 2 devices, the rest paged on the host",
		Device:  "cuda:0,vulkan:1 (2 device(s))",
		Blocks:  b,
		Alloc: app.Allocation{
			NBlocks: n, CPUBlocks: 34, DeviceBlocks: 14,
			HostUsed: 30 * qwenPage, HostBudget: 30*qwenPage + 200<<20, Dense: 622_000_000,
			PageBytes: qwenPage, MaxSeq: 32_768,
			Devices: []app.DeviceUse{
				{Name: gpuName, Short: "cuda:0", Blocks: 5, Used: 2_300_000_000, Limit: 3_800_000_000,
					Pool: "sm_86", Weights: 5 * qwenPage, KVPool: 256 << 20},
				{Name: "1:Intel(R) Iris(R) Xe Graphics [vulkan]", Short: "vulkan:1", Blocks: 9,
					Used: 9 * qwenPage, Limit: 20 << 30, Pool: "host", Shared: true, Weights: 9 * qwenPage},
			},
		},
		Pager: app.PagerStat{Frames: 30, Resident: 30 * qwenPage, PageIns: 412, PageOuts: 382,
			BytesRead: 150 << 30, Reads: 9_120, ChunkBytes: 1 << 20},
		Think: "The user wants an HTTP server in Rust. The standard library has no HTTP server, " +
			"so this means a crate; axum is the common choice and keeps the example short.",
		Reply: "Use axum: add axum and tokio to Cargo.toml, build a Router with one route, " +
			"and serve it with axum::serve on a TcpListener bound to 0.0.0.0:3000.",
	}
}

// Models is every fixture, in catalog order.
func Models() []Model { return []Model{Llama(), Qwen()} }

// Sources are the files the Convert tab lists: a GGUF not yet converted, one
// whose container is there, and a vision tower that goes with its model.
func Sources() []catalog.Entry {
	return []catalog.Entry{
		{Path: "/mock/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", Name: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
			Kind: catalog.KindGGUF, Size: 4372812000, Quant: "Q4_K_M",
			Target: "/mock/Mistral-7B-Instruct-v0.3-Q4_K_M.jlm"},
		{Path: "/mock/Llama-3.2-1B-Instruct-Q4_K_M.gguf", Name: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
			Kind: catalog.KindGGUF, Size: 807694464, Quant: "Q4_K_M",
			Target: "/mock/Llama-3.2-1B-Instruct-Q4_K_M.jlm", TargetVersion: catalog.CurrentVersion},
		{Path: "/mock/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", Name: "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
			Kind: catalog.KindGGUF, Size: 103741440, Quant: "Q8_0", Tower: true},
	}
}

// kvPerPos is the history one position costs on each tier, for the fixtures'
// KV to grow with the conversation the way the engine's does.
const kvPerPos = 32 << 10

// Engine is a screen.Engine with no model behind it.
type Engine struct {
	sh *app.Shell
	// Live streams a reply on its own, a word every Interval, as the real
	// engine's decode loop does, and paces a load through its stages. Off, the
	// caller advances a reply with Step.
	Live     bool
	Interval time.Duration
	// Hold parks a load at its first stage until Release, so a scenario can
	// see what a person sees while a large model opens.
	Hold bool
	held []string

	mu     sync.Mutex
	open   []*Model
	active *Model
	words  []string // the reply in flight, what is left of it
	think  int      // how many of words are reasoning
	reply  int      // the turn index the reply goes into
	sent   int
	start  time.Time
	// Calls is every method called, in order. Read it under no concurrent call.
	Calls []string
}

// New builds a mock engine that publishes into sh.
func New(sh *app.Shell) *Engine {
	var _ screen.Engine = (*Engine)(nil)
	return &Engine{sh: sh, Interval: 40 * time.Millisecond}
}

func (e *Engine) called(format string, a ...any) {
	e.Calls = append(e.Calls, fmt.Sprintf(format, a...))
}

func find(path string) *Model {
	for _, m := range Models() {
		if m.Entry.Path == path {
			return &m
		}
	}
	return nil
}

// Load opens a fixture by path and makes it active. Loads queue, as the
// engine's do: the first is opened, the rest wait.
func (e *Engine) Load(path string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.called("Load %s", path)
	if m := e.lookup(path); m != nil {
		e.activate(m)
		return
	}
	if find(path) == nil {
		e.sh.Post(func() { e.sh.SetStatus("mock: no fixture is called " + path) })
		return
	}
	if slices.Contains(e.held, path) {
		return
	}
	// The stages the engine names, in its order.
	stage := "Reading the model"
	if len(e.held) > 0 {
		stage = "Waiting"
	}
	e.sh.Store.QueueLoad(app.Loading{Path: path, Stage: stage})
	e.held = append(e.held, path)
	switch {
	case e.Hold:
	case e.Live:
		if len(e.held) == 1 {
			go e.work()
		}
	default:
		e.release()
	}
}

// work opens the queue at a pace a person can watch.
func (e *Engine) work() {
	for {
		time.Sleep(900 * time.Millisecond)
		e.Stage("Placing layers on the GPU")
		time.Sleep(900 * time.Millisecond)
		e.Release()
		e.mu.Lock()
		more := len(e.held) > 0
		e.mu.Unlock()
		if !more {
			return
		}
	}
}

// Stage moves the load in progress to its next stage, as the engine reports one.
func (e *Engine) Stage(s string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.held) == 0 {
		return
	}
	path := e.held[0]
	e.sh.Post(func() { e.sh.Store.SetLoadStage(path, s) })
}

// Release finishes the load in progress.
func (e *Engine) Release() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.release()
}

// release opens the first held fixture and starts the next. e.mu is held.
func (e *Engine) release() {
	if len(e.held) == 0 {
		return
	}
	path := e.held[0]
	e.held = e.held[1:]
	if m := find(path); m != nil {
		e.open = append(e.open, m)
		e.activate(m)
	}
	var next string
	if len(e.held) > 0 {
		next = e.held[0]
	}
	e.sh.Post(func() {
		e.sh.Store.EndLoad(path)
		if next != "" {
			e.sh.Store.SetLoadStage(next, "Reading the model")
		}
	})
}

// Use makes an open fixture the active one.
func (e *Engine) Use(path string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.called("Use %s", path)
	if m := e.lookup(path); m != nil {
		e.activate(m)
	}
}

func (e *Engine) lookup(path string) *Model {
	for _, m := range e.open {
		if m.Entry.Path == path {
			return m
		}
	}
	return nil
}

// activate publishes m as the loaded model. e.mu is held.
func (e *Engine) activate(m *Model) {
	e.active = m
	models := make([]app.LoadedModel, len(e.open))
	for i, o := range e.open {
		models[i] = app.LoadedModel{Path: o.Entry.Path, Name: o.Entry.Name,
			Blocks: int(o.Entry.NLayer), Colour: i, Active: o == m, Grant: o.Alloc.HostBudget}
	}
	snap := *m
	e.sh.Post(func() {
		st := e.sh.Store
		st.Models.Set(models)
		st.Active.Set(snap.Entry.Path)
		st.ModelPath.Set(snap.Entry.Path)
		st.ModelSummary.Set(snap.Summary)
		st.ChatCapable.Set(true)
		st.Vision.Set(snap.Vision)
		st.Loaded.Set(true)
		st.DeviceReport.Set(snap.Device)
		st.BlockMap.Set(snap.Blocks)
		st.Pager.Set(snap.Pager)
		st.Alloc.Set(snap.Alloc)
		st.Problem.Set(app.Problem{})
		e.sh.SetStatus("loaded " + snap.Entry.Name)
	})
}

// Unload closes a fixture. Closing the active one leaves nothing in chat.
func (e *Engine) Unload(path string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.called("Unload %s", path)
	m := e.lookup(path)
	if m == nil {
		return
	}
	e.open = slices.DeleteFunc(e.open, func(o *Model) bool { return o == m })
	if e.active == m {
		e.active = nil
	}
	models := make([]app.LoadedModel, len(e.open))
	for i, o := range e.open {
		models[i] = app.LoadedModel{Path: o.Entry.Path, Name: o.Entry.Name,
			Blocks: int(o.Entry.NLayer), Colour: i, Active: o == e.active, Grant: o.Alloc.HostBudget}
	}
	active := e.active != nil
	e.sh.Post(func() {
		st := e.sh.Store
		st.Models.Set(models)
		if !active {
			st.Active.Set("")
			st.Loaded.Set(false)
			st.ModelPath.Set("")
			st.BlockMap.Set(nil)
			st.Alloc.Set(app.Allocation{})
		}
		e.sh.SetStatus("closed " + m.Entry.Name)
	})
}

// SetPriority records the choice; the fixtures' budgets do not move.
func (e *Engine) SetPriority(on bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.called("SetPriority %v", on)
}

// Relocate puts the first n blocks on device 0 and the rest on the host.
func (e *Engine) Relocate(n int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.called("Relocate %d", n)
	m := e.active
	if m == nil || len(m.Alloc.Devices) == 0 {
		return
	}
	n = min(max(n, 0), len(m.Blocks))
	for i := range m.Blocks {
		if i < n {
			m.Blocks[i] = byte(widgets.OnDevice(0) | widgets.PlaceResident)
		} else {
			m.Blocks[i] = byte(widgets.PlaceCPU | widgets.PlaceResident)
		}
	}
	m.Alloc.DeviceBlocks, m.Alloc.CPUBlocks = n, len(m.Blocks)-n
	for k := range m.Alloc.Devices {
		d := &m.Alloc.Devices[k]
		was := d.Weights
		d.Blocks, d.Weights = 0, 0
		if k == 0 {
			d.Blocks, d.Weights = n, uint64(n)*m.Alloc.PageBytes
		}
		// The weights are what moved, so the pool's charge moves with them.
		d.Used = d.Used - min(was, d.Used) + d.Weights
	}
	blocks, alloc := append([]byte(nil), m.Blocks...), m.Alloc
	e.sh.Post(func() {
		e.sh.Store.BlockMap.Set(blocks)
		e.sh.Store.Alloc.Set(alloc)
		e.sh.SetStatus(fmt.Sprintf("moved: %d of %d block(s) on the device", n, len(blocks)))
	})
}

// Run runs a job on its own goroutine, as the engine's worker would.
func (e *Engine) Run(job func()) bool {
	e.mu.Lock()
	e.called("Run")
	e.mu.Unlock()
	go job()
	return true
}

// Send queues the active fixture's reply. Live, it streams on its own;
// otherwise each Step publishes one word.
func (e *Engine) Send(r screen.ChatRequest) {
	e.mu.Lock()
	e.called("Send %q", r.Prompt)
	m := e.active
	if m == nil {
		e.mu.Unlock()
		e.sh.Alert("No model loaded", "Open a model on the Models tab first.")
		return
	}
	think := strings.Fields(m.Think)
	e.words = append(think, strings.Fields(m.Reply)...)
	e.think, e.reply, e.sent, e.start = len(think), r.Reply, 0, time.Now()
	e.sh.Store.Busy.Set(true)
	e.sh.Store.Streaming.Set(true)
	e.sh.Post(func() { e.sh.Store.PromptTokS.Set(396.41) })
	e.mu.Unlock()
	if e.Live {
		go func() {
			for e.Step() {
				time.Sleep(e.Interval)
			}
		}()
	}
}

// Stop finishes the reply in flight with what it has.
func (e *Engine) Stop() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.called("Stop")
	if e.words != nil {
		e.words = e.words[:e.sent]
		e.finish()
	}
}

// Step publishes the next word of the reply in flight and reports whether
// there is more.
func (e *Engine) Step() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.words == nil {
		return false
	}
	if e.sent == len(e.words) {
		e.finish()
		return false
	}
	e.sent++
	think, text := e.split()
	reply, opened := e.reply, e.sent == 1 && e.think > 0
	e.sh.Post(func() {
		st := e.sh.Store
		st.Stream.Set(text)
		st.StreamThink.Set(think)
		st.DecodeTokS.Set(39.61)
		if opened {
			st.SetReplyThinkOpen(reply, true)
		}
		st.Touch()
	})
	return true
}

func (e *Engine) split() (think, text string) {
	n := min(e.sent, e.think)
	return strings.Join(e.words[:n], " "), strings.Join(e.words[n:e.sent], " ")
}

// finish writes the reply into its turn, as the engine's last post does, and
// grows the history the fixture holds. e.mu is held.
func (e *Engine) finish() {
	think, text := e.split()
	n, reply, m := e.sent, e.reply, e.active
	e.words = nil
	m.Alloc.Pos = min(m.Alloc.Pos+n+24, m.Alloc.MaxSeq)
	kv := uint64(m.Alloc.Pos) * kvPerPos
	m.Alloc.CPUKV = kv * uint64(m.Alloc.CPUBlocks) / uint64(max(m.Alloc.NBlocks, 1))
	for k := range m.Alloc.Devices {
		d := &m.Alloc.Devices[k]
		d.KV = kv * uint64(d.Blocks) / uint64(max(m.Alloc.NBlocks, 1))
		d.KVPool = max(d.KVPool, d.KV)
	}
	alloc, name := m.Alloc, m.Entry.Name
	e.sh.Post(func() {
		st := e.sh.Store
		st.Stream.Set(text)
		st.StreamThink.Set(think)
		st.DecodeTokS.Set(39.61)
		st.BytesPerTok.Set(816_010_912)
		st.Alloc.Set(alloc)
		st.SetTurn(reply, app.Turn{Role: app.RoleAssistant, Text: text, Think: think,
			Tokens: n, TokPerSec: 39.61, Model: name})
		st.Touch()
		st.Streaming.Set(false)
		st.Busy.Set(false)
		e.sh.SetStatus(fmt.Sprintf("%d token(s) at 39.61 tok/s", n))
	})
}

// Machine is a screen.Machine that reports the machine the fixtures describe,
// without opening a backend.
type Machine struct{}

// Report is that machine.
func Report() hardware.Report {
	return hardware.Report{
		Probed: true, GOOS: "linux", GOARCH: "amd64", GoVersion: "go1.27.1",
		CPU: "12th Gen Intel(R) Core(TM) i9-12900HK", Logical: 20,
		PCores: []int{0, 2, 4, 6, 8, 10}, ECores: []int{12, 13, 14, 15, 16, 17, 18, 19},
		SMT: []int{1, 3, 5, 7, 9, 11}, Decode: []int{0, 2, 4, 6, 8, 10}, RAMTotal: 31 << 30,
		MemBudget: 15 << 30, GCBudgetCap: 24 << 30, MemWall: 55.7e9,
		Arch: "amd64", ISA: "sse2, sse3, ssse3, sse4.1, sse4.2, popcnt, avx2, fma, f16c, avx-vnni, bmi2",
		Form: "x86, wide passes (AVX2, FMA, F16C), int8 dots by VPDPBUSD",
		GPUs: []hardware.GPU{
			{API: "cuda", Name: "NVIDIA GeForce RTX 3050 Ti Laptop GPU", Slots: 20, Free: 3_800_000_000, Total: 4 << 30},
			{API: "vulkan", Name: "Intel(R) Iris(R) Xe Graphics", Slots: 96, Free: 20 << 30, Total: 20 << 30, Unified: true},
		},
	}
}

// Probe is Report.
func (Machine) Probe() hardware.Report { return Report() }

// Cached is Report, as if probed already.
func (Machine) Cached() (hardware.Report, bool) { return Report(), true }

// Files is a screen.Files whose disk holds the fixtures and nothing else.
type Files struct{}

// Scan returns every fixture's entry.
func (Files) Scan([]string) []catalog.Entry {
	var out []catalog.Entry
	for _, m := range Models() {
		out = append(out, m.Entry)
	}
	return append(out, Sources()...)
}

// Probe does nothing: a fixture's entry is already probed.
func (Files) Probe(*catalog.Entry) {}

// Deps is all three, publishing into sh.
func Deps(sh *app.Shell) (screen.Deps, *Engine) {
	e := New(sh)
	return screen.Deps{Engine: e, Machine: Machine{}, Files: Files{}}, e
}

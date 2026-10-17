// Package server is jitllm's serving layer: a ConnectRPC control plane over
// the engine's own orchestration primitives, plus OpenAI- and
// Anthropic-compatible HTTP shims that are thin adapters over the SAME
// internal generate path.
//
// jitllm is an inference OS: it places and moves blocks across every core and
// accelerator, and the caller can force each of those decisions. So
// PlacementService is a first-class surface here, and the compatibility shims
// are adapters beside it.
package server

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/tok"
)

// Config is what a server is built with. Every field has a working default.
type Config struct {
	// ModelDir is where ListModels scans and where a bare name resolves
	// (default "models", relative to the working directory).
	ModelDir string

	// BatchDir is where the OpenAI Files and Batch APIs keep their files
	// (default ".jitllm-batch" under ModelDir).
	BatchDir string

	// MaxBatchRows bounds how many generates of one device model decode as
	// rows of one step (batch.go). Zero takes the engine's bound, the widest
	// step a device runs across sessions; 1 turns batching off, and every
	// generate then runs alone on its gates.
	MaxBatchRows int

	// PromptChunk is how many prompt tokens the step loop feeds into one step
	// beside the rows already decoding, which bounds how much admitting a
	// prompt holds each of their tokens up. Zero takes a device prefill chunk
	// (nn.MaxDevicePrefillChunk), the pieces the device would cut it into
	// anyway; a step never carries more than model.MaxStepRows rows in all.
	PromptChunk int

	// StepPromptTokens bounds the prompt tokens one step carries beside
	// decoding rows, so admitting a prompt holds each of their tokens up by
	// about one decode step rather than a prompt chunk. Zero measures it
	// (stepBudget): the prompt tokens that cost one decode step's time. A
	// step with no decoding row takes PromptChunk whole.
	StepPromptTokens int

	// StepCost is how many decode steps' time a step carrying prompt tokens
	// beside decoding rows may take, when StepPromptTokens measures the
	// budget: the decoding rows' worst inter-token gap against an arriving
	// prompt's time to its first token. Zero (or up to 1) takes
	// DefaultStepCost.
	StepCost float64

	// Fairness is the step loop's policy between throughput and the spread
	// of the requests' waits, 0 to 100 (batchfair.go): 0 serves first come
	// first served and runs each row to its end, the most tokens a second;
	// above it rows are time-sliced (parked for a waiting request after a
	// quantum that shrinks with the level), the prompts admitted after the
	// oldest are fed beside it, and priority is a head start that age
	// overtakes. nil takes DefaultFairness.
	Fairness *int

	// JointSteps is how a decode step whose rows could run as one joint step
	// does run: measured per row count (the default), always joint, or never.
	JointSteps JointSteps

	// ROCm is the ROCm library directory the AMD backend loads from, or empty
	// for the default search (jit/gpu/hip). jitllmd fills it from JITLLM_ROCM.
	ROCm string

	// TuneCache is the directory the device tuners keep their answers in
	// (tier.WithTuneCache), or empty for none. jitllmd passes the user cache
	// directory.
	TuneCache string

	// DefaultMaxSeq is the KV capacity a session gets when it asks for none.
	// Zero takes the model's own context length.
	DefaultMaxSeq int

	// KVF16 is the KV cache width a load that names none gets: true binary16,
	// false f32, nil the engine's own per-host choice (model.WithKVF16).
	KVF16 *bool

	// PromptStore is where a session created with PromptCache keeps its
	// prompt prefixes; nil refuses such a session. jitllmd's -kv-cache is a
	// model.FileStore.
	PromptStore model.KVStore

	// Images bounds the pictures the HTTP shims accept in a chat request, and
	// says whether a picture named by a remote URL is fetched (images.go).
	Images ImagePolicy

	// Version is reported by GetServerInfo.
	Version string

	// NoMemCache turns the per-model memory cache off: every generate then
	// prefills its whole prompt, as a session given no KVStore does.
	NoMemCache bool
	// MemCacheBytes bounds each model's memory cache; zero is an eighth
	// of the model's host share, and never more than an eighth of what the
	// host has available when the cache grows (storeLimit).
	MemCacheBytes uint64
	// HostBudget is the host memory every loaded model's share is divided
	// from. Zero asks the host at each load: sched.MemBudget, the smallest of
	// the cgroup's limit, the bound NUMA nodes' memory and what is available,
	// less a slice, plus what this engine's models already hold.
	HostBudget uint64
	// SessionPool is how many reset States each model keeps for model_id
	// requests: zero is LoadOptions.Sessions (at least one), negative none.
	SessionPool int
	// MaxQueue is how many requests one model holds, running or waiting,
	// before it refuses the next with ErrOverloaded: zero is 64, negative
	// unbounded.
	MaxQueue int
	// RetryAfter is what an HTTP refusal for ErrOverloaded tells the client
	// to wait; zero is a second.
	RetryAfter time.Duration
	// WarmLoads warms every load, as LoadOptions.Warm does one: the Connect
	// LoadModel call has no field for it.
	WarmLoads bool
	// OffHeap is told, after every load, unload, pin and change of priority,
	// the page budgets every loaded model's frames may grow to, summed: the
	// off-heap memory the loaded models can reach, 0 with none loaded. The
	// collector's memory limit is process-wide, so the engine does not set it;
	// jitllmd hands goheap.OffHeap here, as cmd/jitllm calls it after its one
	// load (AGENTS.md RULE 2f).
	OffHeap func(budgets uint64)

	// Probe enumerates the hardware. nil takes the real one, which is
	// expensive and exclusive: backend.Open() opens and closes every backend.
	// Tests and deployments that know their topology supply it here.
	Probe func() ([]DeviceInfo, error)

	// OnDemand is how a model a request names but nobody loaded is loaded
	// (library.go): its devices, budget and seam. Path and ModelID are
	// filled per model.
	OnDemand LoadOptions
	// NoOnDemand turns on-demand loading off: a request names a loaded model
	// or fails, as before the models folder was indexed.
	NoOnDemand bool
}

// preloadDepth is how many pages a load keeps in flight behind model.Open
// (model.WithPreload): a server's model is opened to be used, so its pages are
// read while the JIT and the device placement run rather than when the first
// request faults them in.
const preloadDepth = 4

func (c *Config) withDefaults() {
	if c.ModelDir == "" {
		c.ModelDir = "models"
	}
	if c.PromptChunk <= 0 {
		c.PromptChunk = nn.MaxDevicePrefillChunk
	}
	if c.Version == "" {
		c.Version = "dev"
	}
}

// Engine owns every loaded model, every open session, and the gates that
// record which sessions run on which hardware (they never queue one). It is the whole of the server's state; the six
// Connect services and the two HTTP shims are projections of it.
type Engine struct {
	cfg     Config
	started time.Time

	mu       sync.RWMutex
	models   map[string]*LoadedModel
	sessions map[string]*Session

	// gateMu is separate from mu on purpose. Generate holds a session's own
	// lock while it asks for a gate, and every read path takes mu; one mutex
	// over both would make the lock order depend on which handler ran.
	gateMu sync.Mutex
	gates  map[string]*gate
	// aliases maps a device id a caller may write ("cuda:1", "gpu:0",
	// "vulkan:0") to the gate key of the physical device a loaded tier
	// resolved it to (deviceGateKeys). Guarded by gateMu.
	aliases map[string]string

	seq atomic.Uint64

	// modelDir is Config.ModelDir until SetModelDir moves it.
	modelDir atomic.Pointer[string]

	// The host budget the models divide (budget.go), guarded by mu: total is
	// read at each load (rereadHostLocked), order is the models in load order,
	// and priority gives the favored model all but an eighth. hostAvail, set
	// by a test, replaces sched.MemBudget as the host's answer.
	total     uint64
	hostAvail func() uint64
	order     []string
	priority  bool
	favored   string

	preempt preemptStats

	// devices is the hardware probe, taken once. See devices.go for why it is
	// not re-taken on every list.
	devMu   sync.Mutex
	devices []DeviceInfo
	devErr  error
	probed  bool

	// libMu orders read-modify-writes of the models folder's state file;
	// demandMu makes on-demand loads one at a time, so two requests naming
	// the same model load it once (library.go).
	libMu    sync.Mutex
	demandMu sync.Mutex
}

// New builds an Engine. It does not touch the hardware; the first ListDevices
// or LoadModel does.
func New(cfg Config) *Engine {
	cfg.withDefaults()
	// One fetch bound for the whole server, shared by every surface.
	cfg.Images = cfg.Images.ready()
	return &Engine{
		cfg:      cfg,
		started:  time.Now(),
		models:   map[string]*LoadedModel{},
		sessions: map[string]*Session{},
		gates:    map[string]*gate{},
		aliases:  map[string]string{},
	}
}

// ModelDir is where ListModels scans and a bare name resolves.
func (e *Engine) ModelDir() string {
	if d := e.modelDir.Load(); d != nil {
		return *d
	}
	return e.cfg.ModelDir
}

// SetModelDir moves ModelDir, for a front end whose model folder is a
// setting the person can change while the engine runs.
func (e *Engine) SetModelDir(dir string) { e.modelDir.Store(&dir) }

// Config returns the configuration the Engine was built with, its defaults
// filled in.
func (e *Engine) Config() Config { return e.cfg }

// gate returns the gate for a device id, creating it on first use.
func (e *Engine) gate(id string) *gate {
	e.gateMu.Lock()
	defer e.gateMu.Unlock()
	if g, ok := e.gates[id]; ok {
		return g
	}
	g := newGate(id)
	e.gates[id] = g
	return g
}

func (e *Engine) nextID(prefix string) string {
	return fmt.Sprintf("%s-%d-%d", prefix, time.Now().UnixNano()/1e6, e.seq.Add(1))
}

// ---------------------------------------------------------------- models

// LoadedModel is one model the Engine holds open: its model.Model, the
// device tier its sessions share, and the sessions created on it.
type LoadedModel struct {
	id       string
	path     string
	name     string
	m        *model.Model
	loadedAt time.Time

	// dev is the tier, shared by every session of this model: the tier owns
	// the model, a session owns the sequence. nil for a host-only model.
	// grammars are the structured-output grammars compiled over this
	// model's tokenizer (grammar.go).
	grammars grammars

	dev       nn.Device
	gpu       *tier.GPU
	closeDev  func()
	deviceIDs []string
	// gateKeys is the physical devices the tier opened, one gate key each
	// (deviceGateKeys): what a session of this model serialises on. deviceIDs
	// is the spec as written and keys nothing, since `cuda` and `cuda:1` are
	// different text for one shared card.
	gateKeys []string
	// budget is the total the pager is divided from -- the load's, or the
	// last SetPageBudget's -- before the subtractions in hostWeightShare.
	// Guarded by mu.
	budget uint64
	// pin is a budget the caller set for this model (LoadOptions or
	// SetPageBudget), taken off the top of the division; 0 is a share.
	pin       uint64
	maxBlocks int
	// kvBudget caps the bytes this model's sessions' histories hold at once
	// (SetKVBudget, preempt.go); 0 is none. Guarded by mu.
	kvBudget uint64

	// loop batches this model's generates, on its device or on the host
	// (batch.go). nil with batching off. Set before the model is published
	// and never changed.
	loop *stepLoop

	// mu guards the model-level mutations: page budget, and the placement
	// default. Session work does not take it.
	mu sync.Mutex

	sessions map[string]*Session

	// embMu guards emb, the idle embedders Embed reuses (embed.go).
	embMu sync.Mutex
	emb   embedders

	// decMu guards dec, the decision model's one Decider (decide.go), and
	// decClosed, set at unload.
	decMu     sync.Mutex
	dec       *model.Decider
	decClosed bool
	// decQ is the decisions waiting for the Decider, guarded by decQMu;
	// decBatchMax the most requests one batch has answered.
	decQMu      sync.Mutex
	decQ        []*decJob
	decBatchMax atomic.Int64

	tokensGenerated atomic.Int64
	tokensPrefilled atomic.Int64

	// ttft is the memory cache, the State pool and the admission bound
	// (ttft.go); warmTook is how long the load's warm-up ran.
	ttft     ttft
	warmTook time.Duration
	// generates, prefillNanos and decodeNanos are every finished generate's
	// count and the time it spent in its prefill (the time to its first
	// token, queueing aside) and its decode: /metrics' summaries.
	generates    atomic.Int64
	prefillNanos atomic.Int64
	decodeNanos  atomic.Int64
}

// finished records one generate: n tokens decoded, after a prefill of prefill,
// in decode.
func (lm *LoadedModel) finished(n int, prefill, decode time.Duration) {
	lm.tokensGenerated.Add(int64(n))
	lm.generates.Add(1)
	lm.prefillNanos.Add(int64(prefill))
	lm.decodeNanos.Add(int64(decode))
}

// gateIDs is what a session of this model must hold to run: one gate per
// physical device its tier opened, keyed on the device and not on the spec
// text, so models loaded with `cuda` and `cuda:1` share card 1's gate and no
// other. A model with no device runs on the host and queues there.
func (lm *LoadedModel) gateIDs() []string {
	if len(lm.gateKeys) == 0 {
		return []string{HostGateID}
	}
	// A partial placement runs its remaining blocks on the host, so the host
	// gate is held too. That is not over-locking: those blocks really do run
	// on the same cores another host session would want.
	return append(append([]string{}, lm.gateKeys...), HostGateID)
}

// deviceGateKeys is the gate key of every device a tier opened -- the
// physical identity it already dedupes on, or the backend and ordinal where
// the backend reports none (tier.Hardware) -- and records the ids a caller may
// name each device by, so GetDeviceQueue and ListSessions find its gate.
func (e *Engine) deviceGateKeys(g *tier.GPU) []string {
	hw := g.Hardware()
	keys := make([]string, 0, len(hw))
	e.gateMu.Lock()
	defer e.gateMu.Unlock()
	for _, h := range hw {
		keys = append(keys, h.Key)
		e.aliases[h.Ref] = h.Key
		if h.Name != "" {
			e.aliases[h.Name] = h.Key
		}
	}
	return keys
}

// gateKey is the gate a device id a caller wrote names: the physical device a
// loaded tier resolved it to, or the probe's identity for it, or the id itself
// when neither knows it (a device no model has opened has no queue).
func (e *Engine) gateKey(id string) string {
	if id == HostGateID {
		return id
	}
	e.gateMu.Lock()
	k, ok := e.aliases[id]
	e.gateMu.Unlock()
	if ok {
		return k
	}
	e.devMu.Lock()
	defer e.devMu.Unlock()
	for _, d := range e.devices {
		if d.ID == id && d.PhysicalID != "" {
			return d.PhysicalID
		}
	}
	return id
}

// infoGateKey is the gate a listed device's queue is on: its physical
// identity when the probe read one, so a card listed under two backends
// reports one queue.
func (e *Engine) infoGateKey(d DeviceInfo) string {
	if d.PhysicalID != "" {
		return d.PhysicalID
	}
	return e.gateKey(d.ID)
}

// LoadOptions is LoadModel's input in Go terms.
type LoadOptions struct {
	Path            string
	ModelID         string
	PageBudgetBytes uint64
	DeviceIDs       []string
	MaxDeviceBlocks int // -1 means "as many as fit"
	KVF16           *bool
	// DeviceSample forces where a sampled token's candidates are selected
	// (model.WithDeviceSample); the zero value, auto, measures.
	DeviceSample model.DeviceSampleMode
	// Placement pins blocks and the head to named devices, in the grammar of
	// model.ParsePlacement ("0-15=cuda:0,16-31=vulkan:0"); the names are the
	// DeviceIDs entries. Empty leaves every block to the engine.
	Placement string
	// PlacementStrict fails the load when a Placement entry cannot be met,
	// instead of placing that block where the engine would have.
	PlacementStrict bool
	// Sessions is how many concurrent sessions every placed linear block
	// reserves a recurrent pair for (tier.Config.Sessions); attention history
	// is paged and reserves nothing. 0 and 1 are one: a second session then
	// gets what the first left over, and on a full card that is the host.
	Sessions int

	// Warm runs a short prefill and a decode step on a pooled State before
	// LoadModel returns, so the first request does not pay for the kernels
	// and recordings those emit (ttft.go).
	Warm bool

	// tierConfig adjusts the tier after Sessions is set. Tests use it to put
	// the device into a configuration a gate has to run (a joint step the
	// device refuses); nothing on the wire reaches it.
	tierConfig func(*tier.Config)
}

// DeviceError is a load that failed on the devices it named rather than on
// the file, so a front end can send the person to the device setting.
type DeviceError struct{ Err error }

func (d *DeviceError) Error() string { return d.Err.Error() }
func (d *DeviceError) Unwrap() error { return d.Err }

// ErrNotFound is returned for an unknown model or session id.
var ErrNotFound = errors.New("server: not found")

// ErrInUse is returned when an unload would orphan live sessions.
var ErrInUse = errors.New("server: still in use")

// ErrPlacement is returned for a placement the session's tier cannot make: a
// host-only session, a device the tier was not opened with, or a block another
// session is using where it is.
var ErrPlacement = errors.New("server: the placement cannot be made")

// ErrExists is returned for a model or session id that is already taken.
var ErrExists = errors.New("server: already exists")

// ErrInvalid is returned for a request the caller has to change before it can
// succeed. It is never a server fault: mapped to Internal, it would tell a
// client to retry the identical request.
var ErrInvalid = errors.New("server: invalid request")

// ResolvePath turns a bare name into a path under ModelDir. An absolute path
// is taken as given.
func (e *Engine) ResolvePath(p string) string {
	if p == "" {
		return p
	}
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(e.ModelDir(), p)
}

// hostInexact, when set, loads a host model the step loop runs without the
// one summation order (LoadModel). False but in a gate's violation: it is how
// the gate shows that a sampled row parts from its run alone without it.
var hostInexact bool

// LoadModel opens a container and, when devices were named, the tier that will
// hold its blocks. A GGUF is refused by model.Open with a NotConvertedError,
// which errors.go turns into a FailedPrecondition carrying the convert command.
func (e *Engine) LoadModel(o LoadOptions) (*LoadedModel, error) {
	if o.MaxDeviceBlocks == 0 {
		o.MaxDeviceBlocks = -1
	}
	path := e.ResolvePath(o.Path)
	id := o.ModelID
	if id == "" {
		id = e.nextID("m")
	}

	e.mu.Lock()
	if _, ok := e.models[id]; ok {
		e.mu.Unlock()
		return nil, fmt.Errorf("%w: model id %q is already loaded", ErrExists, id)
	}
	e.mu.Unlock()

	// The share is computed before the open, because model.Open and the tier
	// otherwise read sched.MemBudget() themselves, ignoring what the loaded
	// models already hold. A budget the caller named is a pin. The host is
	// asked again first: what it has to give moved since the last load.
	hostBudget := o.PageBudgetBytes
	e.mu.Lock()
	if e.cfg.HostBudget == 0 {
		e.rereadHostLocked()
	}
	e.mu.Unlock()
	if hostBudget == 0 {
		e.mu.Lock()
		hostBudget = e.sharesLocked(id)[id]
		e.mu.Unlock()
	}

	var (
		dev      nn.Device
		gpu      *tier.GPU
		closeDev = func() {}
		devIDs   []string
		gateKeys []string
	)
	if spec := strings.Join(o.DeviceIDs, ","); spec != "" && spec != HostGateID {
		es, err := tier.ParseDevices(spec)
		if err != nil {
			return nil, &DeviceError{err}
		}
		wantsDevice := false
		for _, x := range es {
			wantsDevice = wantsDevice || x.Device()
		}
		if wantsDevice {
			g, err := tier.OpenWith(
				tier.WithDevices(spec),
				tier.WithROCm(e.cfg.ROCm),
				tier.WithTuneCache(e.cfg.TuneCache),
				tier.WithLazyTune(e.cfg.TuneCache != ""),
				tier.WithHostBudget(hostBudget),
				tier.WithConfig(func(c *tier.Config) {
					c.Sessions = o.Sessions
					// The step loop (newStepLoop below) runs steps of up to
					// MaxStepRows rows: the device reserves their scratch at
					// placement, or a card the blocks filled refuses the step.
					if e.cfg.MaxBatchRows != 1 {
						c.StepRows = model.MaxStepRows
					}
					if o.tierConfig != nil {
						o.tierConfig(c)
					}
				}),
			)
			if err != nil {
				// Only "auto" degrades to the host. A spec that names a
				// device asked for that device; silently answering on the
				// CPU would be a wrong answer with a right shape.
				if len(es) != 1 || es[0].API != "auto" {
					return nil, &DeviceError{err}
				}
			} else {
				dev, gpu, closeDev = g, g, g.Close
				devIDs = normaliseDeviceIDs(o.DeviceIDs)
				gateKeys = e.deviceGateKeys(g)
			}
		}
	}

	opts := []model.Option{model.WithPageBudget(hostBudget), model.WithPreload(preloadDepth)}
	if kv := cmp.Or(o.KVF16, e.cfg.KVF16); kv != nil {
		opts = append(opts, model.WithKVF16(*kv))
	}
	opts = append(opts, model.WithDeviceSample(o.DeviceSample))
	if o.Placement != "" {
		pl, err := model.ParsePlacement(o.Placement, o.PlacementStrict)
		if err != nil {
			closeDev()
			return nil, &DeviceError{err}
		}
		if err := namedDevicesOpen(pl, gpu); err != nil {
			closeDev()
			return nil, &DeviceError{err}
		}
		opts = append(opts, model.WithPlacement(pl))
	}
	// A host model the step loop runs steps its sessions' rows as one pass
	// (model.StepRuns' host arm, a matmul over the rows) or one session after
	// another (a matvec each), as the joint choice decides per step. Its
	// contract is that a row comes out as it does alone, to the bit, so a
	// seeded sampler draws the same tokens either way. That holds only when
	// the matmul sums a row in the matvec's order: the pre-VNNI x86 and arm64
	// prefill GEMMs fold a k-quant's super-block in integers where the decode
	// matvec rounds per sub-block, arm64's decode picker times forms that sum
	// differently, and a wide pool splits a long matvec over k. GEMMExact is
	// the one summation order every packed kernel shares.
	if dev == nil && e.cfg.MaxBatchRows != 1 && !hostInexact {
		opts = append(opts, model.WithJITOptions(nn.WithGEMMExact(true)))
	}
	m, err := model.Open(path, opts...)
	if err != nil {
		closeDev()
		return nil, err
	}

	lm := &LoadedModel{
		id:        id,
		path:      path,
		name:      strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		m:         m,
		loadedAt:  time.Now(),
		dev:       dev,
		gpu:       gpu,
		closeDev:  closeDev,
		deviceIDs: devIDs,
		gateKeys:  gateKeys,
		budget:    hostBudget,
		pin:       o.PageBudgetBytes,
		maxBlocks: o.MaxDeviceBlocks,
		sessions:  map[string]*Session{},
	}
	if e.cfg.MaxBatchRows != 1 {
		lm.loop = newStepLoop(e, lm)
	}
	e.initTTFT(lm, o)

	e.mu.Lock()
	if _, ok := e.models[id]; ok {
		e.mu.Unlock()
		lm.closeLoop()
		m.Close()
		closeDev()
		return nil, fmt.Errorf("%w: model id %q is already loaded", ErrExists, id)
	}
	e.models[id] = lm
	e.order = append(e.order, id)
	e.mu.Unlock()
	// The models already loaded give up the bytes this one now holds.
	e.rebudget()
	if o.Warm || e.cfg.WarmLoads {
		// A model that cannot run a five-token prompt is not loaded: the
		// first request would find out instead.
		if err := e.warm(lm); err != nil {
			e.UnloadModel(id, true)
			return nil, err
		}
	}
	return lm, nil
}

// normaliseDeviceIDs drops the specs that name no device, so gateIDs does not
// queue on a gate called "auto".
func normaliseDeviceIDs(ids []string) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" || id == HostGateID {
			continue
		}
		// Strip a `=BYTES` budget suffix: it names a limit, not a device.
		if j := strings.IndexByte(id, '='); j >= 0 {
			id = id[:j]
		}
		out = append(out, id)
	}
	return out
}

// Model returns the loaded model with this id, or ErrNotFound.
func (e *Engine) Model(id string) (*LoadedModel, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	lm, ok := e.models[id]
	if !ok {
		return nil, fmt.Errorf("%w: model %q", ErrNotFound, id)
	}
	return lm, nil
}

// Models returns every loaded model, sorted by id.
func (e *Engine) Models() []*LoadedModel {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]*LoadedModel, 0, len(e.models))
	for _, lm := range e.models {
		out = append(out, lm)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

// UnloadModel closes a model. Every State must be closed before its Model, so
// force closes the sessions first rather than letting the ordering contract be
// violated by a caller who did not know about it.
func (e *Engine) UnloadModel(id string, force bool) (closed int, err error) {
	e.mu.Lock()
	lm, ok := e.models[id]
	if !ok {
		e.mu.Unlock()
		return 0, fmt.Errorf("%w: model %q", ErrNotFound, id)
	}
	var victims []*Session
	for _, s := range lm.sessions {
		victims = append(victims, s)
	}
	if len(victims) > 0 && !force {
		e.mu.Unlock()
		return 0, fmt.Errorf("%w: model %q has %d open session(s); pass force to close them",
			ErrInUse, id, len(victims))
	}
	delete(e.models, id)
	e.order = without(e.order, id)
	for _, s := range victims {
		delete(e.sessions, s.id)
	}
	e.mu.Unlock()

	for _, s := range victims {
		s.close()
	}
	closed = len(victims)
	e.closeIdle(lm)
	// The loop goes after the sessions -- their generates hold its rows until
	// they end -- and before the model it steps.
	lm.closeLoop()
	lm.closeEmbedders()
	lm.closeDecider()
	if cerr := lm.m.Close(); cerr != nil {
		err = cerr
	}
	lm.closeDev()
	// The remaining models grow into what this one held.
	e.rebudget()
	return closed, err
}

func without(ids []string, id string) []string {
	out := ids[:0]
	for _, x := range ids {
		if x != id {
			out = append(out, x)
		}
	}
	return out
}

// Close tears the whole engine down in the order the engine requires.
func (e *Engine) Close() {
	e.mu.Lock()
	models := make([]*LoadedModel, 0, len(e.models))
	for _, lm := range e.models {
		models = append(models, lm)
	}
	sessions := make([]*Session, 0, len(e.sessions))
	for _, s := range e.sessions {
		sessions = append(sessions, s)
	}
	e.models = map[string]*LoadedModel{}
	e.sessions = map[string]*Session{}
	e.order = nil
	e.mu.Unlock()

	for _, s := range sessions {
		s.close()
	}
	for _, lm := range models {
		lm.closeLoop()
		e.closeIdle(lm)
		lm.closeEmbedders()
		lm.closeDecider()
		lm.m.Close()
		lm.closeDev()
	}
}

func (lm *LoadedModel) closeLoop() {
	if lm.loop != nil {
		lm.loop.close()
	}
}

// ---------------------------------------------------------------- sessions

// Session is one sequence on a loaded model: a model.State, its sampler and
// the counters the stats calls report. Work on it is serialised by its own
// lock; a generate with only a model_id runs on an ephemeral one.
type Session struct {
	id      string
	modelID string
	lm      *LoadedModel
	st      *model.State
	maxSeq  int

	// mu serialises this session's own work. A session is one sequence, so two
	// concurrent generates on it are a caller error rather than a queue.
	mu sync.Mutex
	// move is a move between devices waiting for a token boundary
	// (Engine.MoveBlocks), under moveMu: a generate serves it between two
	// tokens, and an idle session where it was asked.
	moveMu sync.Mutex
	move   *moveReq

	sampling model.Sampler
	created  time.Time
	// cached is a session with a prompt store: it prefills through it, alone.
	cached bool
	// storeOn says the State is attached to the model's memory cache for the
	// generate in hand (attachStore). Guarded by mu.
	storeOn bool
	// spec is the session's speculation for generates that carry none, and
	// specRan a generate that stopped inside a speculative round: the model
	// ran rows past the last token the reply kept, so the session cannot be
	// continued (under mu).
	spec    Speculation
	specRan bool

	lastUsed  atomic.Int64
	generated atomic.Int64
	prefilled atomic.Int64
	running   atomic.Bool
	queuePos  atomic.Int32

	// Every reported figure is a snapshot taken under s.mu. model.State's
	// fields are written by Forward with no lock, so reading them from a stats
	// handler mid-generate is a data race, and taking s.mu there would block
	// for the whole generation. The session publishes a snapshot at create,
	// after every placement change and at the end of every generate.
	snapPos       atomic.Int32
	snapKV        atomic.Uint64
	snapDevBlocks atomic.Int32
	snapAt        atomic.Int64
	snapPlacement atomic.Pointer[placementSnapshot]
	// snapBatched is whether a generate here would decode as a row of the
	// model's step loop (Engine.joins), as of the snapshot.
	snapBatched atomic.Bool
	// snapSeam and snapSeamSettled are State.SeamTuned as of the snapshot:
	// the tuner is written by Forward, so TuneSeam reports these.
	snapSeam        atomic.Int32
	snapSeamSettled atomic.Bool

	cancelMu sync.Mutex
	cancel   context.CancelFunc

	closed atomic.Bool

	// priority, parked and snapHist are preemption's (preempt.go): the order
	// a session is parked in, whether it is parked, and its history's bytes
	// as of the last snapshot (model.State.HistoryBytes).
	priority atomic.Int32
	parked   atomic.Bool
	snapHist atomic.Uint64
	// parkView is the memory cache view a parked session's pages are pinned
	// in (park), released when it resumes. Guarded by mu.
	parkView *model.PinnedStore
}

// SessionOptions is CreateSession's input.
type SessionOptions struct {
	ModelID string
	// modelName is GenerateOptions.ModelName, for a request's own session.
	modelName       string
	SessionID       string
	MaxSeq          int
	DeviceIDs       []string
	MaxDeviceBlocks int
	Relocate        bool
	// KeepOffHost turns off what a State does by default when its context
	// outgrows the device: hand a block to the host. A device choice that
	// leaves the CPU out is kept, and the token takes the device's refusal.
	KeepOffHost bool
	Sampling    model.Sampler
	// KVStore keeps the session's prompt prefixes, so a prompt that starts as
	// an earlier one did resumes from its cached keys and values; CacheKey
	// names what may be shared (model.State.SetCacheKey). A session with a
	// store prefills alone rather than as a row of the step loop.
	KVStore  model.KVStore
	CacheKey string

	// pooled takes a State from the model's pool when one is idle: a
	// model_id request's session, which asks for nothing a pooled State
	// could lack.
	pooled bool
	// Speculation is the default for generates that carry none.
	Speculation Speculation
	// PromptCache keeps the prompt prefixes in Config.PromptStore: the wire's
	// form of KVStore, refused when the server has no store.
	PromptCache bool
}

// Speculation is speculative decoding for a generate (model.Speculator):
// drafts verified in one pass of the model, by the model's own prediction
// block where it carries one and by prompt lookup otherwise. Greedy output is
// plain greedy decode's token for token; with prompt lookup a sampled
// generate drafts nothing. A generate that continues its session decodes
// plainly, since a Speculator starts on an empty sequence, and so does one
// whose prompt has spans; a speculative generate prefills alone, outside the
// step loop and the session's prompt store.
type Speculation struct {
	Enabled bool
	// Draft is the tokens drafted a round; zero is the engine's choice.
	Draft int
}

// check refuses the options no session can be made from, as ErrInvalid.
func (o *SessionOptions) check(store model.KVStore) error {
	switch {
	case o.MaxSeq < 0:
		return fmt.Errorf("%w: max_seq %d is negative", ErrInvalid, o.MaxSeq)
	case o.MaxDeviceBlocks < -1:
		return fmt.Errorf("%w: max_device_blocks %d: want -1 (as many as fit) or more", ErrInvalid, o.MaxDeviceBlocks)
	case o.Relocate && o.KeepOffHost:
		return fmt.Errorf("%w: relocate_while_serving and keep_off_host contradict each other", ErrInvalid)
	case o.PromptCache && o.KVStore == nil && store == nil:
		return fmt.Errorf("%w: prompt_cache needs a prompt store, and this server has none (jitllmd serve -kv-cache DIR)", ErrInvalid)
	case o.CacheKey != "" && !o.PromptCache && o.KVStore == nil:
		return fmt.Errorf("%w: cache_key names what a prompt store shares; set prompt_cache", ErrInvalid)
	}
	return nil
}

// CreateSession builds a model.State and offers its blocks to the model's
// tier. The tier's Attach gives this State its own device session, which is
// what makes two sessions on one card correct -- see tier.GPU.Attach.
func (e *Engine) CreateSession(o SessionOptions) (*Session, error) {
	if err := o.check(e.cfg.PromptStore); err != nil {
		return nil, err
	}
	lm, err := e.resolveModel(o.ModelID)
	if err != nil {
		return nil, err
	}
	o.ModelID = lm.id
	if o.PromptCache && o.KVStore == nil {
		// Through the model's memory cache when it has one, which layers
		// over the same store (ttft.go, layered).
		o.KVStore = e.cfg.PromptStore
		if lm.ttft.kv != nil {
			o.KVStore = lm.ttft.kv
		}
	}
	if lm.m.IsEncoder() {
		return nil, fmt.Errorf("%w: model %q is an encoder (%s): it has no decoder session to generate "+
			"from; embed with it instead", ErrInvalid, lm.called(o.modelName), lm.m.Cfg.Arch)
	}
	maxSeq := o.MaxSeq
	if maxSeq <= 0 {
		maxSeq = e.defaultMaxSeq(lm)
	}
	id := o.SessionID
	if id == "" {
		id = e.nextID("s")
	}

	e.mu.Lock()
	if _, ok := e.sessions[id]; ok {
		e.mu.Unlock()
		return nil, fmt.Errorf("%w: session id %q", ErrExists, id)
	}
	e.mu.Unlock()

	var st *model.State
	if o.pooled {
		st = e.takeIdle(lm)
	}
	reused := st != nil
	if reused {
		// Placed when it was built, and reset when it was pooled.
		lm.ttft.statesReused.Add(1)
	} else {
		st = lm.m.NewState(maxSeq)
		lm.ttft.statesCreated.Add(1)
	}
	if lm.dev != nil && !reused {
		max := o.MaxDeviceBlocks
		if max == 0 {
			max = lm.maxBlocks
		}
		if max == 0 {
			max = -1
		}
		if err := st.SetDeviceLayers(lm.dev, max); err != nil {
			st.Close()
			return nil, err
		}
	}
	if o.Relocate {
		st.SetRelocate(true)
	}
	if o.KeepOffHost {
		st.SetRelocate(false)
	}
	if o.KVStore != nil {
		st.SetKVStore(o.KVStore)
		if o.CacheKey != "" {
			// A refused key leaves the process-local default: the cache still
			// serves this run, just not the next one.
			st.SetCacheKey(o.CacheKey)
		}
	}
	e.applyPageBudget(lm, st)

	s := &Session{
		id:       id,
		modelID:  lm.id,
		lm:       lm,
		st:       st,
		maxSeq:   maxSeq,
		sampling: o.Sampling,
		created:  time.Now(),
		cached:   o.KVStore != nil,
		spec:     o.Speculation,
	}
	s.lastUsed.Store(time.Now().UnixMilli())
	s.mu.Lock()
	s.refresh()
	s.mu.Unlock()

	e.mu.Lock()
	// The model may have been unloaded while this State was being placed. An
	// unload closes the sessions it finds, so one registered after it would be
	// a State outliving its Model.
	if e.models[lm.id] != lm {
		e.mu.Unlock()
		st.Close()
		return nil, fmt.Errorf("%w: model %q was unloaded", ErrNotFound, lm.called(o.modelName))
	}
	if _, ok := e.sessions[id]; ok {
		e.mu.Unlock()
		st.Close()
		return nil, fmt.Errorf("%w: session id %q", ErrExists, id)
	}
	e.sessions[id] = s
	lm.sessions[id] = s
	e.mu.Unlock()
	return s, nil
}

// placementSnapshot is the session's placement as of the last time it was safe
// to read. See the snap* fields on session.
type placementSnapshot struct {
	runs        [][2]int
	declines    []model.DeviceDecline
	headOnDev   bool
	demotions   int
	relocations int
	reclaims    int
	relocating  bool
	takenAt     time.Time
	// on is the device each block is on, by block; "" for a host block or a
	// device the tier did not name.
	on []string
	// hist is each block's history cost (model.State.BlockHistory).
	hist []blockHistory
}

// blockHistory is one block's model.State.BlockHistory.
type blockHistory struct {
	perPos, state uint64
	window        int
}

// refresh republishes the snapshot. The caller MUST hold s.mu.
func (s *Session) refresh() {
	s.snapPos.Store(int32(s.st.Pos()))
	s.snapKV.Store(s.st.KVBytes())
	s.snapHist.Store(s.st.HistoryBytes())
	s.snapDevBlocks.Store(int32(s.st.GPULayers()))
	s.snapAt.Store(time.Now().UnixMilli())
	s.snapBatched.Store(s.lm.loop != nil && stepsJointly(s.st))
	blocks, settled := s.st.SeamTuned()
	s.snapSeam.Store(int32(blocks))
	s.snapSeamSettled.Store(settled)
	s.snapPlacement.Store(&placementSnapshot{
		runs:        s.st.DeviceBlocks(),
		declines:    s.st.DeviceDeclines(),
		headOnDev:   s.st.HeadOnDevice(),
		demotions:   s.st.DeviceDemotions(),
		relocations: s.st.Relocations(),
		reclaims:    s.st.Reclaims(),
		relocating:  s.st.Relocating(),
		takenAt:     time.Now(),
		on:          blocksOn(s.st, s.lm.m.Cfg.NLayer),
		hist:        historyOf(s.st, s.lm.m.Cfg.NLayer),
	})
}

// blocksOn is the device each block of st is on, "" for the host.
func blocksOn(st *model.State, n int) []string {
	on := make([]string, n)
	for li := range on {
		on[li], _ = st.DeviceOf(li)
	}
	return on
}

// historyOf is what each block of st's history costs.
func historyOf(st *model.State, n int) []blockHistory {
	h := make([]blockHistory, n)
	for li := range h {
		h[li].perPos, h[li].window, h[li].state = st.BlockHistory(li)
	}
	return h
}

// placement returns the last published snapshot, never nil.
func (s *Session) placement() *placementSnapshot {
	if p := s.snapPlacement.Load(); p != nil {
		return p
	}
	return &placementSnapshot{}
}

// Session returns the open session with this id, or ErrNotFound.
func (e *Engine) Session(id string) (*Session, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	s, ok := e.sessions[id]
	if !ok {
		return nil, fmt.Errorf("%w: session %q", ErrNotFound, id)
	}
	return s, nil
}

// Sessions returns every open session, sorted by id.
func (e *Engine) Sessions() []*Session {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]*Session, 0, len(e.sessions))
	for _, s := range e.sessions {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

// CloseSession removes the session from the Engine and its model, cancels a
// generate in flight on it and closes its State once that generate has let
// go. An unknown id is ErrNotFound.
func (e *Engine) CloseSession(id string) error {
	e.mu.Lock()
	s, ok := e.sessions[id]
	if !ok {
		e.mu.Unlock()
		return fmt.Errorf("%w: session %q", ErrNotFound, id)
	}
	delete(e.sessions, id)
	delete(s.lm.sessions, id)
	e.mu.Unlock()
	s.close()
	// Its history is given back: the model's weights may take the bytes.
	e.applyPageBudget(s.lm, nil)
	return nil
}

func (s *Session) close() {
	if s.closed.Swap(true) {
		return
	}
	s.cancelGeneration()
	s.mu.Lock()
	defer s.mu.Unlock()
	// A move waiting for a token boundary will not get one; the State's Close
	// gives back what it prepared.
	s.moveMu.Lock()
	r := s.move
	s.move = nil
	s.moveMu.Unlock()
	if r != nil {
		r.done <- moveRes{err: fmt.Errorf("%w: session %q closed", ErrNotFound, s.id)}
	}
	if s.parkView != nil {
		// A session closed while parked never faults its pages back: they
		// stop being pinned, and the memory cache evicts them in turn.
		s.parkView.Release()
		s.parkView = nil
	}
	s.st.Close()
}

func (s *Session) cancelGeneration() bool {
	s.cancelMu.Lock()
	defer s.cancelMu.Unlock()
	if s.cancel == nil {
		return false
	}
	s.cancel()
	return true
}

// ---------------------------------------------------------------- generate

// PromptKind says which member of Prompt is meant. An explicit tag beats
// guessing from which field is non-empty: a caller may legitimately ask to
// generate from an EMPTY continuation of an existing session.
type PromptKind int

// The prompt kinds: none (a continuation of the session's own sequence),
// text to encode, token ids as given, and chat messages to render through the
// model's template.
const (
	PromptNone PromptKind = iota
	PromptText
	PromptIDs
	PromptChat
	// PromptSpans is token and embedding spans, a chat turn with a picture in
	// its history; it runs alone, never as a row of the step loop.
	PromptSpans
)

// ChatInput is a chat prompt: the messages, an optional system prompt given
// apart from them (HasSystem says it was), and what the template is asked to
// render.
type ChatInput struct {
	Messages            []model.ChatMessage
	System              string
	HasSystem           bool
	AddGenerationPrompt bool
	TemplateName        string
	// Images is the conversation's pictures in the order the client sent
	// them; Messages[i].Images says how many are message i's. The engine lays
	// them out where the model's chat template puts its image markers
	// (Engine.pictureSpans), so a chat with pictures runs as PromptSpans.
	Images []image.Image
	// Tools is the tool list the template renders, a JSON array in the
	// OpenAI shape ([{"type": "function", "function": {...}}]); nil for none.
	Tools []byte
	// ToolChoice is what the reply may do with Tools. One that forces a
	// call (required, or a named tool) holds the reply to the model's own
	// tool-call syntax around the tool's schema (tools.go); every choice is
	// handed to a template that reads tool_choice.
	ToolChoice model.ToolChoice
}

// Prompt is what a generate runs: the member Kind names is the one read. A
// chat with pictures, laid out as PromptSpans, keeps its Chat beside the
// spans for its tools.
type Prompt struct {
	Kind  PromptKind
	Text  string
	IDs   []int32
	Chat  *ChatInput
	Spans []model.Span
}

// GenerateOptions is the one request shape: the Connect InferenceService and
// both HTTP shims build one, so there is no second generate path.
type GenerateOptions struct {
	SessionID string
	ModelID   string
	// ModelName is the name the caller addressed the model by (BindTarget
	// sets it), which errors repeat: a client that sent a file name cannot
	// act on an id it never saw. Empty names the model by ModelID.
	ModelName string

	Prompt       Prompt
	MaxTokens    int
	Sampling     *model.Sampler
	Stop         []string
	Continue     bool
	Echo         bool
	QueueTimeout time.Duration
	// IgnoreEOS keeps generating past an end-of-generation token, up to
	// MaxTokens: the token is emitted and fed like any other (vLLM's
	// ignore_eos). Stop strings still end the generate.
	IgnoreEOS bool
	// CachePrompt is a request that marked a cache breakpoint (Anthropic's
	// cache_control). Every generate that starts its sequence goes through
	// the memory cache already; a marked one also reports the positions it
	// computed as written to the cache.
	CachePrompt bool
	// Priority is a request's claim on the host budget (prioritise): ""
	// leaves it as it is.
	Priority string
	// Speculation overrides the session's; nil takes it.
	Speculation *Speculation
	// Grammar constrains the output to GBNF text (grammar.go): every
	// sampled token keeps it inside the grammar, and the reply ends where the
	// grammar does. A constrained generate decodes alone, plainly: not as a
	// row of the step loop, and not speculatively.
	Grammar string
	// maskSkip, set by a gate, leaves the grammar's mask off at that step
	// (counting from 1) and the output free after it: the break the
	// structured-output gates must see.
	maskSkip int
	// Logprobs puts each sampled token's log-probability on its Token event,
	// with the TopLogprobs (0..model.MaxTopLogprobs) most likely alternatives.
	// They are the model's raw distribution, before temperature and penalties
	// (model.Logprobs).
	Logprobs    bool
	TopLogprobs int
	// Seeds, when it holds more than one, asks for that many continuations
	// of one prompt (OpenAI's n): choice i samples with Seeds[i] and its
	// events carry Event.Choice = i. The prompt is prefilled once; every
	// other choice restores its pages and its logits (generateN).
	Seeds []int64
}

// EventKind discriminates Event.
type EventKind int

// The event kinds, in the order a generate emits them: one Started, a Token
// per sampled token, one Finished.
const (
	EventStarted EventKind = iota
	EventToken
	EventFinished
)

// Started is a generate's first event, sent once the prompt is prefilled:
// how long the request queued and behind how many, where the model's blocks
// ran, and how long the prefill took.
type Started struct {
	SessionID    string
	ModelID      string
	Ephemeral    bool
	PromptTokens int
	QueuedFor    time.Duration
	QueueDepth   int32
	DeviceBlocks int
	HostBlocks   int
	DeviceIDs    []string
	Prefill      time.Duration
	Execution    ExecutionMode
	// Batched is set when the generate decodes as a row of its model's step
	// loop (batch.go) rather than alone on its gates.
	Batched bool
	// Restored is how many of the prompt's positions came out of a prompt
	// store rather than being computed: all of them for every choice of an
	// n > 1 request but the first.
	Restored int
	// Tools reads this choice's tool calls from its tokens, in the model's
	// own syntax, when the request declared tools; nil otherwise. A shim
	// feeds it every Token's ID (tools.go).
	Tools *model.ToolStream
}

// Token is one step of the output. Every sampled token is sent, its Text
// empty while held back for a stop string or a partial rune; ID -1 carries
// text alone (the echoed prompt at Index -1, or the held-back tail).
type Token struct {
	ID    int32
	Text  string
	Index int
	// Logprob is set on a sampled token when the request asked for logprobs.
	Logprob *TokenLogprob
}

// TokenLogprob is a sampled token's log-probability and the most likely
// alternatives at its step, most likely first. Text is the token alone,
// decoded on its own: its bytes, which may be part of a rune.
type TokenLogprob struct {
	Text    string
	Logprob float32
	Top     []TopToken
}

// TopToken is one alternative at a step.
type TopToken struct {
	ID      int32
	Text    string
	Logprob float32
}

// newLogprobs is a generate's logprobs working set, nil when it asked for none.
func newLogprobs(o GenerateOptions) *model.Logprobs {
	if !o.Logprobs {
		return nil
	}
	return &model.Logprobs{N: o.TopLogprobs}
}

// tokenDecoder is what takeLogprob reads a token's text through.
type tokenDecoder interface {
	Decode(ids []int32) string
}

// takeLogprob is the event's copy of one step's logprobs: the working set is
// reused next step, so the event owns its own slice.
func takeLogprob(l *model.Logprobs, v tokenDecoder, logits []float32, next int32) *TokenLogprob {
	if l == nil {
		return nil
	}
	out := &TokenLogprob{Text: v.Decode([]int32{next}), Logprob: l.Take(logits, next)}
	out.Top = make([]TopToken, len(l.Top))
	for i, t := range l.Top {
		out.Top[i] = TopToken{ID: t.ID, Text: v.Decode([]int32{t.ID}), Logprob: t.Logprob}
	}
	return out
}

// FinishReason mirrors the proto enum.
type FinishReason int

// The reasons a generate ends: a stop string matched, MaxTokens reached, an
// end-of-generation token sampled, the context cancelled, or an error.
const (
	FinishUnspecified FinishReason = iota
	FinishStop
	FinishMaxTokens
	FinishEOS
	FinishCancelled
	FinishError
)

// Finished is a generate's last event: why it ended, the token counts, the
// prefill and decode times, the decode rate with the bytes a token reads (so
// the rate can be checked against the read wall), and the session's position.
type Finished struct {
	Reason           FinishReason
	StopMatched      string
	PromptTokens     int
	CompletionTokens int
	Prefill          time.Duration
	Decode           time.Duration
	TokensPerSecond  float64
	BytesPerToken    uint64
	Position         int
	// Restored is how many of PromptTokens came from the memory cache or a prompt store.
	Restored int
}

// Event is one message of a generate's stream: Kind says which of Started,
// Token and Finished is set.
type Event struct {
	// Choice is the continuation an event belongs to, 0 unless the request
	// asked for several (GenerateOptions.Seeds).
	Choice   int
	Kind     EventKind
	Started  *Started
	Token    *Token
	Finished *Finished
}

// tokenLimit is a request's completion length: what it asked for, never more
// than room, the positions left in the session's context. A request that asks
// for none runs until the model ends its reply or fills the context, as
// OpenAI's API reads an absent max_tokens. Either way a reply that reaches the
// end of the context finishes as FinishMaxTokens, not on a full KV cache.
func tokenLimit(asked, room int) int {
	room = max(room, 0)
	if asked <= 0 || asked > room {
		return room
	}
	return asked
}

// Generate is the one generate path. The first event reports how long the
// request queued -- for a gate, or for a row of its model's step loop -- and
// how many were ahead, so a caller does not mistake a queue for a slow model.
func (e *Engine) Generate(ctx context.Context, o GenerateOptions, emit func(Event) error) (err error) {
	// Two admissions, in this order: the model's request queue here, first,
	// so a request it refuses builds nothing and parks nobody; then, once the
	// prompt is known, the KV budget (admitKV), which may park idle sessions
	// to make room for this one.
	//
	// A model named rather than a session is resolved first: by id, file
	// name or alias, loaded on demand when it is in the models folder.
	if o.SessionID == "" && o.ModelID != "" {
		lm, err := e.resolveModel(o.ModelID)
		if err != nil {
			return err
		}
		if o.ModelName == "" && lm.id != o.ModelID {
			o.ModelName = o.ModelID
		}
		o.ModelID = lm.id
	}
	if lm := e.modelOf(o); lm != nil {
		done, err := e.admitQueue(lm)
		if err != nil {
			return err
		}
		defer done()
	}
	if len(o.Seeds) > 1 {
		return e.generateN(ctx, o, emit)
	}
	if len(o.Seeds) == 1 {
		if o.Sampling == nil {
			return fmt.Errorf("%w: a seed with no sampler", ErrInvalid)
		}
		sm := *o.Sampling
		sm.Seed = o.Seeds[0]
		o.Sampling = &sm
	}
	s, ephemeral, err := e.resolveSession(o)
	if err != nil {
		return err
	}
	if ephemeral {
		defer e.releaseEphemeral(s)
	}
	if o.Priority != "" {
		if err := e.prioritise(s.lm, o.Priority); err != nil {
			return err
		}
	}

	// One sequence, one generate. A second concurrent call on the same session
	// is a caller error and blocking here is the kind answer to it.
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed.Load() {
		return fmt.Errorf("%w: session %q is closed", ErrNotFound, s.id)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	s.cancelMu.Lock()
	s.cancel = cancel
	s.cancelMu.Unlock()
	defer func() {
		s.cancelMu.Lock()
		s.cancel = nil
		s.cancelMu.Unlock()
	}()

	lm := s.lm
	if lm.m.Vocab == nil {
		return fmt.Errorf("server: model %q has no tokenizer: %v", lm.called(o.ModelName), lm.m.TokErr)
	}

	if o.Prompt.Kind == PromptChat && o.Prompt.Chat != nil && len(o.Prompt.Chat.Images) > 0 {
		sp, err := pictureSpans(s, o)
		if err != nil {
			return err
		}
		// Chat stays beside the spans: its tools are what a forced call's
		// grammar and the reply's call parsing read.
		o.Prompt = Prompt{Kind: PromptSpans, Spans: sp, Chat: o.Prompt.Chat}
	}
	if o.Prompt.Kind == PromptSpans {
		if len(o.Prompt.Spans) == 0 {
			return fmt.Errorf("%w: the prompt is empty", ErrInvalid)
		}
	}
	ids, err := e.encode(lm, o.Prompt, o.ModelName)
	if err != nil {
		return err
	}
	spans := o.Prompt.Kind == PromptSpans
	if !spans && len(ids) == 0 && !o.Continue {
		return fmt.Errorf("%w: the prompt is empty and continue_session was not set", ErrInvalid)
	}
	if !spans && len(ids) == 0 {
		return fmt.Errorf("%w: continue_session with no prompt has no token to run", ErrInvalid)
	}

	// A request whose client has already gone does no prompt work and sends
	// no GenerateStarted, on whichever path would have run it.
	if err := ctx.Err(); err != nil {
		return context.Cause(ctx)
	}

	// Room for this generate's history, parking idle sessions of the model
	// if its KV budget is short (preempt.go), and this session back if it
	// was the one parked.
	// tail is the prompt's last token ids, which a forced tool call's grammar
	// reads; npos is the positions the prompt takes, a picture's rows included.
	tail, npos := ids, len(ids)
	if spans {
		npos = model.SpanPositions(o.Prompt.Spans, lm.m.Cfg.NEmbd)
		tail = nil
		for i := len(o.Prompt.Spans) - 1; i >= 0 && tail == nil; i-- {
			tail = o.Prompt.Spans[i].Tokens
		}
	}
	if err := e.admitKV(s, npos+tokenLimit(o.MaxTokens, s.st.MaxSeq()-s.st.Pos()-npos)); err != nil {
		return err
	}
	spec := s.spec
	if o.Speculation != nil {
		spec = *o.Speculation
	}
	speculate := spec.Enabled && !spans && !o.Continue
	if speculate && o.Logprobs {
		return fmt.Errorf("%w: speculation with logprobs is not built: a round decides several tokens "+
			"from one verification and reads back no distribution for each", ErrInvalid)
	}
	var con *constraint
	if tc, err := lm.toolConstraint(o, tail); err != nil {
		return err
	} else if tc != nil {
		if o.IgnoreEOS {
			return fmt.Errorf("%w: a forced tool call ends its reply with an end-of-generation token, "+
				"and ignore_eos would run past it", ErrInvalid)
		}
		con = tc
		speculate = false
	} else if o.Grammar != "" {
		if o.IgnoreEOS {
			return fmt.Errorf("%w: a grammar ends its reply with an end-of-generation token, and ignore_eos "+
				"would run past it", ErrInvalid)
		}
		m, err := lm.matcher(o.Grammar)
		if err != nil {
			return err
		}
		con = &constraint{m: m, st: m.Start()}
		speculate = false
	}
	if o.Continue && s.specRan {
		return fmt.Errorf("%w: session %q stopped inside a speculative round, and its model ran past the "+
			"reply; it cannot be continued -- generate without continue_session", ErrInvalid, s.id)
	}

	// The model's memory cache, for a generate that starts its sequence
	// (ttft.go). The State is reset here, before the store is attached or
	// taken away: both want position 0. A speculative generate prefills
	// through its Speculator, which neither restores nor seals, so it runs
	// without the store rather than leave pages named by a prompt it did not
	// record.
	store := e.useStore(s, o) && !speculate
	if !o.Continue {
		s.st.Reset()
		if err := s.attachStore(store); err != nil {
			return err
		}
	}

	if lp := e.joins(s); lp != nil && !spans && !s.cached && !speculate && con == nil {
		return e.generateBatched(ctx, lp, s, o, ids, ephemeral, store, emit)
	}

	// ---- recorded as running on its devices; nothing waits here.
	gs := e.gatesFor(lm.gateIDs())
	var waited time.Duration
	depth := gs.acquire(s.id)
	defer gs.release()

	s.running.Store(true)
	defer s.running.Store(false)
	s.lastUsed.Store(time.Now().UnixMilli())

	// ---- prefill. A sequence that starts here stops between chunks when its
	// client goes; one that continues keeps its history whole, so it does not.
	if !o.Continue {
		s.st.SetPrefillInterrupt(func() bool { return ctx.Err() != nil })
		defer s.st.SetPrefillInterrupt(nil)
		s.specRan = false
	}
	sampler := s.sampling
	if o.Sampling != nil {
		sampler = *o.Sampling
	}
	prefillStart := time.Now()
	var logits []float32
	var sp *model.Speculator
	var first int32
	prompted := len(ids)
	restored := 0
	switch {
	case speculate:
		opts := []model.SpecOption{model.WithSpecDraft(spec.Draft)}
		if !o.IgnoreEOS {
			opts = append(opts, model.WithSpecStop(lm.m.Vocab.IsEOG))
		}
		if sp, err = s.st.Speculate(opts...); err != nil {
			return fmt.Errorf("%w: %v", ErrInvalid, err)
		}
		defer sp.Close()
		first, err = sp.Start(ids, &sampler)
	case spans:
		prompted = model.SpanPositions(o.Prompt.Spans, lm.m.Cfg.NEmbd)
		if o.Continue {
			// A continued session's history holds its earlier pictures as
			// rows already; the new spans follow them.
			logits, err = s.st.PrefillMixed(o.Prompt.Spans...)
		} else {
			logits, err = s.st.PrefillCachedMixed(o.Prompt.Spans...)
		}
		restored = s.st.KVRestored()
	case (s.cached || store) && !o.Continue:
		logits, err = s.st.PrefillCached(ids)
		restored = s.st.KVRestored()
	default:
		logits, err = s.st.Prefill(ids)
	}
	if errors.Is(err, model.ErrPrefillInterrupted) {
		// Part of a prompt nobody is waiting for: give the sequence back
		// whole rather than leave it half-prefilled.
		s.st.Reset()
		return context.Cause(ctx)
	}
	if err != nil {
		return err
	}
	lm.ttft.restored.Add(int64(restored))
	lm.ttft.computed.Add(int64(prompted - restored))
	s.prefilled.Add(int64(prompted))
	lm.tokensPrefilled.Add(int64(prompted))
	prefill := time.Since(prefillStart)

	devBlocks := s.st.GPULayers()
	if err := emit(Event{Kind: EventStarted, Started: &Started{
		SessionID:    s.id,
		ModelID:      lm.id,
		Ephemeral:    ephemeral,
		PromptTokens: prompted,
		QueuedFor:    waited,
		QueueDepth:   depth,
		DeviceBlocks: devBlocks,
		HostBlocks:   lm.m.Cfg.NLayer - devBlocks,
		DeviceIDs:    lm.deviceIDs,
		Prefill:      prefill,
		Execution:    ExecutionParallel,
		Restored:     restored,
		Tools:        lm.toolStream(o),
	}}); err != nil {
		return err
	}

	if o.Echo && !spans {
		if err := emit(Event{Kind: EventToken, Token: &Token{
			ID: -1, Text: lm.m.Vocab.Decode(ids), Index: -1,
		}}); err != nil {
			return err
		}
	}

	// ---- decode.
	maxTokens := tokenLimit(o.MaxTokens, s.st.MaxSeq()-s.st.Pos())

	st := newStreamText(lm.m.Vocab.NewChatStream().Next, o.Stop)
	reason := FinishMaxTokens
	stopMatched := ""
	out := make([]int32, 0, min(maxTokens, 4096))
	decodeStart := time.Now()
	n := 0
	// pend is what a speculative round decided and the reply has not taken
	// yet; the first is Start's.
	var pend []int32
	if sp != nil {
		pend = []int32{first}
	}

	lpw := newLogprobs(o)
	// pending is the token ForwardSample already drew, -1 when the next one
	// is drawn from logits. Only a reply that wants no row after the prompt
	// takes it -- no grammar mask and no logprobs, both of which read the
	// host's logits -- so a device holding the head selects the candidates
	// and only they come home.
	pending := int32(-1)
	for ; n < maxTokens; n++ {
		if ctx.Err() != nil {
			reason = FinishCancelled
			break
		}
		var next int32
		var tlp *TokenLogprob
		switch {
		case sp == nil && pending >= 0:
			next = pending
			sampler.Observe(next)
		case sp == nil:
			lg := logits
			if con != nil && n+1 != o.maskSkip {
				lg = con.mask(logits)
			}
			next = sampler.Sample(lg)
			sampler.Observe(next)
			// The model's raw distribution, before the grammar's mask, as
			// before temperature and penalties: vLLM's default
			// (raw_logprobs).
			tlp = takeLogprob(lpw, lm.m.Vocab, logits, next)
			if con != nil && !con.accept(next) {
				if n+1 != o.maskSkip {
					return fmt.Errorf("server: token %d is outside the grammar its mask allowed", next)
				}
				con = nil // the gate's break: the reply leaves its grammar
			}
		case len(pend) == 0:
			// Every token a round returns is observed into the sampler.
			sp.Limit(maxTokens - n)
			if pend, err = sp.Next(&sampler); err != nil {
				return err
			}
		}
		if sp != nil {
			next, pend = pend[0], pend[1:]
		}
		if !o.IgnoreEOS && lm.m.Vocab.IsEOG(next) {
			reason = FinishEOS
			break
		}
		out = append(out, next)

		// Every token is sent, its text empty while held back (a partial rune,
		// or a possible stop string's start): a client counting ids must see
		// all of them, as GenerateToken says.
		chunk, hit, match := st.push(out)
		if err := emit(Event{Kind: EventToken, Token: &Token{ID: next, Text: chunk, Index: n, Logprob: tlp}}); err != nil {
			return err
		}
		if hit {
			reason, stopMatched = FinishStop, match
			n++
			break
		}
		if sp != nil {
			continue
		}
		// A move between devices asked meanwhile is prepared and committed
		// here, between two tokens (Engine.MoveBlocks).
		s.serveMove(false)
		if con == nil && !o.Logprobs {
			if pending, err = s.st.ForwardSample(next, &sampler); err != nil {
				return err
			}
			continue
		}
		pending = -1
		if logits, err = s.st.Forward(next); err != nil {
			return err
		}
	}
	// A round's last token is decided and not yet run, as plain decode's
	// last sample is; a reply that ended before the round's end leaves rows
	// in the model that it did not keep.
	s.specRan = len(pend) > 0
	if reason == FinishMaxTokens || reason == FinishEOS || reason == FinishCancelled {
		// Flush whatever was being held back for a stop string that never came.
		if tail := st.flush(); tail != "" {
			if err := emit(Event{Kind: EventToken, Token: &Token{ID: -1, Text: tail, Index: n}}); err != nil {
				return err
			}
		}
	}
	decode := time.Since(decodeStart)
	s.generated.Add(int64(n))
	lm.finished(n, prefill, decode)
	s.lastUsed.Store(time.Now().UnixMilli())
	s.refresh()
	// The history grew by what this generate committed.
	e.applyPageBudget(lm, nil)

	rate := 0.0
	if decode > 0 && n > 0 {
		rate = float64(n) / decode.Seconds()
	}
	return emit(Event{Kind: EventFinished, Finished: &Finished{
		Reason:           reason,
		StopMatched:      stopMatched,
		PromptTokens:     prompted,
		CompletionTokens: n,
		Prefill:          prefill,
		Decode:           decode,
		TokensPerSecond:  rate,
		// Bytes per token makes the rate checkable against the read wall.
		BytesPerToken: lm.m.BytesPerToken(),
		Position:      s.st.Pos(),
		Restored:      restored,
	}})
}

// generateN is Generate for several continuations of one prompt (OpenAI's
// n). The prompt is prefilled once: every choice runs on a fresh session of
// the model sharing one request-scoped prompt store under one namespace, so
// the first choice's prefill seals the prompt's pages and its final logits
// into it and every later choice restores them (State.PrefillCached) and
// computes no prompt position -- the prefix cache's own path, which is how
// the engine forks a sequence. Started.Restored reports it per choice.
//
// The choices run one after another, each alone on the model's gates: a
// State owns a pool, and n of them at once on the same cores is the worst
// configuration the engine has. Each samples with its own seed and its events
// carry its index.
func (e *Engine) generateN(ctx context.Context, o GenerateOptions, emit func(Event) error) error {
	if o.SessionID != "" || o.Continue {
		return fmt.Errorf("%w: several choices run on fresh sessions of a model; "+
			"a named session or continue_session takes one", ErrInvalid)
	}
	if o.Speculation != nil && o.Speculation.Enabled {
		return fmt.Errorf("%w: speculation with several choices is not built: the choices prefill "+
			"through a shared prompt store, which a Speculator does not use", ErrInvalid)
	}
	lm, err := e.Model(o.ModelID)
	if err != nil {
		return err
	}
	// The model's memory cache when it has one: the first choice's prompt
	// then also serves later requests, and its pages count against the
	// model's budget. A request-scoped store otherwise, as before. The
	// namespace is the model's, so a choice restores what an earlier request
	// left as well as what the first choice sealed; a page the bounded cache
	// evicts between choices is computed again.
	var store model.KVStore = model.NewMemStore()
	ns := "n/" + lm.id + "/" + e.nextID("fork")
	if lm.ttft.kv != nil {
		store, ns = lm.ttft.kv, lm.ttft.ns
	}
	for i, seed := range o.Seeds {
		if err := ctx.Err(); err != nil {
			return err
		}
		s, err := e.CreateSession(SessionOptions{ModelID: o.ModelID, KVStore: store, CacheKey: ns})
		if err != nil {
			return err
		}
		oi := o
		oi.Seeds = nil
		oi.ModelID, oi.SessionID = "", s.id
		sm := s.sampling
		if o.Sampling != nil {
			sm = *o.Sampling
		}
		sm.Seed = seed
		oi.Sampling = &sm
		err = e.Generate(ctx, oi, func(ev Event) error {
			ev.Choice = i
			if ev.Started != nil {
				st := *ev.Started
				st.Ephemeral = true
				ev.Started = &st
			}
			return emit(ev)
		})
		e.CloseSession(s.id)
		if err != nil {
			return err
		}
	}
	return nil
}

// resolveSession returns the named session, or creates an ephemeral one for a
// model_id request.
func (e *Engine) resolveSession(o GenerateOptions) (*Session, bool, error) {
	if o.SessionID != "" && o.ModelID != "" {
		return nil, false, fmt.Errorf("%w: give session_id or model_id, not both", ErrInvalid)
	}
	if o.SessionID != "" {
		s, err := e.Session(o.SessionID)
		return s, false, err
	}
	if o.ModelID == "" {
		return nil, false, fmt.Errorf("%w: one of session_id or model_id is required", ErrInvalid)
	}
	s, err := e.CreateSession(SessionOptions{ModelID: o.ModelID, modelName: o.ModelName, pooled: true})
	if err != nil {
		return nil, false, err
	}
	return s, true, nil
}

// encode turns a Prompt into token ids. A rendered chat template is a
// complete prompt and must be encoded without added specials, or the first
// turn gets a doubled BOS; model.ChatIDs owns that pairing, so chat goes
// through it rather than ChatPrompt-then-Encode.
func (e *Engine) encode(lm *LoadedModel, p Prompt, name string) ([]int32, error) {
	switch p.Kind {
	case PromptIDs:
		return p.IDs, nil
	case PromptText:
		return lm.m.Vocab.Encode(p.Text, true), nil
	case PromptChat:
		if p.Chat == nil {
			return nil, fmt.Errorf("%w: chat prompt with no messages", ErrInvalid)
		}
		if !lm.m.ChatCapable() {
			return nil, fmt.Errorf("%w: model %q carries no chat template, so a chat request "+
				"would run as a RAW COMPLETION -- which produces fluent output and no signal at all; "+
				"send a text prompt instead", ErrInvalid, lm.called(name))
		}
		msgs := p.Chat.Messages
		if p.Chat.HasSystem {
			// Anthropic puts the system prompt in a top-level field; the
			// template takes a message list, so it is prepended.
			msgs = append([]model.ChatMessage{{Role: "system", Content: p.Chat.System}}, msgs...)
		}
		return lm.m.ChatIDsToolChoice(msgs, p.Chat.Tools, p.Chat.ToolChoice, p.Chat.AddGenerationPrompt)
	case PromptNone, PromptSpans:
		return nil, nil
	}
	return nil, fmt.Errorf("%w: unknown prompt kind %d", ErrInvalid, p.Kind)
}

// acceptsImages refuses a picture for a model with no vision tower by name.
func acceptsImages(lm *LoadedModel) error {
	if lm.m.Tower() == nil {
		return fmt.Errorf("%w: model %q does not accept images: its container carries no vision tower",
			ErrInvalid, lm.id)
	}
	return nil
}

// pictureSpans lays a chat with pictures out as spans on s's State: the
// template's text, and each picture where the template puts its image markers.
// A model with no vision tower is refused by name, never run with the picture
// left out. Such a prompt runs alone rather than as a row of the step loop,
// as every span prompt does.
func pictureSpans(s *Session, o GenerateOptions) ([]model.Span, error) {
	lm, chat := s.lm, o.Prompt.Chat
	if err := acceptsImages(lm); err != nil {
		return nil, err
	}
	if !lm.m.ChatCapable() {
		return nil, fmt.Errorf("%w: model %q carries no chat template, which is where a picture's "+
			"markers go", ErrInvalid, lm.id)
	}
	n := 0
	for _, m := range chat.Messages {
		n += m.Images
	}
	if n != len(chat.Images) {
		return nil, fmt.Errorf("%w: the messages count %d picture(s) and the request carries %d",
			ErrInvalid, n, len(chat.Images))
	}
	msgs := chat.Messages
	if chat.HasSystem {
		msgs = append([]model.ChatMessage{{Role: "system", Content: chat.System}}, msgs...)
	}
	sp, err := s.st.ChatPictureSpans(msgs, chat.Images, chat.Tools, chat.ToolChoice, chat.AddGenerationPrompt)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	return sp, nil
}

// ---------------------------------------------------------------- streaming text

// streamText turns a growing token id list into incremental text, and applies
// stop strings.
//
// Each new id is decoded once, through tok.ChatStream, whose pieces
// concatenate to DecodeChat of the whole list at every prefix -- so the text
// is what re-decoding the whole list would give, at a cost per token that does
// not grow with the completion. DecodeChat, not Decode, so a harmony vocabulary
// does not leave role and channel headers glued to the text.
//
// The tail is held back so the start of a stop string never reaches the
// client before the match completes; the window is one byte short of the
// longest stop string.
type streamText struct {
	next    func(id int32) string
	stops   []string
	hold    int
	n       int // ids decoded so far
	decoded []byte
	emitted int
}

// newStreamText streams through next, one id's text at a time: a
// tok.ChatStream's Next for a model's vocabulary.
func newStreamText(next func(id int32) string, stops []string) *streamText {
	hold := 0
	for _, s := range stops {
		if len(s) > hold {
			hold = len(s)
		}
	}
	if hold > 0 {
		hold--
	}
	return &streamText{next: next, stops: stops, hold: hold}
}

// push takes the full id list so far and returns the text safe to emit now,
// whether a stop string matched, and which one. Only the ids past the last
// push are decoded.
func (t *streamText) push(ids []int32) (chunk string, stopped bool, match string) {
	before := len(t.decoded)
	for _, id := range ids[t.n:] {
		t.decoded = append(t.decoded, t.next(id)...)
	}
	t.n = len(ids)

	// A stop string is matched against the whole completion, because a stop
	// may straddle any number of tokens. The text before this push held no
	// match, so a new one ends in the new bytes and starts no earlier than
	// len(s)-1 before them: only that window is searched.
	for _, s := range t.stops {
		if s == "" {
			continue
		}
		from := max(0, before-len(s)+1)
		if i := bytes.Index(t.decoded[from:], []byte(s)); i >= 0 {
			i += from
			// Emit up to the stop and no further. The stop string itself is
			// never part of the completion.
			if i > t.emitted {
				chunk = string(t.decoded[t.emitted:i])
			}
			t.emitted = len(t.decoded)
			return truncPartialRune(chunk), true, s
		}
	}

	safe := len(t.decoded) - t.hold
	if safe <= t.emitted {
		return "", false, ""
	}
	chunk = truncPartialRune(string(t.decoded[t.emitted:safe]))
	t.emitted += len(chunk)
	return chunk, false, ""
}

// flush releases the held-back tail once no stop string can still match it.
func (t *streamText) flush() string {
	if len(t.decoded) <= t.emitted {
		return ""
	}
	chunk := string(t.decoded[t.emitted:])
	t.emitted = len(t.decoded)
	return chunk
}

// truncPartialRune drops a trailing incomplete UTF-8 sequence so a multi-byte
// rune is never split across two writes.
func truncPartialRune(s string) string {
	return s[:tok.Complete(s)]
}

// ---------------------------------------------------------------- scanning

// ModelFileInfo is one entry ListModels reports from disk.
type ModelFileInfo struct {
	Path           string
	Name           string
	Size           uint64
	IsContainer    bool
	ConvertCommand string
	// Aliases and Disabled are the file's entry in the folder's state file
	// (library.go); ModelID is the id it is loaded under, when it is.
	Aliases  []string
	Disabled bool
	ModelID  string
}

// ScanModels lists containers and GGUFs in a directory. A GGUF is listed WITH
// the command that would convert it, because the engine reads one weight
// format and a caller who cannot see the other file has no way to learn why
// their model is missing.
func (e *Engine) ScanModels(dir string) ([]ModelFileInfo, string, error) {
	if dir == "" {
		dir = e.ModelDir()
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, dir, err
	}
	var out []ModelFileInfo
	for _, ent := range ents {
		if ent.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(ent.Name()))
		if ext != model.Ext && ext != ".gguf" {
			continue
		}
		fi, err := ent.Info()
		if err != nil {
			continue
		}
		p := filepath.Join(dir, ent.Name())
		m := ModelFileInfo{
			Path:        p,
			Name:        strings.TrimSuffix(ent.Name(), filepath.Ext(ent.Name())),
			Size:        uint64(fi.Size()),
			IsContainer: ext == model.Ext,
		}
		if !m.IsContainer {
			m.ConvertCommand = fmt.Sprintf("jitllm convert %s %s",
				p, strings.TrimSuffix(p, filepath.Ext(p))+model.Ext)
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	e.applyState(dir, out)
	return out, dir, nil
}

// ---------------------------------------------------------------- accessors
//
// The engine's own fields stay unexported so a caller cannot reach past the
// gates into a model.State. These are the read-only handles the command and
// the handlers need.

// ID is the stable id this model was loaded under.
func (lm *LoadedModel) ID() string { return lm.id }

// called is how an error names lm: as the request addressed it (name, its id
// or its file name), or by its id when the request named none.
func (lm *LoadedModel) called(name string) string {
	if name != "" {
		return name
	}
	return lm.id
}

// Name is the container's file name without its extension. The container
// carries no model name, so this is not an identity.
func (lm *LoadedModel) Name() string { return lm.name }

// Path is the container this model was opened from.
func (lm *LoadedModel) Path() string { return lm.path }

// DeviceIDs is the tier this model's blocks are offered to. Empty means the
// model is host-only and has no tier at all.
func (lm *LoadedModel) DeviceIDs() []string {
	return append([]string(nil), lm.deviceIDs...)
}

// ID is the stable id this session was created under.
func (s *Session) ID() string { return s.id }

// ModelID is the model this session holds a sequence of.
func (s *Session) ModelID() string { return s.modelID }

// Model is the loaded model's engine handle, for an in-process front end that
// renders prompts and reads the model's configuration and counters.
func (lm *LoadedModel) Model() *model.Model { return lm.m }

// GPU is the model's device tier, nil for a host-only model.
func (lm *LoadedModel) GPU() *tier.GPU { return lm.gpu }

// Inspect runs f with the session's State under the session's lock, so it
// never overlaps a generate: what a front end reads between turns (placement,
// the prompt cache's counters) or does on the session's own JIT (a picture).
func (s *Session) Inspect(f func(st *model.State)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f(s.st)
}

// SetDeviceBlocks moves the seam: it asks the devices to hold n blocks of this
// session and returns how many they took. Growing is best effort, and a shrink
// whose history cannot move leaves the seam where it was.
func (e *Engine) SetDeviceBlocks(s *Session, n int) int {
	s.mu.Lock()
	got := s.st.SetGPULayers(n)
	s.refresh()
	s.mu.Unlock()
	e.applyPageBudget(s.lm, nil)
	return got
}

// onDevices reports that every block of first..last is on a device, as the
// last snapshot has it: asked without the session's lock, so a generate in
// flight does not hold the answer back.
func (s *Session) onDevices(first, last int) bool {
	on := make([]bool, last+1)
	for _, r := range s.placement().runs {
		for li := r[0]; li < r[1] && li <= last; li++ {
			on[li] = true
		}
	}
	for li := first; li <= last; li++ {
		if !on[li] {
			return false
		}
	}
	return true
}

// moveReq is a move between devices handed to whoever holds the session
// next: its generate, between two tokens, or MoveBlocks itself when idle.
type moveReq struct {
	first, last int
	to          string
	pm          *model.PendingMove
	done        chan moveRes
}

type moveRes struct {
	mv  model.BlockMove
	err error
}

// serveMove advances the session's pending move; the caller holds s.mu, at a
// token boundary. The first call prepares it (model.State.PrepareMove: the
// target takes the blocks while the sequence keeps generating), and a later
// one commits it once the target is ready -- or at once when final, which is
// a session no generate is holding.
func (s *Session) serveMove(final bool) {
	s.moveMu.Lock()
	r := s.move
	s.moveMu.Unlock()
	if r == nil {
		return
	}
	if r.pm == nil {
		pm, err := s.st.PrepareMove(r.first, r.last+1, r.to)
		if err != nil {
			s.finishMove(r, moveRes{err: err})
			return
		}
		s.moveMu.Lock()
		r.pm = pm
		s.moveMu.Unlock()
	}
	if !final && !r.pm.Ready() {
		return
	}
	mv, err := r.pm.Commit()
	s.refresh()
	s.finishMove(r, moveRes{mv, err})
}

// finishMove hands the result back and clears the slot.
func (s *Session) finishMove(r *moveReq, res moveRes) {
	s.moveMu.Lock()
	if s.move == r {
		s.move = nil
	}
	s.moveMu.Unlock()
	r.done <- res
}

// MoveBlocks moves blocks first..last of a session from the devices they are
// on to the device named to, history and all. The device must be one of the
// session's tier: the tier owns the devices and is opened at load.
//
// A session generating keeps generating: its generate prepares the move
// between two tokens (model.State.PrepareMove), keeps decoding on the source
// while the target takes the blocks, and commits at the first token boundary
// the target is ready, so the reply waits for the history's copy alone. An
// idle session is moved here. A move asked while another is pending is
// refused.
func (e *Engine) MoveBlocks(ctx context.Context, s *Session, first, last int, to string) (model.BlockMove, error) {
	if s.lm.dev == nil {
		return model.BlockMove{}, fmt.Errorf("%w: session %q was loaded host-only; there is no "+
			"device to move blocks to", ErrPlacement, s.id)
	}
	if !hasString(s.lm.deviceIDs, to) {
		return model.BlockMove{}, fmt.Errorf("%w: session %q is on %v and %q is not one of them: "+
			"the TIER owns the devices and is opened at load. Load the model with every device "+
			"you want to move between", ErrPlacement, s.id, s.lm.deviceIDs, to)
	}
	r := &moveReq{first: first, last: last, to: to, done: make(chan moveRes, 1)}
	s.moveMu.Lock()
	if s.move != nil {
		s.moveMu.Unlock()
		return model.BlockMove{}, fmt.Errorf("%w: session %q has a move of blocks %d..%d to %s pending",
			ErrPlacement, s.id, s.move.first, s.move.last, s.move.to)
	}
	s.move = r
	s.moveMu.Unlock()
	// The pool's States are no sequence's, and would keep the blocks where
	// they are (dropIdle).
	e.dropIdle(s.lm)
	tick := time.NewTicker(2 * time.Millisecond)
	defer tick.Stop()
	for {
		// Idle -- no generate holds the session -- the move is made here.
		if s.mu.TryLock() {
			s.serveMove(true)
			s.mu.Unlock()
		}
		select {
		case res := <-r.done:
			e.applyPageBudget(s.lm, nil)
			if res.err != nil {
				res.err = fmt.Errorf("%w: %v", ErrPlacement, res.err)
			}
			return res.mv, res.err
		case <-ctx.Done():
			// Withdrawn only while nothing was prepared; past that the move
			// finishes and the State is left consistent either way.
			s.moveMu.Lock()
			if s.move == r && r.pm == nil {
				s.move = nil
				s.moveMu.Unlock()
				return model.BlockMove{}, ctx.Err()
			}
			s.moveMu.Unlock()
		case <-tick.C:
		}
	}
}

// hostWeightShare is what the HOST may spend on paged weights, given a total
// budget: the dense region never pages, and skipping that subtraction
// double-spends -- a budget one page short costs a read every token. What a
// tier on the host's own memory holds (tier.GPU.HostReserved) is not taken
// here: Model.SetPageBudget takes it off itself, and again whenever a block
// moves, so taking it here too would count it twice.
func (lm *LoadedModel) hostWeightShare(total uint64) uint64 {
	avail := total
	if d := lm.m.DenseBytes(); d < avail {
		avail -= d
	} else {
		return 0
	}
	return avail
}

// pagerBudget is a share as the pager must be handed it. The pager reads ZERO
// as "unlimited", so a share that came to nothing -- a total smaller than what
// the dense region and the KV already claim -- becomes
// one byte: one page at a time, faulted in and out, which is what so small a
// budget means. Handed over as zero it would make the smallest budget a
// caller can ask for into the largest.
func pagerBudget(share uint64) uint64 { return max(share, 1) }

// applyPageBudget re-sizes the pager now that this model's placement and its
// sessions' KV caches are known: the host needs pages only for the blocks the
// devices did not take, and KV is committed memory the weight budget must not
// also claim. KV is summed over every session of the model.
//
// It is an Engine method because lm.sessions is written under e.mu, not
// lm.mu.
func (e *Engine) applyPageBudget(lm *LoadedModel, newest *model.State) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	if lm.m.PageSize() == 0 || lm.budget == 0 {
		return // not a container, or no budget to divide
	}
	avail := lm.hostWeightShare(lm.budget)

	// The newest session is not in the map yet, so it is counted separately;
	// a rebudget has none. A history is what it has committed (KVBytes), not
	// its context: pages are committed as it grows, and charging the whole
	// context up front left a long-context model's pager one byte. The share
	// is re-divided as histories grow (after every generate) and when one is
	// given back (CloseSession).
	var kv uint64
	minDev := lm.m.Cfg.NLayer
	if newest != nil {
		kv = newest.KVBytes()
		minDev = newest.GPULayers()
	}
	e.mu.RLock()
	for _, s := range lm.sessions {
		kv += s.snapKV.Load()
		if d := int(s.snapDevBlocks.Load()); d < minDev {
			minDev = d
		}
	}
	if newest == nil && len(lm.sessions) == 0 {
		minDev = 0 // no session has placed anything yet
	}
	kv += lm.idleKV()
	e.mu.RUnlock()
	// The model's host step scratch (model.Model.StepScratchBytes): one step's
	// rows across sessions, held between steps for the next.
	kv += lm.m.StepScratchBytes()

	// The pooled States' history (above) and the memory cache are committed
	// host memory too. The store's bound is a share of the model's budget,
	// set here so it follows every re-division.
	if st := lm.ttft.store; st != nil {
		st.SetLimit(e.storeLimit(avail, st.Bytes()))
		kv += st.Bytes()
	}
	if kv < avail {
		avail -= kv
	} else {
		avail = 0
	}
	// The host block count follows the session with the fewest device
	// blocks: any block some session runs on the host needs a page.
	if lm.m.Cfg.NLayer-minDev <= 0 {
		return // every block is on a device for every session; nothing faults here
	}
	lm.m.SetPageBudget(pagerBudget(avail))
}

// defaultMaxSeq is the KV capacity a session of lm gets when it asks for none.
func (e *Engine) defaultMaxSeq(lm *LoadedModel) int {
	if e.cfg.DefaultMaxSeq > 0 {
		return e.cfg.DefaultMaxSeq
	}
	return lm.m.Cfg.NCtx
}

// namedDevicesOpen refuses a placement that names a device the load did not
// open: it would otherwise load and fail at the first session, far from the
// setting that caused it.
func namedDevicesOpen(pl model.Placement, g *tier.GPU) error {
	open := map[string]bool{"host": true}
	if g != nil {
		for _, h := range g.Hardware() {
			open[nn.DeviceName(h.Name)] = true
		}
	}
	named := make([]model.Place, 0, len(pl.Blocks)+2)
	for _, b := range pl.Blocks {
		named = append(named, b)
	}
	for _, x := range []*model.Place{pl.Rest, pl.Head} {
		if x != nil {
			named = append(named, *x)
		}
	}
	for _, x := range named {
		if !open[x.On] {
			return fmt.Errorf("placement names %q, which this load did not open (it opened %d device(s); "+
				"name each in the device list, e.g. cuda:0,vulkan:0)", x.On, len(open)-1)
		}
	}
	return nil
}

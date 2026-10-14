package model

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDecodeDoesNotAllocate holds a warm decode token at ZERO heap
// allocations, on the host and on every device backend present, for a
// dense model and a mixture. A count of allocations does not depend on the
// clock -- the same token allocates the same objects on a hot box and a cold
// one -- so this is a correctness-style gate and not a timing one (RULE 4).
// What does vary is the runtime's own caches refilling, which countAllocs
// tells apart from the engine's allocations by the stack that made them.
//
// What it guards is the closure handed to the worker pool or a device's owner
// thread: one that captures locals escapes and costs an allocation per region,
// which is how a token came to make 32-125 of them. The allocator and the
// collector are part of the token and no kernel counter sees them.
func TestDecodeDoesNotAllocate(t *testing.T) {
	// A dense model, a mixture, and a hybrid -- whose linear blocks keep a
	// recurrent state per session in the device's pools. JITLLM_ALLOC_MODELS
	// adds more: a step's arms need every block on the card, so a model only a
	// large card holds (a latent-attention mixture, gpt-oss's biased banks) is
	// covered only where one is named.
	names := []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "olmoe-1b-7b-0924-instruct-Q4_K_M.gguf",
		"qwen35/Qwen3.5-0.8B-Q4_K_M.gguf", "synth-glm4moe.gguf",
		"synth-qwen2moe.gguf", "synth-ernie45moe.gguf", "synth-hunyuanmoe.gguf", "synth-minimaxm2.gguf",
		"hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf"}
	// Every principle fixture (paging_test.go): one graph per architecture
	// that has no real model in the list above.
	names = append(names, principleFixtures...)
	names = append(names, principleMixtures...)
	if v := os.Getenv("JITLLM_ALLOC_MODELS"); v != "" {
		names = append(names, strings.Split(v, ",")...)
	}
	// The backends, by kind unless JITLLM_STEP_DEVICES names the devices: a
	// box whose default index is not the card under test pins them there.
	specs := []string{"cuda", "vulkan", "metal"}
	if v := os.Getenv("JITLLM_STEP_DEVICES"); v != "" {
		specs = strings.Split(v, ",")
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			// jlmOf skips naming the model when neither it nor its container
			// is here, and opens a container alone on a host that keeps no
			// GGUFs (arm64).
			p := testmodels.Path(name)
			// The host's tuners pinned: they settle over hundreds of tokens,
			// allocating as they choose, and the gate cannot tell when that is
			// done. The device's split tuner runs at load and keeps its default.
			// The arms that need a model opened their own way run first and close
			// it before the shared one opens, so one copy is resident at a time:
			// olmoe twice is 9 GiB, past scripts/cap 8G's MemoryHigh, where the
			// kernel throttles the cgroup into reclaim and the gate crawls.
			// The k-sliced fused matvec (nn's fusedSliced) runs where a pool is
			// wide enough to make a balanced row chunk short -- olmoe's head on
			// a 14-worker two-socket host -- and a six-worker host never takes it
			// at these shapes, so its closure allocated two objects a token
			// there and nowhere this gate ran. Pinned here, it runs on every
			// fused matrix of every host.
			t.Run("host-ksliced", func(t *testing.T) {
				mk, err := Open(jlmOf(t, p), noTune, WithJITOptions(nn.WithFusedKSlices(2)))
				if err != nil {
					t.Fatal(err)
				}
				defer mk.Close()
				// An all-float model (the synth fixtures, every tensor F32) has
				// no fused packed matvec, which is the only kernel k-slicing
				// splits; the host arm above is its whole decode.
				if !anyPackedBlock(mk) {
					t.Skip("every block weight is float: there is no fused packed matvec to split over k")
				}
				if decodeAllocs(t, mk, nil).ksliced == 0 {
					t.Fatal("no matvec of the window was split over k: this arm measured the plain host decode twice")
				}
			})
			// The node-affine fused matvec (sched.DoNodes), which cmd/jitllm
			// turns on wherever the pool spans two NUMA nodes: the shipping
			// host decode of a two-socket box, and a path no one-node host has.
			t.Run("host-numa", func(t *testing.T) {
				if len(sched.NUMANodes()) < 2 {
					t.Skip("one NUMA node: DoNodes is the plain region here, which the host arm measured")
				}
				mn, err := Open(jlmOf(t, p), noTune, WithJITOptions(nn.WithSched(sched.WithNUMA(true))))
				if err != nil {
					t.Fatal(err)
				}
				defer mn.Close()
				// DoNodes splits the fused packed matvec, as k-slicing does:
				// an all-float model has none, and its host arm is its whole
				// decode.
				if !anyPackedBlock(mn) {
					t.Skip("every block weight is float: there is no fused packed matvec to run node-affine")
				}
				s := mn.NewState(8)
				nodes, _ := s.jit.NUMA()
				s.Close()
				if nodes < 2 {
					t.Skipf("the pool spans %d node(s) under this process's CPU mask: no DoNodes region can be "+
						"node-affine", nodes)
				}
				if decodeAllocs(t, mn, nil).affine == 0 {
					t.Fatal("no region of the window ran node-affine: this arm measured the plain host decode twice")
				}
			})
			// The stream trial pinned off with the tuners: it migrates blocks
			// between its runs, and a migration's page-ins are not a decode.
			m, err := Open(jlmOf(t, p), noTune, WithStreamTrial(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			t.Run("host", func(t *testing.T) {
				decodeAllocs(t, m, nil)
			})
			// The step across host sessions (stepHost) a server's step loop
			// runs a host-only model's requests through.
			t.Run("host-step", func(t *testing.T) { hostStepAllocs(t, m, false) })
			t.Run("host-step-chunk", func(t *testing.T) { hostStepAllocs(t, m, true) })
			// The q8_0 cache: its append is the generated quantizer, with
			// padding and spill space reserved at load, so a warm token
			// allocates nothing either.
			t.Run("host-q8", func(t *testing.T) {
				if err := m.Cfg.KVTypeRefusal(KVQ8_0); err != nil {
					t.Skipf("refused by name: %v", err)
				}
				if err := m.SetKVType(KVQ8_0); err != nil {
					t.Fatal(err)
				}
				defer m.ClearKVF16()
				s := m.NewState(8)
				q8 := s.KVType() == KVQ8_0
				s.Close()
				if !q8 {
					t.Fatal("a q8_0 cache was asked for and not selected")
				}
				decodeAllocs(t, m, nil)
			})
			for _, spec := range specs {
				t.Run(spec, func(t *testing.T) {
					g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					t.Logf("%s is %s", spec, g.Name())
					t.Run("decode", func(t *testing.T) { decodeAllocs(t, m, g) })
					t.Run("step", func(t *testing.T) { stepAllocs(t, m, g, stepAll) })
					t.Run("step-one-head", func(t *testing.T) { stepAllocs(t, m, g, stepOneHead) })
					t.Run("step-chunk", func(t *testing.T) { stepAllocs(t, m, g, stepChunk) })
					t.Run("step-solo", func(t *testing.T) { stepAllocs(t, m, g, stepSolo) })
					checkLinearRowsForm(t, m, g)
				})
			}
		})
	}
}

// TestLongDecodeDoesNotAllocate is TestDecodeDoesNotAllocate's host arm past
// four KV pages of context, which that gate's window -- under one page, by
// design -- never reaches. The attention's page walk once collected a window's
// spans into a four-span array, so every head of every token allocated past
// 1024 positions: a picture's prompt alone (Phi-4-reasoning-vision's 2003)
// was enough, and every gate decoded short.
func TestLongDecodeDoesNotAllocate(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// Past four pages, and short of the next page boundary over the window so
	// no page is committed inside it.
	const ctx, warm, n = 5*kvPageTarget + 16, 16, 64
	s := m.NewState(ctx + warm + n + 8)
	defer s.Close()
	ids := make([]int32, ctx)
	for i := range ids {
		ids[i] = int32(5 + i%500)
	}
	if _, err := s.Prefill(ids); err != nil {
		t.Fatal(err)
	}
	for i := range warm {
		if _, err := s.Forward(int32(7 + i)); err != nil {
			t.Fatal(err)
		}
	}
	if pg := s.kv.layers[0].p; s.pos/pg < 5 {
		t.Fatalf("the window starts at position %d of %d-position pages: under five pages, which is not "+
			"what this gate is for", s.pos, pg)
	}
	r0 := m.container.Reads()
	w := countAllocs(func() {
		for i := range n {
			if _, err := s.Forward(int32(3 + i)); err != nil {
				t.Fatal(err)
			}
		}
	})
	allocVerdict(t, fmt.Sprintf("host, positions %d..%d", s.pos-n, s.pos), w, n, m.container.Reads()-r0, 0)
}

// stepMode is which shape of step stepAllocs runs: the shapes the server's
// step loop makes (server/batch.go).
type stepMode int

const (
	stepAll     stepMode = iota // every session decodes and wants logits
	stepOneHead                 // only the first wants logits: the one-row head
	stepChunk                   // the third feeds a prompt chunk, no logits, beside two decoding
	stepSolo                    // one run alone, which StepRuns takes through Forward
)

// stepAllocs is decodeAllocs for StepRuns: three sessions on g, every block
// and the head on the device so the runs go as rows of one ragged step, in
// the shape mode names.
func stepAllocs(t *testing.T, m *Model, g *tier.GPU, mode stepMode) {
	prompts := []string{"The capital of France is", "Water boils at a temperature of",
		"The clerk counted barrels of salt on the upper floor, and"}
	// As decodeAllocs: the window sits inside one 128-position grain, past the
	// warm-up that crosses the first, and short of 256. A chunk run moves two
	// positions a step, so that arm takes half the steps.
	warm, n, chunk := 128, 64, 1
	if mode == stepChunk {
		warm, n, chunk = 64, 32, 2
	}
	var runs []Run
	toks := make([][]int32, len(prompts))
	for i, pr := range prompts {
		ids := m.Vocab.Encode(pr, true)
		st := m.NewState(len(ids) + chunk*(warm+n) + 8)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			t.Skipf("CARD TOO SMALL: %d of %d blocks, head %v -- a joint step needs them all, this arm "+
				"proved nothing", st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice())
		}
		if _, err := st.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		k, logits := 1, true
		switch {
		case mode == stepOneHead:
			logits = i == 0
		case mode == stepChunk && i == 2:
			k, logits = chunk, false
		}
		toks[i] = make([]int32, k)
		runs = append(runs, Run{State: st, Tokens: toks[i], Logits: logits})
	}
	if mode == stepSolo {
		runs = runs[:1]
	}
	step := func(at int) {
		for i, tk := range toks {
			for j := range tk {
				tk[j] = int32(3 + (at+i+j)%64)
			}
		}
		if _, err := StepRuns(runs); err != nil {
			t.Fatal(err)
		}
	}
	for k := range warm {
		step(k)
	}
	// The placement is checked again after the warm-up: a State whose
	// history outgrew what the card had left hands its blocks home
	// (SetRelocate), which is the answer a too-small card gives, not a
	// failure of the step -- but it is said, with what moved.
	if mode != stepSolo {
		for k, r := range runs {
			if err := r.State.StepRefusal(); err != nil && r.State.Relocations() > 0 {
				t.Skipf("CARD TOO SMALL after the warm-up: run %d moved %d block(s) home (%v), so the "+
					"step cannot be joint -- this arm proved nothing", k, r.State.Relocations(), err)
			}
		}
	}
	if err := prefaultExperts(m); err != nil {
		t.Fatal(err)
	}
	c0, d0 := g.Stats(), g.DevStats()
	r0 := m.container.Reads()
	w := countAllocs(func() {
		for k := range n {
			step(warm + k)
		}
	})
	c1 := g.Stats()
	rows := 0
	for _, r := range runs {
		rows += len(r.Tokens)
	}
	want := n * rows
	if mode == stepSolo {
		want = 0 // one run is no step across sessions
	}
	// Each device counts the rows it ran, so on a tier over several devices
	// every device holding blocks counts the whole step and the rest none.
	d1, placed := g.DevStats(), g.Placed()
	for i := range d1 {
		w := want
		if i >= len(placed) || placed[i] == 0 {
			w = 0
		}
		if got := d1[i].SessionRows - d0[i].SessionRows; got != w {
			why := make([]string, len(runs))
			for k, r := range runs {
				why[k] = fmt.Sprint(r.State.StepRefusal())
				if k > 0 && r.State.ldCand != runs[0].State.ldCand {
					why[k] += " (another device candidate than run 0's)"
				}
			}
			t.Fatalf("device %d (%d blocks) stepped %d row(s) across sessions, want %d: the arm did not run "+
				"the step it names; each run's refusal: %v", i, placed[i], got, w, why)
		}
	}
	// An AltUp model's head reads the streams' mean on the host (altup.go),
	// and DeepSeek V4's their collapse (ds4.go): none of their steps takes
	// the device's head.
	wantOne := n
	if m.Cfg.streamHead() {
		wantOne = 0
	}
	if one := c1.RagHeadOne - c0.RagHeadOne; mode == stepOneHead && one != wantOne {
		t.Fatalf("%d of %d steps took the one-row head, want %d", one, n, wantOne)
	}
	allocVerdict(t, fmt.Sprintf("%d run(s) of %d row(s) a step", len(runs), rows), w, n,
		m.container.Reads()-r0, g.Stats().Captures-c0.Captures)
}

// decodeAllocs decodes warm tokens on a fresh State -- on g when it is not
// nil -- and fails on any heap allocation a token makes. It returns what the
// host ran in the window, for an arm's selection check.
func decodeAllocs(t *testing.T, m *Model, g *tier.GPU) decodeWindow {
	// The window is positions warm..warm+n-1, inside one of the device's
	// 128-position score grains (tier's scoreGrain): crossing one re-captures
	// the launch graph, which allocates its wrappers once per 128 positions
	// rather than per token. The warm-up crosses the first. It also stops
	// short of position 256, where the host's KV cache commits its next page:
	// growth, paid once per page of context, not per token.
	const warm, n = 160, 64
	s := m.NewState(warm + n + 8)
	defer s.Close()
	if g != nil {
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		// The selection check: a device arm with nothing placed measured
		// the host twice.
		if s.GPULayers() == 0 {
			t.Skipf("no block placed on the device -- this arm proved nothing")
		}
	}
	// The warm-up settles the tuners, the kernel caches and the pages, whose
	// one-off allocations are not a token's.
	for i := range warm {
		if _, err := s.Forward(int32(5 + i)); err != nil {
			t.Fatal(err)
		}
	}
	// Every routed expert resident, which a long decode reaches by itself: a
	// cold expert page is a read, and its read goroutine is an allocation
	// per page-in rather than per token.
	if err := prefaultExperts(m); err != nil {
		t.Fatal(err)
	}
	var c0 tier.Stats
	if g != nil {
		c0 = g.Stats()
	}
	r0 := m.container.Reads()
	k0 := s.jit.KSliced()
	_, a0 := s.jit.NUMA()
	w := countAllocs(func() {
		for i := range n {
			if _, err := s.Forward(int32(3 + i%64)); err != nil {
				t.Fatal(err)
			}
		}
	})
	captures := 0
	if g != nil {
		captures = g.Stats().Captures - c0.Captures
	}
	_, a1 := s.jit.NUMA()
	dw := decodeWindow{ksliced: s.jit.KSliced() - k0, affine: a1 - a0}
	allocVerdict(t, fmt.Sprintf("%d of %d blocks placed, %d matvec(s) split over k, %d region(s) node-affine",
		s.GPULayers(), m.Cfg.NLayer, dw.ksliced, dw.affine), w, n, m.container.Reads()-r0, captures)
	return dw
}

// decodeWindow is what the host ran over decodeAllocs' window: matvecs split
// over k (nn.JIT.KSliced) and regions dispatched node-affine (nn.JIT.NUMA).
type decodeWindow struct{ ksliced, affine int64 }

// allocVerdict reports a window of n tokens or steps and fails it: on an
// allocation by the engine, on a window the profile cannot account for, and
// on a window that was not the steady state it claims to be -- one that read
// a page or recorded the launch sequence again.
func allocVerdict(t *testing.T, what string, w windowAllocs, n int, reads int64, captures int) {
	t.Helper()
	t.Logf("%s: %d engine allocation(s) over %d; %d in the runtime's per-P caches (sync.Pool, sudogs, "+
		"timers); %d page read(s)", what, w.engine, n, w.caches, reads)
	if captures != 0 {
		t.Fatalf("the window recorded the launch sequence %d time(s): it measured a re-capture, not a "+
			"replayed token", captures)
	}
	if reads != 0 {
		t.Fatalf("the window read %d page(s): this measured page-ins, not a resident model's decode", reads)
	}
	if w.engine != 0 || w.unattributed > 0 {
		// The profile cannot see a tiny allocation (under 16 bytes, no
		// pointers) that lands in a block the allocator already has open, so
		// those count as unattributed; the stacks below still show the first
		// of each block.
		t.Fatalf("a warm decode allocates: %d object(s) over %d, of which\n%s"+
			"and %d the profile did not show (tiny allocations past the first in each 16-byte block)",
			w.engine+max(w.unattributed, 0), n, w.where, max(w.unattributed, 0))
	}
}

// prefaultExperts makes every routed expert page of a mixture resident, as a
// decode does over enough tokens. Nothing on a dense model.
func prefaultExperts(m *Model) error {
	all := make([]int32, m.Cfg.NExpert)
	for i := range all {
		all[i] = int32(i)
	}
	var h expertHold
	for li := range m.Cfg.NLayer {
		if !m.Cfg.MoEAt(li) {
			continue
		}
		err := m.ensureExperts(li, all, &h)
		h.release()
		if err != nil {
			return err
		}
	}
	return nil
}

// windowAllocs is what countAllocs found.
type windowAllocs struct {
	// engine is every allocation not below a cache of the runtime's or of
	// goffi's, and where the stacks that made them.
	engine int64
	where  string
	// caches is the allocations below a per-P cache of the runtime's -- a
	// sync.Pool, the sudog cache, a P's timer heap: refills after a
	// collection, or a P's first use of one, which a decode that allocates
	// nothing never triggers again -- and the runtime starting an OS thread
	// (runtime.allocm: an M, its g0 and profiling stacks), which a woken P
	// does when no idle thread is parked. That last read seven allocations
	// over a host decode and failed the gate on stacks with no
	// engine frame in them. countAllocs collects before the window,
	// so some are always here.
	caches int64
	// unattributed is what MemStats counted beyond the profile. Above zero
	// it is tiny allocations: under 16 bytes and pointer-free, they share a
	// block, and only the first in each block is profiled -- engine
	// allocations, since no runtime cache allocates that small. Below zero is
	// the profile seeing other goroutines in the moment between its snapshot
	// and the window's first count, which is harmless: those stacks are
	// classified with the rest.
	unattributed int64
}

// countAllocs runs f and splits the heap allocations it made by who made
// them, from an allocation profile at rate 1 taken around it.
//
// The split is the point. A process allocates in its runtime's per-P caches at
// random -- a sync.Pool (goffi's call frames, under every device call), the
// sudogs a parking pool worker takes, a P's timer heap -- each refilled or
// grown the first time a P needs it after a collection, and a gate that
// counted those failed one run in a few for the runtime's reasons. The
// engine's own allocations are exact either way.
func countAllocs(f func()) windowAllocs {
	was := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() { runtime.MemProfileRate = was }()
	resample()
	before := memRecords()
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	f()
	runtime.ReadMemStats(&b)
	after := memRecords()

	var w windowAllocs
	seen := int64(0)
	for k, n := range after {
		n -= before[k]
		if n <= 0 {
			continue
		}
		frames := runtime.CallersFrames(k[:])
		cache, ours := false, false
		var top []string
		for {
			fr, more := frames.Next()
			switch {
			case strings.HasPrefix(fr.Function, "sync.(*Pool)."),
				strings.HasPrefix(fr.Function, "sync.(*poolChain)."),
				fr.Function == "runtime.acquireSudog",
				fr.Function == "runtime.(*timers).addHeap",
				fr.Function == "runtime.allocm",
				// A type assertion's per-site cache, grown the first time a
				// concrete type reaches the site.
				fr.Function == "runtime.buildTypeAssertCache":
				cache = true
			case strings.HasSuffix(fr.Function, ".memRecords"):
				ours = true
			}
			if len(top) < 6 {
				top = append(top, fr.Function)
			}
			if !more {
				break
			}
		}
		if ours {
			continue
		}
		seen += n
		if cache {
			w.caches += n
			continue
		}
		w.engine += n
		w.where += fmt.Sprintf("  %d at %s\n", n, strings.Join(top, " <- "))
	}
	w.unattributed = int64(b.Mallocs-a.Mallocs) - seen
	return w
}

// resample makes every P take up a new MemProfileRate. Each P samples its
// next allocation after a byte count drawn at the old rate -- half a megabyte
// on average at the default -- so until a P has allocated past it, its
// allocations go unprofiled, and the window would read as unattributed. More
// goroutines than Ps, each allocating megabytes at once, take every P past
// its count.
func resample() {
	var wg sync.WaitGroup
	for range 8 * runtime.GOMAXPROCS(0) {
		wg.Go(func() {
			for range 8 {
				runtime.KeepAlive(make([]byte, 1<<20))
				runtime.Gosched()
			}
		})
	}
	wg.Wait()
}

// memRecords is the allocation profile by stack, complete up to the call: the
// profile can be two collections behind, so it collects three times first.
func memRecords() map[[32]uintptr]int64 {
	for range 3 {
		runtime.GC()
	}
	var rs []runtime.MemProfileRecord
	n, _ := runtime.MemProfile(nil, true)
	for {
		rs = make([]runtime.MemProfileRecord, n+64)
		var ok bool
		if n, ok = runtime.MemProfile(rs, true); ok {
			break
		}
	}
	m := make(map[[32]uintptr]int64, n)
	for _, r := range rs[:n] {
		m[r.Stack0] += r.AllocObjects
	}
	return m
}

// anyPackedBlock reports whether any block weight of m is in a packed
// (quantized) layout -- what the fused matvec, and so k-slicing, runs on.
func anyPackedBlock(m *Model) bool {
	for i := range m.layers {
		l := &m.layers[i]
		for _, w := range []*tensor{&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down} {
			if w.packed != nil {
				return true
			}
		}
	}
	return m.output.packed != nil
}

// TestOffCardDecodeDoesNotAllocate is TestDecodeDoesNotAllocate's device arms
// on mixture blocks whose experts run off the card: on the host (hybrid), and
// sent to the card every token. A warm token of either makes no engine heap
// allocation, and each arm counts its path having run inside the window.
func TestOffCardDecodeDoesNotAllocate(t *testing.T) {
	for _, name := range []string{"synth-kimik3.gguf", "kimik3/Kimi-K3-0.40B.Q8_0.gguf"} {
		t.Run(name, func(t *testing.T) {
			for _, where := range []string{"host", "card"} {
				// An F32 bank has no sheet layout to send (only hybrid needs none).
				if where == "card" && strings.HasPrefix(name, "synth") {
					continue
				}
				t.Run(where, func(t *testing.T) {
					m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithExperts(where), WithStreamTrial(false))
					if err != nil {
						t.Fatal(err)
					}
					defer m.Close()
					g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("gpu:0")}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, "gpu:0", err)
					}
					defer g.Close()
					t.Run("decode", func(t *testing.T) { decodeAllocs(t, m, g) })
					if where == "host" {
						t.Run("step", func(t *testing.T) { stepAllocs(t, m, g, stepAll) })
					}
					// The selection check, on a state of its own: the windows'
					// states are gone and their counts with them.
					st := m.NewState(8)
					defer st.Close()
					st.SetDeviceLayers(g, m.Cfg.NLayer)
					s0 := g.Stats()
					for _, id := range []int32{5, 6} {
						if _, err := st.Forward(id); err != nil {
							t.Fatal(err)
						}
					}
					s1 := g.Stats()
					if where == "host" && s1.HybridRuns == s0.HybridRuns || where == "card" && s1.StreamFills == s0.StreamFills {
						t.Fatalf("experts %s: %d hybrid block-steps, %d fills -- the path under test never ran",
							where, s1.HybridRuns-s0.HybridRuns, s1.StreamFills-s0.StreamFills)
					}
				})
			}
		})
	}
}

package srcgate

import (
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoMutablePackageState keeps package-level state out of the module.
//
// jitllm runs several models, devices and instances in one process, so a
// switch, a counter or a cache written at package level makes two instances
// interfere. State belongs on the instance that owns it (the JIT, the device,
// the pool, the model) and is set through a With...() option; the libraries
// read no environment.
//
// A package-level var is reported when its declared type or initialiser is
// mutable by nature (sync and atomic types, channels, make, new), or when a
// function other than init assigns to it, writes through it (an element or a
// field on the left of an assignment), increments it, or deletes from it.
//
// An exported var initialised with a slice, array, map or struct literal is
// reported too, since any importer can write it: make it unexported with a
// function, or allowlist it as a table.
// Anything reported must be on globalAllowlist with the reason shared state is
// right for it: a cache of immutable deterministic results, a table written
// once, a process resource, a UI singleton, or a counter with no instance to
// hold it. Test files are read as writers only: a switch a test sets is state. Generated code (gen/), testdata, dev/ and scripts/
// (measurement harnesses and generators, which are programs and not the
// engine) and files that exist only in a jitllmtest, jitllmbench or
// jitllmfault build are not read. A sync.Mutex, RWMutex, Map, Once or atomic
// at package level is reported whatever it guards: the answer is a narrower
// scope, not a lock around a wider one.
func TestNoMutablePackageState(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]string{}
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			switch d.Name() {
			case "gen", "testdata", "node_modules", "vendor", "models":
				return filepath.SkipDir
			}
			if rel == "dev" || rel == "scripts" {
				return filepath.SkipDir
			}
			if rel != "." && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(rel, ".go") {
			b, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			if instrumentOnly(string(b)) {
				return nil
			}
			files[rel] = string(b)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(files) < 500 {
		t.Fatalf("read %d Go files under %s: the tree was not found, so this gate proved nothing", len(files), root)
	}
	found := mutableGlobals(t, files)
	used := map[string]bool{}
	for _, f := range found {
		if k := allowed(f.key); k != "" {
			used[k] = true
			continue
		}
		t.Errorf("%s: package-level %s is mutable state (%s); move it onto the instance that owns it, or allowlist it in globals_test.go with the reason shared is right", f.pos, f.key, f.why)
	}
	for k := range globalAllowlist {
		if !used[k] {
			t.Errorf("globalAllowlist entry %s matches no mutable package-level var; delete it", k)
		}
	}
}

// TestMutablePackageStateGateFires runs the gate's detector over a source
// with each kind of violation and over one without, so a gate that stopped
// seeing them cannot stay green.
func TestMutablePackageStateGateFires(t *testing.T) {
	bad := map[string]string{"p/a.go": `package p
import ("sync"; "sync/atomic")
var flag bool
var counter int
var table = map[string]int{}
var hook func()
var mu sync.Mutex
var n atomic.Int64
var buf []int
var Exported = []int{1}
func Set() { flag = true }
func Count() { counter++ }
func Put() { table["a"] = 1 }
func Hook() { hook = func() {} }
func Add() { buf = append(buf, 1) }
`}
	got := map[string]bool{}
	for _, f := range mutableGlobals(t, bad) {
		got[f.key] = true
	}
	for _, name := range []string{"flag", "counter", "table", "hook", "mu", "n", "buf", "Exported"} {
		if !got["p."+name] {
			t.Errorf("the detector missed p.%s", name)
		}
	}
	good := map[string]string{"p/a.go": `package p
import "errors"
var ErrX = errors.New("x")
var tbl = map[string]int{"a": 1}
var names = []string{"a"}
var set = map[string]int{}
func init() { set["a"] = 1 }
func Get() int { flag := 1; flag = 2; return flag + tbl["a"] + len(names) + set["a"] }
`}
	if g := mutableGlobals(t, good); len(g) != 0 {
		t.Errorf("the detector reported %v on immutable state", g)
	}
}

// allowed is the allowlist entry covering key, an exact name or a path.Match
// pattern ("jit/gpu/vulkan.vk*"), or "".
func allowed(key string) string {
	if _, ok := globalAllowlist[key]; ok {
		return key
	}
	for k := range globalAllowlist {
		if ok, _ := path.Match(k, key); ok {
			return k
		}
	}
	return ""
}

// instrumentOnly is true for a file that exists only in a build with a test,
// bench or fault-injection tag: it is not in a release binary, so state in it
// is an instrument of that build and not of the engine.
func instrumentOnly(src string) bool {
	for _, line := range strings.SplitN(src, "\n", 40) {
		if !constraint.IsGoBuild(line) {
			continue
		}
		x, err := constraint.Parse(line)
		if err != nil {
			return false
		}
		instr := func(tag string) bool {
			return tag == "jitllmtest" || tag == "jitllmbench" || tag == "jitllmfault"
		}
		on := x.Eval(func(string) bool { return true })
		off := x.Eval(func(tag string) bool { return !instr(tag) })
		return on && !off
	}
	return false
}

type globalFinding struct{ key, pos, why string }

// mutableGlobals parses files (path relative to the module -> source) and
// returns the package-level vars that are mutable state.
func mutableGlobals(t *testing.T, files map[string]string) []globalFinding {
	t.Helper()
	fset := token.NewFileSet()
	type decl struct {
		pos, why string
	}
	byDir := map[string][]*ast.File{}
	var names []string
	for n := range files {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		f, err := parser.ParseFile(fset, n, files[n], 0)
		if err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		d := filepath.ToSlash(filepath.Dir(n))
		byDir[d] = append(byDir[d], f)
	}
	var out []globalFinding
	for dir, fs := range byDir {
		vars := map[string]*decl{}
		for _, f := range fs {
			if strings.HasSuffix(fset.Position(f.Pos()).Filename, "_test.go") {
				continue // a test declares no engine state; it is read below as a writer
			}
			for _, d := range f.Decls {
				g, ok := d.(*ast.GenDecl)
				if !ok || g.Tok != token.VAR {
					continue
				}
				for _, s := range g.Specs {
					v := s.(*ast.ValueSpec)
					why := ""
					if v.Type != nil {
						why = mutableType(v.Type)
					}
					for _, e := range v.Values {
						if why == "" {
							why = mutableInit(e)
						}
					}
					for i, n := range v.Names {
						if n.Name == "_" {
							continue
						}
						w := why
						if w == "" && n.IsExported() && i < len(v.Values) {
							w = exportedTable(v.Values[i])
						}
						vars[n.Name] = &decl{pos: fset.Position(n.Pos()).String(), why: w}
					}
				}
			}
		}
		for _, f := range fs {
			for _, d := range f.Decls {
				fn, ok := d.(*ast.FuncDecl)
				if !ok || fn.Body == nil || (fn.Recv == nil && fn.Name.Name == "init") {
					continue
				}
				local := localNames(fn)
				note := func(e ast.Expr, how string) {
					id := rootIdent(e)
					if id == nil || local[id.Name] {
						return
					}
					if v := vars[id.Name]; v != nil && v.why == "" {
						v.why = how + " in " + fn.Name.Name
					}
				}
				ast.Inspect(fn.Body, func(n ast.Node) bool {
					switch n := n.(type) {
					case *ast.AssignStmt:
						if n.Tok == token.DEFINE {
							return true
						}
						for _, l := range n.Lhs {
							note(l, "assigned")
						}
					case *ast.IncDecStmt:
						note(n.X, "incremented")
					case *ast.CallExpr:
						if id, ok := n.Fun.(*ast.Ident); ok && len(n.Args) > 0 && (id.Name == "delete" || id.Name == "clear") {
							note(n.Args[0], id.Name+"d")
						}
					}
					return true
				})
			}
		}
		for name, v := range vars {
			if v.why != "" {
				out = append(out, globalFinding{key: dir + "." + name, pos: v.pos, why: v.why})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].key < out[j].key })
	return out
}

// exportedTable names what makes an exported var's value writable by every
// importer: a slice, array, map or struct literal, or a func literal called
// to build one.
func exportedTable(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.CompositeLit:
		return "exported composite literal, writable by every importer"
	case *ast.CallExpr:
		if f, ok := x.Fun.(*ast.FuncLit); ok && f.Type.Results != nil && len(f.Type.Results.List) == 1 {
			switch f.Type.Results.List[0].Type.(type) {
			case *ast.ArrayType, *ast.MapType:
				return "exported table built by a function, writable by every importer"
			}
		}
	}
	return ""
}

func mutableType(e ast.Expr) string {
	why := ""
	ast.Inspect(e, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.ChanType:
			why = "channel"
		case *ast.SelectorExpr:
			if id, ok := n.X.(*ast.Ident); ok && (id.Name == "sync" || id.Name == "atomic") {
				why = id.Name + "." + n.Sel.Name
			}
		}
		return why == ""
	})
	return why
}

func mutableInit(e ast.Expr) string {
	why := ""
	ast.Inspect(e, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.CallExpr:
			if id, ok := n.Fun.(*ast.Ident); ok && (id.Name == "make" || id.Name == "new") {
				why = id.Name + " in the initialiser"
			}
		case *ast.CompositeLit:
			if n.Type != nil && why == "" {
				why = mutableType(n.Type)
			}
		case *ast.SelectorExpr:
			if id, ok := n.X.(*ast.Ident); ok && (id.Name == "sync" || id.Name == "atomic") {
				why = id.Name + "." + n.Sel.Name
			}
		}
		return why == ""
	})
	return why
}

// rootIdent is the variable an assignable expression writes through.
func rootIdent(e ast.Expr) *ast.Ident {
	for {
		switch x := e.(type) {
		case *ast.Ident:
			return x
		case *ast.IndexExpr:
			e = x.X
		case *ast.SelectorExpr:
			e = x.X
		case *ast.StarExpr:
			e = x.X
		case *ast.ParenExpr:
			e = x.X
		case *ast.SliceExpr:
			e = x.X
		default:
			return nil
		}
	}
}

// localNames is every name a function declares itself (parameters, results,
// receiver, :=, var, range), which shadows a package-level var of that name.
func localNames(fn *ast.FuncDecl) map[string]bool {
	l := map[string]bool{}
	fields := func(fl *ast.FieldList) {
		if fl == nil {
			return
		}
		for _, f := range fl.List {
			for _, n := range f.Names {
				l[n.Name] = true
			}
		}
	}
	fields(fn.Recv)
	fields(fn.Type.Params)
	fields(fn.Type.Results)
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.AssignStmt:
			if n.Tok == token.DEFINE {
				for _, x := range n.Lhs {
					if id, ok := x.(*ast.Ident); ok {
						l[id.Name] = true
					}
				}
			}
		case *ast.ValueSpec:
			for _, id := range n.Names {
				l[id.Name] = true
			}
		case *ast.RangeStmt:
			if n.Tok == token.DEFINE {
				for _, x := range []ast.Expr{n.Key, n.Value} {
					if id, ok := x.(*ast.Ident); ok {
						l[id.Name] = true
					}
				}
			}
		case *ast.FuncLit:
			fields(n.Type.Params)
			fields(n.Type.Results)
		}
		return true
	})
	return l
}

// globalAllowlist is every package-level var that is allowed to be mutable
// state, by "dir.name" (dir relative to the module root; a path.Match pattern
// is allowed), with why. An entry that matches nothing fails the test, so the
// list cannot outlive the code.
//
// The reasons come from a short list. cache: results that are a pure function
// of their key, so sharing them between instances changes nothing. table: set
// up once before anything reads it. process: the thing it describes is one per
// process (the dlopen'd driver, the CPU's features, the Go collector, the
// profiler, the cores). ui: a UI process shows one window. counter: tallies
// events across every instance for a gate or a report that has no instance to
// ask. debt: not yet moved onto its instance, and named so it is found.
var globalAllowlist = map[string]string{
	// common
	"common/config.env":      "ui: the app's environment, handed over once at startup by the command (UseEnv)",
	"common/crash.*":         "process: the crash handler is one per process (signal handlers, the report directory)",
	"common/hardware.cached": "cache: the machine probe's last report, serialised by mu",
	"common/hardware.mu":     "cache: serialises the machine probe, which opens and closes every backend",

	// convert
	"convert/gguf.mapped":    "process: the live mappings' address ranges, which Evict consults to refuse MADV_DONTNEED on anonymous memory",
	"convert/gguf.probeOnce": "cache: the kernel's residency answer, asked once",
	"convert/gguf.probed":    "cache: the kernel's residency answer, asked once",

	// engine/model
	"engine/model.fuseTriesTotal": "counter: fused runs asked for across every State, read by the gates that wrap many States",
	"engine/model.fusedRunsTotal": "counter: fused runs opened across every State, read by the gates that wrap many States",
	"engine/model.modelIDs":       "process: the id allocator for loaded models",
	"engine/model.opNanos":        "counter: per-op time, process-wide by its API (model.ResetProfile has no receiver)",

	// engine/nn: kernels generated for a shape and a tier are a pure function of both
	"engine/nn.RaggedRows":     "counter: ragged activation rows quantized across every JIT, read by gates and a report with no JIT in hand",
	"engine/nn.argmaxCode":     "cache: generated kernel per tier",
	"engine/nn.boundK":         "cache: generated kernel per tier",
	"engine/nn.gateK":          "cache: generated kernel per tier",
	"engine/nn.iq4nlK":         "cache: generated kernel per tier",
	"engine/nn.narrowK":        "cache: generated kernel per tier",
	"engine/nn.ssdGateK":       "cache: generated kernel per tier",
	"engine/nn.widenK":         "cache: generated kernel per tier",
	"engine/nn.colPools":       "cache: generated kernels keyed by shape and tier",
	"engine/nn.hcConsts":       "cache: constants keyed by shape",
	"engine/nn.hcMixes":        "cache: generated kernels keyed by shape and tier",
	"engine/nn.imgKernels":     "cache: generated kernels keyed by shape and tier",
	"engine/nn.lns":            "cache: generated kernels keyed by shape and tier",
	"engine/nn.moeSigmoids":    "cache: generated kernels keyed by shape and tier",
	"engine/nn.moeTopKs":       "cache: generated kernels keyed by shape and tier",
	"engine/nn.resampleConsts": "cache: constants keyed by shape",
	"engine/nn.ropeCodes":      "cache: generated kernels keyed by shape and tier",
	"engine/nn.ropeTabs":       "cache: generated kernels keyed by shape and tier",
	"engine/nn.ropes":          "cache: rotary tables, a pure function of the rotary description",
	"engine/nn.ropeSplits":     "cache: rotary tables, a pure function of the rotary description",
	"engine/nn.ngramHashes":    "cache: hash tables, a pure function of the n-gram description",
	"engine/nn.rowKernels":     "cache: generated kernels keyed by shape and tier",
	"engine/nn.sampleKernels":  "cache: generated kernels keyed by shape and tier",
	"engine/nn.sinCosConsts":   "cache: constants keyed by shape",
	"engine/nn.sparseMixers":   "cache: generated kernels keyed by shape and tier",
	"engine/nn.mm*":            "counter: MatMulProfile's phase times, process-wide by its API and armed per JIT",
	"engine/nn.ns*":            "counter: ProfileNanos's phase times, process-wide by its API and armed per JIT",
	"engine/nn.ropeTabCalls":   "counter: rotary tables built, read by gates with no JIT in hand (the kernels are free functions)",
	"engine/nn.rowMajorQuant":  "counter: GGUF-layout reads, which must be zero across any container inference; the callers are free functions",
	"engine/nn.prefillDuels":   "process: a prefill duel measures the machine and is shared by every JIT on it by design (tune.go)",
	"engine/nn.tuneCache":      "process: the tune cache on disk, one file per user, loaded once",
	"engine/nn.tuneCacheMu":    "process: the tune cache on disk, one file per user, loaded once",
	"engine/nn.tuneCacheOnce":  "process: the tune cache on disk, one file per user, loaded once",

	// engine/sched
	"engine/sched.crews":          "process: the registry of shared crews, because cores are one set per process",
	"engine/sched.procs":          "process: GOMAXPROCS holds, because the Go runtime has one",
	"engine/sched.gcReserve":      "process: the Go collector is one per process; this is its budget reserve",
	"engine/sched.weightsOffHeap": "process: whether the collector sees weight memory, which sizes the process's budget",
	"engine/sched.regionLabels":   "process: pprof labels serve the one CPU profiler a process has",
	"engine/sched.qos*":           "process: macOS QoS entry points, loaded once",
	"engine/sched.eClass":         "process: the E-core QoS class, a property of the machine",

	// format/jlm
	"format/jlm.offHeapBytes": "process: the Go memory limit has to subtract every mapping outside the heap (internal/cmd/goheap)",

	// internal
	"internal/testdevices.*":         "process: test support, which restricts the devices a test binary may open",
	"internal/testmodels/fresh.done": "cache: models converted once per test binary",

	// jit/cpu
	"jit/cpu.baselineErr":  "cache: the baseline emitter's self-check, run once",
	"jit/cpu.baselineOnce": "cache: the baseline emitter's self-check, run once",
	"jit/cpu.dotEmulated":  "counter: SDOTs widened across every JIT, read by gates (the emitters are free functions)",
	"jit/cpu.dotOnce":      "process: the CPU's dot-product feature probe",
	"jit/cpu.hasDot":       "process: the CPU's dot-product feature probe",
	"jit/cpu.features":     "process: the CPU's feature probe",
	"jit/cpu.featuresOnce": "process: the CPU's feature probe",
	"jit/cpu.hostTier":     "process: the CPU's tier, from its feature probe",
	"jit/cpu.forceTier":    "process: an input to the feature probe, set by a test before any JIT is built",
	"jit/cpu.tierForced":   "process: an input to the feature probe, set by a test before any JIT is built",
	"jit/cpu.probeMu":      "process: serialises the feature probe",
	"jit/cpu.mapped":       "process: bytes of executable memory mapped, a ledger of address space",
	"jit/cpu.regLedger":    "process: executable mappings by address, for the fault handler",
	"jit/cpu.perfMap*":     "process: the profiler's perf map is one file named for the process",
	"jit/cpu.qa*":          "counter: activation quantizer paths taken across every JIT, read by gates (the quantizer is a free function)",

	// jit/gpu/backend
	"jit/gpu/backend.allocRefused":   "counter: refused allocations across every device, read by gates with no device in hand",
	"jit/gpu/backend.compiles":       "counter: kernels compiled across every device, read by gates and a report with no device in hand",
	"jit/gpu/backend.hostReads":      "counter: host reads across every device, read by gates with no device in hand",
	"jit/gpu/backend.hostReadBytes":  "counter: host reads across every device, read by gates with no device in hand",
	"jit/gpu/backend.ownedDevices":   "counter: CUDA devices open across the process",
	"jit/gpu/backend.ownedQueues":    "counter: CUDA queues open across the process",
	"jit/gpu/backend.ownerCalls":     "counter: owner-thread calls across every device, read by gates",
	"jit/gpu/backend.postedCalls":    "counter: owner-thread posts across every device, read by gates",
	"jit/gpu/backend.cudaCopies":     "cache: scratch pool holding no state between uses",
	"jit/gpu/backend.cudaCopyLists":  "cache: scratch pool holding no state between uses",
	"jit/gpu/backend.cudaWrites":     "cache: scratch pool holding no state between uses",
	"jit/gpu/backend.restriction":    "process: which devices this process may open, installed by test support",
	"jit/gpu/backend.verbose":        "process: log verbosity",
	"jit/gpu/backend.cudaTiming":     "debt: device-clock timing switch, read by cmd/jitllm through a free function; belongs on the CUDA device",
	"jit/gpu/backend.cudaKernTiming": "debt: device-clock timing switch, read by cmd/jitllm through a free function; belongs on the CUDA device",
	"jit/gpu/backend.graphTimes":     "debt: replay times accumulated for cmd/jitllm's report; belongs on the CUDA device",
	"jit/gpu/backend.kernTimes":      "debt: kernel times accumulated for cmd/jitllm's report; belongs on the CUDA device",
	"jit/gpu/backend.mtlResidency":   "debt: a Metal open option read from a package var; belongs in the open options",
	"jit/gpu/backend.mtlUnretained":  "debt: a Metal open option read from a package var; belongs in the open options",

	// jit/gpu: driver bindings, loaded once with the library
	"jit/gpu/cuda.*":   "process: the dlopen'd CUDA driver's entry points, bound once under loadOnce",
	"jit/gpu/hip.*":    "process: the dlopen'd HIP runtime, bound once",
	"jit/gpu/vulkan.*": "process: the dlopen'd Vulkan loader's entry points, bound once",
	"jit/gpu/metal.*":  "process: the Metal framework's selectors and entry points, bound once at load; the stat counters and kernel timing are debt, to move onto the device",

	// jit/gpu/tier: the tier is being refactored elsewhere; narrow this to its caches when that lands
	"jit/gpu/tier.binaryIDOnce": "process: a memoised hash of the running executable, the same for every tier",

	// tables exported as data: nothing writes them after init, but any importer could
	"common/discover.BalanceSteps":   "table: exported list, written once",
	"common/session.SamplingKnobs":   "table: exported list, written once",
	"convert/library.Models":         "table: exported list, written once",
	"engine/graph.DeltaRuleRoles":    "table: exported list, written once",
	"engine/graph.MixerRoles":        "table: exported list, written once",
	"engine/graph.MixtureRoles":      "table: exported list, written once",
	"engine/graph.Rewrites":          "table: exported list, written once",
	"engine/model.UncoveredReasons":  "table: exported list, written once",
	"format/quant.*":                 "table: exported dequantization tables and the packed-type list, written once",
	"internal/oracle.MoEDivergences": "table: exported list, written once",
	"internal/oracle.MoEFaults":      "table: exported list, written once",
	"jit/gpu/kernels.GELUErfA":       "table: exported constants of an approximation, written once",
	"jit/gpu/kernels.TanhP":          "table: exported constants of an approximation, written once",
	"jit/gpu/kernels.TanhQ":          "table: exported constants of an approximation, written once",
	"jit/gpu/kernels.Gated":          "table: exported list of activation kinds, written once",
	"jit/gpu/kernels.Ungated":        "table: exported list of activation kinds, written once",
	"tok/pretok.LlamaCppDefault":     "table: exported pre-tokenizer description, written once",
	"tok/pretok.Table":               "table: the generated pre-tokenizer cache, written once",
	"ui/app.Mono":                    "ui: the window's mono face",
	"ui/app.Prose":                   "ui: the window's prose face",
	"ui/app.TabNames":                "ui: the window's tab names",
	"ui/install.Programs":            "table: exported list, written once",

	// seams a test replaces to simulate a machine or a service; each is read-only in a real run
	"engine/sched.affinity":          "seam: sched_getaffinity, replaced by a test to simulate a mask",
	"engine/sched.bindNodes":         "seam: the memory policy read, replaced by a test",
	"engine/sched.procRoot":          "seam: where /proc is read from, pointed at a fake tree by a test",
	"engine/sched.sysRoot":           "seam: where /sys is read from, pointed at a fake tree by a test",
	"format/jlm.migrations":          "seam: the container migrations, replaced by a test",
	"jit/cpu.force*":                 "process: the machine description a test overrides to simulate another CPU (an input to the feature probe; a JIT reads the probe once at construction)",
	"jit/cpu.sseEmitters":            "seam: the SSE emitter table, replaced by a test",
	"engine/nn.ropeTabConsts":        "debt: the rotary-table kernel's constants, swapped by a gate as its violation; belongs on the JIT",
	"cmd/jitllm.addrOpts":            "debt: a test seam for the service address options; belongs on the command's context",
	"server/cmd/jitllmd.addrOpts":    "debt: a test seam for the service address options; belongs on the command's context",
	"server/cmd/jitllmd.thisService": "debt: a test seam for the service under systemd; belongs on the command's context",

	// tok
	"tok.bpeScratches": "cache: scratch pool holding no state between uses",

	// ui: one window per process
	"ui.stdio":             "ui: the process's replaced stdio",
	"ui/app.UIFonts":       "ui: the window's fonts, loaded once",
	"ui/app.parsedFonts":   "ui: the window's fonts, loaded once",
	"ui/screen.*":          "ui: the screens the one window shows",
	"ui/widgets.dialogSeq": "ui: the one window's dialog ids",
	"ui/widgets.thumbs":    "ui: the one window's thumbnails",
}

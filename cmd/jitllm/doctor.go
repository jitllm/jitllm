package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"runtime"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/capabilities"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// `jitllm doctor [-json] [model.jlm]` says whether this machine and this build
// can run, and for a container whether this build reads it; for each thing
// missing it names the next action. It inspects only: it opens no model
// (jlm.Open reads the header and tables, no page), downloads nothing and runs
// no kernel. Logs go to stderr; stdout is the report.
//
// The exit code is the answer, so a script needs no parsing to branch:
const (
	doctorOK      = 0 // everything a run needs is here
	doctorError   = 1 // the inspection itself failed: bad arguments, an unreadable file
	doctorMissing = 2 // inspected, and something is missing; Problems names it and the next action
)

// doctorReport is what -json prints.
type doctorReport struct {
	// Status is "ok", "missing" or "error", matching the exit code.
	Status    string                 `json:"status"`
	Build     capabilities.BuildInfo `json:"build"`
	Host      doctorHost             `json:"host"`
	Memory    doctorMemory           `json:"memory"`
	Devices   []doctorDevice         `json:"devices"`
	Backends  []doctorBackend        `json:"backends"`
	Container *doctorContainer       `json:"container,omitempty"`
	Problems  []doctorProblem        `json:"problems"`
	// Notes are facts that need no action: no GPU is a CPU machine, not a
	// fault.
	Notes []string `json:"notes"`
	// Error is set with Status "error".
	Error *doctorProblem `json:"error,omitempty"`
}

type doctorHost struct {
	OS   string `json:"os"`
	Arch string `json:"arch"`
	CPU  string `json:"cpu"`
	// Tier is the code generator's tier on this CPU.
	Tier string `json:"tier"`
	// JIT is whether this build has a code generator for this host at all.
	JIT bool `json:"jit"`
}

type doctorMemory struct {
	// Total is the machine's RAM, 0 when unknown.
	Total uint64 `json:"total_bytes"`
	// Budget is what the engine sizes its page frames from: the smallest of
	// available memory, a cgroup limit and the bound NUMA nodes, less
	// headroom. 0 when unknown.
	Budget uint64 `json:"budget_bytes"`
}

type doctorDevice struct {
	Backend string `json:"backend"`
	Name    string `json:"name"`
	// FreeBytes and TotalBytes are 0 when the device cannot say.
	FreeBytes  uint64 `json:"free_bytes"`
	TotalBytes uint64 `json:"total_bytes"`
	Unified    bool   `json:"unified_memory,omitempty"`
}

type doctorBackend struct {
	Name   string `json:"name"`
	Usable bool   `json:"usable"`
	// Why is the reason when it is not usable.
	Why string `json:"why,omitempty"`
}

type doctorContainer struct {
	Path string `json:"path"`
	// Version is the file's container version, 0 when it is not a container.
	Version uint32 `json:"version"`
	// Reads is the version this build reads.
	Reads uint32 `json:"reads"`
	Arch  string `json:"arch,omitempty"`
	// Runs says whether this build opens it: the version matches and the
	// architecture is one it knows.
	Runs   bool `json:"runs"`
	Vision bool `json:"vision,omitempty"`
}

// doctorProblem is one missing requirement and what to do about it.
type doctorProblem struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Next    string `json:"next"`
}

// doctorEnv is everything the doctor reads from the machine, so the tests
// can hand it a fake one.
type doctorEnv struct {
	version   string
	cpuName   string
	tier      string
	jit       bool
	hostErr   error // nn.HostRefusal
	totalRAM  uint64
	budget    uint64
	devices   []doctorDevice
	hipAbsent string // backend.HIPMissing: why ROCm was not loaded, "" when it was
	// archKnown says whether this build runs a container's architecture.
	archKnown func(jlm.Arch) bool
}

// realDoctorEnv probes this machine the way `jitllm hardware` does.
func realDoctorEnv() doctorEnv {
	env := doctorEnv{
		version:   version,
		cpuName:   cpuBrand(),
		tier:      cpu.TierReport(),
		jit:       nn.Available(),
		hostErr:   nn.HostRefusal(),
		budget:    sched.MemBudget(),
		archKnown: jlm.Arch.Valid,
	}
	if b, ok := totalRAM(); ok {
		env.totalRAM = b
	}
	hipCfg := backend.HIPConfig{Path: rocmDir()}
	env.hipAbsent = backend.HIPMissing(hipCfg)
	devs := backend.OpenWith(backend.Opts{HIP: hipCfg})
	for _, d := range devs {
		dd := doctorDevice{Backend: backendName(d.API()), Name: d.Name()}
		if free, total, err := d.Mem(); err == nil {
			dd.FreeBytes, dd.TotalBytes = free, total
		}
		if u, ok := d.(backend.Unified); ok {
			dd.Unified = u.UnifiedMemory()
		}
		env.devices = append(env.devices, dd)
		d.Close()
	}
	return env
}

// backendName is a device's backend as -devices spells it.
func backendName(api string) string {
	switch strings.ToLower(api) {
	case "ptx", "cuda":
		return "cuda"
	case "amdgcn", "hip":
		return "hip"
	case "spirv", "vulkan":
		return "vulkan"
	case "msl", "metal":
		return "metal"
	}
	return api
}

// diagnose builds the report. container is "" for none.
func diagnose(env doctorEnv, container string) (doctorReport, error) {
	r := doctorReport{
		Build:    capabilities.Identity(env.version),
		Host:     doctorHost{OS: runtime.GOOS, Arch: runtime.GOARCH, CPU: env.cpuName, Tier: env.tier, JIT: env.jit},
		Memory:   doctorMemory{Total: env.totalRAM, Budget: env.budget},
		Devices:  env.devices,
		Problems: []doctorProblem{},
		Notes:    []string{},
	}
	if r.Devices == nil {
		r.Devices = []doctorDevice{}
	}
	if !env.jit {
		r.Problems = append(r.Problems, doctorProblem{Code: "no-code-generator",
			Message: fmt.Sprintf("this build has no code generator for %s/%s, and every op a token runs is generated code", runtime.GOOS, runtime.GOARCH),
			Next:    "run jitllm on linux, macOS or windows on amd64 or arm64"})
	} else if env.hostErr != nil {
		r.Problems = append(r.Problems, doctorProblem{Code: "cpu-below-baseline", Message: env.hostErr.Error(),
			Next: "run on a CPU with the extensions named (x86-64 needs SSE4.1 and SSSE3)"})
	}
	if env.budget == 0 {
		r.Notes = append(r.Notes, "the host memory budget is unknown on this platform; pass -maxmem to size the page frames")
	}

	have := map[string]bool{}
	for _, d := range env.devices {
		have[d.Backend] = true
	}
	for _, b := range []string{"cuda", "hip", "vulkan", "metal"} {
		db := doctorBackend{Name: b, Usable: have[b]}
		if !db.Usable {
			db.Why = "no device opened"
			if b == "hip" && env.hipAbsent != "" {
				db.Why = env.hipAbsent
			}
		}
		r.Backends = append(r.Backends, db)
	}
	if len(env.devices) == 0 {
		r.Notes = append(r.Notes, "no GPU backend opened a device: every block runs on the CPU, which is a supported configuration, not a fault")
	}

	if container != "" {
		c, problems, err := checkContainer(container, env.archKnown)
		if err != nil {
			return r, err
		}
		r.Container = c
		r.Problems = append(r.Problems, problems...)
	}
	r.Status = "ok"
	if len(r.Problems) > 0 {
		r.Status = "missing"
	}
	return r, nil
}

// errNotInspectable marks a container path the doctor could not read at all,
// as distinct from one it read and found wanting.
var errNotInspectable = errors.New("doctor: cannot inspect")

// checkContainer reads a container's version and, when this build reads that
// version, its architecture. A file that is absent or not a container is a
// problem with a next action; a file that cannot be read is an error.
func checkContainer(path string, archKnown func(jlm.Arch) bool) (*doctorContainer, []doctorProblem, error) {
	c := &doctorContainer{Path: path, Reads: jlm.Version}
	if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
		return c, []doctorProblem{{Code: "container-missing", Message: path + " does not exist",
			Next: "convert a model: jitllm convert <model.gguf|hf-dir|hf://owner/repo|NAME> " + path +
				" (`jitllm library` lists the names)"}}, nil
	} else if err != nil {
		return nil, nil, fmt.Errorf("%w %s: %v", errNotInspectable, path, err)
	}
	v, err := containerVersion(path)
	if err != nil {
		next := "convert the source model: jitllm convert <source> " + strings.TrimSuffix(path, ".gguf") + jlm.Ext
		return c, []doctorProblem{{Code: "not-a-container", Message: err.Error(), Next: next}}, nil
	}
	c.Version = v
	if v != jlm.Version {
		next := "re-run jitllm convert on the model this container was made from, with this build"
		if jlm.Upgradable(v) {
			next = "upgrade it in place: jitllm upgrade " + path
		}
		return c, []doctorProblem{{Code: "container-version",
			Message: fmt.Sprintf("%s is container version %d and this build reads %d", path, v, jlm.Version),
			Next:    next}}, nil
	}
	f, err := jlm.Open(path)
	if err != nil {
		return c, []doctorProblem{{Code: "container-unreadable", Message: err.Error(),
			Next: "re-run jitllm convert on the model this container was made from; the file is damaged or truncated"}}, nil
	}
	defer f.Close()
	a := f.Config().Arch
	c.Arch = a.String()
	c.Vision = f.H.NVisBlocks > 0
	if !archKnown(a) {
		return c, []doctorProblem{{Code: "architecture-unknown",
			Message: fmt.Sprintf("%s's architecture %s is not one this build runs", path, c.Arch),
			Next:    "install the jitllm release that wrote it (`jitllm capabilities` lists what this one runs)"}}, nil
	}
	c.Runs = true
	return c, nil, nil
}

// containerVersion reads the version from a container's first twelve bytes,
// which every version has kept: the magic, then the u32 version. It answers
// for a container of another version, which jlm.Open refuses.
func containerVersion(path string) (uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	var b [12]byte
	if _, err := io.ReadFull(f, b[:]); err != nil {
		return 0, fmt.Errorf("%s is too short to be a container", path)
	}
	if string(b[:8]) != jlm.Magic {
		return 0, fmt.Errorf("%s is not a jlm container (magic %q)", path, b[:8])
	}
	return binary.LittleEndian.Uint32(b[8:]), nil
}

func doctorCmd(args []string) int {
	set := flag.NewFlagSet("doctor", flag.ContinueOnError)
	set.SetOutput(os.Stderr)
	asJSON := set.Bool("json", false, "print the report as JSON on stdout")
	if err := set.Parse(args); err != nil {
		// A parse failure leaves the flag unset; a -json among the arguments
		// still asks for a JSON error.
		return doctorFail(os.Stdout, slices.Contains(args, "-json") || slices.Contains(args, "--json"), "usage", err)
	}
	if set.NArg() > 1 {
		return doctorFail(os.Stdout, *asJSON, "usage", fmt.Errorf("jitllm doctor takes at most one container, got %d arguments", set.NArg()))
	}
	r, err := diagnose(realDoctorEnv(), set.Arg(0))
	if err != nil {
		return doctorFail(os.Stdout, *asJSON, "inspect", err)
	}
	return doctorPrint(os.Stdout, *asJSON, r)
}

// doctorFail reports an error the doctor could not inspect past.
func doctorFail(w io.Writer, asJSON bool, code string, err error) int {
	if asJSON {
		writeDoctorJSON(w, doctorReport{Status: "error", Error: &doctorProblem{Code: code, Message: err.Error(),
			Next: "see `jitllm doctor -h`"}, Problems: []doctorProblem{}, Notes: []string{}, Devices: []doctorDevice{}})
	} else {
		fmt.Fprintln(os.Stderr, "jitllm doctor:", err)
	}
	return doctorError
}

func writeDoctorJSON(w io.Writer, r doctorReport) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(r); err != nil {
		fmt.Fprintln(os.Stderr, "jitllm doctor:", err)
	}
}

// doctorPrint writes the report and returns its exit code.
func doctorPrint(w io.Writer, asJSON bool, r doctorReport) int {
	code := doctorOK
	if r.Status == "missing" {
		code = doctorMissing
	}
	if asJSON {
		writeDoctorJSON(w, r)
		return code
	}
	rev := r.Build.Revision
	if r.Build.Modified {
		rev += "+dirty"
	}
	fmt.Fprintf(w, "build     %s (%s) %s\n", r.Build.Version, rev, r.Build.Go)
	fmt.Fprintf(w, "host      %s/%s  %s\n", r.Host.OS, r.Host.Arch, r.Host.CPU)
	fmt.Fprintf(w, "          tier %s; code generator %v\n", r.Host.Tier, r.Host.JIT)
	fmt.Fprintf(w, "memory    %s total, budget %s\n", gib(r.Memory.Total), gib(r.Memory.Budget))
	for _, d := range r.Devices {
		fmt.Fprintf(w, "device    %-6s %s  %s free of %s\n", d.Backend, d.Name, gib(d.FreeBytes), gib(d.TotalBytes))
	}
	for _, b := range r.Backends {
		if b.Usable {
			fmt.Fprintf(w, "backend   %-6s usable\n", b.Name)
		} else {
			fmt.Fprintf(w, "backend   %-6s not usable: %s\n", b.Name, b.Why)
		}
	}
	if c := r.Container; c != nil {
		fmt.Fprintf(w, "container %s: version %d (this build reads %d), arch %s, runs %v\n", c.Path, c.Version, c.Reads, c.Arch, c.Runs)
	}
	for _, n := range r.Notes {
		fmt.Fprintf(w, "note      %s\n", n)
	}
	for _, p := range r.Problems {
		fmt.Fprintf(w, "MISSING   %s: %s\n          next: %s\n", p.Code, p.Message, p.Next)
	}
	fmt.Fprintf(w, "status    %s\n", r.Status)
	return code
}

func gib(b uint64) string {
	if b == 0 {
		return "unknown"
	}
	return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30))
}

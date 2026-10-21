// Package hip binds AMD's ROCm runtime (libamdhip64 on Linux, amdhip64_N.dll
// from the HIP SDK on Windows) and code object manager (libamd_comgr) through
// jit_old/gpu/ffi, with no cgo.
//
// Both libraries are found and loaded at run time, as jit_old/gpu/cuda finds
// libcuda: a binary built with CGO_ENABLED=0 runs on a host with ROCm and on
// one without, and on the second it simply has no HIP devices. Nothing here
// runs an external program -- no hipcc, clang, ld.lld or rocminfo. Kernels are
// jitllm's own LLVM IR (jit_old/gpu/amdgpu), turned into a code object by comgr's
// library actions (Compile), and devices are described by HIP calls alone.
//
// The package reads no environment. A caller that wants a ROCm outside the
// places searched by default names it in Config.Path, which cmd/jitllm and
// jitllmd fill from JITLLM_ROCM (on Windows, from the HIP SDK's HIP_PATH when
// JITLLM_ROCM is unset).
package hip

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/jitllm/jitllm/jit_old/gpu/ffi"
)

// Config is where ROCm is looked for. The zero value searches the default
// places (the loader's own path by library name, then DefaultDirs).
type Config struct {
	// Path is a ROCm library directory -- the one holding libamdhip64.so on
	// Linux, the HIP SDK's bin directory holding amdhip64_N.dll on Windows --
	// or empty for the default search. When set, ONLY that directory is
	// tried: a caller who named a ROCm gets that one or none, never a
	// different one found elsewhere.
	Path string
}

// layout is where one operating system's ROCm keeps its two libraries: the
// names tried, newest ABI first; the file patterns a directory is also
// searched with, for names the list does not know yet (the Windows HIP SDK
// puts its version in comgr's file name: amd_comgr0604.dll); and the
// directories an install uses under a root.
type layout struct {
	hip, comgr      []string
	hipGlob, cgGlob string
	dirs            func(root string) []string
	root            string
}

var layouts = map[string]layout{
	"linux": {
		hip:   []string{"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"},
		comgr: []string{"libamd_comgr.so.3", "libamd_comgr.so.2", "libamd_comgr.so"},
		dirs:  linuxDirs,
		root:  "/",
	},
	// AMD's HIP SDK: amdhip64_N.dll, which the display driver also puts in
	// System32 (so the loader's own search finds it), and comgr, which only
	// the SDK installs, under C:\Program Files\AMD\ROCm\X.Y\bin.
	"windows": {
		hip:     []string{"amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"},
		comgr:   []string{"amd_comgr_3.dll", "amd_comgr_2.dll"},
		hipGlob: "amdhip64*.dll",
		cgGlob:  "amd_comgr*.dll",
		dirs:    windowsDirs,
		root:    `C:\Program Files`,
	},
}

// HIPNames and ComgrNames are the library names tried on this system, newest
// ABI first. Package variables so a gate can read them.
var (
	HIPNames   = layouts[runtime.GOOS].hip
	ComgrNames = layouts[runtime.GOOS].comgr
)

// DefaultDirs is where a ROCm install puts its libraries on this system, in
// the order tried after the loader's own path.
func DefaultDirs() []string {
	lo, ok := layouts[runtime.GOOS]
	if !ok {
		return nil
	}
	return lo.dirs(lo.root)
}

// linuxDirs is /opt/rocm (the distribution's symlink to the active version),
// then every /opt/rocm-X.Y.Z newest first, then the distribution's own
// library directory.
func linuxDirs(root string) []string {
	dirs := []string{filepath.Join(root, "opt", "rocm", "lib")}
	vers, _ := filepath.Glob(filepath.Join(root, "opt", "rocm-*", "lib"))
	sort.Slice(vers, func(i, j int) bool { return versionLess(vers[j], vers[i]) })
	dirs = append(dirs, vers...)
	return append(dirs, filepath.Join(root, "usr", "lib", "x86_64-linux-gnu"))
}

// windowsDirs is every HIP SDK under root (Program Files), newest first:
// AMD\ROCm\X.Y\bin. HIP_PATH, which the SDK's installer sets, is an
// environment variable and so the caller's to read.
func windowsDirs(root string) []string {
	vers, _ := filepath.Glob(filepath.Join(root, "AMD", "ROCm", "*", "bin"))
	sort.Slice(vers, func(i, j int) bool { return versionLess(vers[j], vers[i]) })
	return vers
}

// versionLess orders two .../<version>/<lib or bin> paths by their numeric
// version, so 6.10 sorts after 6.9. The version directory is rocm-X.Y.Z on
// Linux and X.Y on Windows.
func versionLess(a, b string) bool {
	pa, pb := versionOf(a), versionOf(b)
	for i := 0; i < len(pa) && i < len(pb); i++ {
		if pa[i] != pb[i] {
			return pa[i] < pb[i]
		}
	}
	return len(pa) < len(pb)
}

func versionOf(p string) []int {
	base := filepath.Base(filepath.Dir(p))
	var out []int
	for _, f := range strings.Split(strings.TrimPrefix(base, "rocm-"), ".") {
		n := 0
		for _, c := range f {
			if c < '0' || c > '9' {
				break
			}
			n = n*10 + int(c-'0')
		}
		out = append(out, n)
	}
	return out
}

// inDir is every file in dir a library may be: the known names that exist,
// then whatever else matches the pattern, highest name first (amd_comgr0605
// before amd_comgr0604).
func inDir(dir string, names []string, glob string) []string {
	var out []string
	seen := map[string]bool{}
	for _, n := range names {
		p := filepath.Join(dir, n)
		if _, err := os.Stat(p); err == nil {
			out = append(out, p)
			seen[p] = true
		}
	}
	if glob == "" {
		return out
	}
	more, _ := filepath.Glob(filepath.Join(dir, glob))
	sort.Sort(sort.Reverse(sort.StringSlice(more)))
	for _, p := range more {
		if !seen[p] {
			out = append(out, p)
		}
	}
	return out
}

// candidates is every path tried for one library under a config and a
// layout: the named directory alone, or the names bare (the loader's search
// path) and then whatever each default directory holds.
func candidates(c Config, lo layout, comgr bool) []string {
	names, glob := lo.hip, lo.hipGlob
	if comgr {
		names, glob = lo.comgr, lo.cgGlob
	}
	if c.Path != "" {
		out := inDir(c.Path, names, glob)
		if len(out) == 0 {
			// Nothing there: the names under the directory, so the load fails
			// naming them rather than with no candidate at all.
			for _, n := range names {
				out = append(out, filepath.Join(c.Path, n))
			}
		}
		return out
	}
	out := append([]string(nil), names...)
	for _, d := range lo.dirs(lo.root) {
		out = append(out, inDir(d, names, glob)...)
	}
	return out
}

// lib is one load of ROCm: both libraries' entry points, bound once per
// configured path. Two configs naming different directories are two loads.
type lib struct {
	hip   *hipAPI
	comgr *comgrAPI
	// hipErr and comgrErr say why a half is missing. comgr alone is enough to
	// compile (the offline gates), and HIP alone is not enough to run.
	hipErr, comgrErr error
}

var (
	libsMu sync.Mutex
	libs   = map[string]*lib{}
)

// load binds the libraries a config names, once per path. A library that is
// absent, or present without an entry point this package needs, is an error in
// the result and never a panic: ffi panics on a missing symbol, which here is
// recovered into the reason.
func load(c Config) *lib {
	libsMu.Lock()
	defer libsMu.Unlock()
	if l, ok := libs[c.Path]; ok {
		return l
	}
	l := &lib{}
	lo, ok := layouts[runtime.GOOS]
	if !ok {
		err := fmt.Errorf("hip: ROCm runs on Linux and Windows; this is %s", runtime.GOOS)
		l.hipErr, l.comgrErr = err, err
		libs[c.Path] = l
		return l
	}
	l.comgr, l.comgrErr = bindComgr(candidates(c, lo, true))
	l.hip, l.hipErr = bindHIP(candidates(c, lo, false))
	libs[c.Path] = l
	return l
}

// bindSafely runs a binder that panics on a missing symbol and returns the
// panic as an error naming it.
func bindSafely(what string, f func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("hip: %s: %v", what, r)
		}
	}()
	f()
	return nil
}

// openFirst is ffi.Open with the candidate list in its error.
func openFirst(what string, names []string) (*ffi.Lib, error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("hip: no %s candidate to load", what)
	}
	l, err := ffi.Open(names...)
	if err != nil {
		return nil, fmt.Errorf("hip: %s is not loadable: %w", what, err)
	}
	return l, nil
}

// Searched is where a config looks for ROCm, for a report: the named
// directory, or the loader's search path and every default directory.
func Searched(c Config) string {
	if c.Path != "" {
		return c.Path
	}
	if d := DefaultDirs(); len(d) > 0 {
		return "the loader's path, " + strings.Join(d, ", ")
	}
	if runtime.GOOS == "windows" {
		return `the loader's path; no HIP SDK under C:\Program Files\AMD\ROCm`
	}
	return "the loader's path"
}

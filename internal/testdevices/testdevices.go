// Package testdevices is the one place a test binary learns which devices it
// may open.
//
// On a box with several cards a gate runs on the card it locked and must not
// touch another job's: CUDA_VISIBLE_DEVICES hides CUDA cards from a process but
// Vulkan ignores it, so a gate that sweeps "every Vulkan device", or names
// vulkan:0, reaches cards it does not own. JITLLM_TEST_DEVICES is the
// selection, a comma-separated list of device specs:
//
//	JITLLM_TEST_DEVICES=cuda:2,vulkan:2 go test ./engine/model
//
// Unset, nothing changes: every test opens what it always opened. Set, no test
// in the binary opens any other device. Two things make that true.
//
// The refusal: Install hands the selection to the backend, which asks it before
// every device open (backend.Restrict). A device outside the selection is not
// opened; a search for devices (the default pick, a loop over ordinals) is
// narrowed to the selection, and a test that NAMES a device outside it is
// recorded as a violation, which Fail turns into a failed binary.
//
// The helpers: Specs, One, Spec and Take give a test the device list it would
// have hardcoded, replaced by the selection when there is one, so a gate that
// wants "a device" gets a selected one instead of failing.
//
// A device spec's ordinal is the backend's own: a CUDA ordinal is the
// driver's, so under CUDA_VISIBLE_DEVICES=2 the card is cuda:0, and without
// it cuda:2. A Vulkan index is the loader's enumeration index.
//
// Tests only. A shipped binary reads no environment and never imports this
// package (guard_test.go).
package testdevices

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Var is the environment variable that holds the selection.
const Var = "JITLLM_TEST_DEVICES"

// device is one selected device in the backend's own terms.
type device struct {
	api string // "ptx", "spirv", "msl", "amdgcn"
	ord int    // the ordinal, 0 for Metal
}

func (d device) String() string {
	switch d.api {
	case "ptx":
		return fmt.Sprintf("cuda:%d", d.ord)
	case "spirv":
		return fmt.Sprintf("vulkan:%d", d.ord)
	case "amdgcn":
		return fmt.Sprintf("hip:%d", d.ord)
	}
	return "metal"
}

var (
	selection []device
	selected  bool
)

func init() {
	v := strings.TrimSpace(os.Getenv(Var))
	if v == "" {
		return
	}
	ds, err := parse(v)
	if err != nil {
		panic(fmt.Sprintf("%s=%q: %v", Var, v, err))
	}
	selection, selected = ds, true
}

// parse reads a selection: backend:ordinal entries, metal alone. A bare
// backend name is refused, because "every Vulkan device" is the thing a
// selection exists to say no to.
func parse(v string) ([]device, error) {
	var out []device
	seen := map[device]bool{}
	for _, raw := range strings.Split(v, ",") {
		text := strings.ToLower(strings.TrimSpace(raw))
		name, sel, has := strings.Cut(text, ":")
		var d device
		switch name {
		case "cuda", "ptx":
			d.api = "ptx"
		case "vulkan", "spirv":
			d.api = "spirv"
		case "hip", "rocm", "amdgcn":
			d.api = "amdgcn"
		case "metal", "msl":
			d.api = "msl"
			if has {
				return nil, fmt.Errorf("%q: metal takes no ordinal", raw)
			}
		default:
			return nil, fmt.Errorf("%q names no backend; want cuda:N, vulkan:N, hip:N or metal", raw)
		}
		if d.api != "msl" {
			n, err := strconv.Atoi(sel)
			if !has || err != nil || n < 0 {
				return nil, fmt.Errorf("%q: want %s:ORDINAL", raw, name)
			}
			d.ord = n
		}
		if !seen[d] {
			seen[d] = true
			out = append(out, d)
		}
	}
	return out, nil
}

// Selected reports whether JITLLM_TEST_DEVICES is set.
func Selected() bool { return selected }

// Selection is the selected devices as specs ("cuda:2", "vulkan:2", "metal"),
// nil when unset.
func Selection() []string {
	var out []string
	for _, d := range selection {
		out = append(out, d.String())
	}
	return out
}

// Specs is the device list a sweep runs over: def when nothing is selected,
// else the selection. Each entry is one device, so a sweep over a bare backend
// name ("cuda", "vulkan", "metal") becomes a sweep over the selected devices.
func Specs(def ...string) []string {
	if !selected {
		return def
	}
	return Selection()
}

// One is the spec of the single device a gate runs on: def when nothing is
// selected, else the first selected device.
func One(def string) string {
	if !selected {
		return def
	}
	return selection[0].String()
}

// Spec is a device list as one -devices string, for a tier that places blocks
// over several: def when nothing is selected, else the whole selection.
func Spec(def string) string {
	if !selected {
		return def
	}
	return strings.Join(Selection(), ",")
}

// Take is n devices for a gate that needs that many distinct ones: def (n specs
// joined by commas) when nothing is selected, else the first n selected, and a
// skip naming the selection when it holds fewer. A gate that moves blocks
// between two cards cannot run on one, and says so rather than opening a
// second card that was not given.
func Take(t testing.TB, n int, def string) []string {
	t.Helper()
	if !selected {
		return strings.Split(def, ",")
	}
	if len(selection) < n {
		t.Skipf("needs %d devices; %s=%s selects %d", n, Var, strings.Join(Selection(), ","), len(selection))
	}
	return Selection()[:n]
}

// Require skips unless every device a spec names is selected, for a gate whose
// devices are part of what it asserts (a placement that puts blocks on
// "cuda:0" and "vulkan:1" by name). A spec is a -devices list, budgets
// included; gpu, auto and all name no device and pass. Nothing selected runs
// it.
func Require(t testing.TB, specs ...string) {
	t.Helper()
	if !selected {
		return
	}
	for _, s := range specs {
		for _, entry := range strings.Split(s, ",") {
			entry, _, _ = strings.Cut(entry, "=") // a budget is not part of the device
			switch name, _, _ := strings.Cut(strings.ToLower(strings.TrimSpace(entry)), ":"); name {
			case "", "host", "cpu", "gpu", "auto", "all":
				continue // not a named device: the backends' own default pick, narrowed by the selection
			}
			ds, err := parse(entry)
			if err != nil {
				t.Fatalf("testdevices.Require(%q): %v", s, err)
			}
			if !allowed(ds[0]) {
				t.Skipf("names %s; %s=%s does not select it", ds[0], Var, strings.Join(Selection(), ","))
			}
		}
	}
}

func allowed(d device) bool {
	for _, s := range selection {
		if s == d {
			return true
		}
	}
	return false
}

func anyOf(api string) (device, bool) {
	for _, s := range selection {
		if s.api == api {
			return s, true
		}
	}
	return device{}, false
}

var (
	mu       sync.Mutex
	violated = map[string]int{}
	narrowed = map[string]int{}
	refusing = map[device]int{}
)

// admit is the backend.Restriction: the selection's answer to one device open.
func admit(api string, ord int, enumerated bool) (int, error) {
	if !selected {
		return ord, nil
	}
	switch {
	case ord == -2:
		// A Vulkan name: the backend asks again with its index once it is open.
		if _, ok := anyOf(api); !ok {
			return refuse(api, "by name", enumerated)
		}
	case ord < 0:
		// The default pick: the first selected device of the backend.
		d, ok := anyOf(api)
		if !ok {
			return refuse(api, "default pick", enumerated)
		}
		return d.ord, nil
	case api == "msl":
		if !allowed(device{api, 0}) {
			return refuse(api, "", enumerated)
		}
	case !allowed(device{api, ord}):
		if expected(device{api, ord}) {
			return -1, fmt.Errorf("%s is outside %s=%s", device{api, ord}, Var, strings.Join(Selection(), ","))
		}
		return refuse(api, strconv.Itoa(ord), enumerated)
	}
	return ord, nil
}

func expected(d device) bool {
	mu.Lock()
	defer mu.Unlock()
	return refusing[d] > 0
}

// Refusing declares spec ("cuda:7") a device the test names on purpose to see
// it refused. While the test runs, an open of it outside the selection is
// refused as every such open is, and not recorded as a violation; an open of
// any other device outside the selection still is. A spec the selection holds
// is a test error: that device would open.
func Refusing(t testing.TB, spec string) {
	t.Helper()
	ds, err := parse(spec)
	if err != nil || len(ds) != 1 {
		t.Fatalf("testdevices.Refusing(%q): want one device spec (%v)", spec, err)
	}
	d := ds[0]
	if selected && allowed(d) {
		t.Fatalf("testdevices.Refusing(%q): %s=%s selects it, so it would open", spec, Var, strings.Join(Selection(), ","))
	}
	mu.Lock()
	refusing[d]++
	mu.Unlock()
	t.Cleanup(func() {
		mu.Lock()
		if refusing[d]--; refusing[d] == 0 {
			delete(refusing, d)
		}
		mu.Unlock()
	})
}

// refuse records an open the selection does not allow, and the error that
// refuses it.
func refuse(api, what string, enumerated bool) (int, error) {
	name := device{api, 0}.String()
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[:i]
	}
	if what != "" {
		name += ":" + what
	}
	mu.Lock()
	if enumerated {
		narrowed[name]++
	} else {
		violated[name]++
	}
	mu.Unlock()
	return -1, fmt.Errorf("%s is outside %s=%s", name, Var, strings.Join(Selection(), ","))
}

// Install puts the selection in force in this process through set, the
// backend's Restrict. It is generic over the restriction's named type so this
// package imports nothing from the engine (the backend's own tests import it).
// With nothing selected it installs nothing.
func Install[R ~func(string, int, bool) (int, error)](set func(R)) {
	if selected {
		set(R(admit))
	}
}

// Violations are the devices a test named that the selection does not hold,
// with how many opens asked for each.
func Violations() []string {
	mu.Lock()
	defer mu.Unlock()
	return list(violated)
}

// Narrowed are the devices a search for devices skipped because the selection
// does not hold them. It is information: a sweep over every device was cut to
// the selection.
func Narrowed() []string {
	mu.Lock()
	defer mu.Unlock()
	return list(narrowed)
}

func list(m map[string]int) []string {
	var out []string
	for k, n := range m {
		out = append(out, fmt.Sprintf("%s x%d", k, n))
	}
	sort.Strings(out)
	return out
}

// Fail is the exit code of a test binary under the selection: code, or 1 when
// a test named a device outside it. A test binary's TestMain wraps its run in
// it, as it does the Vulkan-vanished check:
//
//	os.Exit(testdevices.Fail(m.Run()))
func Fail(code int) int {
	if v := Violations(); len(v) > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: a test opened a device outside %s=%s: %s\n",
			Var, strings.Join(Selection(), ","), strings.Join(v, ", "))
		return 1
	}
	return code
}

// Reset forgets the ledger. For this package's own gate, which provokes a
// violation on purpose.
func Reset() {
	mu.Lock()
	violated, narrowed = map[string]int{}, map[string]int{}
	mu.Unlock()
}

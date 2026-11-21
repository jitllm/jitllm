package testdevices

import (
	"strings"
	"testing"
)

// apiOf is the lowering's name for a backend name as a test writes it.
func apiOf(t testing.TB, backend string) string {
	t.Helper()
	switch strings.ToLower(backend) {
	case "cuda", "ptx":
		return "ptx"
	case "vulkan", "spirv":
		return "spirv"
	case "hip", "rocm", "amdgcn":
		return "amdgcn"
	case "metal", "msl":
		return "msl"
	}
	t.Fatalf("testdevices: %q names no backend", backend)
	return ""
}

// Has reports whether the device at ordinal ord of a backend ("cuda",
// "vulkan", "hip") may be opened: true when nothing is selected. A loop over
// every ordinal a backend has skips the ones that are false.
func Has(t testing.TB, backend string, ord int) bool {
	t.Helper()
	api := apiOf(t, backend)
	return !isSelected() || allowed(device{api, ord})
}

// Ordinal is the ordinal of the device a gate that names one backend's device
// opens: def when nothing is selected, else the first selected device of that
// backend, and a skip when the selection holds none of it.
func Ordinal(t testing.TB, backend string, def int) int {
	t.Helper()
	if !isSelected() {
		return def
	}
	d, ok := anyOf(apiOf(t, backend))
	if !ok {
		t.Skipf("needs a %s device; %s=%s selects none", backend, Var, strings.Join(Selection(), ","))
	}
	return d.ord
}

// Like is spec ("cuda:0", "vulkan:1", "metal") moved onto the selected device
// of the same backend: spec itself when nothing is selected, a skip when the
// selection holds no device of that backend. It is for a gate that is about
// one backend's behaviour and takes any card of it.
func Like(t testing.TB, spec string) string {
	t.Helper()
	if !isSelected() {
		return spec
	}
	name, _, _ := strings.Cut(spec, ":")
	d, ok := anyOf(apiOf(t, name))
	if !ok {
		t.Skipf("needs a %s device; %s=%s selects none", name, Var, strings.Join(Selection(), ","))
	}
	return d.String()
}

// SkipSelected skips the test when a selection is set. For a gate whose point
// is a device the box does not have (an ordinal past the last card, a backend
// that is not there): a selection cannot hold it, and opening it anyway would
// be exactly the open the selection forbids.
func SkipSelected(t testing.TB, why string) {
	t.Helper()
	if isSelected() {
		t.Skipf("%s=%s is set and %s", Var, strings.Join(Selection(), ","), why)
	}
}

// Within is the candidates the selection allows, in their order: all of them
// when nothing is selected. A gate that tries devices one after another and
// takes the first that suits it tries only selected ones.
func Within(candidates []string) []string {
	if !isSelected() {
		return candidates
	}
	var out []string
	for _, c := range candidates {
		ds, err := parse(c)
		if err != nil {
			continue
		}
		if allowed(ds[0]) {
			out = append(out, c)
		}
	}
	return out
}

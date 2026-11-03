package testdevices_test

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// ledger counts what the restriction reports.
type ledger struct{ violated, narrowed map[string]int }

func (l *ledger) list(m map[string]int) []string {
	var out []string
	for k, n := range m {
		out = append(out, fmt.Sprintf("%s x%d", k, n))
	}
	sort.Strings(out)
	return out
}

// install puts the selection in force at the backend for the test with a
// ledger of its own, and takes it out again.
func install(t *testing.T) *ledger {
	t.Helper()
	l := &ledger{violated: map[string]int{}, narrowed: map[string]int{}}
	backend.Restrict(testdevices.RestrictionForTest(
		func(name string) { l.violated[name]++ },
		func(name string) { l.narrowed[name]++ }))
	t.Cleanup(func() { backend.Restrict(nil) })
	return l
}

func TestSelectionIsParsed(t *testing.T) {
	testdevices.SetForTest(t, "cuda:2, VULKAN:2 ,metal,cuda:2")
	want := []string{"cuda:2", "vulkan:2", "metal"}
	if got := testdevices.Selection(); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("Selection = %v, want %v (a duplicate collapses)", got, want)
	}
	if got := testdevices.One("gpu:0"); got != "cuda:2" {
		t.Errorf("One = %q, want the first selected device", got)
	}
	if got := testdevices.Spec("gpu"); got != "cuda:2,vulkan:2,metal" {
		t.Errorf("Spec = %q", got)
	}
	if got := testdevices.Specs("cuda", "vulkan"); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("Specs = %v, want the selection and not the defaults", got)
	}
	if got := testdevices.Within([]string{"vulkan:0", "vulkan:2", "metal"}); strings.Join(got, ",") != "vulkan:2,metal" {
		t.Errorf("Within = %v", got)
	}
	if got := testdevices.Like(t, "vulkan:0"); got != "vulkan:2" {
		t.Errorf("Like(vulkan:0) = %q: the same backend's selected device", got)
	}
	if testdevices.Has(t, "cuda", 0) || !testdevices.Has(t, "cuda", 2) {
		t.Error("Has does not follow the selection")
	}
}

func TestNothingSelectedChangesNothing(t *testing.T) {
	testdevices.SetForTest(t, "")
	if got := testdevices.One("cuda:0"); got != "cuda:0" {
		t.Errorf("One = %q", got)
	}
	if got := testdevices.Specs("cuda", "vulkan"); len(got) != 2 {
		t.Errorf("Specs = %v", got)
	}
	if got := testdevices.Take(t, 2, "cuda:0,vulkan:1"); len(got) != 2 || got[1] != "vulkan:1" {
		t.Errorf("Take = %v", got)
	}
	if !testdevices.Has(t, "vulkan", 7) || testdevices.Ordinal(t, "cuda", 3) != 3 {
		t.Error("with nothing selected every device is allowed")
	}
}

// TestARefusalIsAViolationOnlyWhenTheTestNamedTheDevice is the gate's own
// violation run, at the backend and on any host: the selection names a Vulkan
// device no box has, so every other open is outside it. A device a test names
// is refused and recorded, and Install makes a violation end the binary; a
// search for devices is narrowed without a record; and with the selection
// withdrawn the ledger is not consulted at all.
func TestARefusalIsAViolationOnlyWhenTheTestNamedTheDevice(t *testing.T) {
	testdevices.SetForTest(t, "vulkan:99")
	l := install(t)

	if d, err := backend.OpenCUDA(0); err == nil || !strings.Contains(err.Error(), "outside JITLLM_TEST_DEVICES") {
		if d != nil {
			d.Close()
		}
		t.Fatalf("OpenCUDA(0) under vulkan:99 = %v, want a refusal", err)
	}
	if d, err := backend.OpenVulkan("0"); err == nil || !strings.Contains(err.Error(), "outside JITLLM_TEST_DEVICES") {
		if d != nil {
			d.Close()
		}
		t.Fatalf("OpenVulkan(0) under vulkan:99 = %v, want a refusal", err)
	}
	if v := l.list(l.violated); len(v) != 2 {
		t.Fatalf("violations = %v, want the two named opens", v)
	}

	l.violated, l.narrowed = map[string]int{}, map[string]int{}
	for _, d := range backend.Open() {
		name := d.Name()
		d.Close()
		t.Errorf("backend.Open under vulkan:99 returned %s", name)
	}
	if v := l.list(l.violated); len(v) != 0 {
		t.Errorf("a search for devices was recorded as a violation: %v", v)
	}
	if n := l.list(l.narrowed); len(n) == 0 {
		t.Error("the search was not narrowed: the restriction was never consulted")
	}
}

// TestADeliberateRefusalIsNotAViolation: a device a test declares with
// Refusing is refused and not recorded, every other device outside the
// selection still is, and the declaration ends with its test.
func TestADeliberateRefusalIsNotAViolation(t *testing.T) {
	testdevices.SetForTest(t, "vulkan:99")
	l := install(t)
	open := func(ord int) error {
		d, err := backend.OpenCUDA(ord)
		if d != nil {
			d.Close()
		}
		return err
	}

	t.Run("declared", func(t *testing.T) {
		testdevices.Refusing(t, "cuda:7")
		if err := open(7); err == nil || !strings.Contains(err.Error(), "outside JITLLM_TEST_DEVICES") {
			t.Fatalf("OpenCUDA(7) declared refused = %v, want a refusal", err)
		}
		if v := l.list(l.violated); len(v) != 0 {
			t.Fatalf("a declared refusal was recorded: %v", v)
		}
		if err := open(6); err == nil {
			t.Fatal("OpenCUDA(6) under vulkan:99 opened")
		}
		if v := l.list(l.violated); len(v) != 1 || v[0] != "cuda:6 x1" {
			t.Fatalf("violations = %v, want the undeclared cuda:6 alone", v)
		}
	})
	l.violated = map[string]int{}
	if err := open(7); err == nil {
		t.Fatal("OpenCUDA(7) under vulkan:99 opened")
	}
	if v := l.list(l.violated); len(v) != 1 || v[0] != "cuda:7 x1" {
		t.Fatalf("after its test ended the declaration still holds: violations = %v", v)
	}
}

// TestOnlyTheSelectedDevicesOpen runs on a host with the variable set, which
// is what a shared box does: it asks every ordinal of every backend the host
// has to open, by name, and the devices that open are exactly the selected
// ones. The refusals are the violation run, and are counted in the test's own
// ledger so the binary does not fail on them. With nothing selected there is nothing to hold.
func TestOnlyTheSelectedDevicesOpen(t *testing.T) {
	if !testdevices.Selected() {
		t.Skip("JITLLM_TEST_DEVICES is unset: this gate holds a selection and has none")
	}
	l := install(t)
	opened := map[string]bool{}
	try := func(spec string, open func() (backend.Device, error)) {
		d, err := open()
		if err != nil {
			return
		}
		opened[spec] = true
		d.Close()
	}
	if n, err := backend.CUDACount(); err == nil {
		for i := 0; i < n; i++ {
			try("cuda:"+strconv.Itoa(i), func() (backend.Device, error) { return backend.OpenCUDA(i) })
		}
	}
	if infos, err := backend.VulkanDevices(); err == nil {
		for _, in := range infos {
			try("vulkan:"+strconv.Itoa(in.Index), func() (backend.Device, error) {
				return backend.OpenVulkan(strconv.Itoa(in.Index))
			})
		}
	}
	if len(opened) == 0 {
		t.Skipf("no device of %v opened on this host", testdevices.Selection())
	}
	sel := map[string]bool{}
	for _, s := range testdevices.Selection() {
		sel[s] = true
	}
	for spec := range opened {
		if !sel[spec] {
			t.Errorf("%s opened and is outside the selection %v", spec, testdevices.Selection())
		}
	}
	t.Logf("opened %d of the selected %v; the rest were refused: %v",
		len(opened), testdevices.Selection(), l.list(l.violated))
}

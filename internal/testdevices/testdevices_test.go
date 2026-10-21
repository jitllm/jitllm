package testdevices_test

import (
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// install puts the selection in force at the backend for the test and takes it
// out again.
func install(t *testing.T) {
	t.Helper()
	testdevices.Install(backend.Restrict)
	t.Cleanup(func() { backend.Restrict(nil) })
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
// is refused and recorded, and Fail turns the record into a failed binary; a
// search for devices is narrowed without a record; and with the selection
// withdrawn the ledger is not consulted at all.
func TestARefusalIsAViolationOnlyWhenTheTestNamedTheDevice(t *testing.T) {
	testdevices.SetForTest(t, "vulkan:99")
	install(t)

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
	if v := testdevices.Violations(); len(v) != 2 {
		t.Fatalf("Violations = %v, want the two named opens", v)
	}
	if code := testdevices.Fail(0); code != 1 {
		t.Fatalf("Fail(0) = %d with violations recorded: the binary would have passed", code)
	}

	testdevices.Reset()
	for _, d := range backend.Open() {
		name := d.Name()
		d.Close()
		t.Errorf("backend.Open under vulkan:99 returned %s", name)
	}
	if v := testdevices.Violations(); len(v) != 0 {
		t.Errorf("a search for devices was recorded as a violation: %v", v)
	}
	if n := testdevices.Narrowed(); len(n) == 0 {
		t.Error("the search was not narrowed: the restriction was never consulted")
	}
	if code := testdevices.Fail(0); code != 0 {
		t.Errorf("Fail(0) = %d with nothing named outside the selection", code)
	}
}

// TestOnlyTheSelectedDevicesOpen runs on a host with the variable set, which
// is what a shared box does: it asks every ordinal of every backend the host
// has to open, by name, and the devices that open are exactly the selected
// ones. The refusals are the violation run, and are cleared so the binary does
// not fail on them. With nothing selected there is nothing to hold.
func TestOnlyTheSelectedDevicesOpen(t *testing.T) {
	if !testdevices.Selected() {
		t.Skip("JITLLM_TEST_DEVICES is unset: this gate holds a selection and has none")
	}
	install(t)
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
		len(opened), testdevices.Selection(), testdevices.Violations())
	testdevices.Reset()
}

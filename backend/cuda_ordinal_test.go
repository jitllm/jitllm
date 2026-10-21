package backend_test

import (
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/backend"
)

// The CUDA backend opens a device by ordinal, and backend.Open still takes the
// first one. Gated here: the count comes from the driver, an ordinal outside it
// is refused by an error naming the count (not served by device 0), and ordinal
// 0 is the device backend.Open returns. Two ordinals need a two-card host.

func TestCUDAOrdinalZeroIsWhatOpenReturns(t *testing.T) {
	gpuLock(t)
	n, err := backend.CUDACount()
	if err != nil || n < 1 {
		t.Skip("no CUDA device here:", err)
	}

	ord := testdevices.Ordinal(t, "cuda", 0)
	d, err := backend.OpenCUDA(ord)
	if err != nil {
		t.Fatalf("OpenCUDA(%d) on a box with %d device(s): %v", ord, n, err)
	}
	name, api := d.Name(), d.API()
	o, ok := d.(backend.OrdinalDevice)
	if !ok {
		t.Fatal("the CUDA device does not report its Ordinal")
	}
	if o.Ordinal() != ord {
		t.Errorf("OpenCUDA(%d) opened ordinal %d", ord, o.Ordinal())
	}
	free, total, err := d.Mem()
	if err != nil || total == 0 {
		t.Errorf("Mem: %d/%d, %v", free, total, err)
	}
	d.Close()

	if api != "ptx" {
		t.Errorf("API %q", api)
	}
	// The name must identify a card, not a capability: two cards share a
	// PTX target.
	if !strings.HasPrefix(name, "#"+strconv.Itoa(ord)+" ") {
		t.Errorf("Name %q does not lead with its ordinal", name)
	}
	if !strings.Contains(name, "(sm_") {
		t.Errorf("Name %q does not carry the PTX target in parentheses", name)
	}
	if len(strings.TrimSpace(strings.TrimPrefix(name, "#0 "))) < len("(sm_86)")+1 {
		t.Errorf("Name %q is the target and nothing else", name)
	}
	t.Logf("OpenCUDA(0): %s %s, %.2f of %.2f GiB free", api, name,
		float64(free)/(1<<30), float64(total)/(1<<30))

	// backend.Open must still return exactly this device for the ptx arm.
	var seen string
	for _, dd := range backend.Open() {
		if dd.API() == "ptx" {
			seen = dd.Name()
		}
		dd.Close()
	}
	if seen != name {
		t.Errorf("backend.Open's ptx device is %q, OpenCUDA(0) is %q", seen, name)
	}
}

// TestCUDAOrdinalOutOfRange: the refusal is an error that names the count, and
// it opens nothing. A fallback to device 0 here would place blocks on a card
// the caller did not ask for, and every number downstream would look fine.
func TestCUDAOrdinalOutOfRange(t *testing.T) {
	testdevices.SkipSelected(t, "it opens an ordinal past the box's last card, which no selection holds")
	gpuLock(t)
	n, err := backend.CUDACount()
	if err != nil {
		t.Skip("no CUDA driver here:", err)
	}
	d, err := backend.OpenCUDA(n)
	if err == nil {
		d.Close()
		t.Fatalf("OpenCUDA(%d) opened a device on a box with %d", n, n)
	}
	if !strings.Contains(err.Error(), strconv.Itoa(n)) {
		t.Errorf("the error does not name the count %d: %v", n, err)
	}
	t.Logf("OpenCUDA(%d) -> %v", n, err)
}

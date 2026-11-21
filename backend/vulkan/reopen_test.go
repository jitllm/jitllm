package vulkan

import (
	"os"
	"strconv"
	"testing"
)

// TestReopeningKeepsEveryDevice opens and closes a hardware device over and
// over in one process, enumerating after each close, and fails the moment the
// enumeration shrinks. Every test process does this -- a gate opens a device,
// runs, closes it and the next gate opens it again -- and a driver that stops
// enumerating partway through turns every later Vulkan arm into a skip that
// reads as green.
//
// JITLLM_VK_REOPENS sets the number of cycles (default 200).
func TestReopeningKeepsEveryDevice(t *testing.T) {
	infos, err := List()
	if err != nil {
		t.Skipf("no Vulkan here: %v", err)
	}
	pick := -1
	for _, in := range infos {
		if in.Compute && !in.Software {
			pick = in.Index
			break
		}
	}
	if pick < 0 {
		t.Skipf("no hardware Vulkan device among %d -- this gate proved nothing here", len(infos))
	}
	n := 200
	if v := os.Getenv("JITLLM_VK_REOPENS"); v != "" {
		if n, err = strconv.Atoi(v); err != nil {
			t.Fatalf("JITLLM_VK_REOPENS=%q: %v", v, err)
		}
	}
	for i := 0; i < n; i++ {
		c, err := OpenDevice(strconv.Itoa(pick))
		if err != nil {
			t.Fatalf("open %d of %d: %v", i+1, n, err)
		}
		c.Close()
		got, err := List()
		if err != nil {
			t.Fatalf("enumerate after close %d of %d: %v", i+1, n, err)
		}
		if len(got) != len(infos) {
			t.Fatalf("after %d open/close cycle(s) the loader enumerates %d device(s), where the first "+
				"enumeration in this process saw %d: a driver stopped loading", i+1, len(got), len(infos))
		}
	}
	t.Logf("%d open/close cycles of device %d, %d device(s) enumerated after every one", n, pick, len(infos))
}

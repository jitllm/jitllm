package server

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestLoadWithNoDevicesTakesEveryDevice: a load that names no device is offered
// all of them ("auto"), the host alone is asked for as "cpu", and an explicit
// list is left as the caller wrote it.
//
// VIOLATION SIGNATURE. Return ids unchanged and the empty case stays nil, which
// the engine reads as host only.
func TestLoadWithNoDevicesTakesEveryDevice(t *testing.T) {
	if got := DefaultDevices(nil); !slices.Equal(got, []string{"auto"}) {
		t.Fatalf("nil -> %v, want [auto]", got)
	}
	if got := DefaultDevices([]string{}); !slices.Equal(got, []string{"auto"}) {
		t.Fatalf("empty -> %v, want [auto]", got)
	}
	for _, in := range [][]string{{"cpu"}, {"cuda:0"}, {"cuda:0", "cpu"}} {
		if got := DefaultDevices(in); !slices.Equal(got, in) {
			t.Fatalf("%v -> %v, want it unchanged", in, got)
		}
	}
}

// TestSelectorsResolveToTheOpenedDevices: "auto" becomes the ids of the devices
// the tier opened, so a session on it can be relocated by id; an explicit id is the device it names.
//
// VIOLATION SIGNATURE. Return ids unchanged and the first case keeps "auto".
func TestSelectorsResolveToTheOpenedDevices(t *testing.T) {
	hw := []tier.Hardware{{Ref: "cuda:0"}, {Ref: "vulkan:1"}}
	if got := resolveDeviceIDs([]string{"auto"}, hw); !slices.Equal(got, []string{"cuda:0", "vulkan:1"}) {
		t.Fatalf("auto -> %v", got)
	}
	if got := resolveDeviceIDs([]string{"cuda:0"}, hw[:1]); !slices.Equal(got, []string{"cuda:0"}) {
		t.Fatalf("explicit -> %v", got)
	}
}

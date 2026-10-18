package server

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestLoadPlacementPutsBlocksOnTheNamedDevices: LoadOptions.Placement (the CLI's
// -placement, the TUI's and the desktop app's "Block placement") puts each
// block on the device the map names. The first half of the blocks goes to one
// device and the second to the other, then the map is swapped and every block
// must follow it -- a load that ignored the map would put the blocks in the
// same place twice. A map naming a device the load did not open is refused as
// a device error, not placed elsewhere. JITLLM_TEST_DEVICES selects the two
// devices (default the laptop's one card under CUDA and Vulkan).
func TestLoadPlacementPutsBlocksOnTheNamedDevices(t *testing.T) {
	devs := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	path := modelPath(t, deviceModel)
	const nb = 6
	mapOf := func(a, b string) string {
		return fmt.Sprintf("0-%d=%s,%d-%d=%s", nb/2-1, a, nb/2, nb-1, b)
	}
	ctx := context.Background()
	placed := func(t *testing.T, place string) map[int32]string {
		t.Helper()
		e := New(Config{Version: "test", NoMemCache: true})
		t.Cleanup(e.Close)
		if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "m", DeviceIDs: devs, Placement: place, PlacementStrict: true}); err != nil {
			t.Skipf("NO DEVICE: loading onto -devices %s with %q failed (%v) -- this gate proved nothing",
				strings.Join(devs, ","), place, err)
		}
		c := serveEngine(t, e)
		if _, err := e.CreateSession(SessionOptions{ModelID: "m", SessionID: "s", MaxSeq: 128}); err != nil {
			t.Fatal(err)
		}
		pl, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: "s"}))
		if err != nil {
			t.Fatal(err)
		}
		got := map[int32]string{}
		for _, b := range pl.Msg.GetPlacement().GetBlocks() {
			got[b.GetIndex()] = b.GetDeviceId()
		}
		return got
	}
	check := func(t *testing.T, got map[int32]string, first, second string) {
		t.Helper()
		if len(got) != nb {
			t.Fatalf("%d blocks placed, want %d: %v", len(got), nb, got)
		}
		for i := int32(0); i < nb; i++ {
			want := first
			if i >= nb/2 {
				want = second
			}
			if got[i] != want {
				t.Fatalf("block %d is on %q, the map names %q: %v", i, got[i], want, got)
			}
		}
	}
	forward := placed(t, mapOf(devs[0], devs[1]))
	check(t, forward, devs[0], devs[1])
	swapped := placed(t, mapOf(devs[1], devs[0]))
	check(t, swapped, devs[1], devs[0])
	if forward[0] == swapped[0] {
		t.Fatalf("the swapped map left block 0 on %q: the map is not what places", forward[0])
	}

	t.Run("a device the load did not open is refused", func(t *testing.T) {
		e := New(Config{Version: "test", NoMemCache: true})
		t.Cleanup(e.Close)
		_, err := e.LoadModel(LoadOptions{Path: path, ModelID: "m", DeviceIDs: devs[:1], Placement: "0-5=" + devs[1], PlacementStrict: true})
		var de *DeviceError
		if err == nil || !errors.As(err, &de) {
			t.Fatalf("a map naming %s over -devices %s loaded or failed unnamed: %v", devs[1], devs[0], err)
		}
	})
	t.Run("a malformed map is refused", func(t *testing.T) {
		e := New(Config{Version: "test", NoMemCache: true})
		t.Cleanup(e.Close)
		_, err := e.LoadModel(LoadOptions{Path: path, ModelID: "m", DeviceIDs: devs, Placement: "0-5"})
		var de *DeviceError
		if err == nil || !errors.As(err, &de) {
			t.Fatalf("0-5 (no =WHERE) was accepted: %v", err)
		}
	})
}

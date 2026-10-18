package tier

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// idDev is a fake device that names its hardware and driver, as CUDA and
// Vulkan do, so the tuners may cache against it.
type idDev struct {
	*fakeDev
	uuid, drv string
}

func (d idDev) Identity() backend.Identity {
	return backend.Identity{Key: "uuid:" + d.uuid, Source: "test"}
}
func (d idDev) DriverVersion() string { return d.drv }

// TestSplitTunerPersistsPerDeviceDriverAndRevision: a second tier on the same
// device, driver and executable takes the split tuner's answer from the
// WithTuneCache directory with no timed trial; a different UUID, driver or
// executable times again; a corrupt file or an entry naming a split the shape
// cannot take is measured again; a device that cannot name itself writes
// nothing; and TuneOff still pins the table's answer whatever the file says.
func TestSplitTunerPersistsPerDeviceDriverAndRevision(t *testing.T) {
	const rows, k = 4096, 4096
	table := chooseSplit(rows, k/32, 0)
	win := 8
	if table == win {
		win = 16
	}
	trials := 0
	var hooks trialHooks
	hooks.split = func(_ *devTier, _, _ backend.Kernel, _ *resident, _, _ backend.Buf, _, split int, grp bool) (float64, bool) {
		trials++
		if split == win && !grp {
			return 2, true
		}
		return 1, true
	}
	rev := "revA"
	hooks.binaryID = func() string { return rev }

	dir := t.TempDir()
	tune := func(dev backend.Device, opts ...Option) (int, int) {
		t.Helper()
		g, err := New([]Slot{{Dev: dev, Bytes: 1 << 30}}, append([]Option{WithTuneCache(dir), withTrialHooks(hooks)}, opts...)...)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		trials = 0
		s := g.devs[0].tuneSplit(kernels.Q4_0, rows, k, &resident{})
		return s, trials
	}
	card := func(uuid, drv string) backend.Device { return idDev{&fakeDev{name: "same name"}, uuid, drv} }

	if s, n := tune(card("aa", "cuda12040")); s != win || n == 0 {
		t.Fatalf("first process: split %d after %d trials, want %d measured", s, n, win)
	}
	if s, n := tune(card("aa", "cuda12040")); s != win || n != 0 {
		t.Fatalf("second process: split %d after %d trials, want %d from the cache and none timed", s, n, win)
	}
	for _, c := range []struct {
		what string
		dev  backend.Device
		rev  string
	}{
		{"another card of the same name", card("bb", "cuda12040"), "revA"},
		{"another driver", card("aa", "cuda12050"), "revA"},
		{"another executable", card("aa", "cuda12040"), "revB"},
	} {
		rev = c.rev
		if s, n := tune(c.dev); s != win || n == 0 {
			t.Errorf("%s: split %d after %d trials, want it measured again", c.what, s, n)
		}
	}
	rev = "revA"

	// A corrupt file is no entry; neither is a split the shape cannot take.
	p := filepath.Join(dir, "gpusplit1")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"\x00\xff garbage\nno-space-here\n", strings.ReplaceAll(string(b), " "+strconv.Itoa(win)+" 0", " 7 0")} {
		if err := os.WriteFile(p, []byte(bad), 0o644); err != nil {
			t.Fatal(err)
		}
		if s, n := tune(card("aa", "cuda12040")); s != win || n == 0 {
			t.Errorf("file %q: split %d after %d trials, want it measured again", bad[:min(len(bad), 40)], s, n)
		}
	}

	// TuneOff pins the table, cache or no cache.
	if s, n := tune(card("aa", "cuda12040"), WithDeviceTune(TuneOff)); s != table || n != 0 {
		t.Errorf("TuneOff: split %d after %d trials, want the table's %d untimed", s, n, table)
	}

	// No identity, no entry.
	dir = t.TempDir()
	if s, n := tune(&fakeDev{name: "anonymous"}); s != win || n == 0 {
		t.Fatalf("anonymous device: split %d after %d trials", s, n)
	}
	if _, err := os.Stat(filepath.Join(dir, "gpusplit1")); !os.IsNotExist(err) {
		t.Errorf("a device with no UUID wrote a cache entry (stat: %v)", err)
	}
}

// TestLazyTunerRunsOffThePlacementPath: with WithLazyTune a shape the cache
// does not hold gets the table's split at once with no timed trial on the
// placement path; the background tuner times it, adopts the winner and writes
// it to the cache, which the next tier reads with no trial at all. TuneOff
// queues nothing. The control arm, lazy off, times on the placement path.
func TestLazyTunerRunsOffThePlacementPath(t *testing.T) {
	const rows, k = 4096, 4096
	table := chooseSplit(rows, k/32, 0)
	win := 8
	if table == win {
		win = 16
	}
	hot, bg := 0, 0
	var hooks trialHooks
	hooks.split = func(_ *devTier, _, _ backend.Kernel, _ *resident, _, _ backend.Buf, _, split int, grp bool) (float64, bool) {
		hot++
		if split == win && !grp {
			return 2, true
		}
		return 1, true
	}
	hooks.bg = func(_ *devTier, _, _ backend.Kernel, _ *resident, _, _, _ backend.Buf, _, split int, grp bool) (float64, bool) {
		bg++
		if split == win && !grp {
			return 2, true
		}
		return 1, true
	}
	hooks.binaryID = func() string { return "revA" }
	dir := t.TempDir()
	open := func(opts ...Option) (*GPU, *devTier) {
		g, err := New([]Slot{{Dev: idDev{&fakeDev{name: "card"}, "aa", "cuda12040"}, Bytes: 1 << 30}},
			append([]Option{WithTuneCache(dir), withTrialHooks(hooks)}, opts...)...)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return g, g.devs[0]
	}
	key := splitKey{kernels.Q4_0, rows, k}

	g, d := open(WithLazyTune(true))
	d.mu.Lock()
	d.sizePart(rows * win)
	s := d.tuneSplit(kernels.Q4_0, rows, k, &resident{ok: true})
	d.mu.Unlock()
	if s != table || hot != 0 {
		t.Fatalf("lazy: split %d after %d placement-path trials, want the table's %d and none", s, hot, table)
	}
	d.tuneWG.Wait()
	d.mu.Lock()
	adopted := d.splits[key]
	d.mu.Unlock()
	g.Close()
	if bg == 0 || adopted != win {
		t.Fatalf("background tuner: %d trials, adopted split %d, want %d", bg, adopted, win)
	}

	g, d = open()
	if s := d.tuneSplit(kernels.Q4_0, rows, k, &resident{ok: true}); s != win || hot != 0 {
		t.Fatalf("next tier: split %d after %d trials, want the background's %d from the cache", s, hot, win)
	}
	g.Close()

	bg = 0
	g, d = open(WithLazyTune(true), WithDeviceTune(TuneOff))
	if s := d.tuneSplit(kernels.Q4_0, rows, k, &resident{ok: true}); s != table {
		t.Fatalf("TuneOff: split %d, want the table's %d", s, table)
	}
	d.tuneWG.Wait()
	g.Close()
	if bg != 0 {
		t.Fatalf("TuneOff queued %d background trials", bg)
	}

	dir = t.TempDir()
	g, d = open()
	d.tuneSplit(kernels.Q4_0, rows, k, &resident{ok: true})
	g.Close()
	if hot == 0 {
		t.Fatalf("control: lazy off timed nothing on the placement path; the gate cannot tell the arms apart")
	}
}

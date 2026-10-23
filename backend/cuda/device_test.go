package cuda

import (
	"os"
	"os/exec"
	"runtime"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Ordinals: the count comes from the driver and agrees with an independent
// source, an ordinal outside it is refused by name, and ordinal 0 behaves as it
// did when it was the only one reachable. On a one-card host this cannot prove
// two ordinals work; only a second card can.

func TestCountIsReadFromTheDriver(t *testing.T) {
	n, err := Count()
	if err != nil {
		t.Skip("no CUDA driver here:", err)
	}
	if n < 1 {
		t.Skip("driver present, no CUDA device")
	}
	t.Logf("cuDeviceGetCount: %d", n)

	// An independent oracle: nvidia-smi walks the same driver through a
	// different library. The set of names is compared, not name[i] against
	// ordinal i, because nvidia-smi orders by PCI bus and CUDA by speed.
	smi, err := exec.Command("nvidia-smi", "-L").Output()
	if err != nil {
		t.Skip("nvidia-smi is not here to cross-check with:", err)
	}
	// nvidia-smi lists every card on the machine; the driver counts only
	// those CUDA_VISIBLE_DEVICES names, so the oracle takes the same subset.
	var cards []smiCard
	for _, ln := range strings.Split(string(smi), "\n") {
		if !strings.HasPrefix(ln, "GPU ") {
			continue
		}
		head, rest, _ := strings.Cut(ln, ": ")
		idx, err := strconv.Atoi(strings.TrimPrefix(head, "GPU "))
		if err != nil {
			t.Fatalf("nvidia-smi -L line %q: %v", ln, err)
		}
		name, uuid, _ := strings.Cut(rest, " (UUID:")
		cards = append(cards, smiCard{idx, strings.TrimSpace(name), strings.TrimSpace(strings.TrimSuffix(uuid, ")"))})
	}
	vis, set := os.LookupEnv("CUDA_VISIBLE_DEVICES")
	want, why := visibleCards(cards, vis, set)
	if why != "" {
		t.Skipf("CUDA_VISIBLE_DEVICES=%q %s -- this gate proved nothing", vis, why)
	}
	if len(want) != n {
		t.Fatalf("cuDeviceGetCount says %d device(s), nvidia-smi -L lists %d visible under CUDA_VISIBLE_DEVICES=%q: %q",
			n, len(want), vis, want)
	}

	var got []string
	for i := 0; i < n; i++ {
		d, err := OpenDevice(i)
		if err != nil {
			t.Fatalf("OpenDevice(%d): %v", i, err)
		}
		got = append(got, d.Name())
		t.Logf("ordinal %d: %s  slots %d", d.Ordinal(), d.Label(), d.Slots())
		d.Close()
	}
	sort.Strings(got)
	sort.Strings(want)
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("cuDeviceGetName %q, nvidia-smi %q", got[i], want[i])
		}
	}
}

// smiCard is one line of nvidia-smi -L.
type smiCard struct {
	idx        int
	name, uuid string
}

// visibleCards is the names of the cards the CUDA driver enumerates under a
// CUDA_VISIBLE_DEVICES of vis (set: whether it is set at all), in the order it
// lists them: an index or a UUID prefix each, the list ending at the first
// entry that names no card, as the driver reads it. why is non-empty when the
// list names something this oracle cannot follow (a MIG instance).
func visibleCards(cards []smiCard, vis string, set bool) (names []string, why string) {
	if !set {
		for _, c := range cards {
			names = append(names, c.name)
		}
		return names, ""
	}
	for _, e := range strings.Split(vis, ",") {
		e = strings.TrimSpace(e)
		if strings.HasPrefix(e, "MIG-") {
			return nil, "names a MIG instance, which nvidia-smi -L does not list as a card"
		}
		found := false
		for _, c := range cards {
			if i, err := strconv.Atoi(e); err == nil && i == c.idx || e != "" && !isDigits(e) && strings.HasPrefix(c.uuid, e) {
				names, found = append(names, c.name), true
				break
			}
		}
		if !found {
			break
		}
	}
	return names, ""
}

func isDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return s != ""
}

// TestOrdinalZeroIsUnchanged: Open() is OpenDevice(0) and reports the same
// things.
func TestOrdinalZeroIsUnchanged(t *testing.T) {
	a, err := Open()
	if err != nil {
		t.Skip("no CUDA device:", err)
	}
	target, slots, name, ord := a.Target, a.Slots(), a.Name(), a.Ordinal()
	a.Close()

	b, err := OpenDevice(0)
	if err != nil {
		t.Fatalf("OpenDevice(0) after a working Open(): %v", err)
	}
	defer b.Close()

	if ord != 0 || b.Ordinal() != 0 {
		t.Errorf("ordinals %d and %d, want 0 from both", ord, b.Ordinal())
	}
	if b.Target != target || b.Slots() != slots || b.Name() != name {
		t.Errorf("Open() gave %q/%d/%q, OpenDevice(0) gave %q/%d/%q",
			target, slots, name, b.Target, b.Slots(), b.Name())
	}
	// The label must carry the ordinal: two cards of one generation share a
	// target.
	if !strings.HasPrefix(b.Label(), "#0 ") || !strings.Contains(b.Label(), target) {
		t.Errorf("Label %q does not carry both the ordinal and the target", b.Label())
	}
	if name == "" || name == target {
		t.Errorf("Name %q is not a device name (target is %q)", name, target)
	}
}

// TestOrdinalOutOfRangeNamesTheCount: asking for a device that is not there is
// an error that says how many there are, never a crash or a silent device 0.
func TestOrdinalOutOfRangeNamesTheCount(t *testing.T) {
	n, err := Count()
	if err != nil {
		t.Skip("no CUDA driver here:", err)
	}
	for _, ord := range []int{n, n + 7, -1} {
		d, err := OpenDevice(ord)
		if err == nil {
			d.Close()
			t.Fatalf("OpenDevice(%d) opened a device on a box with %d", ord, n)
		}
		if !strings.Contains(err.Error(), "device(s)") ||
			!strings.Contains(err.Error(), strconv.Itoa(n)) {
			t.Errorf("OpenDevice(%d) error does not name the count %d: %v", ord, n, err)
		}
		t.Logf("OpenDevice(%d) -> %v", ord, err)
	}
}

// TestMemBindsItsOwnContext: cuMemGetInfo answers for the current context, so
// Mem must bind its own. With one card the wrong-card half cannot be shown,
// but the binding can: with the current context cleared, an unbound Mem fails
// with CUDA_ERROR_INVALID_CONTEXT, and a bound one answers and leaves its own
// device current.
func TestMemBindsItsOwnContext(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip("no CUDA device:", err)
	}
	defer d.Close()

	free0, total0, err := d.Mem()
	if err != nil {
		t.Fatalf("Mem with the context current: %v", err)
	}

	// Open locked this goroutine to its thread, so this is the context's own
	// thread, the only place it can be taken away.
	if err := call(cuCtxSetCurrent(0), "cuCtxSetCurrent"); err != nil {
		t.Fatal(err)
	}
	free1, total1, err := d.Mem()
	if err != nil {
		t.Fatalf("Mem with NO current context: %v", err)
	}
	if total1 != total0 {
		t.Errorf("total %d then %d: Mem answered for two different devices", total0, total1)
	}
	if free1 == 0 {
		t.Errorf("free came back 0 (was %d)", free0)
	}
	var cur CUdevice
	if r := cuCtxGetDevice(&cur); r != 0 {
		t.Fatalf("cuCtxGetDevice: %s", errStr(r))
	}
	if cur != d.dev {
		t.Errorf("Mem left device %d current, this Device is %d (#%d)", cur, d.dev, d.ord)
	}
	t.Logf("#%d %s: %.2f of %.2f GiB free, asked with no context current",
		d.Ordinal(), d.Name(), float64(free1)/(1<<30), float64(total1)/(1<<30))
}

// TestFailedOpenDoesNotExitLocked: OpenDevice locks its caller's OS thread
// before it knows whether it will succeed, so every failure must unlock. A
// goroutine that exits locked has its thread destroyed without glibc teardown,
// which crashes the next thread the runtime builds; so this makes the failures
// and then forces thread creation. The gate is the crash (the violation
// segfaults the binary); the thread count is only logged, since it is noisy.
// It needs no GPU: the failing open is an impossible ordinal.
func TestFailedOpenDoesNotExitLocked(t *testing.T) {
	const n = 40
	fail := func() {
		done := make(chan struct{})
		go func() {
			defer close(done)
			if d, err := OpenDevice(-1); err == nil {
				d.Close()
				t.Error("OpenDevice(-1) opened a device")
			}
		}()
		<-done
	}
	before := pprof.Lookup("threadcreate").Count()
	for i := 0; i < n; i++ {
		fail()
	}
	t.Logf("%d failed opens, threadcreate moved by %d (a number, not the gate)",
		n, pprof.Lookup("threadcreate").Count()-before)

	// Each of these locks its thread, so the runtime must build real threads,
	// which is where a half-dismantled pthread would land.
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			s := 0
			for j := 0; j < 1<<16; j++ {
				s += j
			}
			runtime.Gosched()
			_ = s
		}()
	}
	wg.Wait()
}

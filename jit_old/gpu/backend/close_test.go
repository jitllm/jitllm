package backend

import (
	"github.com/jitllm/jitllm/internal/testdevices"
	"runtime"
	"sync"
	"testing"
)

// TestCloseThenSpawnThreads is the regression for closing one backend
// corrupting thread creation for the whole process: the CUDA owner goroutine
// exited with its OS thread still locked, Go destroyed that thread with a raw
// exit that skips glibc's teardown (a real pthread under goffi's fakecgo), and
// the next thread the runtime created crashed.
//
// The test opens everything, closes all but one, then forces the runtime to
// build threads.
func TestCloseThenSpawnThreads(t *testing.T) {
	devs := Open()
	if len(devs) == 0 {
		t.Skip("no GPU on this host")
	}
	for _, d := range devs[1:] {
		d.Close()
	}
	devs[0].Close()

	// Every one of these locks its thread, so the runtime must create real
	// threads rather than reuse one.
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

// TestAClosedCUDADeviceRefuses: a closed device drops the work it is handed
// rather than hang (cudaDev.do), which is right for a release and was wrong
// for every call that answers something -- Alloc handed back a buffer with no
// memory behind it, and the first Copy into it dereferenced nil. Each such
// call must refuse, and a release must still be quiet.
func TestAClosedCUDADeviceRefuses(t *testing.T) {
	d, err := OpenCUDA(testdevices.Ordinal(t, "cuda", 0))
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	b, err := d.Alloc(64)
	if err != nil {
		t.Fatal(err)
	}
	d.Close()
	p := make([]byte, 64)
	if nb, err := d.Alloc(64); err == nil {
		t.Fatalf("Alloc on a closed device answered %v and no error", nb)
	}
	for name, err := range map[string]error{
		"Write":   b.Write(p),
		"WriteAt": b.WriteAt(0, p),
		"Read":    b.Read(p),
		"Copy":    d.Copy(b, 0, b, 32, 32),
	} {
		if err == nil {
			t.Errorf("%s on a closed device reported success", name)
		}
	}
	if _, _, err := d.Mem(); err == nil {
		t.Error("Mem on a closed device reported success")
	}
	b.Free()
}

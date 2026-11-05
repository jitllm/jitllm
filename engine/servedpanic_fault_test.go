//go:build jitllmfault

package engine

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestServedPanicFailsTheStep panics inside a served step's submission on
// every device (JITLLM_DEVICE_PANIC_AT, the prompt's submission and a
// decode step's): the step returns the panic as the device's failure, the
// Instance finishes the steps, and Close returns, where a submission that
// panicked left its ticket in flight and Close waited on it forever.
//
// Build with -tags jitllmfault; the tier reads JITLLM_DEVICE_PANIC_AT at Open.
func TestServedPanicFailsTheStep(t *testing.T) {
	p := testmodels.Path("stories15M-q8_0.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("MODEL MISSING: %v (set JITLLM_MODELS)", err)
	}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("Once upon a time", true)
	ran := 0
	for _, at := range []string{"1", "3"} {
		for _, dev := range stepDevices() {
			t.Run(dev+"/"+at, func(t *testing.T) {
				t.Setenv("JITLLM_DEVICE_PANIC_AT", at)
				g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				s := testInstance(t, m, len(ids)+8)
				if err := s.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				if s.GPULayers() != m.Cfg.NLayer {
					t.Fatalf("the device took %d of %d blocks", s.GPULayers(), m.Cfg.NLayer)
				}
				ran++
				done := make(chan error, 1)
				go func() {
					// A panic that reaches the caller is caught here, so the
					// Close below is the caller's after it, as a deferred one is.
					defer func() {
						if r := recover(); r != nil {
							done <- fmt.Errorf("the step panicked: %v", r)
						}
					}()
					_, err := s.Prefill(ids)
					for i := 0; err == nil && i < 4; i++ {
						_, err = s.Forward(ids[i])
					}
					done <- err
				}()
				select {
				case err := <-done:
					if err != nil && strings.Contains(err.Error(), "the step panicked") {
						t.Error(err)
					}
					t.Logf("the steps ended with %v; %d demotion(s), the device's last error %q", err, s.DeviceDemotions(), g.Err())
				case <-time.After(5 * time.Minute):
					t.Fatal("the steps did not end")
				}
				if !strings.Contains(g.Err(), "panicked") {
					t.Fatalf("the device's last error is %q: the panic did not fail the step", g.Err())
				}
				closed := make(chan struct{})
				go func() {
					s.Close()
					close(closed)
				}()
				select {
				case <-closed:
				case <-time.After(time.Minute):
					t.Fatal("Instance.Close did not return after a submission panicked")
				}
			})
		}
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}

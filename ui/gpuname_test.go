package main

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/crash"
)

// gogpu's own record of the adapter it picked names the GPU in the crash
// reports, backend included, and still reaches the log. Against the violation
// (the tap passing records on untouched) the GPU stays unnamed.
func TestTheAdapterGogpuPicksNamesTheGPU(t *testing.T) {
	var out bytes.Buffer
	var guard crash.Guard
	l := slog.New(adapterTap{slog.NewTextHandler(&out, nil), &guard})
	l.Info("adapter selected", "name", "Microsoft Basic Render Driver", "backend", "DX12", "type", "IntegratedGPU")
	if got, want := guard.GPU(), "Microsoft Basic Render Driver (DX12, IntegratedGPU)"; got != want {
		t.Errorf("crash reports name the GPU %q, want %q", got, want)
	}
	if !strings.Contains(out.String(), "adapter selected") {
		t.Errorf("the record did not reach the log: %q", out.String())
	}
}

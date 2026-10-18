package model

import (
	"cmp"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDeviceProgramCoverage reports, for every synthetic fixture whose
// blocks a device takes, whether its first device block runs as its device program, and
// where it does not, why: the work list the block encoder's deletion (D5)
// drives to empty. A report until then; at D5 it fails while any fixture
// still runs the encoder.
func TestDeviceProgramCoverage(t *testing.T) {
	g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("cuda")}, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, "cuda", err)
	}
	defer g.Close()
	paths, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "synth-*.jlm"))
	reasons := map[string][]string{}
	covered, tried := 0, 0
	for _, p := range paths {
		name := strings.TrimSuffix(filepath.Base(p), ".jlm")
		if strings.HasSuffix(name, "-vlm") || strings.Contains(name, "mmproj") {
			continue
		}
		m, err := Open(p, noTune)
		if err != nil {
			continue
		}
		func() {
			defer m.Close()
			if m.IsEncoder() {
				return
			}
			s := m.NewState(32)
			defer s.Close()
			if err := s.SetDevice(g); err != nil || s.GPULayers() == 0 {
				return
			}
			tried++
			if _, err := s.Prefill(m.Vocab.Encode("Hello", true)); err != nil {
				reasons["prefill: "+err.Error()] = append(reasons["prefill: "+err.Error()], name)
				return
			}
			// The first block the device holds (a sliding or shared layer
			// may leave block 0 home).
			first := 0
			if runs := s.DeviceBlocks(); len(runs) > 0 {
				first = runs[0][0]
			}
			prog, why := g.BlockProgram(first)
			if prog != "" && why == "" {
				covered++
				return
			}
			if why == "" {
				why = fmt.Sprintf("(no program, no reason: block %d of runs %v, %d device blocks; declines %v; device %q)", first, s.DeviceBlocks(), s.GPULayers(), s.DeviceDeclines(), s.devWhy())
			}
			reasons[why] = append(reasons[why], name)
		}()
		g.ReleaseLayers(0, 1<<20)
	}
	keys := make([]string, 0, len(reasons))
	for k := range reasons {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b string) int { return cmp.Compare(len(reasons[b]), len(reasons[a])) })
	t.Logf("%d of %d fixtures run their first device block as a device program", covered, tried)
	for _, k := range keys {
		t.Logf("%3d  %s: %s", len(reasons[k]), k, strings.Join(reasons[k], " "))
	}
}

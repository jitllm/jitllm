//go:build jitllmfault && (amd64 || arm64)

package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSpecSoftcapGateDiscriminates runs TestSpecOverAFinalSoftcap's device
// comparison with the per-row head skipping the final softcap ("nocap") and
// demands it fail by far: the cap a speculation step's verify rows carry must
// be what the gate compares. The host's break (rowsOut.noCap) runs in the
// untagged gate itself.
//
//	go test -tags jitllmfault ./engine/model -run TestSpecSoftcapGateDiscriminates
func TestSpecSoftcapGateDiscriminates(t *testing.T) {
	for _, path := range specCapModels() {
		name := strings.TrimSuffix(path[strings.LastIndex(path, "/")+1:], ".gguf")
		t.Run(name, func(t *testing.T) {
			m, _ := openSpecCapped(t, path)
			defer m.Close()
			prompt := m.Vocab.Encode(specPrompts[0], true)
			want := plainGreedy(t, m, prompt, 32, nil)
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					bound := specCapBound(t, m, spec, prompt, want)
					g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
						tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					g.SetFault("rows", "nocap")
					bad := specVerifyCapped(t, m, g, prompt, want, false)
					t.Logf("nocap on %s: verify rows against decode, worst logit NMSE %.3e, %.0fx the bound %.3e",
						spec, bad, bad/bound, bound)
					if !(bad > 10*bound) {
						t.Fatalf("the gate passed with the verify rows uncapped on %s: NMSE %.3e, bound %.3e",
							spec, bad, bound)
					}
				})
			}
		})
	}
}

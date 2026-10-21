package tier_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestDeviceBlocksDoNotStayHostResident: a block is a page placed on one
// resource, so a block uploaded to the device must give up its host frame; a
// fully offloaded model once kept a second copy on the host.
//
// It also checks that loading faulted no page: the per-block norms live in
// the dense region, so build() resolves shapes without reading pages, which is
// what lets placement happen before loading.
//
// Pages are the observable, not RSS: a released page is garbage, and when it
// shows up is the collector's decision.
func TestDeviceBlocksDoNotStayHostResident(t *testing.T) {
	src := testmodels.Path("Qwen3-1.7B-Q4_K_M.gguf")
	if _, err := os.Stat(src); err != nil {
		t.Skipf("model not present: %s (set JITLLM_MODELS to the model directory)", src)
	}
	dst := filepath.Join(t.TempDir(), "m"+jlm.Ext)
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatalf("convert: %v", err)
	}

	m, err := model.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if n := m.HostResidentBlocks(); n != 0 {
		t.Errorf("Open left %d of %d blocks host-resident; build() should fault none",
			n, m.Cfg.NLayer)
	}

	st := m.NewState(64)
	defer st.Close()
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"))
	if err != nil || g == nil {
		t.Skipf("no accelerator: %v", err)
	}
	defer g.Close()
	st.SetDevice(g)
	placed := st.GPULayers()
	if placed == 0 {
		t.Skip("the device took no blocks")
	}
	// Every block the device took must have given up its host page. Blocks the
	// device declined are the host's and may be resident.
	if got, max := m.HostResidentBlocks(), m.Cfg.NLayer-placed; got > max {
		t.Errorf("%d blocks are on the device, yet %d of %d are still host-resident "+
			"(at most %d should be) -- the page was copied, not placed",
			placed, got, m.Cfg.NLayer, max)
	}
	t.Logf("%d of %d blocks on the device, %d host-resident",
		placed, m.Cfg.NLayer, m.HostResidentBlocks())
}

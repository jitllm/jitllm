//go:build !jitllmtest

package tier

import (
	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
)

// The release build: no shadow (shadow.go). A device's shadow is always nil,
// so each of the emit's checks of it is false and none of these runs.

type shadowTier struct{}

func openShadow() *shadowTier { return nil }

func (*shadowTier) close() {}

func (*shadowTier) step(int) {}

func (*shadowTier) block(bool) {}

func (*shadowTier) capture(*devTier, backend.Session, *layer, int, *nn.LayerPlan, backend.Buf, backend.Buf, int, string) {
}

func (*shadowTier) attnIn(*devTier, *launcher, backend.Session, *layer, int, *nn.LayerPlan, backend.Buf, backend.Buf, backend.Buf, backend.Buf, int, int, string) {
}

func (*shadowTier) postAttention(*devTier, *launcher, backend.Session, *layer, int, *nn.LayerPlan, int, backend.Buf, backend.Buf, backend.Buf) {
}

func (*shadowTier) query(*devTier, *launcher, backend.Session, *layer, int, *nn.LayerPlan, backend.Buf, backend.Buf, backend.Buf, backend.Buf, int, string) {
}

func (*shadowTier) keyAttend(*devTier, *launcher, backend.Session, *layer, int, *nn.LayerPlan, *kvLayerPool,
	backend.Buf, backend.Buf, backend.Buf, backend.Buf, backend.Buf, backend.Buf, backend.Buf, backend.Buf, backend.Buf, int, int, bool, string) {
}

func (*shadowTier) head(*devTier, *launcher, backend.Session, *nn.LayerPlan, backend.Buf, backend.Buf, *resident, backend.Buf, string) {
}

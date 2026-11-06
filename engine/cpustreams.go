package engine

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// blockWeights is a part's weights as block li reads them (l its copy, nil
// for the model's dense tensors): where the device binds a part's form.
type blockWeights struct {
	s  *Instance
	li int
	l  *model.Layer
}

func (b blockWeights) Bind(f *cpudev.Form) ([]unsafe.Pointer, nn.Decline) {
	return b.s.bindWeights(f, b.li, b.l)
}

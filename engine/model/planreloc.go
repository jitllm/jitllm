package model

import (
	"errors"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit"
)

// SetPlanDevice moves a planned sequence to dev, gpu the composer's
// description of it, mid-sequence: its blocks, KV pages and recurrent state
// with it, the next token the same answer. Not built: the plan runs on the
// device it opened on.
func (s *State) SetPlanDevice(dev backend.Device, gpu jit.Device) error {
	return errors.New("model: moving a planned sequence to another device is not built: the plan runs on the device it opened on")
}

// SetPlanLayers places a planned sequence's first n blocks on the plan's
// device and the rest on the CPU device, mid-sequence, their KV pages with
// them. Not built: every block of the plan is on the one device.
func (s *State) SetPlanLayers(n int) error {
	return errors.New("model: placing a planned sequence's blocks on the CPU device is not built: every block of the plan is on its one device")
}

// PlanLayers is how many of a planned sequence's blocks are on the plan's
// device; the rest are on the CPU device.
func (s *State) PlanLayers() int {
	if s.planSess == nil {
		return 0
	}
	return s.c.NLayer
}

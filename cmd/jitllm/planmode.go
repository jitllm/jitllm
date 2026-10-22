package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/jit"
)

// planDevice is JITLLM_PLAN_DEVICE, API:ORDINAL (ptx:0, spirv:1, msl:0):
// the model decodes through the composer's plan on that device
// (model.WithPlanDevice), the device described to the composer by what it
// states of itself. Unset, nil.
func planDevice() (model.Option, error) {
	v := os.Getenv("JITLLM_PLAN_DEVICE")
	if v == "" {
		return nil, nil
	}
	api, ord, _ := strings.Cut(v, ":")
	n, err := strconv.Atoi(ord)
	if err != nil {
		return nil, fmt.Errorf("JITLLM_PLAN_DEVICE=%s: want API:ORDINAL", v)
	}
	var dev backend.Device
	i := 0
	for _, d := range backend.Open() {
		if d.API() == api && i == n && dev == nil {
			dev = d
			continue
		}
		if d.API() == api {
			i++
		}
		d.Close()
	}
	if dev == nil {
		return nil, fmt.Errorf("JITLLM_PLAN_DEVICE=%s: no such device", v)
	}
	// A device that states no resident thread count takes the composer's
	// default, a V100's 80 units of 2048.
	slots := dev.Slots()
	if slots <= 0 {
		slots = 80 * 2048
	}
	return model.WithPlanDevice(dev, jit.Device{Kind: jit.DeviceGPU, Slots: slots, Lanes: 32}), nil
}

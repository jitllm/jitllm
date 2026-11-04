package main

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/model"
)

// registerReport prints what each composed kernel cost the CPU and each
// device, worst first: spill, then occupancy (or vector registers named),
// then registers.
func registerReport(w io.Writer, host []model.CPUKernel, devs []gpu.DeviceKernels, asJSON bool) error {
	sort.SliceStable(host, func(a, b int) bool {
		if host[a].Stack != host[b].Stack {
			return host[a].Stack > host[b].Stack
		}
		return host[a].Vec > host[b].Vec
	})
	for i := range devs {
		ks := devs[i].Kernels
		sort.SliceStable(ks, func(a, b int) bool {
			if ks[a].SpillBytes != ks[b].SpillBytes {
				return ks[a].SpillBytes > ks[b].SpillBytes
			}
			if oa, ob := ks[a].Occupancy(), ks[b].Occupancy(); oa != ob {
				return oa < ob
			}
			return ks[a].Regs > ks[b].Regs
		})
	}
	if asJSON {
		e := json.NewEncoder(w)
		e.SetIndent("", "  ")
		return e.Encode(struct {
			Tier    string              `json:"cpu_tier"`
			Host    []model.CPUKernel   `json:"cpu"`
			Devices []gpu.DeviceKernels `json:"devices"`
		}{cpu.New().Tier().String(), host, devs})
	}
	fmt.Fprintf(w, "\ncpu (%s)\n  %-48s %8s %6s %7s %5s\n", cpu.New().Tier(), "kernel", "vec/file", "stack", "bytes", "forms")
	for _, k := range host {
		vec := "-"
		if k.Vec >= 0 {
			vec = fmt.Sprintf("%d/%d", k.Vec, k.File)
		}
		fmt.Fprintf(w, "  %-48s %8s %6d %7d %5d\n", k.Name, vec, k.Stack, k.Bytes, k.Forms)
	}
	for _, d := range devs {
		fmt.Fprintf(w, "\n%s\n", d.Device)
		if !d.Known {
			fmt.Fprintf(w, "  this backend does not report its compiler's register allocation\n")
			continue
		}
		fmt.Fprintf(w, "  %-44s %5s %5s %6s %7s %5s %9s %6s %4s\n",
			"kernel", "group", "regs", "spill", "shared", "res", "occupancy", "shapes", "ran")
		for _, k := range d.Kernels {
			ran := "yes"
			if !k.Ran {
				ran = "no"
			}
			fmt.Fprintf(w, "  %-44s %5d %5d %6d %7d %5d %8.0f%% %6d %4s\n",
				k.Name, k.Group, k.Regs, k.SpillBytes, k.SharedBytes, k.Resident, 100*k.Occupancy(), k.Compiles, ran)
		}
	}
	return nil
}

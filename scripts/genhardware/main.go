// Command genhardware measures this machine into the estimator's hardware
// table:
//
//	go run ./scripts/genhardware -probe engine/cost/hardware_table.go
//
// probes the host and each device with a clock to time kernels on, once per
// hardware model, puts each profile in the table in place of the entry of
// its name and backend or beside the others, and writes the table. A machine
// without the source runs the compiled command, which carries the table it
// was built with, and its table comes back.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

func main() {
	measure := flag.Bool("probe", false, "probe this machine into the table")
	flag.Parse()
	if !*measure || flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: genhardware -probe TABLE")
		os.Exit(2)
	}
	ps, err := probeMachine()
	if err == nil {
		var b []byte
		if b, err = cost.RenderHardware(cost.Merge(cost.Hardware(), ps)); err == nil {
			err = os.WriteFile(flag.Arg(0), b, 0o644)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// probeMachine is this machine's profiles: its host, and each device with a
// clock to time kernels on, once per hardware model.
func probeMachine() ([]cost.Profile, error) {
	var out []cost.Profile
	if name, err := probe.HostName(); err != nil {
		fmt.Fprintln(os.Stderr, "host:", err)
	} else if d, err := probe.Host(); err != nil {
		fmt.Fprintln(os.Stderr, "host:", err)
	} else {
		out = append(out, cost.Profile{Name: name, API: runtime.GOARCH, Source: "probe", Device: d})
	}
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	seen := map[string]bool{}
	for _, dev := range devs {
		name := probe.GPUName(dev)
		if _, ok := probe.Timer(dev); !ok || seen[name+dev.API()] {
			continue
		}
		seen[name+dev.API()] = true
		d, err := probe.GPU(dev)
		if err != nil {
			fmt.Fprintln(os.Stderr, name+":", err)
			continue
		}
		out = append(out, cost.Profile{Name: name, API: dev.API(), Source: "probe", Device: d})
		fmt.Fprintln(os.Stderr, "measured", name, "on", dev.API())
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("genhardware: nothing on this machine was measured")
	}
	return out, nil
}

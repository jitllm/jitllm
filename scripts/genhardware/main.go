// Command genhardware measures machines into the estimator's hardware table:
//
//	go run ./scripts/genhardware -probe engine/cost/hardware_table.go
//
// probes this machine's host and each device with a clock to time kernels
// on, once per hardware model, and merges the profiles into the table: each
// in place of the entry of its hardware model and backend, or beside the
// others. The merge reads the table's syntax, not the table the command was
// built with, so the file given is what it adds to. On a machine without the
// source, the compiled command probes into a new file, which comes back and
// is merged:
//
//	genhardware -probe m4.go                         # on the machine
//	go run ./scripts/genhardware -merge m4.go engine/cost/hardware_table.go
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/probe"
)

func main() {
	measure := flag.Bool("probe", false, "probe this machine into TABLE")
	from := flag.String("merge", "", "merge the profiles of this table file into TABLE")
	host := flag.Bool("host", true, "probe the host CPU too: off on a machine whose CPU is no fixed hardware model, a cloud VM's")
	flag.Parse()
	if *measure == (*from != "") || flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: genhardware -probe TABLE | genhardware -merge FROM TABLE")
		os.Exit(2)
	}
	if err := run(*measure, *host, *from, flag.Arg(0)); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(measure, host bool, from, table string) error {
	var add []byte
	if measure {
		ps, err := probeMachine(host)
		if err != nil {
			return err
		}
		if add, err = cost.RenderHardware(ps); err != nil {
			return err
		}
	} else {
		var err error
		if add, err = os.ReadFile(from); err != nil {
			return err
		}
	}
	into, err := os.ReadFile(table)
	if os.IsNotExist(err) {
		into, err = cost.RenderHardware(nil)
	}
	if err != nil {
		return err
	}
	b, err := merge(into, add)
	if err != nil {
		return err
	}
	return os.WriteFile(table, b, 0o644)
}

// probeMachine is this machine's profiles: its host where host is set, and
// each device with a clock to time kernels on, once per hardware model.
func probeMachine(host bool) ([]cost.Profile, error) {
	var out []cost.Profile
	if host {
		if p, err := probeHost(); err != nil {
			fmt.Fprintln(os.Stderr, "host:", err)
		} else {
			out = append(out, p)
		}
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

// probeHost is the host CPU's profile.
func probeHost() (cost.Profile, error) {
	name, err := probe.HostName()
	if err != nil {
		return cost.Profile{}, err
	}
	d, err := probe.Host()
	if err != nil {
		return cost.Profile{}, err
	}
	return cost.Profile{Name: name, API: runtime.GOARCH, Source: "probe", Device: d}, nil
}

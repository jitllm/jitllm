package gpu

import (
	"fmt"
	"os"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/cost"
	hw "github.com/jitllm/jitllm/engine/probe"
)

// choose picks which GPU API to run on when a host offers more than one.
//
// The answer is the devices' measured profiles (cost.Known), not an
// assumption: which API is faster depends on the card and the driver, and an
// integrated GPU sharing host memory can be slower than the CPU. WithAPI pins
// it; TuneOff takes the first device.
//
// One physical card enumerated by two backends is one candidate, not two:
// backend.APIRank decides between them (see AGENTS.md, Scope), and the
// profiles only compare genuinely different devices.
func choose(devs []backend.Device, kb knobs) backend.Device {
	if len(devs) == 1 {
		return devs[0]
	}
	keep := func(i int) backend.Device {
		for j, d := range devs {
			if j != i {
				d.Close()
			}
		}
		return devs[i]
	}
	if want := kb.api; want != "" {
		for i, d := range devs {
			if d.API() == want {
				return keep(i)
			}
		}
		fmt.Fprintf(os.Stderr, "jitllm: no %q device; using %s\n", want, devs[0].API())
		return keep(0)
	}
	cand, merges := preferOnePerCard(devs)
	if kb.verbose {
		for _, m := range merges {
			fmt.Fprintf(os.Stderr, "jitllm: %s\n", m)
		}
	}
	if len(cand) == 1 || kb.tune == TuneOff {
		return keep(cand[0])
	}
	sub := make([]backend.Device, len(cand))
	for i, j := range cand {
		sub[i] = devs[j]
	}
	best, rates := probe(sub)
	if kb.verbose {
		for i, d := range sub {
			fmt.Fprintf(os.Stderr, "jitllm: profile %-6s %-40s %8.1f GB/s\n", d.API(), d.Name(), rates[i]/1e9)
		}
	}
	return keep(cand[best])
}

// probeSet is the read an unprofiled device is ranked by: past any card's
// last cache, short against opening the device.
const probeSet = 256 << 20

// probe is each device's memory read rate, and the fastest: decode reads
// every weight once a token, so the rate a card's memory serves is the rate it
// decodes at. A device's measured profile states it; one with none is
// measured here (a read of probeSet bytes), and one that cannot be reads 0,
// behind the rest in enumeration order.
func probe(devs []backend.Device) (best int, rates []float64) {
	rates = make([]float64, len(devs))
	for i, d := range devs {
		if p, ok := cost.Known(hw.GPUName(d), d.API()); ok {
			rates[i] = p.Device.Bandwidth
		} else if r, err := hw.ReadRate(d, probeSet, 3); err == nil {
			rates[i] = r
		}
		if rates[i] > rates[best] {
			best = i
		}
	}
	return best, rates
}

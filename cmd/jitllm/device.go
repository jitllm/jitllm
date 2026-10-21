package main

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/engine/sched"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// One binary: importing jit_old/gpu costs only libc and libdl, since the CUDA,
// Vulkan and Metal drivers are dlopen'd. The no-cgo property is untouched:
// goffi reaches the drivers through dlopen.

// openDevices resolves -devices. The flag is plural because a host can have
// several cards and the placement question is "which of them", not "GPU or
// not". tier.ParseDevices is the grammar:
//
//	auto                 one device, measured -- and no GPU is a CPU machine
//	all, gpu             every distinct device on the host, fastest first
//	cpu                  no device at all
//	gpu:N                the Nth device backend.Open reports
//	cuda, vulkan         every device of that backend, fastest first
//	cuda:0, vulkan:1     one named device, by ordinal (CUDA) or index/name (Vulkan)
//	...=3G               that device's weight budget, overriding -vram
//
// auto stays one device on purpose, and a bare backend name or `all` asks for
// the rest: each change of device costs the residual stream a round trip home,
// so the router fills the fastest card first and a second one takes only what
// the first could not hold. A selector pins exactly one device.
//
// -vram 0 (the default) asks each device what it has free and leaves headroom.
// See tier.budgetFor.
//
// streamGroups is the container's own answer (jlm.streamGroupsFor), or 0 to let
// the tier choose; the environment override in gpuOptions runs after this and
// wins.
func openDevices(spec string, vram, hostBudget uint64, streamGroups, stepRows int) (nn.Device, func(), string, error) {
	if hostBudget == 0 {
		hostBudget = sched.MemBudget()
	}
	es, err := tier.ParseDevices(spec)
	if err != nil {
		return nil, nil, "", err
	}
	// cpu alone is the one spec that opens nothing. Beside a device it is
	// explicit rather than empty -- the host always runs what no device took.
	device := false
	for _, e := range es {
		device = device || e.Device()
	}
	if !device {
		return nil, func() {}, "cpu", nil
	}
	base := []tier.Option{
		tier.WithDevices(spec),
		tier.WithBudget(vram),
		tier.WithHostBudget(hostBudget),
	}
	if streamGroups > 0 {
		base = append(base, tier.WithConfig(func(c *tier.Config) { c.StreamGroupsStart = streamGroups }))
	}
	if stepRows > 0 {
		base = append(base, tier.WithConfig(func(c *tier.Config) { c.StepRows = stepRows }))
	}
	g, err := tier.OpenWith(append(base, gpuOptions()...)...)
	if err != nil {
		// auto is the only spec for which a missing GPU is not an error:
		// anything that names a device asked for that device.
		if len(es) == 1 && es[0].API == "auto" {
			return nil, func() {}, "cpu (" + err.Error() + ")", nil
		}
		return nil, nil, "", err
	}
	return g, g.Close, g.Name(), nil
}

// relocatable refuses -relocate on a device spec that leaves the host out.
// Relocating moves blocks to the CPU, and a spec that names only devices asked
// for them and nothing else -- the same reading openDevices gives it.
func relocatable(spec string) error {
	ok, err := tier.AdmitsHost(spec)
	if err != nil || ok {
		return err
	}
	return fmt.Errorf("-relocate moves blocks to the host, and -devices %q leaves it out: "+
		"use auto, or add cpu (-devices %s,cpu)", spec, spec)
}

// deviceStats is what the tier will admit about a run, which matters because
// "over the configured budget" and "the card is full" are opposite facts.
func deviceStats(d nn.Device) string {
	g, ok := d.(*tier.GPU)
	if !ok {
		return ""
	}
	st := g.Stats()
	s := fmt.Sprintf("%.2f GiB resident", float64(g.Bytes())/(1<<30))
	// The decode row tile, printed whenever it is on -- "it made no difference"
	// and "it never ran" are otherwise the same number.
	if st.RowtTiles > 0 || st.RowtPlain > 0 {
		s += fmt.Sprintf(", rowt %d tiled / %d plain", st.RowtTiles, st.RowtPlain)
	}
	if st.TileMV > 0 || st.TileWhy != "" {
		s += fmt.Sprintf(", %d collective-tile prompt matvecs (%d f16 conversions, %d quantizes skipped)",
			st.TileMV, st.ActF16Launches, st.QuantSkipped)
		if st.TileWhy != "" {
			s += " -- refused: " + st.TileWhy
		}
	}
	// Whether the decode replayed a captured graph or issued its launches one
	// by one: the same tokens either way, a different per-token cost.
	if st.Captures > 0 {
		s += fmt.Sprintf(", %d graph capture(s)", st.Captures)
	}
	if st.RagGroupMV > 0 {
		s += fmt.Sprintf(", %d few-sequence decode matvecs", st.RagGroupMV)
	}
	if st.VoltaMV > 0 {
		s += fmt.Sprintf(", %d sm_70 f16 tensor-core matvecs (%d GemmVolta; %d f16 conversions, %d quantizes skipped)",
			st.VoltaMV, st.VoltaGemm, st.ActF16Launches, st.QuantSkipped)
	}
	// Which kernel a batched mixture's experts ran on: "it made no difference"
	// and "it never ran" are otherwise the same number.
	if st.GroupedMoE > 0 {
		s += fmt.Sprintf(", %d grouped mixture block(s), %d expert matvecs as a binary16 GEMM (sm_70 tensor cores or Metal simdgroup_matrix)",
			st.GroupedMoE, st.GroupedVolta)
	}
	if st.MLABatched > 0 {
		s += fmt.Sprintf(", %d batched latent block(s), %d scores and %d accumulates on sm_70 tensor cores",
			st.MLABatched, st.MLAScores70, st.MLAAcc70)
	}
	// Why the rotary table is uploaded from the host, when it is: the
	// fallback is correct, so the absence is invisible unless printed. The
	// counters are printed at the end of the run (main.go), since this runs at
	// placement, before any token.
	if st.RopeTableWhy != "" {
		s += fmt.Sprintf("\n          rotary table on the host: %s", st.RopeTableWhy)
	}
	// With more than one device, say what each took: the total cannot tell
	// "both took half" from "the second took nothing".
	if g.Devices() > 1 {
		per := g.DevStats()
		for i, n := range g.Placed() {
			s += fmt.Sprintf("\n          %-30s %d block(s), %d matvec(s)",
				per[i].Device, n, per[i].Served)
		}
		if c := g.Crossings(); c > 0 {
			s += fmt.Sprintf("\n          %d device change(s) per token: the residual stream "+
				"comes home and goes out again at each one", c)
		}
	}
	if st.NoRoom > 0 {
		s += fmt.Sprintf("; %d tensors the DEVICE refused (out of memory), so those matvecs "+
			"ran on the CPU and this rate is not comparable with a run that had the card to itself",
			st.NoRoom)
	}
	// Device KV cache competes with the weights and scales with -n; unlike the
	// host's, it is a real allocation the moment a block is taken.
	if st.KVBytes > 0 {
		s += fmt.Sprintf(" (%.2f GiB of it KV cache, which scales with -n)",
			float64(st.KVBytes)/(1<<30))
	}
	// Load is pack-or-(upload-then-unpack), so all three terms are printed. The
	// end-of-run "device cost" line prints the same three over the whole run;
	// the difference is what the tokens paid.
	if st.TPack > 0 || st.TUpload > 0 || st.TUnpack > 0 {
		s += fmt.Sprintf("; prep %.0f ms pack + %.0f ms upload + %.0f ms device unpack",
			float64(st.TPack.Microseconds())/1000, float64(st.TUpload.Microseconds())/1000,
			float64(st.TUnpack.Microseconds())/1000)
	}
	if st.Declined > 0 {
		s += fmt.Sprintf("; %d over the -vram budget (%s)", st.Declined, st.DeclineWhy)
	}
	// A paging device uploads blocks every token, which otherwise looks like a
	// slow kernel.
	if st.SessionDeclines > 0 {
		s += fmt.Sprintf("; %d refusal(s) of a later session's history (tier.Config.Sessions "+
			"reserves room for it)", st.SessionDeclines)
	}
	if st.PageIns > 0 || st.PageOuts > 0 {
		s += fmt.Sprintf("; %d slot(s), %d page-in(s), %d page-out(s)", st.Slots, st.PageIns, st.PageOuts)
	}
	return s
}

// hostReserved is how much of the host's weight budget the tier may fill with
// device weights: nonzero only when a device's memory IS the host's.
func hostReserved(d nn.Device) uint64 {
	if g, ok := d.(*tier.GPU); ok {
		return g.HostReserved()
	}
	return 0
}

// deviceBudgets is one line per device: what it may hold, where that number came
// from, and which heap it comes out of.
//
// The pool is named on every line: two 6 GiB budgets are 12 GiB on two cards
// but come off the host's own budget for an integrated GPU.
func deviceBudgets(d nn.Device) string {
	g, ok := d.(*tier.GPU)
	if !ok {
		return ""
	}
	var b strings.Builder
	for _, x := range g.Budgets() {
		fmt.Fprintf(&b, "\n          %-30s budget %6.2f GiB  (%s)", x.Device,
			float64(x.Limit)/(1<<30), x.Why)
		if x.Pool != "" {
			fmt.Fprintf(&b, "\n          %-30s   from pool %q, %.2f GiB", "",
				x.Pool, float64(x.PoolLimit)/(1<<30))
			if x.Host {
				fmt.Fprint(&b, " -- the HOST's own memory, taken off its weight budget")
			}
		}
	}
	return b.String()
}

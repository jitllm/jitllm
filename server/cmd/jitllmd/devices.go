package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// cmdDevices is DeviceService: what hardware the daemon found, and how many
// bytes are actually spendable on it. It prints the server's spendable_total
// rather than summing the list itself: an integrated GPU's heap is system RAM,
// and one card seen by two backends (same_device_as) is one budget.
func cmdDevices(ctx context.Context, c cli, args []string) error {
	fs := flag.NewFlagSet("jitllmd devices", flag.ContinueOnError)
	addr := addrFlag(fs)
	if err := parse(c, fs, args); err != nil {
		return err
	}
	cn, err := target(ctx, c, fs, *addr)
	if err != nil {
		return err
	}

	if id := strings.Join(fs.Args(), " "); id != "" {
		resp, err := cn.Device.GetDevice(ctx, connect.NewRequest(&v1.GetDeviceRequest{DeviceId: id}))
		if err != nil {
			return clientError("describing "+id, cn.base, err)
		}
		printDevice(c.out, resp.Msg.GetDevice(), true)
		return nil
	}

	resp, err := cn.Device.GetMemoryTopology(ctx, connect.NewRequest(&v1.GetMemoryTopologyRequest{}))
	if err != nil {
		return clientError("reading the memory topology", cn.base, err)
	}
	t := resp.Msg.GetTopology()
	printHost(c.out, t.GetHost())
	fmt.Fprintln(c.out)
	for _, d := range t.GetDevices() {
		printDevice(c.out, d, false)
	}
	fmt.Fprintf(c.out, "\nspendable %s   host weight budget plus every PHYSICAL device whose heap is\n"+
		"                     NOT host memory, counted once however many backends enumerate it\n",
		bytesOf(t.GetSpendableTotal()))
	return nil
}

func printHost(w io.Writer, h *v1.HostMemory) {
	if h == nil {
		return
	}
	// The weight budget is a fraction of the cgroup, not the machine's
	// memory; printing both shows whether a model will page before it loads.
	fmt.Fprintf(w, "host     weight budget %s of %s total, %s available\n",
		bytesOf(h.GetWeightBudget()), bytesOf(h.GetTotal()), bytesOf(h.GetAvailable()))
	fmt.Fprintf(w, "         %d decode core(s) from %dP + %dE, %d SMT sibling(s)\n",
		h.GetDecodeCores(), h.GetPerformanceCores(), h.GetEfficiencyCores(), h.GetSmtSiblings())
}

func printDevice(w io.Writer, d *v1.Device, verbose bool) {
	if d == nil {
		return
	}
	id := d.GetRef().GetId()
	fmt.Fprintf(w, "%-9s %-30s %-11s %10s total, %10s free\n",
		id, trunc(d.GetName(), 30), deviceKindName(d.GetKind()),
		bytesOf(d.GetTotalMemory()), bytesOf(d.GetFreeMemory()))

	if d.GetCountsTowardHostBudget() && d.GetKind() != v1.DeviceKind_DEVICE_KIND_CPU {
		fmt.Fprintf(w, "          ★ this heap IS host memory -- SUBTRACT it from the host budget, "+
			"never add it as a second pool\n")
	}
	// A duplicate is printed with the reason it is not counted; it remains a
	// legitimate -devices target.
	if o := d.GetSameDeviceAs(); o != "" {
		fmt.Fprintf(w, "          ★ the SAME physical device as %s -- listed, and NOT counted in "+
			"spendable (%s)\n", o, d.GetPhysicalId())
	}
	if !d.GetAvailable() {
		// An unavailable device is listed with its reason rather than
		// disappearing.
		fmt.Fprintf(w, "          unavailable: %s\n", d.GetUnavailableReason())
		return
	}
	fmt.Fprintf(w, "          %s   %d session(s) attached, %d running, %d queued\n",
		execName(d.GetExecution()), d.GetAttachedSessions(),
		d.GetRunningSessions(), d.GetQueuedRequests())
	if note := d.GetExecutionNote(); note != "" && verbose {
		fmt.Fprintf(w, "          %s\n", note)
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

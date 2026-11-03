package discover

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/convert/library"
)

func TestParamsReadsTotalAndActive(t *testing.T) {
	for _, c := range []struct {
		in            string
		total, active float64
	}{
		{"8B", 8e9, 8e9}, {"30B-A3B", 30e9, 3e9}, {"360M", 3.6e8, 3.6e8}, {"0.6B", 6e8, 6e8}, {"", 0, 0},
	} {
		if tot, act := Params(c.in); tot != c.total || act != c.active {
			t.Errorf("Params(%q) = %g, %g; want %g, %g", c.in, tot, act, c.total, c.active)
		}
	}
}

// The machine of these cases: a 4 GiB card, 15 GiB of weight budget.
func discoverMachine() session.MachineReport {
	return session.MachineReport{Probed: true, MemBudget: 15 << 30, MemWall: 50e9,
		GPUs: []session.GPUInfo{{Name: "card", Mem: 4 << 30}}}
}

func discoverLibrary() []library.Model {
	return []library.Model{
		{Title: "Tiny", Params: "1B", Bytes: 700 << 20},
		{Title: "Mid", Params: "8B", Bytes: 4 << 30},
		{Title: "Big", Params: "14B", Bytes: 8 << 30},
		{Title: "Huge", Params: "70B", Bytes: 40 << 30},
	}
}

// Fastest leads with the smallest model that fits and Smartest with the
// largest; a model larger than memory is never ranked above one that fits.
func TestTheBalanceRanksTheEndsAndKeepsPagedModelsLast(t *testing.T) {
	first := func(pref int) string {
		return discoverLibrary()[Rank(discoverLibrary(), discoverMachine(), pref)[0]].Title
	}
	if got := first(0); got != "Tiny" {
		t.Errorf("Fastest leads with %s, want Tiny", got)
	}
	if got := first(len(BalanceSteps()) - 1); got != "Big" {
		t.Errorf("Smartest leads with %s, want Big: Huge does not fit", got)
	}
	for pref := range BalanceSteps() {
		o := Rank(discoverLibrary(), discoverMachine(), pref)
		if last := discoverLibrary()[o[len(o)-1]].Title; last != "Huge" {
			t.Errorf("balance %d ranks %s last, want the paged Huge", pref, last)
		}
	}
}

// Decode reads every active weight once a token: the estimate follows the
// measured bandwidth, counts a mixture's active share only, is a floor on the
// GPU, and admits a paged model is disk-bound.
func TestTheSpeedEstimateFollowsTheBandwidth(t *testing.T) {
	ram := library.Model{Params: "8B", Bytes: 5 << 30}
	if got := Speed(ram, discoverMachine()); got != "about 7 tok/s" {
		t.Errorf("an 8B in memory reads %q", got)
	}
	moe := library.Model{Params: "30B-A3B", Bytes: 10 << 30}
	if got := Speed(moe, discoverMachine()); got != "about 37 tok/s" {
		t.Errorf("a 30B-A3B reads %q: only its active tenth is read a token", got)
	}
	if got := Speed(library.Model{Params: "1B", Bytes: 700 << 20}, discoverMachine()); !strings.HasPrefix(got, "over ") {
		t.Errorf("a model on the GPU reads %q, want a floor", got)
	}
	if got := Speed(library.Model{Params: "70B", Bytes: 40 << 30}, discoverMachine()); !strings.Contains(got, "disk") {
		t.Errorf("a paged model reads %q", got)
	}
	if got := Speed(ram, session.MachineReport{Probed: true, MemBudget: 15 << 30}); got != "" {
		t.Errorf("with no measured bandwidth the estimate reads %q, want nothing", got)
	}
}

// "Fits" must say where a model would live on this machine.
//
// A GPU that shares system memory must never make a model fit the GPU that
// does not fit RAM.
func TestFitsSaysWhereAModelWouldLive(t *testing.T) {
	gib := int64(1 << 30)
	mr := session.MachineReport{Probed: true, MemBudget: uint64(24 * gib), GPUs: []session.GPUInfo{
		{Name: "RTX 3050 Ti", Mem: uint64(4 * gib)},
		{Name: "Iris Xe", Mem: uint64(20 * gib), Unified: true},
	}}
	for _, c := range []struct {
		size int64
		want string
	}{
		{1 * gib, "GPU"},
		{8 * gib, "RAM"}, // fits the unified GPU's number, which is not more memory
		{40 * gib, "paged"},
	} {
		if got := EntryFits(catalog.Entry{Size: c.size}, mr); got != c.want {
			t.Errorf("a %d GiB model fits %q, want %q", c.size/gib, got, c.want)
		}
	}
	if got := EntryFits(catalog.Entry{Size: gib}, session.MachineReport{}); got != "--" {
		t.Errorf("before the probe it reads %q, want --", got)
	}
}

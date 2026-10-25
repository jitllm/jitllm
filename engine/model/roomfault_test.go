//go:build jitllmfault

package model

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The paths a device short of room takes rather than failing a step, each
// forced by tier.SetRoomFault and each held to a violation it must not
// survive (RULE 10): a recurrent seat reset from host-written zeros, a
// Scheduler clamped to the batched scratch the device holds (nn.RowsWidther),
// and a step whose recurrent seats do not fit run with the blocks home
// (nn.RowsRoomer, SchedulerStats.RoomHome). Every forced arm's tokens are the
// host's.

// roomBatch is the hybrid fixture's batch on the host: the tokens each forced
// arm must give.
func roomBatch(t *testing.T, m *Model, g *tier.GPU, width int) ([][]int32, SchedulerStats, error) {
	t.Helper()
	sc := NewScheduler(m, 3, 300, WithPromptBudget(width))
	defer sc.Close()
	if g != nil {
		if err := sc.st.SetDeviceLayers(g, -1); err != nil {
			t.Fatal(err)
		}
		if sc.st.GPULayers() == 0 {
			t.Fatalf("no block on the device: %s", g.Err())
		}
	}
	vocab := min(m.Cfg.NVocab, 1000)
	seqs := make([]Seq, 3)
	for i := range seqs {
		seqs[i] = Seq{Prompt: linPrompt(90+40*i, vocab)[i:], MaxTokens: 6}
		sc.Submit(&seqs[i])
	}
	for sc.Pending() > 0 || sc.Live() > 0 {
		done, err := sc.Step()
		if err != nil {
			return nil, sc.Stats(), err
		}
		for _, s := range done {
			if s.Err != nil {
				return nil, sc.Stats(), s.Err
			}
		}
	}
	out := make([][]int32, len(seqs))
	for i := range seqs {
		out[i] = seqs[i].Out
	}
	return out, sc.Stats(), nil
}

func TestRoomShortagesAreSurvived(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{ctx: 320})
	defer m.Close()
	want, _, err := roomBatch(t, m, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	same := func(t *testing.T, got [][]int32) {
		t.Helper()
		for i := range want {
			for j := range want[i] {
				if j >= len(got[i]) || got[i][j] != want[i][j] {
					t.Fatalf("sequence %d: %v on the device, %v on the host", i, got[i], want[i])
				}
			}
		}
	}
	defer tier.SetRoomFault("")

	t.Run("reset-zeros-from-host", func(t *testing.T) {
		// Each arm on a fresh device: a kept zeros buffer from an earlier
		// reset would serve the next without asking for one.
		run := func(fault string) (error, int) {
			g := swaTier(t)
			defer g.Close()
			st := m.NewState(300)
			defer st.Close()
			if err := st.SetDeviceLayers(g, -1); err != nil || st.GPULayers() == 0 {
				t.Fatalf("placement: %v, %d blocks: %s", err, st.GPULayers(), g.Err())
			}
			prompt := linPrompt(40, min(m.Cfg.NVocab, 1000))
			if _, err := st.Prefill(prompt); err != nil {
				t.Fatal(err)
			}
			tier.SetRoomFault(fault)
			st.Reset()
			tier.SetRoomFault("")
			_, err := st.Prefill(prompt)
			return err, g.Stats().RecZeroHost
		}
		err, host := run("nozeros")
		if err != nil {
			t.Fatalf("a reset with no room for the kept zeros: %v", err)
		}
		if host == 0 {
			t.Fatal("no seat was zeroed from the host: the forced path never ran")
		}
		if err, _ := run("nozeros-nohost"); err == nil || !strings.Contains(err.Error(), "could not be reset") {
			t.Fatalf("the violation (no host write) gave %v, want the reset's failure", err)
		}
	})

	t.Run("width-clamped", func(t *testing.T) {
		g := swaTier(t)
		defer g.Close()
		tier.SetRoomFault("narrow")
		got, st, err := roomBatch(t, m, g, 0)
		tier.SetRoomFault("")
		if err != nil {
			t.Fatalf("a device holding only 64-row scratch: %v", err)
		}
		if st.RoomHome != 0 {
			t.Fatalf("%d step(s) came home: the clamp, not the fallback, was to carry it", st.RoomHome)
		}
		// Against the same device stepped at 64 rows by a stated budget: the
		// same steps, so the same tokens (the host is another reduction
		// order, RULE 11c).
		ref, _, err := roomBatch(t, m, g, 64)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ref {
			if !slices.Equal(got[i], ref[i]) {
				t.Fatalf("sequence %d: %v clamped, %v at a stated 64-row budget", i, got[i], ref[i])
			}
		}
		tier.SetRoomFault("narrow-lie")
		_, _, err = roomBatch(t, m, g, 0)
		tier.SetRoomFault("")
		if err == nil {
			t.Fatal("the violation (RowsWidth ignoring the scratch) ran: the clamp is untested")
		}
	})

	t.Run("seats-home", func(t *testing.T) {
		g := swaTier(t)
		defer g.Close()
		tier.SetRoomFault("noseat")
		got, st, err := roomBatch(t, m, g, 0)
		tier.SetRoomFault("")
		if err != nil {
			t.Fatalf("a device with no room for the batch's seats: %v", err)
		}
		if st.RoomHome == 0 {
			t.Fatal("no step came home: the forced path never ran")
		}
		same(t, got)
		tier.SetRoomFault("noseat-lie")
		_, _, err = roomBatch(t, m, g, 0)
		tier.SetRoomFault("")
		if err == nil {
			t.Fatal("the violation (RowsRoom saying there is room) ran: the fallback is untested")
		}
	})

	// A step whose scratch does not build is refused by name: no move gives a
	// kernel the device refused room, so none is made, where reading it as no
	// room sent every block home and the batch ran on the host, green.
	t.Run("build-refused", func(t *testing.T) {
		g := swaTier(t)
		defer g.Close()
		tier.SetRoomFault("nobuild")
		_, st, err := roomBatch(t, m, g, 0)
		tier.SetRoomFault("")
		if err == nil || !strings.Contains(err.Error(), "does not build") {
			t.Fatalf("a step whose twins do not build gave %v, want the step refused by name", err)
		}
		if st.RoomHome != 0 {
			t.Fatalf("%d step(s) came home for a scratch that does not build", st.RoomHome)
		}
		t.Logf("refused: %v", err)
		tier.SetRoomFault("nobuild-room")
		_, st, err = roomBatch(t, m, g, 0)
		tier.SetRoomFault("")
		if err != nil || st.RoomHome == 0 {
			t.Fatalf("the violation (a build failure read as no room) gave %v with %d step(s) home: "+
				"the refusal is untested", err, st.RoomHome)
		}
		t.Logf("the violation ran on the host: %d step(s) home", st.RoomHome)
	})
}

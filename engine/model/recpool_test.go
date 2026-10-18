package model

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestRecurrentPoolKeepsASessionAcrossOthers: a hybrid device keeps every
// session's recurrent state for a block in one pool (tier/recpool.go), so a
// session arriving grows every pool, and the last-seated one leaving shrinks
// them -- each a new pair of buffers, with every other session's state carried
// across. A session that decodes through both must read exactly what it would
// alone: same tier, same blocks, same arithmetic, so the bar is bit equality.
//
// The session under test does not sit at slot 0: one seated ahead of it
// leaves before the last shrink, so that shrink carries a run of seats that
// starts above the pool's first slot, and a carry that lands at the wrong
// offset is a wrong state rather than an accident of addressing from zero.
// Stats.RecBytes is the check that the pools did move: grown by the arrivals,
// back to two slots when they have gone. Stats.RecCarryBytes is the check that
// the states crossed, and backend.HostReads that they crossed on the device:
// every arrival (SetDevice) and departure (Close) reads nothing to the host.
func TestRecurrentPoolKeepsASessionAcrossOthers(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	ids := m.Vocab.Encode("The capital of France is", true)
	const steps = 6
	for _, dev := range stepDevices() {
		for _, shared := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/shared=%v", dev, shared), func(t *testing.T) {
				r := recPoolCase(t, m, dev, shared, ids, steps)
				if r.p >= 0 {
					t.Fatalf("token %d: logit %d is %g after other sessions came and went and %g alone: "+
						"a pool rebuilt around them did not carry this session's state", r.p, r.i, r.got, r.want)
				}
				if r.carried == 0 || r.reads != 0 {
					t.Fatalf("two arrivals and two departures carried %d bytes of recurrent state on the "+
						"device and read %d time(s) to the host: the rebuilds must carry the states, and "+
						"without the host", r.carried, r.reads)
				}
				t.Logf("rebuilds carried %d B on the device, 0 host reads", r.carried)
			})
		}
	}
}

// recPoolResult is the first logit that differs between a session decoded
// alone and with others coming and going (p is -1 when none does), the bytes
// of recurrent state the pools' rebuilds carried across meanwhile, and the
// reads to the host the arrivals and departures made.
type recPoolResult struct {
	p, i           int
	got, want      float32
	carried, reads uint64
}

// recPoolCase decodes one session alone and once with others arriving and
// leaving around it, and compares the two.
func recPoolCase(t *testing.T, m *Model, dev string, shared bool, ids []int32, steps int, extra ...tier.Option) recPoolResult {
	t.Helper()
	// decode prefills a session, then decodes steps tokens teacher-forced by
	// fed (its own greedy choice when fed is nil). before runs ahead of its
	// SetDevice and between after its prompt, when given.
	decode := func(g *tier.GPU, fed []int32, before, between func()) ([][]float32, []int32) {
		if before != nil {
			before()
		}
		st := m.NewState(len(ids) + steps + 1)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.GPULayers() != m.Cfg.NLayer {
			t.Skipf("CARD TOO SMALL: %d of %d blocks -- this gate proved nothing here (%s)",
				st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		if between != nil {
			between()
		}
		var out [][]float32
		var toks []int32
		tok := Greedy(lg)
		for i := range steps {
			if fed != nil {
				tok = fed[i]
			}
			toks = append(toks, tok)
			if lg, err = st.Forward(tok); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
			tok = Greedy(lg)
		}
		return out, toks
	}

	g := stepTier(t, m, dev, shared, extra...)
	alone, fed := decode(g, nil, nil, nil)
	g.Close()

	g = stepTier(t, m, dev, shared, extra...)
	defer g.Close()
	var grown, after, carried, reads uint64
	// noHost runs f and counts what it read to the host.
	noHost := func(f func()) {
		r0, _ := backend.HostReads()
		f()
		r1, _ := backend.HostReads()
		reads += r1 - r0
	}
	// arrive seats another session and prefills pr; leave closes one.
	arrive := func(pr string) *State {
		o := m.NewState(16)
		noHost(func() {
			if err := o.SetDevice(g); err != nil {
				t.Fatal(err)
			}
		})
		if _, err := o.Prefill(m.Vocab.Encode(pr, true)); err != nil {
			t.Fatal(err)
		}
		return o
	}
	leave := func(o *State) {
		noHost(func() {
			if err := o.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
	var ahead *State
	with, _ := decode(g, fed, func() { ahead = arrive("Once upon a time") }, func() {
		// Slots: ahead 0, this session 1. Two arrivals at 2 and 3 grow
		// every pool twice, carrying the run [0,2) and then [0,3); ahead
		// leaving frees slot 0 and shrinks nothing; the last-seated one
		// leaving shrinks to 3, carrying the run [1,3); the next to 2,
		// carrying slot 1 alone.
		two, carry0 := g.Stats().RecBytes, g.Stats().RecCarryBytes
		slot := two / 2
		x, y := arrive("Water boils at"), arrive("1, 2, 3,")
		grown = g.Stats().RecBytes
		leave(ahead)
		leave(y)
		after = g.Stats().RecBytes
		leave(x)
		if g.Stats().RecBytes != two || grown != 4*slot || after != 3*slot {
			t.Fatalf("recurrent bytes %d with two seats, %d with four, %d after the last-seated "+
				"left and %d after the rest: the pools did not grow and shrink by whole seats",
				two, grown, after, g.Stats().RecBytes)
		}
		carried = g.Stats().RecCarryBytes - carry0
	})
	t.Logf("recurrent pools: %d B with four seats, %d B with three", grown, after)
	for p := range alone {
		for i := range alone[p] {
			if with[p][i] != alone[p][i] {
				return recPoolResult{p, i, with[p][i], alone[p][i], carried, reads}
			}
		}
	}
	return recPoolResult{p: -1, carried: carried, reads: reads}
}

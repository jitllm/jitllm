package model

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The serving split is on by default (WithServingSplit): while one session of a
// model decodes, another session's prefill runs on the E-cores' own pool. The
// gates here are the A/B on the M4 (opt-in, a model on the disk), the
// starvation case under a foreign load, a single session that must not change,
// and a model-free gate that the split is taken when it should be and only
// then.

// burners starts n busy loops in processes of their own at default QoS, the
// foreign load of the starvation case, and returns what stops them. Not
// goroutines: those would take the Go scheduler's Ps from the pools under
// test.
func burners(t *testing.T, n int) func() {
	t.Helper()
	var cmds []*exec.Cmd
	for range n {
		c := exec.Command("/usr/bin/yes")
		if err := c.Start(); err != nil {
			t.Fatalf("starting a burner: %v", err)
		}
		cmds = append(cmds, c)
	}
	return func() {
		for _, c := range cmds {
			c.Process.Kill()
			c.Wait()
		}
	}
}

// splitEnv reads the opt-in A/B's variables: the model, the rounds, the E-core
// QoS class to compare (JITLLM_QOS_CLASS, as sys/qos.h spells it) and the
// number of foreign burners.
func splitEnv(t *testing.T) (path string, rounds, burn int) {
	if os.Getenv("JITLLM_SPLIT_AB") == "" {
		t.Skip("set JITLLM_SPLIT_AB=1")
	}
	p := testmodels.Resolve(os.Getenv("JITLLM_MODEL"))
	if p == "" {
		t.Skip("set JITLLM_MODEL")
	}
	rounds = 20
	if n, err := strconv.Atoi(os.Getenv("JITLLM_AB_ROUNDS")); err == nil && n > 0 {
		rounds = n
	}
	if v := os.Getenv("JITLLM_QOS_CLASS"); v != "" {
		c, err := strconv.ParseUint(v, 0, 32)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("E-core QoS class %#x (was %#x)", c, sched.SetEClass(uint32(c)))
	}
	burn, _ = strconv.Atoi(os.Getenv("JITLLM_SPLIT_BURN"))
	return jlmOf(t, p), rounds, burn
}

// TestServingSplitAB is one session decoding while another prefills, on two
// arms of one model, in one process, interleaved (RULE 2), an A/A first and the
// comparison twice:
//
//   - base: WithServingSplit(false), the pools as they were. Both sessions
//     decode on the P-cores and the prefiller prefills on the P and E cores;
//     the sets overlap, so every region of either waits its turn on one lock,
//     and the decoder, never alone on its crew, never fuses.
//   - split: the default. The decoder keeps the P-cores and fuses; the
//     prefiller, seeing a session decoding, prefills on the E-cores' own pool
//     and lock, its threads at the E-core QoS class.
//
// Each turn opens a State per session, prefills the decoder to the depth,
// starts the prefiller on plen-token prompts in a loop, and times run decoded
// tokens while it runs; the prefiller's rate is what it finished in that time.
// Decode ratios are base/split time: above 1, split decodes faster.
//
// With JITLLM_SPLIT_BURN=n, n foreign busy loops run throughout: the
// starvation case. The split arm's prefill must still finish prompts there
// (a prefiller at background QoS finished none under a foreign E-core load).
//
//	JITLLM_SPLIT_AB=1 JITLLM_MODEL=Llama-3.2-1B-Instruct-Q4_K_M.gguf [JITLLM_AB_ROUNDS=20]
//	  [JITLLM_QOS_CLASS=0x11] [JITLLM_SPLIT_BURN=6]
func TestServingSplitAB(t *testing.T) {
	p, rounds, burn := splitEnv(t)
	open := func(o ...Option) *Model {
		m, err := Open(p, append([]Option{noTune}, o...)...)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	mBase, mSplit := open(WithServingSplit(false)), open()
	defer mBase.Close()
	defer mSplit.Close()
	const depth, run, plen = 64, 64, 32
	prompt := make([]int32, depth)
	for i := range prompt {
		prompt[i] = int32(1 + (i*7919)%1000)
	}
	type result struct {
		dec   time.Duration
		pre   float64 // the prefiller's tokens a second over the decode
		fused int64
		onE   int64
	}
	type arm struct {
		name string
		m    *Model
	}
	turn := func(a arm) result {
		d := a.m.NewState(depth + run + 64)
		defer d.Close()
		s := a.m.NewState(plen + 8)
		defer s.Close()
		if _, err := d.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		var stop atomic.Bool
		var toks atomic.Int64
		var wg sync.WaitGroup
		wg.Go(func() {
			for !stop.Load() {
				s.Reset()
				if _, err := s.Prefill(prompt[:plen]); err != nil {
					t.Error(err)
					return
				}
				toks.Add(plen)
			}
		})
		time.Sleep(20 * time.Millisecond)
		n0, r0 := toks.Load(), d.FusedRuns
		t0 := time.Now()
		for i := range run {
			if _, err := d.ForwardGreedy(int32(1 + i%100)); err != nil {
				t.Fatal(err)
			}
		}
		el := time.Since(t0)
		n1 := toks.Load()
		stop.Store(true)
		wg.Wait()
		return result{dec: el, pre: float64(n1-n0) / el.Seconds(), fused: d.FusedRuns - r0, onE: s.ServedOnE}
	}
	var starved, onEBase, onESplit int
	ab := func(a, b arm) string {
		turn(a)
		turn(b)
		var ratio, da, db, pa, pb []float64
		var fa, fb int64
		for r := range rounds {
			var x, y result
			if r%2 == 0 {
				x, y = turn(a), turn(b)
			} else {
				y, x = turn(b), turn(a)
			}
			ratio = append(ratio, float64(x.dec)/float64(y.dec))
			da, db = append(da, run/x.dec.Seconds()), append(db, run/y.dec.Seconds())
			pa, pb = append(pa, x.pre), append(pb, y.pre)
			fa, fb = fa+x.fused, fb+y.fused
			if b.name == "split" {
				onEBase += int(x.onE)
				onESplit += int(y.onE)
				if y.pre == 0 {
					starved++
				}
			}
		}
		m, iqr := medianIQR(ratio)
		mda, _ := medianIQR(da)
		mdb, _ := medianIQR(db)
		mpa, _ := medianIQR(pa)
		mpb, _ := medianIQR(pb)
		v := ""
		if iqr/m > 0.10 {
			v = "  UNSTABLE -- do not quote this"
		}
		return fmt.Sprintf("decode %s/%s time %.4fx (IQR/median %.1f%%, n=%d): %s %.1f tok/s, %s %.1f tok/s; "+
			"prefill beside it %s %.0f tok/s, %s %.0f tok/s; fused runs %d, %d%s",
			a.name, b.name, m, 100*iqr/m, rounds, a.name, mda, b.name, mdb, a.name, mpa, b.name, mpb, fa, fb, v)
	}
	if burn > 0 {
		defer burners(t, burn)()
		t.Logf("%d foreign busy loops running", burn)
	}
	base, split := arm{"base", mBase}, arm{"split", mSplit}
	t.Logf("A/A: %s", ab(base, arm{"base-B", mBase}))
	for pass := 1; pass <= 2; pass++ {
		t.Logf("pass %d: %s", pass, ab(base, split))
	}
	// The configuration under test was selected (RULE 10): the split arm took
	// the E-cores, the base arm never did.
	if sched.CanSplitE() && (onESplit == 0 || onEBase != 0) {
		t.Fatalf("the split arm placed %d prefills on the E-cores and the base arm %d: the A/B did not compare the two layouts",
			onESplit, onEBase)
	}
	if starved > 0 {
		t.Fatalf("the E-core prefill finished no prompt in %d of the split rounds' decode windows: starved", starved)
	}
}

// TestServingSplitSingleSessionUnchanged is one session that decodes and
// prefills in turn with no other beside it: the split must change nothing. It
// is the default arm against WithServingSplit(false), interleaved, an A/A
// first and the comparison twice; the ratios are base/split rates, so 1.0 is
// unchanged, and the split arm must have placed no prefill on the E-cores.
//
//	JITLLM_SPLIT_AB=1 JITLLM_MODEL=Llama-3.2-1B-Instruct-Q4_K_M.gguf
func TestServingSplitSingleSessionUnchanged(t *testing.T) {
	p, rounds, _ := splitEnv(t)
	open := func(o ...Option) *Model {
		m, err := Open(p, append([]Option{noTune}, o...)...)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	mBase, mSplit := open(WithServingSplit(false)), open()
	defer mBase.Close()
	defer mSplit.Close()
	const run, plen = 64, 128
	prompt := make([]int32, plen)
	for i := range prompt {
		prompt[i] = int32(1 + (i*7919)%1000)
	}
	type result struct{ dec, pre float64 }
	var onE int64
	turn := func(m *Model) result {
		s := m.NewState(plen + run + 64)
		defer s.Close()
		t0 := time.Now()
		if _, err := s.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		pre := float64(plen) / time.Since(t0).Seconds()
		t0 = time.Now()
		for i := range run {
			if _, err := s.ForwardGreedy(int32(1 + i%100)); err != nil {
				t.Fatal(err)
			}
		}
		// A second prefill after the decode, on the same session.
		s.Reset()
		if _, err := s.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		onE += s.ServedOnE
		return result{dec: run / time.Since(t0).Seconds(), pre: pre}
	}
	ab := func(an string, a *Model, bn string, b *Model) string {
		turn(a)
		turn(b)
		var rd, rp, da, db []float64
		for r := range rounds {
			var x, y result
			if r%2 == 0 {
				x, y = turn(a), turn(b)
			} else {
				y, x = turn(b), turn(a)
			}
			rd, rp = append(rd, x.dec/y.dec), append(rp, x.pre/y.pre)
			da, db = append(da, x.dec), append(db, y.dec)
		}
		md, id := medianIQR(rd)
		mp, ip := medianIQR(rp)
		ma, _ := medianIQR(da)
		mb, _ := medianIQR(db)
		return fmt.Sprintf("%s/%s decode+second prefill rate %.4fx (IQR/median %.1f%%), first prefill rate %.4fx (%.1f%%), n=%d; %s %.1f, %s %.1f tok/s over the decode window",
			an, bn, md, 100*id/md, mp, 100*ip/mp, rounds, an, ma, bn, mb)
	}
	t.Logf("A/A: %s", ab("base", mBase, "base-B", mBase))
	onE = 0
	for pass := 1; pass <= 2; pass++ {
		t.Logf("pass %d: %s", pass, ab("base", mBase, "split", mSplit))
	}
	if onE != 0 {
		t.Fatalf("a single session placed %d prefills on the E-cores", onE)
	}
}

// TestServingSplitIsTakenBesideADecoderOnly is the split's placement, on the
// committed fixture and with no model on the disk: where the host has E-cores
// to give, a prefill beside a decoding session goes to the E-cores' pool, and
// a prefill alone, a session alone and a model opened WithServingSplit(false)
// do not; and the answer is the one a prefill alone gives.
func TestServingSplitIsTakenBesideADecoderOnly(t *testing.T) {
	if !sched.CanSplitE() {
		t.Skip("no E-cores a thread can be kept on here: the serving split has nothing to place")
	}
	p := jlmOf(t, "../../testdata/models/stories260K.gguf")
	prompt := make([]int32, 48)
	for i := range prompt {
		prompt[i] = int32(1 + (i*37)%200)
	}
	// beside runs a decoder in the background and prefills prompt on another
	// State of the same model, returning the prefill's logits and its
	// placement count.
	beside := func(m *Model) ([]float32, int64) {
		d := m.NewState(4096)
		defer d.Close()
		s := m.NewState(256)
		defer s.Close()
		var stop atomic.Bool
		var wg sync.WaitGroup
		wg.Go(func() {
			for i := 0; !stop.Load(); i++ {
				if _, err := d.ForwardGreedy(int32(1 + i%100)); err != nil {
					d.Reset()
				}
			}
		})
		defer wg.Wait()
		defer stop.Store(true)
		time.Sleep(20 * time.Millisecond)
		var lg []float32
		for range 8 {
			s.Reset()
			l, err := s.Prefill(prompt)
			if err != nil {
				t.Fatal(err)
			}
			lg = append(lg[:0], l...)
		}
		return lg, s.ServedOnE
	}
	alone := func(m *Model) ([]float32, int64) {
		s := m.NewState(256)
		defer s.Close()
		l, err := s.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), l...), s.ServedOnE
	}
	on, err := Open(p, noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer on.Close()
	off, err := Open(p, noTune, WithServingSplit(false))
	if err != nil {
		t.Fatal(err)
	}
	defer off.Close()

	want, n := alone(on)
	if n != 0 {
		t.Fatalf("a session alone placed %d prefills on the E-cores", n)
	}
	got, n := beside(on)
	if n == 0 {
		t.Fatal("a prefill beside a decoding session never went to the E-cores: the split was not taken")
	}
	var worst, peak float64
	for i := range want {
		worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
		peak = math.Max(peak, math.Abs(float64(want[i])))
	}
	if worst > 1e-3*peak {
		t.Fatalf("a prefill on the E-cores' pool differs from one alone by %.3g (largest logit %.3g)", worst, peak)
	}
	if _, n := beside(off); n != 0 {
		t.Fatalf("WithServingSplit(false) placed %d prefills on the E-cores", n)
	}
	t.Logf("beside a decoder: %d prefills on the E-cores, worst logit difference %.2g of %.2g", n, worst, peak)
}

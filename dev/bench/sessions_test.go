package bench

import (
	"encoding/binary"
	"math"
	"os"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/jit/ir"
)

// The perf floors of sessions running at once, beside the correctness gates
// that hold their answers (backend.TestQueuesRunSessionsAtOnce,
// model.TestSessionsStepAtOnce) and not inside them (AGENTS.md RULE 4). Each
// alternates its arms round by round in one process and holds the median of
// the per-round ratios (RULE 2).

// ratioMedian is the median of the per-round ratios with their IQR.
func ratioMedian(r []float64) (med, q1, q3 float64) {
	sort.Float64s(r)
	return r[len(r)/2], r[len(r)/4], r[3*len(r)/4]
}

// TestQueuesOverlapOnTheDevice: two sessions on two queues of one device,
// each one block of a long dependent chain, take about as long as one alone
// where the driver runs two queues' work at once -- CUDA streams, Metal
// command queues -- and twice as long where it does not. NVIDIA's Vulkan
// driver runs neither two queues' dispatches nor two submissions' to one
// queue at once (2.00x both ways), so on Vulkan the bar is that queues cost
// nothing over taking turns.
func TestQueuesOverlapOnTheDevice(t *testing.T) {
	for _, d := range backend.Open() {
		defer d.Close()
		qd, ok := d.(backend.Queued)
		if !ok {
			continue
		}
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			limit := 1.5
			if d.API() == "spirv" {
				limit = 2.2
			}
			med, q1, q3 := queuesRatio(t, d, qd)
			t.Logf("two queues at once against one alone: median %.2fx (IQR %.2f-%.2f)", med, q1, q3)
			if med > limit {
				t.Fatalf("two queues took %.2fx of one alone, bar %.1fx", med, limit)
			}
		})
	}
}

// spinTrip is the chain's length, read from a buffer so no driver folds it.
const spinTrip = 1 << 22

func queuesRatio(t *testing.T, d backend.Device, qd backend.Queued) (med, q1, q3 float64) {
	b := ir.New("spin", [3]int{32, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pTrip := b.Param("pTrip", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.TID(), b.Const(ir.U32, 31))
	x0 := b.Load(ir.F32, pIn, i, 0)
	one := b.Load(ir.F32, pIn, b.Const(ir.U32, 32), 0)
	b.LoopN(b.Load(ir.U32, pTrip, b.Const(ir.U32, 0), 0))
	x := b.Phi(ir.F32, x0)
	b.SetPhi(x, b.Fma(x, one, one))
	b.EndLoop()
	b.Store(pOut, i, x, 0)
	built, err := b.Done()
	if err != nil {
		t.Fatal(err)
	}
	kern, err := d.Compile(built)
	if err != nil {
		t.Skipf("compile: %v", err)
	}
	defer kern.Close()
	alloc := func(n int, p []byte) backend.Buf {
		buf, err := d.Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(buf.Free)
		if p != nil {
			if err := buf.Write(p); err != nil {
				t.Fatal(err)
			}
		}
		return buf
	}
	in := make([]byte, 33*4)
	binary.LittleEndian.PutUint32(in[32*4:], math.Float32bits(1))
	tb := make([]byte, 4)
	binary.LittleEndian.PutUint32(tb, spinTrip)
	bin, btrip := alloc(len(in), in), alloc(4, tb)
	var qs [2]backend.Queue
	var outs [2]backend.Buf
	for k := range qs {
		q, err := qd.NewQueue()
		if err != nil {
			t.Fatal(err)
		}
		defer q.Close()
		qs[k], outs[k] = q, alloc(32*4, nil)
	}
	host := [2][]byte{make([]byte, 32*4), make([]byte, 32*4)}
	run := func(k int) {
		qd.SessionOn(qs[k], func(s backend.Session) {
			if err := s.Launch(kern, 1, 32, bin, btrip, outs[k]); err != nil {
				t.Error(err)
				return
			}
			if err := s.Read(outs[k], host[k]); err != nil {
				t.Error(err)
			}
		})
	}
	const rounds, per, warm = 12, 6, 2
	var ratios []float64
	for r := range rounds + warm {
		t0 := time.Now()
		for range per {
			run(0)
		}
		alone := time.Since(t0)
		t0 = time.Now()
		var wg sync.WaitGroup
		for k := range qs {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range per {
					run(k)
				}
			}()
		}
		wg.Wait()
		if r >= warm {
			ratios = append(ratios, float64(time.Since(t0))/float64(alone))
		}
	}
	return ratioMedian(ratios)
}

// TestSessionsOverlapOnCUDA: two sessions decoding stories15M at once on one
// CUDA card, every block and the head placed, take clearly less than twice
// one alone: its decode leaves the card mostly idle between launches, which
// two streams fill.
func TestSessionsOverlapOnCUDA(t *testing.T) {
	path := testmodels.Path("stories15M-q8_0.jlm")
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this floor proved nothing", err)
	}
	m, err := model.Open(path, model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := tier.OpenWith(tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		t.Skipf("NO CUDA DEVICE (%v): this floor proved nothing", err)
	}
	defer g.Close()
	prompts := [2][]int32{
		m.Vocab.Encode("Once upon a time, there was a little girl named", true),
		m.Vocab.Encode("The big dog ran to the park and", true),
	}
	// stories15M's context is 128 positions, so every round places fresh
	// States.
	const rounds, n, warm = 10, 96, 2
	seq := max(len(prompts[0]), len(prompts[1])) + n + 2
	place := func(prompt []int32) (*model.State, int32) {
		st := m.NewState(seq)
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			t.Fatalf("%d of %d blocks on the card, head there %v: %s", st.GPULayers(), m.Cfg.NLayer,
				st.HeadOnDevice(), g.Err())
		}
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		return st, model.Greedy(lg)
	}
	decode := func(st *model.State, next int32, n int) int32 {
		for range n {
			lg, err := st.Forward(next)
			if err != nil {
				t.Error(err)
				return next
			}
			next = model.Greedy(lg)
		}
		return next
	}
	both := func(sts [2]*model.State, next [2]int32, n int) [2]int32 {
		var wg sync.WaitGroup
		for k := range sts {
			wg.Add(1)
			go func() {
				defer wg.Done()
				next[k] = decode(sts[k], next[k], n)
			}()
		}
		wg.Wait()
		return next
	}
	var ratios []float64
	for r := range rounds + warm {
		a, na := place(prompts[0])
		t0 := time.Now()
		decode(a, na, n)
		one := time.Since(t0)
		a.Close()
		a, na = place(prompts[0])
		b, nb := place(prompts[1])
		// One step untimed: the second session's lane is built at its first
		// step beside the first.
		next := both([2]*model.State{a, b}, [2]int32{na, nb}, 1)
		t0 = time.Now()
		both([2]*model.State{a, b}, next, n)
		two := time.Since(t0)
		a.Close()
		b.Close()
		if r >= warm {
			ratios = append(ratios, float64(two)/float64(one))
		}
	}
	med, q1, q3 := ratioMedian(ratios)
	t.Logf("%s: %d tokens of two sessions at once against one alone: median %.2fx (IQR %.2f-%.2f)",
		g.Name(), n, med, q1, q3)
	// Taking turns, two sessions' steps take about twice one's.
	if med >= 1.6 {
		t.Fatalf("two sessions at once took %.2fx one alone: their submissions did not overlap", med)
	}
}

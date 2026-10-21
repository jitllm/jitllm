//go:build linux && amd64

package nn

import (
	"math/rand"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestTunerOnThisHardware is a measurement, not a correctness gate:
// it runs only under JITLLM_TUNER_HW=1, on an idle box, inside
// `scripts/cap 8G --` (which takes the exclusive lock).
//
// It checks the one thing a runtime tuner owes every host -- its pick is never
// slower than no prefetch, and within 2% of the best rung measured directly --
// and then, per hardware profile, what that profile is known to want. A profile
// whose hardware is not present SKIPS by name, so the list documents every
// machine the tuner has been validated on and says which one this is.
func TestTunerOnThisHardware(t *testing.T) {
	if os.Getenv("JITLLM_TUNER_HW") == "" {
		t.Skip("a measurement: JITLLM_TUNER_HW=1 ./scripts/cap 8G -- go test ./engine/nn -run TunerOnThisHardware")
	}
	t.Setenv("XDG_CACHE_HOME", t.TempDir()) // never write the user's cache
	if _, err := cpu.Native().PackedFusedAhead(quant.Q4_K, 4); err != nil {
		t.Skipf("this tier has no prefetch form: %v", err)
	}
	// The token is a model's layers, not a big matrix: long uniform runs are
	// what the hardware prefetcher already handles, and the software prefetch
	// rescues short runs over mixed shapes. So this is Llama-3.1-8B's Q4_K_M
	// block, four distinct copies (~0.8 GiB).
	type mat struct {
		t       quant.Type
		rows, k int
		w       *Packed
	}
	var ws []mat
	rng := rand.New(rand.NewSource(1))
	for range 4 {
		for _, m := range []mat{
			{quant.Q4_K, 4096, 4096, nil}, {quant.Q4_K, 1024, 4096, nil}, {quant.Q6_K, 1024, 4096, nil},
			{quant.Q4_K, 4096, 4096, nil}, {quant.Q4_K, 14336, 4096, nil}, {quant.Q4_K, 14336, 4096, nil},
			{quant.Q6_K, 4096, 14336, nil},
		} {
			q, _ := kernels.QuantOf(m.t)
			src := make([]byte, uint64(m.rows*m.k)/m.t.BlockElems()*m.t.BlockBytes())
			rng.Read(src)
			quant.PlantScales(m.t, src, 0)
			qs, d, sc, err := kernels.PackWeights(q, src, m.rows, m.k)
			if err != nil {
				t.Fatal(err)
			}
			m.w = &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
			ws = append(ws, m)
		}
	}
	const k, rows = 14336, 14336
	types := []quant.Type{quant.Q4_K, quant.Q6_K}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	out := make([]float32, rows)
	var bytesPer int
	for _, m := range ws {
		bytesPer += len(m.w.QS) + len(m.w.D) + len(m.w.SC)
	}
	token := func(j *JIT) time.Duration {
		t0 := time.Now()
		for _, m := range ws {
			if !j.MatVecPacked(out, m.t, m.w, x[:m.k], m.rows, m.k) {
				t.Fatal("declined")
			}
		}
		return time.Since(t0)
	}

	// 1. Let the duel run exactly as decode drives it.
	j := NewJIT(k, rows, types, WithTune(TuneForce), WithQuietTuner(true))
	defer j.Close()
	for i := 0; i < 400 && (j.fpf == nil || !j.fpf.done); i++ {
		j.TokenStart()
		token(j)
		j.TokenEnd()
	}
	if j.fpf == nil || !j.fpf.done {
		t.Fatal("the prefetch duel did not settle in 400 tokens")
	}
	pick := j.fpf.cur

	// 2. Measure every rung against distance 0 directly, ABBA, pinned JITs.
	pinned := map[int]*JIT{}
	for _, d := range prefetchDistances {
		p := NewJIT(k, rows, types, WithFusedPrefetch(d), WithTune(TuneOff))
		p.TokenStart()
		p.TokenEnd()
		defer p.Close()
		pinned[d] = p
	}
	gain := map[int]float64{0: 1} // rate at d over rate at 0
	for _, d := range prefetchDistances {
		if d == 0 {
			continue // the baseline every rung is measured against
		}
		var r []float64
		for range 12 {
			a1, b1, b2, a2 := token(pinned[0]), token(pinned[d]), token(pinned[d]), token(pinned[0])
			r = append(r, float64(a1)/float64(b1), float64(a2)/float64(b2))
		}
		slices.Sort(r)
		gain[d] = r[len(r)/2]
	}
	best := 0
	for d, g := range gain {
		if g > gain[best] {
			best = d
		}
	}
	gbs := float64(bytesPer) / token(pinned[pick]).Seconds() / 1e9
	t.Logf("host %q, %d cores, %d NUMA node(s): tuner picked %d; gain over 0: %v; %.1f GB/s at the pick",
		cpuModel(), pinned[0].pool.Max(), len(sched.NUMANodes()), pick, gain, gbs)

	t.Run("every host: the pick is never a loss", func(t *testing.T) {
		if gain[pick] < 0.98 {
			t.Errorf("picked %d at %.3fx of no prefetch", pick, gain[pick])
		}
		if gain[pick] < gain[best]/1.02 {
			t.Errorf("picked %d at %.3fx; %d measures %.3fx", pick, gain[pick], best, gain[best])
		}
	})

	// Each profile names hardware the tuner has been measured on and what the
	// measurement said it wants. Add a row when a new machine is characterised.
	for _, p := range []struct {
		name  string
		match func() bool
		want  func(t *testing.T)
	}{
		{"Broadwell-EP (family 6 model 79): prefetch pays",
			func() bool { return cpuField("cpu family") == "6" && cpuField("model") == "79" },
			func(t *testing.T) {
				// Two-socket Broadwell-EP, 14 cores of one node: prefetch
				// roughly 1.5x at 2-8 row groups against none.
				if pick < 2 || gain[pick] < 1.15 {
					t.Errorf("picked %d at %.3fx; this part measured 1.5x at 2-8", pick, gain[pick])
				}
			}},
		{"Alder Lake P-cores (family 6 model 154): prefetch does not pay",
			func() bool { return cpuField("cpu family") == "6" && cpuField("model") == "154" },
			func(t *testing.T) {
				// Alder Lake, six P-cores: prefetch at 2/4/8 row groups is
				// neutral to slightly slower than none, and the tuner picks 0.
				for _, d := range prefetchDistances {
					if d == 0 {
						continue // the baseline every rung is measured against
					}
					if gain[d] > 1.05 {
						t.Errorf("distance %d measures %.3fx; this part measured no gain -- re-characterise it", d, gain[d])
					}
				}
			}},
	} {
		t.Run(p.name, func(t *testing.T) {
			if !p.match() {
				t.Skipf("not this hardware (%q)", cpuModel())
			}
			p.want(t)
		})
	}
}

func cpuModel() string { return cpuField("model name") }

// cpuField is the first value of a /proc/cpuinfo key, "" when absent.
func cpuField(key string) string {
	b, _ := os.ReadFile("/proc/cpuinfo")
	for _, l := range strings.Split(string(b), "\n") {
		k, v, ok := strings.Cut(l, ":")
		if ok && strings.TrimSpace(k) == key {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

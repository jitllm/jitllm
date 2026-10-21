package model

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestHybridPrefillBatchesOnTheDevice gates the chunked gated delta rule: a
// hybrid with every block placed prefills a prompt longer than one device chunk
// in batched chunks, and every position's final residual is held to the
// row-by-row device path (Config.NoBatch, the single-row oracle) and to the
// host. Every position, because a chunked recurrence that drops the padded
// tail or restarts at a boundary gets row 0 of each chunk right and drifts from
// there. Stats.LinearBatched asserts the scan actually ran.
//
// Fixtures: the three qwen3next shapes (dense, routed mixture, tiled key
// pairing) and Kimi-Linear's (per-channel decay, latent full blocks).
func TestHybridPrefillBatchesOnTheDevice(t *testing.T) {
	const ctx = 1024
	for _, c := range []struct {
		name string
		o    hyOpt
	}{{"dense", hyOpt{ctx: ctx}}, {"moe", hyOpt{moe: true, ctx: ctx}},
		{"tiled", hyOpt{tiled: true, ctx: ctx}}} {
		t.Run(c.name, func(t *testing.T) { hybridBatchedPrefill(t, hybridModelOpt(t, c.o)) })
	}
	// Kimi Delta Attention's per-channel decay through a batch, beside latent
	// full blocks whose float absorb banks the batched latent attention reads.
	t.Run("synth-kimilinear", func(t *testing.T) {
		m, err := Open(hfContainer(t, "synth-kimilinear"), noTune)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		hybridBatchedPrefill(t, m)
	})
	// Real models by name. JITLLM_HYBRID_MODEL is a comma-separated list of containers,
	// JITLLM_HYBRID_DEVICES the device spec for models bigger than one card, and
	// JITLLM_HYBRID_HOST=1 adds the host arm (it needs the whole model in host
	// memory).
	for _, name := range strings.Split(os.Getenv("JITLLM_HYBRID_MODEL"), ",") {
		if name == "" {
			continue
		}
		t.Run(filepath.Base(name), func(t *testing.T) {
			p := testmodels.Path(name)
			if _, err := os.Stat(p); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v", err)
			}
			m, err := Open(p, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			hybridBatchedPrefill(t, m)
		})
	}
}

// hybridBatchBound is what the batched chunk may differ from the row-by-row
// device path by at any position. They differ in the batched matvec twins' and
// tiled attention's reduction order, and in the scan's lane-split dots.
const hybridBatchBound = 1e-2

func hybridBatchedPrefill(t *testing.T, m *Model) {
	c := m.Cfg
	if !c.Hybrid() {
		t.Fatal("not a hybrid: this gate would prove nothing")
	}
	chunk := nn.MaxDevicePrefillChunk
	plen := chunk + 88
	if plen+8 > c.NCtx {
		t.Fatalf("the fixture holds %d positions; the prompt must cross a %d-row chunk", c.NCtx, chunk)
	}
	rnd := rand.New(rand.NewSource(3))
	prompt := make([]int32, plen)
	for i := range prompt {
		prompt[i] = int32(2 + rnd.Intn(min(c.NVocab, 4000)-2))
	}
	forced := []int32{5, 9, 3, 7}

	type arm struct {
		rows   []float32 // [plen][NEmbd], the final residual at every position
		logits [][]float32
		st     tier.Stats
		err    string
	}
	volta := os.Getenv("JITLLM_HYBRID_VOLTA") != ""
	// A named result: the Stats are read in a defer, after an unnamed one
	// would already have been copied out.
	run := func(dev, noBatch bool) (a arm) {
		a.rows = make([]float32, plen*c.NEmbd)
		st := m.NewState(plen + len(forced) + 1)
		defer st.Close()
		if dev {
			spec := testdevices.One("gpu:0")
			g, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff),
				tier.WithPoisonScratch(true),
				tier.WithConfig(func(cf *tier.Config) {
					cf.NoBatch = noBatch
					// sm_70's batched matvecs (off by default) read f16
					// activations where every other path reads int8, which
					// alone moves a residual far more than the recurrence
					// under test. JITLLM_HYBRID_VOLTA=1 runs the shipping
					// sm_70 path, held to the bulk and logit bounds only.
					cf.NoVolta = !volta
				}))
			if err != nil || g == nil {
				noDevice(t, "device", err)
			}
			defer g.Close()
			st.SetDeviceLayers(g, c.NLayer)
			if st.GPULayers() != c.NLayer {
				t.Fatalf("the device took %d of %d blocks (%s) -- this gate needs every one",
					st.GPULayers(), c.NLayer, g.Err())
			}
			defer func() { a.st, a.err = g.Stats(), g.Err() }()
		}
		st.rowTap = func(base int, rows []float32) { copy(a.rows[base*c.NEmbd:], rows) }
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatalf("prefill (device %v, rows %v): %v", dev, noBatch, err)
		}
		a.logits = append(a.logits, append([]float32(nil), lg...))
		for _, id := range forced {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			a.logits = append(a.logits, append([]float32(nil), lg...))
		}
		if d := st.DeviceDemotions(); d != 0 {
			t.Fatalf("the device demoted %d time(s)", d)
		}
		return a
	}
	var host arm
	withHost := m.Cfg.NEmbd <= 64 || os.Getenv("JITLLM_HYBRID_HOST") != ""
	if withHost {
		host = run(false, false)
	}
	rows := run(true, true)
	bat := run(true, false)
	if !withHost {
		host = rows // the host columns then read against the row-by-row arm
	}

	// The fused rule must have run on CUDA (which always has a subgroup), or
	// this says nothing about kernels.GatedDeltaFused.
	if spec := testdevices.One("gpu:0"); (spec == "gpu:0" || strings.HasPrefix(spec, "cuda")) &&
		(rows.st.LinearFused == 0 || bat.st.LinearFused == 0) {
		t.Fatalf("recurrent blocks through the fused rule: %d row by row, %d batched -- CUDA "+
			"should take it on both arms (%s)", rows.st.LinearFused, bat.st.LinearFused, bat.err)
	}
	if rows.st.LinearBatched != 0 || bat.st.LinearBatched == 0 {
		t.Fatalf("recurrent blocks through the scan: %d row by row, %d batched -- the arms are not "+
			"the two paths (%s)", rows.st.LinearBatched, bat.st.LinearBatched, bat.err)
	}
	nmse := func(got, want []float32) float64 {
		var num, den float64
		for i := range want {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				return math.Inf(1)
			}
			d := float64(got[i] - want[i])
			num += d * d
			den += float64(want[i]) * float64(want[i])
		}
		return num / den
	}
	at := func(a arm, p int) []float32 { return a.rows[p*c.NEmbd : (p+1)*c.NEmbd] }
	worstRows, worstHost, worstOracle, wrAt, whAt := 0.0, 0.0, 0.0, -1, -1
	per := make([]float64, 0, plen)
	for p := 0; p < plen; p++ {
		e := nmse(at(bat, p), at(rows, p))
		per = append(per, e)
		if e > worstRows || math.IsInf(e, 0) {
			worstRows, wrAt = e, p
		}
		if e := nmse(at(bat, p), at(host, p)); e > worstHost || math.IsInf(e, 0) {
			worstHost, whAt = e, p
		}
		worstOracle = max(worstOracle, nmse(at(rows, p), at(host, p)))
	}
	for _, p := range []int{0, chunk - 1, chunk, plen - 1} {
		t.Logf("pos %4d: residual NMSE batched/rows %.3e, batched/host %.3e, rows/host %.3e",
			p, nmse(at(bat, p), at(rows, p)), nmse(at(bat, p), at(host, p)),
			nmse(at(rows, p), at(host, p)))
	}
	sort.Float64s(per)
	p50, p90 := per[len(per)/2], per[len(per)*9/10]
	t.Logf("%d positions, %d recurrent block-chunks through the scan: residual NMSE "+
		"batched/rows p50 %.3e p90 %.3e worst %.3e at %d; batched/host worst %.3e at %d, rows/host %.3e",
		plen, bat.st.LinearBatched, p50, p90, worstRows, wrAt, worstHost, whAt, worstOracle)
	var lgs, lgh, lgr string
	worstLg := 0.0
	for i := range bat.logits {
		e := nmse(bat.logits[i], rows.logits[i])
		lgs += fmt.Sprintf(" %.2e", e)
		lgh += fmt.Sprintf(" %.2e", nmse(bat.logits[i], host.logits[i]))
		lgr += fmt.Sprintf(" %.2e", nmse(rows.logits[i], host.logits[i]))
		worstLg = max(worstLg, e)
	}
	t.Logf("logits after the prompt and %d forced tokens\n  batched/rows:%s\n  batched/host:%s\n  rows/host:   %s",
		len(forced), lgs, lgh, lgr)
	// The bulk is bound, and the worst position looser: the arms reduce in
	// different orders, so a real mixture can take a different expert at one
	// position on a routing tie. A wrong chunked recurrence is wrong at most
	// positions, far past this bound at its 90th percentile.
	if !(p90 <= hybridBatchBound) || (!volta && !(worstRows <= 10*hybridBatchBound)) {
		t.Fatalf("residual NMSE between the batched chunk and row-by-row device prefill: "+
			"p90 %.3e, worst %.3e at position %d", p90, worstRows, wrAt)
	}
	if !(worstLg <= hybridBatchBound) {
		t.Fatalf("logit NMSE %.3e between the batched and row-by-row arms", worstLg)
	}
	// Against the host the bar is the row-by-row device path's own distance,
	// not a constant: one position on an int8 or routing tie moves both device
	// arms alike.
	if withHost && !volta && !(worstHost <= 2*worstOracle+1e-6) {
		t.Fatalf("position %d: residual NMSE %.3e between the batched chunk and the host, "+
			"against the row-by-row device path's own %.3e", whAt, worstHost, worstOracle)
	}
}

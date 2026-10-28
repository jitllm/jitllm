package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// An encoder's composed arms (bert, nomic-bert): every block the front end
// states served by the composer on the CPU device (WithComposedCPU) and on
// each card (tier.ComposedTower: a non-causal call's blocks), held to main's
// host encoder and to the references TestEmbeddingsMatchReference holds it
// to, the served blocks counted -- as many as the encoder has, a text, none
// declined -- and each arm run against a violation once. An encoder is
// chosen by what it loaded (its blocks are BERT's post-norm blocks), never
// by its name.

// encoderGoldens is the goldens whose model loads BERT-shaped encoder
// blocks, each opened with opts.
func encoderGoldens(t *testing.T, opts ...Option) (out []embedGolden, ms []*Model) {
	t.Helper()
	for _, g := range loadEmbedGoldens(t) {
		m := openEmbedModel(t, g, opts...)
		if !m.IsEncoder() || m.enc.mb != nil || len(m.enc.blocks) == 0 {
			continue
		}
		out, ms = append(out, g), append(ms, m)
	}
	if len(out) == 0 {
		t.Fatal("no BERT-shaped encoder among the embedding goldens: the gate proved nothing")
	}
	return out, ms
}

// encodeTexts embeds every text of g and keeps, per text, the residual
// after block 0.
func encodeTexts(t *testing.T, m *Model, e *Embedder, g embedGolden) (after0, out [][]float32) {
	t.Helper()
	var cur []float32
	e.afterBlock = func(li int, x []float32) {
		if li == 0 {
			cur = append(cur[:0], x...)
		}
	}
	defer func() { e.afterBlock = nil }()
	for _, text := range g.Texts {
		cur = nil
		v, err := e.Embed(m.EmbedIDs(text))
		if err != nil {
			t.Fatal(err)
		}
		after0 = append(after0, append([]float32(nil), cur...))
		out = append(out, append([]float32(nil), v...))
	}
	return after0, out
}

// worstCos is the lowest cosine between got's and want's vectors, text by
// text.
func worstCos(got, want [][]float32) float64 {
	w := 1.0
	for i := range want {
		h := make([]float64, len(want[i]))
		for j, x := range want[i] {
			h[j] = float64(x)
		}
		w = math.Min(w, cosine(got[i], h))
	}
	return w
}

// worstNMSE is the largest NMSE between got's and want's rows, text by text.
func worstNMSE(got, want [][]float32) float64 {
	w := 0.0
	for i := range want {
		n, _ := nmseOf(got[i], want[i])
		if math.IsNaN(n) {
			return math.Inf(1)
		}
		w = math.Max(w, n)
	}
	return w
}

// llamaCos checks vs's vectors against llama.cpp's where the golden takes
// them on the same file, at the reference gate's bound.
func llamaCos(t *testing.T, g embedGolden, vs [][]float32, arm string) {
	t.Helper()
	if g.LGGUF != "" && g.LGGUF != g.GGUF {
		return
	}
	// TestEmbeddingsMatchReference's bound: 0.999 where the file is
	// quantized, llama.cpp itself short of the float32 model by more than
	// the gate's bar.
	bound := embedCosBound
	if len(g.ST) == len(g.Llamacpp) {
		var floor []float64
		for i := range g.ST {
			floor = append(floor, cos64(g.Llamacpp[i], g.ST[i]))
		}
		if worst(floor) < embedCosBound {
			bound = 0.999
		}
	}
	for i := range vs {
		if i < len(g.Llamacpp) {
			if lc := cosine(vs[i], g.Llamacpp[i]); !(lc >= bound) {
				t.Errorf("%s text %d: cosine to llama.cpp %.7f < %v", arm, i, lc, bound)
			}
		}
	}
}

func TestComposedEncoderOnCPU(t *testing.T) {
	gs, ms := encoderGoldens(t, WithComposedCPU(true))
	for i, g := range gs {
		t.Run(g.Model, func(t *testing.T) {
			ref := openEmbedModel(t, g)
			re, err := ref.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			defer re.Close()
			want0, want := encodeTexts(t, ref, re, g)

			m := ms[i]
			nb := m.Cfg.NLayer
			if n := coveredBlocks(m); n != nb {
				t.Fatalf("the front end states %d of %d blocks (block 0: %s, %v)", n, nb, m.Uncovered(0), m.UncoveredErr(0))
			}
			e, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			got0, got := encodeTexts(t, m, e, g)
			served, declined := e.seg.CPUBlockSteps()
			t.Logf("%s: %d block runs served on the composed CPU, declines %v (form error %v)", g.Model, served,
				nonZero(declined[:]), e.seg.CPUFormErr())
			if served != nb*len(g.Texts) || len(nonZero(declined[:])) != 0 {
				t.Fatalf("the CPU served %d of %d block runs (%d blocks, %d texts), declines %v, form error %v", served,
					nb*len(g.Texts), nb, len(g.Texts), nonZero(declined[:]), e.seg.CPUFormErr())
			}
			ps, pd := e.seg.CPUPoolSteps()
			t.Logf("%s: %d readouts served by the composed pool, declines %v", g.Model, ps, nonZero(pd[:]))
			if ps != len(g.Texts) || len(nonZero(pd[:])) != 0 {
				t.Fatalf("the CPU's composed pool served %d of %d readouts, declines %v", ps, len(g.Texts), nonZero(pd[:]))
			}
			n0, c := worstNMSE(got0, want0), worstCos(got, want)
			t.Logf("%s: after block 0 worst NMSE %.3e; worst cosine to main's encoder %.9f", g.Model, n0, c)
			if !(n0 < 5e-5) {
				t.Fatalf("NMSE %.3e after ONE block (bound 5e-5): the composed CPU is not running this block's graph", n0)
			}
			if !(c >= embedDevCos) {
				t.Errorf("cosine to main's encoder %.9f < %v", c, embedDevCos)
			}
			llamaCos(t, g, got, "composed CPU")

			ids := m.EmbedIDs(g.Texts[0])
			allocs := testing.AllocsPerRun(2, func() {
				if _, err := e.Embed(ids); err != nil {
					t.Fatal(err)
				}
			})
			mainAllocs := testing.AllocsPerRun(2, func() {
				if _, err := re.Embed(ids); err != nil {
					t.Fatal(err)
				}
			})
			t.Logf("%s: a warm composed embed makes %.1f allocations, main's %.1f", g.Model, allocs, mainAllocs)
			if allocs > mainAllocs {
				t.Errorf("a warm composed embed made %.1f heap allocations, main's %.1f", allocs, mainAllocs)
			}

			// The violation: the rows attend causally, not both ways.
			e.seg.enc.causalFault = true
			bad0, _ := encodeTexts(t, m, e, g)
			e.seg.enc.causalFault = false
			nbad := worstNMSE(bad0, want0)
			t.Logf("%s: violation (causal rows): NMSE %.3e after block 0", g.Model, nbad)
			if !(nbad > 100*n0) || !(nbad > 5e-4) {
				t.Fatalf("the gate cannot see causal rows: NMSE %.3e against a clean %.3e", nbad, n0)
			}

			// The readout's violation: the other pooling than the container's.
			was := m.Cfg.Pooling
			m.Cfg.Pooling = jlm.PoolMean
			if was == jlm.PoolMean {
				m.Cfg.Pooling = jlm.PoolCLS
			}
			ve, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			_, vbad := encodeTexts(t, m, ve, g)
			vs, _ := ve.seg.CPUPoolSteps()
			ve.Close()
			m.Cfg.Pooling = was
			cb := worstCos(vbad, want)
			t.Logf("%s: violation (pooled other than %v): worst cosine to main's %.6f, %d readouts served", g.Model, was, cb, vs)
			if vs != len(g.Texts) || cb >= embedDevCos {
				t.Fatalf("the gate cannot see the wrong pooling: cosine %.9f, %d composed readouts", cb, vs)
			}
		})
	}
}

// encParts is what an encoder gate's composed arm asks of a tier: the
// non-causal call's blocks (tier.ComposedTower).
const encParts = composedParts | tier.ComposedTower

func TestComposedEncoderOnDevice(t *testing.T) {
	gs, ms := encoderGoldens(t, WithComposedCPU(true))
	for i, g := range gs {
		t.Run(g.Model, func(t *testing.T) {
			m := ms[i]
			nb := m.encBlocks()
			if n := coveredBlocks(m); n != nb {
				t.Fatalf("the front end states %d of %d blocks (block 0: %s, %v)", n, nb, m.Uncovered(0), m.UncoveredErr(0))
			}
			ref := openEmbedModel(t, g)
			he, err := ref.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			want0, want := encodeTexts(t, ref, he, g)
			he.Close()
			ran := 0
			for _, spec := range stepDevices() {
				probe, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || probe == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				probe.Close()
				ran++
				t.Run(spec, func(t *testing.T) {
					arm := func(n int) (after0, out [][]float32) { return encDeviceArm(t, m, g, spec, n) }
					got0, _ := arm(1)
					n0 := worstNMSE(got0, want0)
					bound := 5e-5
					t.Logf("%s, block 0 composed: worst NMSE %.3e after it (bound %.0e)", spec, n0, bound)
					if !(n0 < bound) {
						t.Fatalf("%s: NMSE %.3e after ONE composed block (bound %.0e)", spec, n0, bound)
					}
					_, got := arm(-1)
					c := worstCos(got, want)
					t.Logf("%s, all %d blocks composed: worst cosine to the host %.9f", spec, nb, c)
					if !(c >= embedDevCos) {
						t.Errorf("%s: cosine to the host %.9f < %v", spec, c, embedDevCos)
					}
					llamaCos(t, g, got, spec)

				})
			}
			if ran == 0 {
				t.Fatal("no device on this host: the gate proved nothing")
			}
		})
	}
}

// encDeviceArm embeds g's texts with n of m's encoder blocks on device spec
// (-1 every one), a tier asked for non-causal blocks (encParts), and demands
// every placed block run served by the composer, none declined.
func encDeviceArm(t *testing.T, m *Model, g embedGolden, spec string, n int) (after0, out [][]float32) {
	t.Helper()
	nb := m.encBlocks()
	gpu, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
		tier.WithDeviceTune(tier.TuneOff), composedOpt(encParts)}, testTierOpts(t)...)...)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	e, err := m.NewEmbedder()
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if err := e.SetDeviceLayers(gpu, n); err != nil {
		t.Fatal(err)
	}
	placed := e.DeviceBlocks()
	if n < 0 && placed != nb || n >= 0 && placed != n {
		t.Fatalf("placed %d blocks, offered %d of %d: %v %s", placed, n, nb, e.DeviceDeclines(), gpu.Err())
	}
	s0, _, _ := gpu.Stats().ServedPart(tier.ComposedTower)
	after0, out = encodeTexts(t, m, e, g)
	st := gpu.Stats()
	served, declined, unasked := st.ServedPart(tier.ComposedTower)
	t.Logf("%s: %d of %d blocks placed, %d block runs served by the composer, %d declined %v, %d unasked (%s)",
		spec, placed, nb, served-s0, declined, nonZero(st.BlockDeclines[:]), unasked, gpu.Err())
	if served-s0 != placed*len(g.Texts) || declined != 0 {
		t.Fatalf("%s: the composer served %d of %d block runs, declined %d %v (%s)", spec,
			served-s0, placed*len(g.Texts), declined, nonZero(st.BlockDeclines[:]), gpu.Err())
	}
	if e.seg.nc() == nil || e.seg.nc().arena == nil {
		return after0, out
	}
	// The blocks left on the CPU and the readout are the CPU device's.
	cs, cd := e.seg.CPUBlockSteps()
	ps, pd := e.seg.CPUPoolSteps()
	t.Logf("%s: %d block runs and %d readouts served on the composed CPU, declines %v %v", spec, cs, ps, nonZero(cd[:]), nonZero(pd[:]))
	if cs != (nb-placed)*len(g.Texts) || ps != len(g.Texts) || len(nonZero(cd[:]))+len(nonZero(pd[:])) != 0 {
		t.Fatalf("%s: the composed CPU served %d of %d block runs and %d of %d readouts, declines %v %v", spec,
			cs, (nb-placed)*len(g.Texts), ps, len(g.Texts), nonZero(cd[:]), nonZero(pd[:]))
	}
	return after0, out
}

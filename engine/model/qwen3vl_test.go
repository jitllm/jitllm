package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// qwen3TowerGate is TestQwenTowerMatchesTransformers' half for a deepstack
// tower (Qwen3-VL): each tap's rows held to transformers' deepstack_features,
// and each of the tower's own pieces taken out of the engine's copy must move
// the output far past the bound -- the 2-D rotary, the resampled position
// table and the deepstack mergers' post-shuffle LayerNorm.
func qwen3TowerGate(t *testing.T, tw *Tower, g *qvlGolden, got []float32, clean float64) {
	t.Helper()
	n := g.grid().Rows() * tw.Cfg.ProjDim
	if len(got) != n*(1+len(tw.Cfg.Deep)) || len(g.Image.DeepA8) != n*len(tw.Cfg.Deep) {
		t.Fatalf("the tower emitted %d floats and the golden %d deepstack floats for %d taps of %d",
			len(got), len(g.Image.DeepA8), len(tw.Cfg.Deep), n)
	}
	deep, worst := nmse32(g.Image.DeepA8, got[n:])
	f32, _ := nmse32(g.Image.Deep, got[n:])
	t.Logf("%d deepstack taps %v: NMSE %.3e, max|d| %.4f against transformers at the engine's int8 (%.3e f32)",
		len(tw.Cfg.Deep), tw.Cfg.Deep, deep, worst, f32)
	if math.IsNaN(deep) || deep > towerBound {
		t.Fatalf("the deepstack rows are not transformers': NMSE %.3e (bound %.0e)", deep, towerBound)
	}
	for _, v := range []struct {
		name  string
		mut   func(*Instance)
		fault towerFault
		deep  bool // the violation shows in the taps' rows, not the merged ones
	}{
		{"no 2-D rotary", noTowerRotary, towerFaultNone, false},
		{"the position table read unresampled", nil, faultNoPosLerp, false},
		{"the deepstack mergers without their LayerNorm", nil, faultDeepPreNorm, true},
	} {
		tw.Cfg.fault = v.fault
		bad, _ := qvlEncode(t, tw, g, v.mut)
		tw.Cfg.fault = towerFaultNone
		want, have := g.Image.EmbdA8, bad[:n]
		if v.deep {
			want, have = g.Image.DeepA8, bad[n:]
		}
		nb, _ := nmse32(want, have)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		if !(nb > 10*towerBound) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}

// TestQwen3VLPictureCarriesItsTaps runs the golden's prompt with the picture
// as a Picture span -- the prefill encodes it, the taps ride its rows -- and
// holds it to transformers end to end; then a second prompt that extends the
// first restores every position past the picture from the prefix cache, runs
// no tower block and decodes what a cold prefill does; and a third, cold instance
// takes the picture's rows, taps included, from the image cache.
func TestQwen3VLPictureCarriesItsTaps(t *testing.T) {
	m, g := openQVL(t, "synth-qwen3vl")
	defer m.Close()
	n := len(g.Pre) + g.grid().Rows() + len(g.Post) + len(g.Cont) + 1
	store := NewMemStore()
	type result struct {
		rows            [][]float32
		restored, block int64
	}
	run := func(cached bool, cont []int32) result {
		st := m.NewInstance(n)
		defer st.Close()
		if cached {
			st.SetKVStore(store)
			mustKey(t, st, "qwen3vl/taps")
		}
		p, err := st.Picture(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
		if err != nil {
			t.Fatal(err)
		}
		spans := []Span{{Tokens: g.Pre}, {Picture: p}, {Tokens: append(append([]int32(nil), g.Post...), cont...)}}
		var lg []float32
		if cached {
			lg, err = st.PrefillCachedMixed(spans...)
		} else {
			lg, err = st.PrefillMixed(spans...)
		}
		if err != nil {
			t.Fatal(err)
		}
		r := result{rows: [][]float32{append([]float32(nil), lg...)}, restored: int64(st.KVRestored())}
		for _, id := range g.Cont[len(cont) : len(g.Cont)-1] {
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
			r.rows = append(r.rows, append([]float32(nil), lg...))
		}
		if st.visState != nil {
			r.block = st.visState.vis.hostBlocks + st.visState.vis.devBlocks
		}
		return r
	}
	cold := run(false, nil)
	w, flips := qvlWorst(g, cold.rows)
	t.Logf("a Picture span, taps and all: worst logit NMSE %.3e against transformers, %d flips", w, flips)
	if math.IsNaN(w) || w > 2e-2 || flips > 0 {
		t.Fatalf("a Picture span reads %.3e, %d flips against transformers", w, flips)
	}
	// The cache: turn 1 carries one continuation token, so its prompt ends on
	// a whole chunk past the picture (a partial last chunk is found only by a
	// prompt that ends there); turn 2 carries three and must restore it.
	run(true, g.Cont[:1])
	m.imgCache.clearForTest()
	t2 := run(true, g.Cont[:3])
	imgEnd := int64(len(g.Pre) + g.grid().Rows())
	if t2.restored <= imgEnd || t2.block != 0 {
		t.Fatalf("turn 2 restored %d positions (the picture ends at %d) and ran %d tower block(s)",
			t2.restored, imgEnd, t2.block)
	}
	for i := range t2.rows {
		if d, _ := nmse32(cold.rows[i+3], t2.rows[i]); d > 1e-10 {
			t.Fatalf("restored row %d is %.3e from the cold prefill's", i, d)
		}
	}
	// The image cache: once one instance has encoded the picture, a fresh
	// instance's encode is answered, taps included.
	run(false, nil)
	h0, _ := m.ImageCacheStats()
	again := run(false, nil)
	h1, _ := m.ImageCacheStats()
	if h1 == h0 || again.block != 0 {
		t.Fatalf("a second encode of the picture ran %d tower block(s) and the image cache answered %d time(s)",
			again.block, h1-h0)
	}
	for i := range cold.rows {
		if d, _ := nmse32(cold.rows[i], again.rows[i]); d != 0 {
			t.Fatalf("the cached picture's row %d is %.3e from the encoded one's", i, d)
		}
	}
	t.Logf("turn 2 restored %d positions past the picture's end at %d with no tower block; "+
		"the image cache answered a fresh instance bit for bit", t2.restored, imgEnd)
}

// TestQwen3VLRotaryIsInterleaved holds the text model's M-RoPE to
// transformers' interleaved split on the golden's own image rows, and fails
// the split laid out as Qwen2-VL's contiguous runs.
func TestQwen3VLRotaryIsInterleaved(t *testing.T) {
	for _, name := range []string{"synth-qwen3vl", "synth-qwen3vlmoe", "synth-qwen35vl"} {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
			defer m.Close()
			if !m.Cfg.RopeInterleaved {
				t.Fatal("the container's text model is not an interleaved M-RoPE one")
			}
			run := func() (float64, error) {
				st := m.NewInstance(len(g.Pre) + g.grid().Rows() + len(g.Post) + len(g.Cont) + 1)
				defer st.Close()
				rows, err := qvlRunErr(st, g, true)
				if err != nil {
					return 0, err
				}
				w, flips := qvlWorst(g, rows)
				if flips > 0 {
					w = math.Max(w, 1)
				}
				return w, nil
			}
			clean, err := run()
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("interleaved: worst logit NMSE %.3e against transformers", clean)
			if math.IsNaN(clean) || clean > 1e-6 {
				t.Fatalf("the text model on transformers' own image rows reads %.3e", clean)
			}
			was := m.rope.Runs
			m.rope.Runs = nn.MRopeRuns(m.Cfg.RopeSections[:])
			bad, err := run()
			m.rope.Runs = was
			if err != nil {
				t.Logf("contiguous sections (the violation): %v -- refused by name, which the gate sees", err)
				return
			}
			t.Logf("contiguous sections (the violation): %.3e", bad)
			if !(bad > 1000*clean) || bad < 1e-6 {
				t.Fatalf("contiguous sections read %.3e against %.3e: the gate cannot see the interleave", bad, clean)
			}
		})
	}
}

package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestActInKeepsThePrinciples runs the decode matvecs that build their own
// operand (kernels.ActIn, tier's fuseActIn) through what every device path
// keeps: blocks moving between the device and the host mid-sequence and back,
// a host page budget below the model, two sessions taking turns on one tier,
// and a warm decode that allocates nothing. Each arm is held to the same
// sequence run with Config.NoActFuse -- bit for bit on a CUDA tier, whose
// lowering pins the float contraction, and within the reduction-order band
// elsewhere -- and Stats().ActInLaunches is the selection check: the
// prologue arm must have encoded it, the other never.
//
// In-group splits in both arms (ForceGroupSplit): the prologue is built only
// where few workgroups share a row (tier.actInPays), and the table split
// TuneOff picks is a partial-writing one.
func TestActInKeepsThePrinciples(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		testmodels.Missing(t, "Llama-3.2-1B-Instruct-Q4_K_M: %v", err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The lighthouse keeper counted the ships that never came back, and", true)
	open := func(noAct bool) *tier.GPU {
		g, err := openTier(t, tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) {
			c.NoActFuse, c.ForceGroupSplit = noAct, true
			c.ActInSites = 11 // forced: TuneOff runs no prologue tuner
		}))
		if err != nil || g == nil {
			t.Skipf("no GPU: %v", err)
		}
		return g
	}
	half := m.Cfg.NLayer / 2
	// seams is the placement before each stretch of tokens: everything,
	// half home, everything back.
	seams := []int{-1, half, -1}
	const per = 6
	// arm runs two sessions taking turns, moving the seam between stretches,
	// and returns every step's logits, session-major.
	arm := func(g *tier.GPU) [][]float32 {
		var out [][]float32
		var ss [2]*State
		var last [2][]float32
		for i := range ss {
			ss[i] = m.NewState(len(ids) + len(seams)*per + 8)
			defer ss[i].Close()
			if err := ss[i].SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if last[i], err = ss[i].Prefill(ids); err != nil {
				t.Fatal(err)
			}
		}
		for _, seam := range seams {
			for i := range ss {
				n := m.Cfg.NLayer
				if seam >= 0 {
					n = seam
				}
				if got := ss[i].SetGPULayers(n); got != n {
					t.Fatalf("session %d: the device took %d blocks, want %d: %s", i, got, n, g.Err())
				}
			}
			for step := 0; step < per; step++ {
				for i := range ss {
					out = append(out, append([]float32(nil), last[i]...))
					if last[i], err = ss[i].Forward(argmaxID(last[i])); err != nil {
						t.Fatal(err)
					}
				}
			}
		}
		return out
	}
	// One tier at a time, closed before the next opens: two tiers on one
	// card would place differently and run different prompts.
	run := func(noAct bool) (out [][]float32, launches int, exact bool) {
		g := open(noAct)
		defer g.Close()
		out = arm(g)
		return out, g.Stats().ActInLaunches, strings.Contains(g.Name(), "[ptx]") && g.Devices() == 1
	}
	compare := func(what string) {
		on, nOn, exact := run(false)
		off, nOff, _ := run(true)
		if nOn == 0 {
			t.Fatalf("%s: the prologue arm encoded no operand prologue", what)
		}
		if nOff != 0 {
			t.Fatalf("%s: NoActFuse encoded %d operand prologues", what, nOff)
		}
		for i := range on {
			var num, den float64
			for j := range on[i] {
				if exact && math.Float32bits(on[i][j]) != math.Float32bits(off[i][j]) {
					t.Fatalf("%s step %d logit %d: prologue %v, two launches %v", what, i, j, on[i][j], off[i][j])
				}
				d := float64(on[i][j] - off[i][j])
				num, den = num+d*d, den+float64(off[i][j])*float64(off[i][j])
			}
			if num > 1e-6*den || argmaxID(on[i]) != argmaxID(off[i]) {
				t.Fatalf("%s step %d: NMSE %.3e, argmax %d vs %d", what, i, num/den, argmaxID(on[i]), argmaxID(off[i]))
			}
		}
		t.Logf("%s: %d steps over two sessions and seams %v match the two-launch arm (%d prologue launches, exact %v)",
			what, len(on), seams, nOn, exact)
	}
	compare("relocating")
	t.Run("paged", func(t *testing.T) {
		// A host page budget of a quarter of the blocks with half of them on
		// the device: the host half pages, the device half runs the prologue.
		was := m.PageBudget()
		m.SetPageBudget(uint64(max(m.Cfg.NLayer/4, 2)) * m.PageSize())
		defer m.SetPageBudget(was)
		seams = []int{half, half}
		compare("paged")
	})
	t.Run("tuned", func(t *testing.T) {
		// The shipping configuration: the split tuner and the prologue tuner
		// both measure on this card (no cache, so nothing is read back), and
		// ActInSites is left to them. The tuner must have decided every site
		// key it found, the prologue must run exactly where it kept one, and
		// the two arms -- one tier, so one set of splits, NoActFuse toggled
		// between two fresh States -- must agree as the forced arms do.
		g, err := openTier(t, tier.WithDeviceTune(tier.TuneAuto))
		if err != nil || g == nil {
			t.Skipf("no GPU: %v", err)
		}
		defer g.Close()
		seq := func() [][]float32 {
			s := m.NewState(len(ids) + 3*per + 8)
			defer s.Close()
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			last, err := s.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			var out [][]float32
			for i := 0; i < 3*per; i++ {
				out = append(out, append([]float32(nil), last...))
				if last, err = s.Forward(argmaxID(last)); err != nil {
					t.Fatal(err)
				}
			}
			return out
		}
		seq() // the retune, and with it the prologue tuner, runs here
		st := g.Stats()
		if st.ActInTimed == 0 {
			t.Fatalf("the prologue tuner decided no site (%s)", g.Err())
		}
		before := st.ActInLaunches
		on := seq()
		ran := g.Stats().ActInLaunches - before
		if (st.ActInKept > 0) != (ran > 0) {
			t.Fatalf("the tuner kept %d of %d sites and the decode encoded %d prologue launches",
				st.ActInKept, st.ActInTimed, ran)
		}
		g.NoActFuse = true
		before = g.Stats().ActInLaunches
		off := seq()
		if n := g.Stats().ActInLaunches - before; n != 0 {
			t.Fatalf("NoActFuse encoded %d operand prologues", n)
		}
		exact := strings.Contains(g.Name(), "[ptx]") && g.Devices() == 1
		for i := range on {
			var num, den float64
			for j := range on[i] {
				if exact && math.Float32bits(on[i][j]) != math.Float32bits(off[i][j]) {
					t.Fatalf("step %d logit %d: tuned %v, two launches %v", i, j, on[i][j], off[i][j])
				}
				d := float64(on[i][j] - off[i][j])
				num, den = num+d*d, den+float64(off[i][j])*float64(off[i][j])
			}
			if num > 1e-6*den || argmaxID(on[i]) != argmaxID(off[i]) {
				t.Fatalf("step %d: NMSE %.3e, argmax %d vs %d", i, num/den, argmaxID(on[i]), argmaxID(off[i]))
			}
		}
		t.Logf("the tuner kept %d of %d sites on %s; %d prologue launches over %d steps, matching the two-launch arm (exact %v)",
			st.ActInKept, st.ActInTimed, g.Name(), ran, len(on), exact)
	})
	t.Run("alloc", func(t *testing.T) {
		g := open(false)
		defer g.Close()
		decodeAllocs(t, m, g)
		if g.Stats().ActInLaunches == 0 {
			t.Fatal("the allocation window ran no operand prologue")
		}
	})
}

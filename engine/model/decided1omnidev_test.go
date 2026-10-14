package model

import (
	"math"
	"path/filepath"
	"testing"
)

// d1OmniScores is every question's scores of the support request (text) and
// of the pictures request (a prefix), in order, on d.
func d1OmniScores(t *testing.T, d *Decider) [][]float32 {
	t.Helper()
	var out [][]float32
	state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
	for i := range qs {
		s, err := d.run(state, &qs[i], 0)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), s...))
	}
	req, imgs := d1OmniPictures(t)
	if err := d.d1OmniPrefix(imgs); err != nil {
		t.Fatal(err)
	}
	defer func() { d.omniPre = d.omniPre[:0] }()
	for i := range req.Questions {
		s, err := d.run(req.State, &req.Questions[i], 0)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), s...))
	}
	return out
}

// TestD1OmniOnEveryDevice runs d1-omni with every block on each device --
// the trunk's convolution and attention blocks and the head's -- and with a
// seam that leaves half the trunk home, against the host: text alone and a
// picture prefix, whose rows attend to the prefix alone and whose last row's
// convolution does not read the text. The violation lets the prefix read the
// text on the device and must move the picture questions' scores.
func TestD1OmniOnEveryDevice(t *testing.T) {
	m := openD1Omni(t)
	defer m.Close()
	host, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	want := d1OmniScores(t, host)
	host.Close()
	worst := func(got [][]float32) float64 {
		w := 0.0
		for i := range want {
			for j := range want[i] {
				if math.IsNaN(float64(got[i][j])) {
					return math.NaN()
				}
				w = max(w, math.Abs(float64(got[i][j]-want[i][j])))
			}
		}
		return w
	}
	ran := 0
	for _, arm := range layaArms(t) {
		if arm.gpu == nil {
			continue
		}
		ran++
		t.Run(arm.name, func(t *testing.T) {
			for _, n := range []int{3, -1, 8} {
				d, err := m.NewDecider(0)
				if err != nil {
					t.Fatal(err)
				}
				if err := d.SetDevice(arm.gpu, n); err != nil {
					d.Close()
					t.Fatal(err)
				}
				placed, total := d.DeviceBlocks(), m.encBlocks()
				if n < 0 && placed != total || n > 0 && placed != n {
					d.Close()
					t.Fatalf("%d blocks offered, %d of %d placed: %v %s", n, placed, total,
						d.emb.DeviceDeclines(), arm.gpu.Err())
				}
				w := worst(d1OmniScores(t, d))
				t.Logf("%d of %d blocks placed: worst score difference from the host %.3e", placed, total, w)
				// f32 order on the device and Q8_0 tower rows the host and
				// the device compute alike: the band Laya's trunk sits in.
				const bound = 0.05
				if math.IsNaN(w) || w > bound {
					t.Errorf("past %g", bound)
				}
				if n < 0 {
					d.emb.violation = "prefix-reads-text"
					d.emb.seg.enc.violation = d.emb.violation
					v := worst(d1OmniScores(t, d))
					d.emb.violation = ""
					t.Logf("violation, the prefix reads the text on the device: %.4f", v)
					if !(v > 10*max(w, 1e-3)) {
						t.Errorf("letting the prefix read the text moved the scores %.4f, under ten times the clean %.4f", v, w)
					}
				}
				d.Close()
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host: the gate proved nothing (RULE 10)")
	}
}

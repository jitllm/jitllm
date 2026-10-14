package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestD1OmniHearsOnEveryDevice runs the audio tower's FastConformer blocks on
// each device (nn.ConvConformer programs), every block placed, against the
// host's prefix rows, and answers the voice request there. The violation:
// the attention's u and v biases swapped on the device must move the rows.
func TestD1OmniHearsOnEveryDevice(t *testing.T) {
	m := openD1Omni(t)
	defer m.Close()
	req, clip, g := d1OmniVoice(t)
	host, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := host.emb.encodeAudio(host.d1OmniSamples(clip))
	if err != nil {
		t.Fatal(err)
	}
	want := append([]float32(nil), rows...)
	host.Close()
	nmse := func(got []float32) float64 {
		var num, den float64
		for i := range want {
			e := float64(got[i] - want[i])
			num += e * e
			den += float64(want[i]) * float64(want[i])
		}
		return num / den
	}
	ran := 0
	for _, arm := range layaArms(t) {
		if arm.gpu == nil {
			continue
		}
		ran++
		t.Run(arm.name, func(t *testing.T) {
			for _, v := range []string{"", "audio-u-v-swapped"} {
				d, err := m.NewDecider(0)
				if err != nil {
					t.Fatal(err)
				}
				d.emb.violation = v
				// The audio blocks alone: the encoder's would take a small card first.
				if err := d.emb.setAudioDevice(arm.gpu); err != nil {
					d.Close()
					t.Fatal(err)
				}
				if n := d.emb.AudioDeviceBlocks(); n != len(m.audio.blocks) {
					why := d.emb.audSeg.DeviceDeclines()
					d.Close()
					t.Fatalf("%d of %d audio blocks placed: %v %s", n, len(m.audio.blocks), why, arm.gpu.Err())
				}
				got, err := d.emb.encodeAudio(d.d1OmniSamples(clip))
				if err != nil {
					d.Close()
					t.Fatal(err)
				}
				e := nmse(got)
				if v != "" {
					t.Logf("%s: prefix NMSE %.3e against the host", v, e)
					if !(e > 1e-2) {
						t.Errorf("%s moved the device's rows only %.3e", v, e)
					}
					d.Close()
					continue
				}
				t.Logf("every audio block on the device: prefix NMSE %.3e against the host", e)
				if e > 1e-3 {
					t.Errorf("past 1e-3")
				}
				ans, err := d.DecideAudio(req.State, req.Questions, clip)
				if err != nil {
					d.Close()
					t.Fatal(err)
				}
				w := 0.0
				for i, gq := range g.Questions {
					probs := ans[i].Probs
					if req.Questions[i].Type == jlm.QuestionNoul {
						probs = []float64{ans[i].Noul, 1 - ans[i].Noul}
					}
					for j := range probs {
						w = max(w, math.Abs(probs[j]-gq.Probs[j]))
					}
				}
				t.Logf("answers: worst probability %.4f off the reference", w)
				if w > 0.03 {
					t.Errorf("past 0.03")
				}
				if n := testing.AllocsPerRun(2, func() {
					if _, err := d.emb.encodeAudio(d.d1OmniSamples(clip)); err != nil {
						t.Fatal(err)
					}
				}); n != 0 {
					t.Errorf("a warm clip on the device made %v allocations", n)
				}
				d.Close()
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host: the gate proved nothing (RULE 10)")
	}
}

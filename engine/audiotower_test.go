package engine

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestD1OmniStatesItsAudioTower converts d1-omni with its mmproj: the
// container opens, its vision tower lowers, and its audio tower reads back
// stated -- an entry framing the clip, a chained block for each block, a
// head -- with the longest clip the reference reads, and lowers: the entry
// over the clip's 3001 frames into the subsampling's 376 rows of the
// tower's width, each block over them, the head into rows of the text
// model's.
func TestD1OmniStatesItsAudioTower(t *testing.T) {
	path := testmodels.Path("d1-omni/d1-omni-600M-F16.gguf")
	mm := testmodels.Path("d1-omni/mmproj-d1-omni-600M-F16.gguf")
	if !sourcePresent(path, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	m, err := model.Open(jlmOfPair(t, path, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a := m.Container.Vision().Audio
	switch {
	case m.Tower == nil:
		t.Fatal("Open lowered no vision tower")
	case a == nil || a.Tower == nil:
		t.Fatal("the container states no audio tower")
	case a.Tower.Entry.Ops[0].Code != jlm.TOpFrames || len(a.Tower.Chain) != int(a.NLayer) || a.MaxSamples != 30*a.SampleRate:
		t.Fatalf("an audio entry from op %d, %d chained blocks of %d, clips of %d samples",
			a.Tower.Entry.Ops[0].Code, len(a.Tower.Chain), a.NLayer, a.MaxSamples)
	}
	ag := m.Audio
	if ag == nil || !ag.Chain || len(ag.Blocks) != int(a.NLayer) || ag.Head == nil {
		t.Fatalf("Open lowered the audio tower to %+v", ag)
	}
	e := ag.Classes[0].Entry[1]
	eo, in := e.Values[e.Out[0]], ag.Blocks[0].Values[ag.Blocks[0].In[0]]
	ho := ag.Head.Values[ag.Head.Out[0]]
	if ag.Rows != 3001 || eo.Rows != 376 || eo.Width != int(a.NEmbd) || in.Rows != 376 || ho.Rows != 376 || ho.Width != int(a.ProjDim) {
		t.Fatalf("the entry writes %d rows of %d from %d frames, the blocks read %d, the head writes %d of %d",
			eo.Rows, eo.Width, ag.Rows, in.Rows, ho.Rows, ho.Width)
	}
	if got, want := ag.Blocks[0].Block, m.TowerBase+int(m.Container.Vision().NLayer); got != want {
		t.Fatalf("the audio's first block is block %d of the model, want %d (after the vision tower's)", got, want)
	}
}

// d1OmniOpen is d1-omni converted with its mmproj, opened.
func d1OmniOpen(t *testing.T) *model.Model {
	t.Helper()
	path := testmodels.Path("d1-omni/d1-omni-600M-F16.gguf")
	mm := testmodels.Path("d1-omni/mmproj-d1-omni-600M-F16.gguf")
	if !sourcePresent(path, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", path, mm)
	}
	m, err := model.Open(jlmOfPair(t, path, mm))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// voiceClip is the voice request's clip (testdata/decision/voice.json's
// one file, a WAV data URL) and its reference's prefix rows' first values.
func voiceClip(t *testing.T) ([]float32, [][]float64) {
	t.Helper()
	var req struct {
		Files []string `json:"files"`
	}
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "voice.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &req); err != nil || len(req.Files) != 1 {
		t.Fatalf("the voice request: %d files, %v", len(req.Files), err)
	}
	_, data, ok := strings.Cut(req.Files[0], ";base64,")
	if !ok {
		t.Fatal("the voice request's file is not a base64 data URL")
	}
	wav, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		t.Fatal(err)
	}
	clip, err := DecodeWAV(wav)
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Prefix [][]float64 `json:"prefix_rows"`
	}
	b, err = os.ReadFile(filepath.Join("testdata", "decision", "d1-omni.voice.reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return clip, g.Prefix
}

// TestD1OmniHearsAsItsReference encodes the voice request's clip through
// d1-omni's audio tower on the CPU device and holds the prefix rows to the
// reference's own code (audio.py over the F16 weights): as many rows, and
// their first values within an NMSE of 1e-2, the int8 blocks' band. A clip
// read as silence is the violation, and it must part from the reference.
func TestD1OmniHearsAsItsReference(t *testing.T) {
	m := d1OmniOpen(t)
	defer m.Close()
	s := testInstance(t, m, 256)
	defer s.Close()
	clip, want := voiceClip(t)
	nmse := func(clip []float32) (float64, int) {
		rows, err := s.EncodeClip(clip)
		if err != nil {
			t.Fatal(err)
		}
		d := int(m.Container.Vision().Audio.ProjDim)
		var num, den float64
		for r, row := range want {
			for i, w := range row {
				if r*d+i >= len(rows) {
					return math.Inf(1), len(rows) / d
				}
				g := float64(rows[r*d+i])
				num += (g - w) * (g - w)
				den += w * w
			}
		}
		return num / den, len(rows) / d
	}
	e, n := nmse(clip)
	t.Logf("%d samples into %d prefix rows (the reference's %d), NMSE %.3e", len(clip), n, len(want), e)
	if n != len(want) || math.IsNaN(e) || !(e < 1e-2) {
		t.Fatalf("%d rows at NMSE %.3e: past the bound (the reference's %d rows, 1e-2)", n, e, len(want))
	}
	if bad, _ := nmse(make([]float32, len(clip))); !(bad > 0.1) {
		t.Fatalf("silence reads NMSE %.3e: the gate cannot see the clip", bad)
	}
}

// TestD1OmniHearsOnTheCard places d1-omni's audio blocks on each device,
// every block and the first half of them, each a part of its own on the card
// (PlaceAudioChain), and holds the clip's prefix rows to the reference's
// within three times the CPU device's distance from it, or 1e-3. No part
// forms at a clip; the blocks placed back on the CPU device read the CPU
// device's rows exactly.
func TestD1OmniHearsOnTheCard(t *testing.T) {
	m := d1OmniOpen(t)
	defer m.Close()
	clip, ref := voiceClip(t)
	d := int(m.Container.Vision().Audio.ProjDim)
	nmse := func(rows []float32) float64 {
		var num, den float64
		for r, row := range ref {
			for i, w := range row {
				g := float64(rows[r*d+i])
				num += (g - w) * (g - w)
				den += w * w
			}
		}
		return num / den
	}
	host := testInstance(t, m, 256)
	want, err := host.EncodeClip(clip)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	host.Close()
	bound := max(3*nmse(want), 1e-3)
	nb := len(m.Audio.Blocks)
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			gd, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || gd == nil {
				noDevice(t, dev, err)
			}
			defer gd.Close()
			for _, blocks := range []int{nb, nb / 2} {
				s := testInstance(t, m, 256)
				if err := s.PlaceAudioChain(gd, blocks); err != nil {
					s.Close()
					t.Fatalf("%d of %d blocks on %s: %v (%s)", blocks, nb, dev, err, gd.Err())
				}
				formed := gd.Stats().UnformedAtStep
				got, err := s.EncodeClip(clip)
				if err != nil {
					s.Close()
					t.Fatalf("%v (%s)", err, gd.Err())
				}
				got = append([]float32(nil), got...)
				if err := s.PlaceAudioChain(gd, 0); err != nil {
					s.Close()
					t.Fatal(err)
				}
				back, err := s.EncodeClip(clip)
				s.Close()
				if err != nil {
					t.Fatal(err)
				}
				if st := gd.Stats(); st.UnformedAtStep != formed {
					t.Fatalf("%d of the audio's parts formed at a clip on %s, not at the placement", st.UnformedAtStep-formed, dev)
				}
				e := nmse(got)
				t.Logf("%s: %d of %d blocks on the card: NMSE %.3e from the reference (the CPU device's %.3e)", dev, blocks, nb, e, nmse(want))
				if math.IsNaN(e) || e > bound {
					t.Fatalf("%d of %d blocks on %s: NMSE %.3e from the reference (bound %.1e)", blocks, nb, dev, e, bound)
				}
				for j := range back {
					if back[j] != want[j] {
						t.Fatalf("placed back on the CPU device, element %d is %v, the CPU device's %v", j, back[j], want[j])
					}
				}
			}
		})
	}
}

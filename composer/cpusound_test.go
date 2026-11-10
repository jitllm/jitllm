package composer

import (
	le "encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// goldPCM is a golden clip's PCM as its WAV stores it, and its format.
type goldPCM struct {
	format         uint8
	channels, rate int
	mask           uint64
	pcm            []byte
	frames         int
}

func readGoldPCM(t *testing.T, path string) goldPCM {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var c goldPCM
	var tag, bits uint16
	for p := 12; p+8 <= len(b); {
		id, n := string(b[p:p+4]), int(le.LittleEndian.Uint32(b[p+4:]))
		body := b[p+8 : min(p+8+n, len(b))]
		switch id {
		case "fmt ":
			tag, bits = le.LittleEndian.Uint16(body), le.LittleEndian.Uint16(body[14:])
			c.channels, c.rate = int(le.LittleEndian.Uint16(body[2:])), int(le.LittleEndian.Uint32(body[4:]))
			if tag == 0xFFFE {
				c.mask, tag = uint64(le.LittleEndian.Uint32(body[20:])), le.LittleEndian.Uint16(body[24:])
			}
		case "data":
			c.pcm = body
		}
		p += 8 + n + n%2
	}
	switch {
	case tag == 3 && bits == 32:
		c.format = graph.SoundF32
	case tag == 3 && bits == 64:
		c.format = graph.SoundF64
	case bits == 8:
		c.format = graph.SoundU8
	case bits == 16:
		c.format = graph.SoundS16
	case bits == 24:
		c.format = graph.SoundS24
	case bits == 32:
		c.format = graph.SoundS32
	}
	c.frames = len(c.pcm) / (SoundBytes(c.format) * c.channels)
	return c
}

// TestComposedCPUSound runs every golden clip through the sound op on the
// CPU (graph.OpSound, its table laid out by SoundTable) and holds each
// output to ffmpeg's (scripts/soundgold.sh): the clip's outputs within
// soundBand, zeros past them. A clip read as another sample format and a
// mix that averages the channels are the violations it must see.
func TestComposedCPUSound(t *testing.T) {
	const N = 4096
	paths, err := filepath.Glob(filepath.Join("..", "testdata", "golden", "sound", "*.wav"))
	if err != nil || len(paths) < 20 {
		t.Fatalf("%d golden clips (%v): scripts/soundgold.sh records them", len(paths), err)
	}
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	forms := map[uint8][]Code{}
	form := func(t *testing.T, f uint8) []Code {
		if codes, ok := forms[f]; ok {
			return codes
		}
		b := graph.NewBlock(-1)
		b.In("unused", 1)
		b.Out(b.Emit(graph.OpSound, nil, []graph.Result{{Name: "sound", Width: N}}, nil, graph.Attrs{Sound: f, Taps: 1 << 12}, nil, nil)[0])
		if err := graph.Verify(b.G, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		_, codes, err := r.Form(Request{Graph: b.G, Step: Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
		if err != nil {
			t.Fatalf("sample format %d: %v", f, err)
		}
		forms[f] = codes
		return codes
	}
	run := func(t *testing.T, f uint8, words []uint32) []float32 {
		got, err := runCPUBound(hostArm(), form(t, f), []float32{0}, N, [3][]uint32{}, nil, func(o Operand) []uint32 {
			if o.Name == "picture" {
				return words
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	ran, formats := 0, map[uint8]bool{}
	for _, path := range paths {
		c := readGoldPCM(t, path)
		want := readGoldClip(t, path).want
		t.Run(strings.TrimSuffix(filepath.Base(path), ".wav"), func(t *testing.T) {
			mix, err := MonoMix(c.channels, c.mask)
			if err != nil {
				t.Fatal(err)
			}
			s, err := NewSWR(c.rate, 16000)
			if err != nil {
				t.Fatal(err)
			}
			words, outputs := SoundTable(nil, c.pcm, c.channels, c.frames, mix, s, N)
			if outputs != len(want) {
				t.Fatalf("%d outputs, ffmpeg's %d", outputs, len(want))
			}
			got := run(t, c.format, words)
			w := 0.0
			for i, g := range got {
				if math.IsNaN(float64(g)) {
					t.Fatalf("output %d is NaN", i)
				}
				v := float32(0)
				if i < len(want) {
					v = want[i]
				}
				w = max(w, math.Abs(float64(g-v)))
			}
			if !(w <= soundBand) {
				t.Fatalf("worst %.3g from ffmpeg over %d outputs, band %g", w, outputs, soundBand)
			}
			t.Logf("format %d, %d Hz, %d channel(s): %d outputs, worst %.3g", c.format, c.rate, c.channels, outputs, w)
			if outputs > 0 {
				other := graph.SoundU8
				if c.format == graph.SoundU8 {
					other = graph.SoundS16
				}
				if d := worstFrom(run(t, other, words), want); d <= soundBand {
					t.Errorf("violation: the clip read as sample format %d is %.3g from ffmpeg, within the band", other, d)
				}
			}
			if c.channels > 1 && outputs > 0 {
				avg := make([]float32, c.channels)
				for i := range avg {
					avg[i] = 1 / float32(c.channels)
				}
				bad, _ := SoundTable(nil, c.pcm, c.channels, c.frames, avg, s, N)
				if d := worstFrom(run(t, c.format, bad), want); d <= soundBand {
					t.Errorf("violation: the average of the channels is %.3g from ffmpeg, within the band", d)
				}
			}
			ran++
			formats[c.format] = true
		})
	}
	if ran != len(paths) || len(formats) != int(graph.SoundFormats)-1 {
		t.Fatalf("%d of %d clips ran, %d of %d sample formats", ran, len(paths), len(formats), graph.SoundFormats-1)
	}
}

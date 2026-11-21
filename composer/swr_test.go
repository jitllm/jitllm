package composer

import (
	le "encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// soundBand is how far a resampled sample may be from ffmpeg's: the
// exact-rational ratios agree to 2e-7, the ratios swr interpolates between
// phases to 5e-6 (its SIMD sums and interpolation in another order).
const soundBand = 5e-6

// goldClip is one of scripts/soundgold.sh's WAVs: its samples as swr's
// audioconvert makes floats of them, channel-interleaved.
type goldClip struct {
	name           string
	channels, rate int
	mask           uint64
	samples        []float32
	want           []float32
}

func readGoldClip(t *testing.T, path string) goldClip {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	c := goldClip{name: strings.TrimSuffix(filepath.Base(path), ".wav")}
	if len(b) < 12 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		t.Fatalf("%s: not a WAV", path)
	}
	var tag, bits uint16
	for p := 12; p+8 <= len(b); {
		id, n := string(b[p:p+4]), int(le.LittleEndian.Uint32(b[p+4:]))
		body := b[p+8 : min(p+8+n, len(b))]
		switch id {
		case "fmt ":
			tag = le.LittleEndian.Uint16(body)
			c.channels = int(le.LittleEndian.Uint16(body[2:]))
			c.rate = int(le.LittleEndian.Uint32(body[4:]))
			bits = le.LittleEndian.Uint16(body[14:])
			if tag == 0xFFFE {
				c.mask = uint64(le.LittleEndian.Uint32(body[20:]))
				tag = le.LittleEndian.Uint16(body[24:])
			}
		case "data":
			c.samples = decodeGold(t, tag, bits, body)
		}
		p += 8 + n + n%2
	}
	raw, err := os.ReadFile(strings.TrimSuffix(path, ".wav") + ".f32")
	if err != nil {
		t.Fatal(err)
	}
	c.want = make([]float32, len(raw)/4)
	for i := range c.want {
		c.want[i] = math.Float32frombits(le.LittleEndian.Uint32(raw[4*i:]))
	}
	return c
}

// decodeGold is audioconvert.c's conversion to float of a sample format.
func decodeGold(t *testing.T, tag, bits uint16, d []byte) []float32 {
	var out []float32
	switch {
	case tag == 3 && bits == 32:
		for i := 0; i+4 <= len(d); i += 4 {
			out = append(out, math.Float32frombits(le.LittleEndian.Uint32(d[i:])))
		}
	case tag == 3 && bits == 64:
		for i := 0; i+8 <= len(d); i += 8 {
			out = append(out, float32(math.Float64frombits(le.LittleEndian.Uint64(d[i:]))))
		}
	case bits == 8:
		for _, v := range d {
			out = append(out, float32(int(v)-0x80)*(1.0/(1<<7)))
		}
	case bits == 16:
		for i := 0; i+2 <= len(d); i += 2 {
			out = append(out, float32(int16(le.LittleEndian.Uint16(d[i:])))*(1.0/(1<<15)))
		}
	case bits == 24:
		// pcm_s24le decodes to s32, its sample in the top 24 bits.
		for i := 0; i+3 <= len(d); i += 3 {
			v := int32(uint32(d[i])<<8 | uint32(d[i+1])<<16 | uint32(d[i+2])<<24)
			out = append(out, float32(v)*(1.0/(1<<31)))
		}
	case bits == 32:
		for i := 0; i+4 <= len(d); i += 4 {
			out = append(out, float32(int32(le.LittleEndian.Uint32(d[i:])))*(1.0/(1<<31)))
		}
	default:
		t.Fatalf("a WAV of format %d, %d bits", tag, bits)
	}
	return out
}

// resampleGold is clip c mixed by mix and resampled by s, as a sound op
// computes it from the plan: each output the sum of its window's mono
// samples, read where SoundAt says, times its phase's filter.
func resampleGold(c goldClip, mix []float32, s SWR) []float32 {
	n := len(c.samples) / c.channels
	mono := make([]float32, n)
	for j := range mono {
		var v float32
		for ch, w := range mix {
			v += w * c.samples[j*c.channels+ch]
		}
		mono[j] = v
	}
	_, outputs := s.Clip(n)
	steps := make([]uint32, outputs*SWRStepWords)
	s.Steps(0, steps)
	out := make([]float32, outputs)
	for i := range out {
		first, ph, frac := int(int32(steps[3*i])), int(steps[3*i+1]), steps[3*i+2]
		dot := func(row int) float32 {
			var v float32
			for k := range s.Taps {
				v += mono[SoundAt(first+k, n)] * s.Bank[row*s.Taps+k]
			}
			return v
		}
		v := dot(ph)
		if s.Linear() {
			v2 := dot(ph + 1)
			v += (v2 - v) * float32(frac) * float32(1/float64(s.SrcIncr))
		}
		out[i] = v
	}
	return out
}

func worstFrom(got, want []float32) float64 {
	w := 0.0
	for i := range min(len(got), len(want)) {
		w = max(w, math.Abs(float64(got[i]-want[i])))
	}
	return w
}

// TestSWRMatchesFFmpeg holds the resample plan and the mono mix to ffmpeg's
// own samples (scripts/soundgold.sh): each golden clip -- every sample
// format, nine rates, layouts from mono to 7.1, clips shorter than a filter
// -- decoded as swr decodes it, mixed by MonoMix and resampled through its
// SWR's bank, steps and edges, against `ffmpeg -ac 1 -ar 16000`: the output
// count exactly, every sample within soundBand. The violations, a window of
// the wrong beta and a mix that averages, leave the band.
func TestSWRMatchesFFmpeg(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("..", "testdata", "golden", "sound", "*.wav"))
	if err != nil || len(paths) < 20 {
		t.Fatalf("%d golden clips (%v): scripts/soundgold.sh records them", len(paths), err)
	}
	resampled := 0
	for _, path := range paths {
		c := readGoldClip(t, path)
		t.Run(c.name, func(t *testing.T) {
			mix, err := MonoMix(c.channels, c.mask)
			if err != nil {
				t.Fatal(err)
			}
			s, err := NewSWR(c.rate, 16000)
			if err != nil {
				t.Fatal(err)
			}
			got := resampleGold(c, mix, s)
			if len(got) != len(c.want) {
				t.Fatalf("%d outputs of %d samples, ffmpeg's %d", len(got), len(c.samples)/c.channels, len(c.want))
			}
			w := worstFrom(got, c.want)
			if !(w <= soundBand) {
				t.Fatalf("worst %.3g from ffmpeg over %d outputs, band %g", w, len(got), soundBand)
			}
			t.Logf("%d Hz, %d channel(s), %d taps of %d phases: %d outputs, worst %.3g", c.rate, c.channels, s.Taps, s.Phases, len(got), w)
			if c.rate != 16000 && len(got) > 0 {
				resampled++
				bad := s
				bad.Bank = swrBank(min(16000*swrCutoff/float64(c.rate), 1), s.Taps, s.Phases, 8)
				if w := worstFrom(resampleGold(c, mix, bad), c.want); w <= soundBand {
					t.Errorf("violation: a window of beta 8 is %.3g from ffmpeg, within the band", w)
				}
			}
			if c.channels > 1 {
				avg := make([]float32, c.channels)
				for i := range avg {
					avg[i] = 1 / float32(c.channels)
				}
				if w := worstFrom(resampleGold(c, avg, s), c.want); w <= soundBand {
					t.Errorf("violation: the average of the channels is %.3g from ffmpeg, within the band", w)
				}
			}
		})
	}
	if resampled < 15 {
		t.Fatalf("%d clips resampled: the gate runs too few", resampled)
	}
}

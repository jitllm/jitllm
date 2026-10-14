package model

import (
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
)

// DecodeAudioDataURL decodes a data:audio/...;base64 voice clip: a WAV of
// 16 kHz mono 16-bit PCM, the samples scaled to [-1, 1] (audio.waveform's
// int16 / 32768). Any other rate, channel count or encoding is refused by
// name: the engine does not resample.
func DecodeAudioDataURL(u string) ([]float32, error) {
	rest, ok := strings.CutPrefix(u, "data:")
	head, data, found := strings.Cut(rest, ",")
	if !ok || !found || !strings.HasPrefix(head, "audio/") || !strings.HasSuffix(head, ";base64") {
		return nil, fmt.Errorf("audio must be a data:audio/...;base64 URL")
	}
	b, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, fmt.Errorf("audio: %w", err)
	}
	return DecodeWAV(b)
}

// DecodeWAV reads a RIFF WAV of 16 kHz mono 16-bit PCM into samples in
// [-1, 1].
func DecodeWAV(b []byte) ([]float32, error) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, fmt.Errorf("audio: not a WAV file (only 16 kHz mono 16-bit PCM WAV is read)")
	}
	var fmtOK bool
	for p := 12; p+8 <= len(b); {
		id, n := string(b[p:p+4]), int(binary.LittleEndian.Uint32(b[p+4:]))
		body := p + 8
		if n < 0 || body+n > len(b) {
			n = len(b) - body
		}
		switch id {
		case "fmt ":
			if n < 16 {
				return nil, fmt.Errorf("audio: a WAV fmt chunk of %d bytes", n)
			}
			format := binary.LittleEndian.Uint16(b[body:])
			ch := binary.LittleEndian.Uint16(b[body+2:])
			rate := binary.LittleEndian.Uint32(b[body+4:])
			bits := binary.LittleEndian.Uint16(b[body+14:])
			if format != 1 || ch != 1 || rate != 16000 || bits != 16 {
				return nil, fmt.Errorf("audio: a WAV of format %d, %d channel(s), %d Hz, %d bits: only 16 kHz mono "+
					"16-bit PCM is read (the engine does not resample): %w", format, ch, rate, bits, ErrUnsupportedAudio)
			}
			fmtOK = true
		case "data":
			if !fmtOK {
				return nil, fmt.Errorf("audio: a WAV data chunk before its fmt chunk")
			}
			pcm := b[body : body+n-n%2]
			out := make([]float32, len(pcm)/2)
			for i := range out {
				out[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768
			}
			return out, nil
		}
		p = body + n + n%2
	}
	return nil, fmt.Errorf("audio: a WAV with no data chunk")
}

// ErrUnsupportedAudio is a clip in a format the engine does not read.
var ErrUnsupportedAudio = errors.New("unsupported audio")

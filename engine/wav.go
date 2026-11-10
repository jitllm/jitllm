package engine

import (
	"encoding/binary"
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// ReadWAV is a RIFF WAV's clip as the file stores it: its PCM -- unsigned
// 8-bit, signed 16-, 24- or 32-bit, or IEEE float of 32 or 64 bits, a
// WAVE_FORMAT_EXTENSIBLE file's with its speakers' mask -- its channels and
// its rate. Nothing is converted here: the audio tower's entry reads the
// samples as they lie (model.Clip). A file of another codec is refused.
func ReadWAV(b []byte) (model.Clip, error) {
	var c model.Clip
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return c, fmt.Errorf("audio: not a WAV file")
	}
	fmtOK := false
	for p := 12; p+8 <= len(b); {
		id, n := string(b[p:p+4]), int(binary.LittleEndian.Uint32(b[p+4:]))
		body := p + 8
		if n < 0 || body+n > len(b) {
			n = len(b) - body
		}
		switch id {
		case "fmt ":
			if n < 16 {
				return c, fmt.Errorf("audio: a WAV fmt chunk of %d bytes", n)
			}
			tag := binary.LittleEndian.Uint16(b[body:])
			c.Channels = int(binary.LittleEndian.Uint16(b[body+2:]))
			c.Rate = int(binary.LittleEndian.Uint32(b[body+4:]))
			bits := binary.LittleEndian.Uint16(b[body+14:])
			if tag == 0xFFFE {
				if n < 40 {
					return c, fmt.Errorf("audio: a WAVE_FORMAT_EXTENSIBLE fmt chunk of %d bytes", n)
				}
				c.Mask = uint64(binary.LittleEndian.Uint32(b[body+20:]))
				tag = binary.LittleEndian.Uint16(b[body+24:])
			}
			switch {
			case tag == 1 && bits == 8:
				c.Format = graph.SoundU8
			case tag == 1 && bits == 16:
				c.Format = graph.SoundS16
			case tag == 1 && bits == 24:
				c.Format = graph.SoundS24
			case tag == 1 && bits == 32:
				c.Format = graph.SoundS32
			case tag == 3 && bits == 32:
				c.Format = graph.SoundF32
			case tag == 3 && bits == 64:
				c.Format = graph.SoundF64
			default:
				return c, fmt.Errorf("audio: a WAV of codec %#x at %d bits a sample: PCM of 8, 16, 24 or 32 bits and IEEE float of 32 or 64 are read", tag, bits)
			}
			if c.Channels < 1 || c.Rate < 1 {
				return c, fmt.Errorf("audio: a WAV of %d channel(s) at %d Hz", c.Channels, c.Rate)
			}
			fmtOK = true
		case "data":
			if !fmtOK {
				return c, fmt.Errorf("audio: a WAV data chunk before its fmt chunk")
			}
			frame := c.Channels * composer.SoundBytes(c.Format)
			c.PCM = b[body : body+n-n%frame]
			return c, nil
		}
		p = body + n + n%2
	}
	return c, fmt.Errorf("audio: a WAV with no data chunk")
}

package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
)

// Clip is a voice clip as its container stores it: frames of Channels
// samples, interleaved, at Rate samples a second, each in a sample format
// (graph.SoundU8 to SoundF64), the speakers laid out by Mask -- a WAV's
// dwChannelMask, 0 for the default layout of so many channels. The audio
// tower's entry reads it as it is (graph.OpSound): every sample's
// conversion, the mix to mono and the resample to the tower's rate are
// generated code.
type Clip struct {
	Format         uint8
	Channels, Rate int
	Mask           uint64
	PCM            []byte
}

// Frames is the clip's whole frames: a sample of every channel.
func (c Clip) Frames() int {
	if b := composer.SoundBytes(c.Format) * c.Channels; b > 0 {
		return len(c.PCM) / b
	}
	return 0
}

// FloatClip is mono samples at rate as a clip of floats.
func FloatClip(samples []float32, rate int) Clip {
	pcm := make([]byte, 4*len(samples))
	if len(samples) > 0 {
		copy(pcm, unsafe.Slice((*byte)(unsafe.Pointer(&samples[0])), 4*len(samples)))
	}
	return Clip{Format: graph.SoundF32, Channels: 1, Rate: rate, PCM: pcm}
}

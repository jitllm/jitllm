package engine

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
)

// EncodeClip is the rows a voice clip puts in the prompt: the clip -- mono
// samples in [-1, 1] at the tower's rate, cut to the longest the reference
// reads and padded with silence to half a second, as audio.waveform makes
// it -- through the audio tower's entry, its chained blocks and its head on
// the CPU device's composed forms, one row of the text model's width for
// each row the blocks read. The rows are the instance's until its next
// encode.
func (s *Instance) EncodeClip(clip []float32) ([]float32, error) {
	if err := s.audioReady(); err != nil {
		return nil, err
	}
	if len(clip) == 0 {
		return nil, fmt.Errorf("model: an empty clip")
	}
	defer s.fr.Enter()()
	a := s.m.Container.Vision().Audio
	tr, tg := s.audio, s.m.Audio
	n := max(min(len(clip), int(a.MaxSamples)), int(a.SampleRate)/2)
	if tr.clip == nil {
		tr.clip = make([]float32, a.MaxSamples)
	}
	buf := tr.clip[:n]
	clear(buf[copy(buf, clip):])
	// The samples are the step's picture table, f32 words as they lie.
	words := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(buf))), n)
	tr.req[composer.RequestPicW] = uint32(n)
	frames, rows := s.m.ClipRows(n)
	D := int(a.ProjDim)
	if cap(tr.out) < tg.Out*D {
		tr.out = make([]float32, tg.Out*D)
	}
	dst := tr.out[:tg.Out*D]
	if err := tr.s.chainRun(tr, tg, words, graph.PixRGBA8, dst, frames, rows); err != nil {
		return nil, fmt.Errorf("model: the audio tower: %w", err)
	}
	return dst[:rows*D], nil
}

// audioReady is why the instance cannot encode a clip, nil when it can.
func (s *Instance) audioReady() error {
	switch {
	case s.audio != nil:
		return nil
	case s.audioErr != nil:
		return fmt.Errorf("model: the audio tower did not compose: %w", s.audioErr)
	}
	return fmt.Errorf("model: the model has no audio tower")
}

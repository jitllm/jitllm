package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/model"
)

// EncodeClip is the rows a voice clip puts in the prompt: the clip as its
// container stores it, mixed to mono and resampled to the tower's rate as
// ffmpeg does (graph.OpSound), cut and padded as the tower reads it
// (Model.ClipLength), through the audio tower's entry, its chained blocks and
// its head on the CPU device's composed forms, one row of the text model's
// width for each row the blocks read. The rows are the instance's until its
// next encode.
func (s *Instance) EncodeClip(clip model.Clip) ([]float32, error) {
	if err := s.audioReady(); err != nil {
		return nil, err
	}
	defer s.fr.Enter()()
	return s.encodeClip(&clip)
}

// encodeClip is EncodeClip under the pager the caller holds.
func (s *Instance) encodeClip(clip *model.Clip) ([]float32, error) {
	a := s.m.Container.Vision().Audio
	tr, tg := s.audio, s.m.Audio
	if composer.SoundBytes(clip.Format) == 0 || clip.Frames() == 0 {
		return nil, fmt.Errorf("model: a clip of sample format %d and %d frames", clip.Format, clip.Frames())
	}
	if tr.swr.In != clip.Rate || tr.swr.Out != int(a.SampleRate) {
		swr, err := composer.NewSWR(clip.Rate, int(a.SampleRate))
		if err != nil {
			return nil, err
		}
		tr.swr = swr
	}
	if tr.mixOf != [2]uint64{uint64(clip.Channels), clip.Mask} {
		mix, err := composer.MonoMix(clip.Channels, clip.Mask)
		if err != nil {
			return nil, err
		}
		tr.mix, tr.mixOf = mix, [2]uint64{uint64(clip.Channels), clip.Mask}
	}
	if tr.swr.Taps*clip.Channels > composer.SoundTapsMax {
		return nil, fmt.Errorf("model: a clip of %d channel(s) at %d Hz resamples through %d taps of each, past the sound op's %d",
			clip.Channels, clip.Rate, tr.swr.Taps, composer.SoundTapsMax)
	}
	// The clip's PCM as it lies and its plan are the step's picture table.
	words, outputs := composer.SoundTable(tr.clip, clip.PCM, clip.Channels, clip.Frames(), tr.mix, tr.swr, int(a.MaxSamples))
	tr.clip = words
	n := s.m.ClipLength(outputs)
	tr.req[composer.RequestPicW] = uint32(n)
	frames, rows := s.m.ClipRows(n)
	D := int(a.ProjDim)
	if cap(tr.out) < tg.Out*D {
		tr.out = make([]float32, tg.Out*D)
	}
	dst := tr.out[:tg.Out*D]
	if err := tr.s.chainRun(tr, tg, words, clip.Format, dst, frames, rows); err != nil {
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

// encodeClips is spans with every clip span's rows in place: encoded into a
// buffer of the audio run's (one a clip of the prompt), or, for a clip whose
// rows all lie before row from -- restored by the prefix cache -- a buffer
// nothing reads, with no audio block run. The caller's spans are left as
// they were.
func (s *Instance) encodeClips(spans []model.Span, from int) ([]model.Span, error) {
	if err := s.audioReady(); err != nil {
		return nil, err
	}
	tr := s.audio
	D := int(s.m.Container.Vision().Audio.ProjDim)
	tr.spans = append(tr.spans[:0], spans...)
	k, row := 0, 0
	for i := range tr.spans {
		sp := &tr.spans[i]
		r := sp.N(s.c.NEmbd)
		if sp.Clip != nil && sp.Embd == nil {
			if k == len(tr.pics) {
				tr.pics = append(tr.pics, nil)
			}
			if cap(tr.pics[k]) < sp.ClipRows*D {
				tr.pics[k] = make([]float32, sp.ClipRows*D)
			}
			sp.Embd = tr.pics[k][:sp.ClipRows*D]
			k++
			if row+r > from {
				rows, err := s.encodeClip(sp.Clip)
				if err != nil {
					return nil, fmt.Errorf("model: span %d: %w", i, err)
				}
				if len(rows) != len(sp.Embd) {
					return nil, fmt.Errorf("model: span %d is a clip of %d rows and the tower writes %d", i, sp.ClipRows, len(rows)/D)
				}
				copy(sp.Embd, rows)
			}
		}
		row += r
	}
	return tr.spans, nil
}

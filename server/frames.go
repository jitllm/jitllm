package server

import "github.com/jitllm/jitllm/format/jlm"

// frames is the model's frames on its CPU device: the memory the blocks its
// instances run on the CPU are read into, made with its first instance at
// load and budgeted by its instances' quotas. nil before that or after the
// last, which every statistic reads as nothing held.
func (lm *LoadedModel) frames() *jlm.Frames { return lm.cpu.HeldFrames(lm.m.Id) }

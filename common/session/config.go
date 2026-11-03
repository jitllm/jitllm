package session

import (
	"fmt"
	"os"
	"path/filepath"
)

// KVCacheDir is where the prompt cache is kept: this system's cache folder,
// which it may clear.
func KVCacheDir() string {
	root, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(root, "jitllm", "kv")
}

// Sampling is the decode sampler's settings, persisted between runs.
type Sampling struct {
	Temp        float32 `json:"temp"`
	TopK        int     `json:"top_k"`
	TopP        float32 `json:"top_p"`
	MinP        float32 `json:"min_p"`
	RepeatPen   float32 `json:"repeat_pen"`
	RepeatLastN int     `json:"repeat_last_n"`
	Seed        int64   `json:"seed"`
}

// DefaultSampling is the sampler a fresh install starts with.
func DefaultSampling() Sampling {
	return Sampling{
		Temp:        0.8,
		TopK:        40,
		TopP:        0.95,
		MinP:        0.05,
		RepeatPen:   1.1,
		RepeatLastN: 64,
		Seed:        0,
	}
}

// Knob is one sampling setting as a front end offers it: its name, its range
// and step, how to read and write it, and the words for its value.
type Knob struct {
	Name           string
	Min, Max, Step float32
	Get            func(Sampling) float32
	Set            func(*Sampling, float32)
	// Caption is the value in words: "off" where the setting does nothing.
	Caption func(Sampling) string
}

// SamplingKnobs() are the sampler's settings, in the order the sampler applies
// them. The filters run in llama.cpp's order -- top-k, top-p, min-p --
// because the other order gives a different distribution, and matching it is
// what makes a prompt behave comparably in both engines. The seed is not a
// knob: it is a value, not a range.
func SamplingKnobs() []Knob {
	return []Knob{
		// The temperature caption names greedy: at zero model.Sampler returns the
		// argmax before reading any other field, so the other knobs are inert.
		{Name: "Temperature", Min: 0, Max: 2, Step: 0.05,
			Get: func(s Sampling) float32 { return s.Temp },
			Set: func(s *Sampling, v float32) { s.Temp = v },
			Caption: func(s Sampling) string {
				if s.Temp <= 0 {
					return "0, always the likeliest token"
				}
				return fmt.Sprintf("%.2f", s.Temp)
			}},
		{Name: "Top-k", Min: 0, Max: 200, Step: 1,
			Get:     func(s Sampling) float32 { return float32(s.TopK) },
			Set:     func(s *Sampling, v float32) { s.TopK = int(v) },
			Caption: func(s Sampling) string { return OffInt(s.TopK, 0, "most likely tokens") }},
		{Name: "Top-p", Min: 0, Max: 1, Step: 0.01,
			Get: func(s Sampling) float32 { return s.TopP },
			Set: func(s *Sampling, v float32) { s.TopP = v },
			Caption: func(s Sampling) string {
				if s.TopP <= 0 || s.TopP >= 1 {
					return "off"
				}
				return fmt.Sprintf("%.2f", s.TopP)
			}},
		{Name: "Min-p", Min: 0, Max: 0.5, Step: 0.01,
			Get: func(s Sampling) float32 { return s.MinP },
			Set: func(s *Sampling, v float32) { s.MinP = v },
			Caption: func(s Sampling) string {
				if s.MinP <= 0 {
					return "off"
				}
				return fmt.Sprintf("%.2f of the likeliest", s.MinP)
			}},
		{Name: "Repeat penalty", Min: 1, Max: 2, Step: 0.01,
			Get: func(s Sampling) float32 { return s.RepeatPen },
			Set: func(s *Sampling, v float32) { s.RepeatPen = v },
			Caption: func(s Sampling) string {
				if s.RepeatPen <= 1 {
					return "off"
				}
				return fmt.Sprintf("%.2f", s.RepeatPen)
			}},
		{Name: "Repeat window", Min: 0, Max: 512, Step: 8,
			Get:     func(s Sampling) float32 { return float32(s.RepeatLastN) },
			Set:     func(s *Sampling, v float32) { s.RepeatLastN = int(v) },
			Caption: func(s Sampling) string { return fmt.Sprintf("last %d tokens", s.RepeatLastN) }},
	}
}

// OffInt is an integer setting in words: "off" at its off value.
func OffInt(v, off int, what string) string {
	if v == off {
		return "off"
	}
	return fmt.Sprintf("%d %s", v, what)
}

package model

import "github.com/jitllm/jitllm/internal/oracle"

// Greedy is the reference's greedy token over logits: the largest, the lower
// id of two equal.
func Greedy(logits []float32) int32 { return oracle.Argmax(logits) }

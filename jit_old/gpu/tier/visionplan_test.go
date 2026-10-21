package tier

import "github.com/jitllm/jitllm/engine/nn"

// This helper is untagged so the package compiles on darwin: the untagged
// actdecline_test.go uses it, and keeping it in the linux-only
// visionmem_test.go broke the whole package's build on arm64.

// visionPlan is a ViT block: LayerNorm with biases, biased q/k/v/o, an ungated
// MLP with biases, non-causal attention over the call's rows and no rotary.
// NRot is zero, so initScratch emits no RoPE kernel and the copy path carries
// k into the cache.
func visionPlan() *nn.LayerPlan {
	return &nn.LayerPlan{
		NonCausal: true, LayerNorm: true, UngatedFFN: true,
		NEmbd: 256, NHead: 4, NKVHead: 4, HeadDim: 64, NRot: 0, NFFN: 1024,
		MaxSeq: 64, Act: nn.ActGELU, RMSEps: 1e-5,
	}
}

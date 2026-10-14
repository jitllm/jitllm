package cpu

import "testing"

// elementwiseInventory is the definition of "every elementwise op". The test
// below proves every entry produces runnable code on this architecture. Adding
// an elementwise op means adding a row here, so an omission is a visible diff.
//
// The GPU twins are enumerated in jit/gpu/lowertest, which lowers each of them
// on all three backends; jit/gpu/backend/layernorm_test.go runs the two that
// were added last against a float64 reference on real hardware.
var elementwiseInventory = []struct {
	name string
	emit func() []byte
}{
	// The norms. RMSNorm bakes its width; LayerNorm bakes its width and whether
	// the file carries a bias.
	{"rmsnorm/2048", func() []byte { return EmitRMSNorm(2048) }},
	{"rmsnorm/64", func() []byte { return EmitRMSNorm(64) }},
	{"layernorm/768", func() []byte { return EmitLayerNorm(768, false) }},
	{"layernorm/768+bias", func() []byte { return EmitLayerNorm(768, true) }},
	// The layer norm's passes turned to Gemma 3n's gaussian top-k and AltUp's
	// magnitude match (RowMode).
	{"layernorm-gausstopk/8192", func() []byte { return EmitGaussTopK(8192) }},
	{"layernorm-magmatch/2048", func() []byte { return EmitMagMatch(2048) }},

	// The softmax. Its length is runtime -- a score row is pos+1 and changes
	// every token, so baking it would mean a kernel per position.
	{"softmax", EmitSoftmax},

	// The activations, gated (an FFN's SwiGLU) and ungated (a ViT's FFN, which
	// has no `up` to multiply by). The activation itself is baked: it is
	// constant from model.Open, so it is a codegen input.
	{"actmul/silu", func() []byte { return EmitActMul(ActSiLU) }},
	{"actmul/gelu", func() []byte { return EmitActMul(ActGELU) }},
	{"actmul/swiglu-oai", func() []byte { return EmitActMul(ActSwiGLUOAI) }},
	// No activation: the gated product alone (Gemma 3n's AltUp scale).
	{"actmul/identity", func() []byte { return EmitActMul(ActIdentity) }},
	// SwiGLU with the gate and the up clamped: DeepSeek V4's experts.
	{"actmul/swiglu-clamp", func() []byte { return EmitActMul(ActSwiGLUClamp) }},
	// Kimi-K3's situ: both operands bounded by a scaled tanh, the gate's
	// times its sigmoid.
	{"actmul/situ", func() []byte { return EmitActMul(ActSitu) }},
	// GELU with erf, transformers' and torch's default (ModernBERT, Laya).
	{"actmul/gelu-erf", func() []byte { return EmitActMul(ActGELUErf) }},
	{"actmul/relu2", func() []byte { return EmitActMul(ActReLU2) }},
	{"act/silu", func() []byte { return EmitAct(ActSiLU) }},
	{"act/gelu", func() []byte { return EmitAct(ActGELU) }},
	{"act/gelu-erf", func() []byte { return EmitAct(ActGELUErf) }},
	// sigmoid(sign(x)*sqrt(max(|x|, 1e-6))): Qwen3.8-Flash-Next's n-gram gate.
	{"act/sqrt-sigmoid", func() []byte { return EmitAct(ActSqrtSigmoid) }},
	// quick-GELU, x*sigma(1.702x): CLIP's tower.
	{"act/quickgelu", func() []byte { return EmitAct(ActQuickGELU) }},
	// Squared ReLU, max(x,0)^2: Nemotron's ungated FFN.
	{"act/relu2", func() []byte { return EmitAct(ActReLU2) }},
	// ReLU, max(x,0): DeepSeek V3.2's lightning indexer.
	{"act/relu", func() []byte { return EmitAct(ActReLU) }},
	// sqrt(softplus(x)): DeepSeek V4's router gate.
	{"act/sqrt-softplus", func() []byte { return EmitAct(ActSqrtSoftplus) }},

	// sigma(gate) * value, where the two are different vectors: a hybrid's
	// attention and linear-layer output gates. actmul/silu would compute
	// g*sigma(g)*v, plausible and wrong, hence a separate kernel.
	{"sigmoidmul", EmitSigmoidMul},
	// x = ln(x): a log-mel feature's log (d1-omni's audio front end).
	{"ln", EmitLn},

	// dst += alpha*src: the residual adds, every attention and projection bias,
	// and the MoE's weighted accumulate. One kernel, because they are one shape.
	{"axpy", EmitAxpy},

	// dst *= alpha: gemma's embedding scale and every attention score row, on
	// the text path and the vision tower.
	{"scale", EmitScale},
	{"softcap", EmitSoftcap},
	// x = min(max(x, lo), hi): DBRX's clip_qkv on q, k and v.
	{"clamp", EmitClamp},

	// The hybrid's two gates, fused. softplus is the only transcendental in
	// the engine that is not exp.
	{"deltagate", EmitDeltaGate},
	// Kimi-K3's KDA decay under its lower bound.
	{"deltadecaybound", EmitDeltaDecayBound},
}

// blockInventory is the ops that are not elementwise: they read or write more
// than one value per lane, so each entry names a shape. They are listed
// separately because both read a buffer they write (the delta rule's state,
// the convolution's history), which a device kernel may not, so their device
// forms differ and lowertest cannot simply mirror this list.
var blockInventory = []struct {
	name string
	emit func() ([]byte, error)
}{
	// One head of the gated delta rule: n rows of n, decayed, dotted,
	// rank-one updated and dotted again.
	{"gated_delta/128", func() ([]byte, error) { return EmitGatedDelta(128) }},
	{"gated_delta/8", func() ([]byte, error) { return EmitGatedDelta(8) }},
	// The causal convolution and its state shift, plane-major.
	{"conv1d/4x8192", func() ([]byte, error) { return EmitConv1d(4, 8192) }},
	{"conv1d/4x64", func() ([]byte, error) { return EmitConv1d(4, 64) }},
	// DeepSeek V4's hyper-connection mixer (a row's 24 mixes, a Sinkhorn of
	// 20 rounds; and the head's collapse weights alone) and the compressor's
	// softmax pool over positions.
	{"hcmix/20", func() ([]byte, error) { return EmitHCMix(20, false) }},
	{"hcmix/head", func() ([]byte, error) { return EmitHCMix(0, true) }},
	{"colpool", func() ([]byte, error) { return EmitColPool(), nil }},
}

// TestEveryBlockOpIsGenerated is TestEveryElementwiseOpIsGenerated for the
// shaped kernels: every entry emits and maps executable on this architecture.
func TestEveryBlockOpIsGenerated(t *testing.T) {
	if len(blockInventory) == 0 {
		t.Fatal("the inventory is empty; this test would pass and prove nothing")
	}
	for _, e := range blockInventory {
		t.Run(e.name, func(t *testing.T) {
			b, err := e.emit()
			if err != nil {
				t.Fatalf("%s: %v", e.name, err)
			}
			if len(b) == 0 {
				t.Fatalf("%s emitted no code", e.name)
			}
			c := mustMap(t, b)
			c.Close()
		})
	}
	t.Logf("%d block ops generated on this architecture", len(blockInventory))
}

// TestEveryElementwiseOpIsGenerated checks every inventory entry emits and maps
// executable on this architecture. It checks coverage, not correctness (the
// per-op NMSE gates do that): a named op with no emitter, an emitter that
// produces nothing, or one that cannot be mapped.
func TestEveryElementwiseOpIsGenerated(t *testing.T) {
	if len(elementwiseInventory) == 0 {
		t.Fatal("the inventory is empty; this test would pass and prove nothing")
	}
	for _, e := range elementwiseInventory {
		t.Run(e.name, func(t *testing.T) {
			b := e.emit()
			if len(b) == 0 {
				t.Fatalf("%s emitted no code", e.name)
			}
			c := mustMap(t, b)
			c.Close()
		})
	}
	t.Logf("%d elementwise ops generated on this architecture", len(elementwiseInventory))
}

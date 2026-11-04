package oracle

// DecisionAnswer is the reference arithmetic of a decision's answer, in
// float64 as llama.cpp's server-decision.cpp (format_answer) and the models'
// own code compute it: each variant's scores scaled by 1/t and softmaxed,
// the second variant's put back in reverse, the variants averaged; then the
// expected level, the mode, TypeSafe's two confidences and Laya's entropy
// confidence.
type DecisionAnswer struct {
	Probs                       []float64
	Expected                    float64 // sum i * p_i
	Mode                        int
	ConfChoice, ConfScore, Laya float64
}

package jlm

import (
	"reflect"
	"testing"
)

// TestRouterWeightTailIsOptional holds the router's stated weight to the
// fifteenth tail: it round-trips after the score, a record that states no
// weight is the record written before the tail existed, and a weight with no
// score to select by, or not over the logits and the pick's, is refused.
func TestRouterWeightTailIsOptional(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 2, NEmbd: 64, NHead: 4, NKVHead: 2, HeadDim: 16, NRot: 16,
			NFFN: 128, NVocab: 1000, RMSEps: 1e-5, RopeBase: 1e4, NExpert: 4, NExpertUsed: 2,
			Acts: []Expr{{Terms: []Term{{Fn: FnMul, Args: [3]Arg{Operand(0), Operand(1)}}}}}}
	}
	var score, weight ExprBuilder
	score.Fn(FnMul, Operand(0), Num(1))
	ex := weight.Fn(FnExp, weight.Fn(FnSub, Operand(0), Operand(1)))
	weight.Fn(FnDiv, ex, weight.Sum(ex))
	plain := base()
	plain.RouterScore = Expr{Terms: score.T}
	stated := base()
	stated.RouterScore, stated.RouterWeight = Expr{Terms: score.T}, Expr{Terms: weight.T}
	short, long := encodeConfig(plain), encodeConfig(stated)
	if len(long) <= len(short) || string(long[:len(short)]) != string(short) {
		t.Fatalf("the weight's tail is not appended to the record without it (%d and %d bytes)", len(short), len(long))
	}
	for _, c := range []*Config{plain, stated} {
		got, err := decodeConfig(encodeConfig(c))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.RouterScore, c.RouterScore) || !reflect.DeepEqual(got.RouterWeight, c.RouterWeight) {
			t.Fatalf("round trip: score %+v weight %+v, want %+v and %+v", got.RouterScore, got.RouterWeight, c.RouterScore, c.RouterWeight)
		}
	}
	noScore := base()
	noScore.RouterWeight = Expr{Terms: weight.T}
	if err := exprsValid(noScore); err == nil {
		t.Error("a weight with no score to select by is valid")
	}
	oneOperand := base()
	oneOperand.RouterScore, oneOperand.RouterWeight = Expr{Terms: score.T}, Expr{Terms: score.T}
	if err := exprsValid(oneOperand); err == nil {
		t.Error("a weight over the logits alone is valid")
	}
	if err := exprsValid(stated); err != nil {
		t.Errorf("the stated weight is refused: %v", err)
	}
}

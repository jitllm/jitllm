package jlm

import (
	"reflect"
	"testing"
)

// TestLlama4ConfigTailIsOptional holds the two halves of why Llama 4 cost no
// container version: its fields round-trip, and a model without them writes
// exactly the record it wrote before they existed -- so a reader that stops
// after the grouped router reads it unchanged.
func TestLlama4ConfigTailIsOptional(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 4, NEmbd: 64, NHead: 4, NKVHead: 2,
			HeadDim: 16, NRot: 16, NFFN: 128, NVocab: 1000, RMSEps: 1e-5, RopeBase: 5e5}
	}
	plain := encodeConfig(base())
	l4 := base()
	l4.SWAWindow, l4.SWAPeriod = 8192, 4
	l4.MoEStep, l4.AttnTempScale, l4.AttnTempFloor, l4.AttnTempOffset = 2, 0.1, 8192, 1
	l4.Flags = FlagSWAChunked | FlagNoPEGlobal | FlagQKL2Norm | FlagExpertWeightIn |
		FlagExpertSigmoid | FlagNoExpertNorm
	tail := encodeConfig(l4)
	if len(tail) != len(plain)+16 {
		t.Fatalf("the tail is %d bytes, want 16", len(tail)-len(plain))
	}
	got, err := decodeConfig(tail)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, l4) {
		t.Errorf("round trip:\n got %+v\nwant %+v", got, l4)
	}
	// A record with no tail -- every container written before it, and every
	// non-Llama-4 one after -- reads as zeros.
	got, err = decodeConfig(plain)
	if err != nil {
		t.Fatal(err)
	}
	if got.MoEStep != 0 || got.AttnTempScale != 0 || got.AttnTempFloor != 0 || got.AttnTempOffset != 0 {
		t.Errorf("a tailless record read as %+v", got)
	}
	// A truncated tail is a truncated record, not a zero.
	if _, err := decodeConfig(tail[:len(tail)-4]); err == nil {
		t.Error("a truncated tail decoded")
	}
}

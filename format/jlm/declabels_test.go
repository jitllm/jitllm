package jlm

import (
	"bytes"
	"reflect"
	"testing"
)

// TestDecisionLabelsRoundTrip holds a label readout's statement through the
// config's encoding, every field set, and a config without one leaves none.
func TestDecisionLabelsRoundTrip(t *testing.T) {
	l := &DecisionLabels{Codes: []string{"A", "B", "AA"}, WideCodes: []string{"00", "01"}, Pool: []string{"a"},
		MaxOptions: 255, KeyCodes: true, Spaced: true, ShowCodes: true,
		NoulTrue: []string{"yes", "Yes"}, NoulFalse: []string{"no"}, NoulRatings: 9, NoulTrueFirst: true,
		ScoreByKey: true, Reversed: true, SortKeys: true, IDInstructions: true, EntropyConfidence: true,
		SortedLegend: true}
	c := &Config{NLayer: 1, NEmbd: 8, NHead: 1, NKVHead: 1, HeadDim: 8, NRot: 8, NFFN: 16,
		NVocab: 32, NCtx: 64, DecisionLabels: l}
	got, err := decodeConfig(encodeConfig(c))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.DecisionLabels, l) {
		t.Fatalf("the label readout came back as %+v, want %+v", got.DecisionLabels, l)
	}
	c.DecisionLabels = nil
	if got, err = decodeConfig(encodeConfig(c)); err != nil || got.DecisionLabels != nil {
		t.Fatalf("a config with no label readout decoded one: %+v (%v)", got.DecisionLabels, err)
	}
}

// TestAReadoutCodeReadsAsNothing holds the byte that named a readout by code
// before readouts were stated: a container written now carries zero there, one
// written then (code 7) reads as the same config, and each stated
// readout, and nothing else, makes a model that decides.
func TestAReadoutCodeReadsAsNothing(t *testing.T) {
	c := &Config{NLayer: 1, NEmbd: 8, NHead: 1, NKVHead: 1, HeadDim: 8, NRot: 8, NFFN: 16,
		NVocab: 32, NCtx: 64, DecisionBlocks: 0x5a5a5a5a, DecisionLabels: &DecisionLabels{Codes: []string{"A"}}}
	b := encodeConfig(c)
	at := bytes.Index(b, []byte{0x5a, 0x5a, 0x5a, 0x5a}) - 1
	if at < 0 || b[at] != 0 {
		t.Fatalf("the readout's byte (at %d) is not written zero", at)
	}
	want, err := decodeConfig(b)
	if err != nil {
		t.Fatal(err)
	}
	old := bytes.Clone(b)
	old[at] = 7
	got, err := decodeConfig(old)
	if err != nil {
		t.Fatalf("a container naming readout code 7 does not read: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("a container naming readout code 7 reads as %+v, want %+v", got, want)
	}
	for _, s := range []Config{{DecisionLabels: &DecisionLabels{}}, {MarkerPrompt: &MarkerPrompt{}}, {JointHead: &JointHead{}}} {
		if !s.Decides() {
			t.Errorf("%+v states a readout and does not decide", s)
		}
	}
	if (&Config{DecisionBlocks: 2, DecisionTemps: []DecisionTemp{{T: 1}}}).Decides() {
		t.Errorf("a config stating no readout decides")
	}
}

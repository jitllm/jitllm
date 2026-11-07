package jlm

import (
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
	c := &Config{Arch: ArchLlama, NLayer: 1, NEmbd: 8, NHead: 1, NKVHead: 1, HeadDim: 8, NRot: 8, NFFN: 16,
		NVocab: 32, NCtx: 64, Decision: DecisionLev, DecisionLabels: l}
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

package jlm

import "testing"

// TestGenerationTailRoundTrips holds the thirtieth tail to the config's
// contract: the stated draw round-trips, a config without it reads as none
// stated and generates as ReferenceGeneration says, and a draw
// GenerationConfig refuses is refused at open.
func TestGenerationTailRoundTrips(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 4, NEmbd: 64, NHead: 4, NKVHead: 2,
			HeadDim: 16, NRot: 16, NFFN: 128, NVocab: 1000, RMSEps: 1e-5, RopeBase: 1e4}
	}
	c := base()
	c.Generation = &Generation{DoSample: true, Temperature: 0.6, TopK: 20, TopP: 0.95, MinP: 0.05, TypicalP: 1, RepetitionPenalty: 1.1}
	got, err := decodeConfig(encodeConfig(c))
	if err != nil {
		t.Fatal(err)
	}
	if got.Generation == nil || *got.Generation != *c.Generation {
		t.Fatalf("round trip: %+v, want %+v", got.Generation, *c.Generation)
	}
	plain, err := decodeConfig(encodeConfig(base()))
	if err != nil {
		t.Fatal(err)
	}
	if plain.Generation != nil {
		t.Fatalf("a config without the tail states %+v", *plain.Generation)
	}
	if len(encodeConfig(base())) >= len(encodeConfig(c)) {
		t.Fatal("the stated draw wrote no tail")
	}
	// The parts' graphs imply the generation record: stating them and no
	// draw writes the zero record, which reads as none stated.
	parts := base()
	parts.Front = FrontGraphs{Names: []string{FrontHead}, Graphs: []string{"head"}}
	got, err = decodeConfig(encodeConfig(parts))
	if err != nil {
		t.Fatal(err)
	}
	if got.Generation != nil || got.Front.Graph(FrontHead) != "head" {
		t.Fatalf("the parts stated and no draw read as the draw %+v and the head %q", got.Generation, got.Front.Graph(FrontHead))
	}
	for _, bad := range []Generation{
		{Temperature: 1, TopP: 1.5, TypicalP: 1, RepetitionPenalty: 1},
		{Temperature: -1, TopP: 1, TypicalP: 1, RepetitionPenalty: 1},
		{Temperature: 1, TopP: 1, TypicalP: 1, RepetitionPenalty: 0},
		{Temperature: 1, TopP: 1, MinP: 2, TypicalP: 1, RepetitionPenalty: 1},
	} {
		c.Generation = &bad
		if _, err := decodeConfig(encodeConfig(c)); err == nil {
			t.Errorf("the draw %+v was accepted", bad)
		}
	}
}

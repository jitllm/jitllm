package jlm

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// TestRopeTableRoundTrips holds the twenty-ninth tail to the config's
// contract: the table and a picture's layout round-trip, a config without
// them reads none, sections without a table are refused (an upgrade below
// revision 3, a conversion again otherwise), and entries the rotary cannot
// hold are refused.
func TestRopeTableRoundTrips(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 2, NEmbd: 64, NHead: 4, NKVHead: 2,
			HeadDim: 16, NRot: 8, NFFN: 128, NVocab: 1000, RMSEps: 1e-5, RopeBase: 1e4, Revision: StatementRevision}
	}
	c := base()
	c.RopeSections = [4]uint32{1, 1, 1, 1}
	c.RopeTable = []RopePair{{0, 0}, {0, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {3, 2}, {3, 3}}
	c.PictureRope = PictureRope{Side: true, Axes: [4]PictureAxis{{Pos: true}, {Mul: PicOne, Mod: PicCols}, {Mul: PicOne, Div: PicCols}, {Base: PicOrdinal}}}
	got, err := decodeConfig(encodeConfig(c))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.RopeTable, c.RopeTable) || got.PictureRope != c.PictureRope {
		t.Fatalf("round trip: table %v, pictures %+v", got.RopeTable, got.PictureRope)
	}
	pairs, err := got.RopePairs()
	if err != nil || !reflect.DeepEqual(pairs, c.RopeTable) {
		t.Fatalf("the stated table reads %v, %v", pairs, err)
	}
	if plain, err := decodeConfig(encodeConfig(base())); err != nil || plain.RopeTable != nil || plain.PictureRope != (PictureRope{}) {
		t.Fatalf("a config without the tail read a table %v, pictures %+v (%v)", plain.RopeTable, plain.PictureRope, err)
	}
	if pairs, err := base().RopePairs(); pairs != nil || err != nil {
		t.Fatalf("a rotary of one position reads %v, %v", pairs, err)
	}
	bare := base()
	bare.RopeSections = c.RopeSections
	if _, err := bare.RopePairs(); err == nil || !strings.Contains(err.Error(), "convert the model again") {
		t.Fatalf("sections without a table: %v", err)
	}
	bare.Revision = 2
	if _, err := bare.RopePairs(); !errors.As(err, new(*UpgradeError)) {
		t.Fatalf("sections without a table below revision 3: %v", err)
	}
	for _, bad := range [][]RopePair{{{4, 0}}, {{0, 4}}, {{0, -1}}, make([]RopePair, 9)} {
		d := base()
		d.RopeTable = bad
		if _, err := decodeConfig(encodeConfig(d)); err == nil {
			t.Errorf("a table %v over a rotary of 8 was read", bad)
		}
	}
	d := base()
	d.PictureRope.Axes[2].Div = PicCells + 1
	if _, err := decodeConfig(encodeConfig(d)); err == nil {
		t.Error("a picture layout reading an undefined term was read")
	}
}

package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// TestGeomsRoundTrip holds the blocks' own geometries to the config's
// contract: they round-trip, a config without them reads every block at the
// model's, and a statement no block can run is refused.
func TestGeomsRoundTrip(t *testing.T) {
	base := func() *Config {
		return &Config{NLayer: 4, NEmbd: 64, NHead: 4, NKVHead: 1, HeadDim: 32, NRot: 32,
			NFFN: 128, NVocab: 1000, RMSEps: 1e-6, RopeBase: 1e6, SWAWindow: 64, SWAPeriod: 2}
	}
	c := base()
	c.Geoms = []Geom{{2, 16, 16, 128}, {1, 32, 32, 128}, {2, 16, 16, 256}, {1, 32, 32, 256}}
	got, err := decodeConfig(encodeConfig(c))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Geoms, c.Geoms) || got.HeadDimAt(2) != 16 || got.NKVHeadAt(0) != 2 || got.NFFNAt(3) != 256 {
		t.Fatalf("round trip: %v", got.Geoms)
	}
	plain, err := decodeConfig(encodeConfig(base()))
	if err != nil || plain.Geoms != nil || plain.HeadDimAt(0) != 32 || plain.NFFNAt(3) != 128 {
		t.Fatalf("a config without geometries read as %v (%v)", plain.Geoms, err)
	}
	for _, tc := range []struct {
		name string
		gs   []Geom
		want string
	}{
		{"a short statement", c.Geoms[:3], "3 blocks' geometries for 4 blocks"},
		{"kv heads that do not group", []Geom{{3, 16, 16, 128}, {1, 32, 32, 128}, {1, 32, 32, 128}, {1, 32, 32, 128}}, "block 0's 3 kv heads"},
		{"a rotary past its head", []Geom{{1, 16, 32, 128}, {1, 32, 32, 128}, {1, 32, 32, 128}, {1, 32, 32, 128}}, "block 0's rotary width 32 on a 16-wide head"},
		{"no feed-forward", []Geom{{1, 32, 32, 0}, {1, 32, 32, 128}, {1, 32, 32, 128}, {1, 32, 32, 128}}, "block 0 has no feed-forward width"},
	} {
		bad := base()
		bad.Geoms = tc.gs
		if _, err := decodeConfig(encodeConfig(bad)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", tc.name, err, tc.want)
		}
	}
}

// TestOlderGeomsAreStated holds a container an older writer made to the
// geometries it carried in fields of their own: the sliding layers' (zero
// where the model's) on every sliding block, and a double-width feed-forward
// on the KV-sharing blocks, its bit cleared; such fields beside a stated
// geometry, or with no sliding layer to carry them, are refused.
func TestOlderGeomsAreStated(t *testing.T) {
	c := &Config{NLayer: 4, NHead: 4, NKVHead: 1, HeadDim: 32, NRot: 32, NFFN: 128, SWAWindow: 64, SWAPeriod: 2,
		NKVShared: 1, Flags2: Flag2DoubleFFN | Flag2VNorm}
	if err := c.olderGeoms(Geom{HeadDim: 16, NRot: 16}); err != nil {
		t.Fatal(err)
	}
	want := []Geom{{1, 16, 16, 128}, {1, 32, 32, 128}, {1, 16, 16, 128}, {1, 32, 32, 256}}
	if !reflect.DeepEqual(c.Geoms, want) || c.Flags2 != Flag2VNorm {
		t.Fatalf("stated %v with flags %b, want %v and only the v norm's", c.Geoms, c.Flags2, want)
	}
	if err := c.olderGeoms(Geom{HeadDim: 16}); err == nil || !strings.Contains(err.Error(), "twice") {
		t.Errorf("an older geometry beside a stated one: %v", err)
	}
	flat := &Config{NLayer: 2, NHead: 4, NKVHead: 1, HeadDim: 32, NRot: 32, NFFN: 128}
	if err := flat.olderGeoms(Geom{HeadDim: 16}); err == nil || !strings.Contains(err.Error(), "no sliding layers") {
		t.Errorf("a sliding geometry with no sliding layers: %v", err)
	}
	if err := flat.olderGeoms(Geom{}); err != nil || flat.Geoms != nil {
		t.Errorf("a config with no older geometry stated %v (%v)", flat.Geoms, err)
	}
}

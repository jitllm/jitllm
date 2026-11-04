package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// sampleTower is a stated tower of every part, SmolVLM's shape.
func sampleTower(t *testing.T) *Tower {
	t.Helper()
	act, err := (&Vision{Flags: FlagGELU}).Activation(false)
	if err != nil {
		t.Fatal(err)
	}
	return &Tower{
		Version: TowerVersion,
		Pre: []PreStep{
			{Kind: PreResize, Ints: []int32{512, 512, int32(FilterTwoTap), int32(PrecFloat)}},
			{Kind: PreNormalize, Floats: []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}},
		},
		Entry: Program{NIn: 1, Out: 4, Ops: []TowerOp{
			{Code: TOpPatchify, In: []uint16{0}, Span: 16},
			{Code: TOpLinear, In: []uint16{1}, Bind: []Role{RoleVPatchEmbd, RoleVPatchBias}},
			{Code: TOpPosTable, Bind: []Role{RoleVPosEmbd}},
			{Code: TOpAdd, In: []uint16{2, 3}},
		}},
		Block: TowerBlock{NHead: 12, NKVHead: 12, HeadDim: 64, Norm: NormLayer, Eps: 1e-6, Act: act},
		Head: Program{NIn: 1, Out: 4, Ops: []TowerOp{
			{Code: TOpLayerNorm, In: []uint16{0}, Bind: []Role{RoleVPostNorm, RoleVPostNormBias}, Eps: 1e-6},
			{Code: TOpRegroup, In: []uint16{1}, Span: 4},
			{Code: TOpLinear, In: []uint16{2}, Bind: []Role{RoleVProj}},
			{Code: TOpPool, In: []uint16{1}, Span: 2},
		}},
		Span: TowerSpan{Before: []int32{198, 49189, 49152}, After: []int32{49189},
			BeforeText: "\n\n<start_of_image>", AfterText: "<end_of_image>\n\n", Bidir: true},
	}
}

// TestTowerRoundTrips: a stated tower comes back from the vision record as
// it went in, beside an audio tower and with none (the empty audio record a
// tower writes reads back as no audio), and a record with no tower reads as
// none.
func TestTowerRoundTrips(t *testing.T) {
	for _, aud := range []*Audio{nil, {NLayer: 2, NEmbd: 8, NHead: 2, NFFN: 16, ProjDim: 8}} {
		v := &Vision{NLayer: 12, NEmbd: 768, NHead: 12, NFFN: 3072, ImageSize: 512, PatchSize: 16,
			ProjDim: 576, Scale: 4, Eps: 1e-6, Projector: ProjIdefics3, Audio: aud, Tower: sampleTower(t)}
		got, err := decodeVision(encodeVision(v))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, v) {
			t.Fatalf("audio %v: the vision record came back as\n%+v\nnot\n%+v", aud != nil, got, v)
		}
	}
	plain := &Vision{NLayer: 1, NEmbd: 8, NHead: 2, NFFN: 16, ImageSize: 32, PatchSize: 16, Scale: 1, Eps: 1e-6}
	got, err := decodeVision(encodeVision(plain))
	if err != nil || got.Tower != nil || got.Audio != nil {
		t.Fatalf("a record with no tower read back as %+v, %v", got, err)
	}
	if want := []Role{RoleVPatchEmbd, RoleVPatchBias, RoleVPosEmbd, RoleVPostNorm, RoleVPostNormBias, RoleVProj}; !reflect.DeepEqual(sampleTower(t).Roles(), want) {
		t.Fatalf("Roles = %v, want %v", sampleTower(t).Roles(), want)
	}
}

// TestTowerRefusesWhatItCannotRead runs Valid against each violation it
// exists to catch, on the record and through the decoder, and a truncated
// record, which must be an error and not a panic.
func TestTowerRefusesWhatItCannotRead(t *testing.T) {
	cases := []struct {
		name, want string
		break_     func(*Tower)
	}{
		{"a later revision", "revision", func(t *Tower) { t.Version = TowerVersion + 1 }},
		{"an unknown step", "kind", func(t *Tower) { t.Pre[0].Kind = 99 }},
		{"a filter in the wrong arithmetic", "resizes", func(t *Tower) { t.Pre[0].Ints[3] = int32(PrecPIL) }},
		{"a std of zero", "std of 0", func(t *Tower) { t.Pre[1].Floats[4] = 0 }},
		{"an unknown op", "does not know", func(t *Tower) { t.Head.Ops[1].Code = topCodes }},
		{"a value read before it is made", "before it is made", func(t *Tower) { t.Entry.Ops[3].In[1] = 4 }},
		{"a linear with no matrix", "binds 0 weights", func(t *Tower) { t.Head.Ops[2].Bind = nil }},
		{"an add of one value", "reads 1 values", func(t *Tower) { t.Entry.Ops[3].In = []uint16{2} }},
		{"a shuffle of nothing", "groups by 0", func(t *Tower) { t.Head.Ops[1].Span = 0 }},
		{"a pool of nothing", "groups by 0", func(t *Tower) { t.Head.Ops[3].Span = 0 }},
		{"a norm with no epsilon", "epsilon", func(t *Tower) { t.Head.Ops[0].Eps = 0 }},
		{"a result past the last value", "past its last", func(t *Tower) { t.Head.Out = 9 }},
		{"heads over no kv heads", "kv heads", func(t *Tower) { t.Block.NKVHead = 5 }},
		{"no activation", "activation", func(t *Tower) { t.Block.Act = Expr{} }},
	}
	if err := sampleTower(t).Valid(); err != nil {
		t.Fatalf("the sample does not validate: %v", err)
	}
	for _, c := range cases {
		tw := sampleTower(t)
		c.break_(tw)
		err := tw.Valid()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Valid = %v, want an error naming %q", c.name, err, c.want)
			continue
		}
		v := &Vision{NLayer: 1, NEmbd: 8, NHead: 2, ImageSize: 32, PatchSize: 16, Scale: 1, Eps: 1e-6, Tower: tw}
		if _, err := decodeVision(encodeVision(v)); err == nil {
			t.Errorf("%s: the decoder read it", c.name)
		}
	}
	b := encodeVision(&Vision{NLayer: 12, NEmbd: 768, NHead: 12, ImageSize: 512, PatchSize: 16, Scale: 4,
		Eps: 1e-6, Tower: sampleTower(t)})
	for n := len(b) - 1; n > len(b)-120; n-- {
		if _, err := decodeVision(b[:n]); err == nil {
			t.Fatalf("a record cut to %d of %d bytes decoded", n, len(b))
		}
	}
}

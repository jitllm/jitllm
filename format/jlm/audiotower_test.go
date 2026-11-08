package jlm

import (
	"reflect"
	"testing"
)

// TestAudioTowerRoundTrips: an audio tower comes back from the vision record
// as it went in, beside the vision tower, with the longest clip it reads; a
// record whose audio states none reads as none, and a tower whose entry
// reads a value it has not made is refused.
func TestAudioTowerRoundTrips(t *testing.T) {
	var e ExprBuilder
	e.Fn(FnLog, e.Fn(FnAdd, Operand(0), Num(1.0/(1<<24))))
	logx := &Expr{Terms: e.T}
	at := &Tower{Version: TowerVersion,
		Entry: Program{NIn: 0, Out: 4, Ops: []TowerOp{
			{Code: TOpFrames, Width: 512, Stride: 160, Eps: 0.97},
			{Code: TOpLinear, In: []uint16{0}, Bind: []Role{RoleADFTCos}},
			{Code: TOpStated, In: []uint16{1}, Expr: logx},
			{Code: TOpFeatureNorm, In: []uint16{2}, Offset: 1, Eps: 1e-5},
			{Code: TOpClipMask, In: []uint16{3}, GridW: 64, Stride: 160, Span: 1},
		}},
		Chain: []ChainBlock{{Rows: 376, Width: 8, Prog: Program{NIn: 1, Out: 2, Ops: []TowerOp{
			{Code: TOpRowsAttend, In: []uint16{0, 0, 0, 0}, Bind: []Role{RoleAPos}, Span: 2},
			{Code: TOpRowConv, In: []uint16{1}, Bind: []Role{RoleADW}, Span: 9},
		}}}},
		Head: Program{NIn: 1, Out: 1, Ops: []TowerOp{{Code: TOpSplitRows, In: []uint16{0}, Width: 1}}},
	}
	if err := at.Valid(); err != nil {
		t.Fatal(err)
	}
	for _, with := range []bool{true, false} {
		a := &Audio{NLayer: 1, NEmbd: 8, NHead: 2, NFFN: 16, ProjDim: 8, MaxSamples: 480000}
		if with {
			a.Tower = at
		}
		v := &Vision{NLayer: 12, NEmbd: 768, NHead: 12, NFFN: 3072, ImageSize: 512, PatchSize: 16,
			ProjDim: 960, Eps: 1e-6, Projector: ProjIdefics3, Tower: sampleTower(t), Audio: a}
		got, err := decodeVision(encodeVision(v))
		if err != nil {
			t.Fatal(err)
		}
		if !with {
			a.MaxSamples = 0
		}
		if !reflect.DeepEqual(got.Audio, a) || !reflect.DeepEqual(got.Tower, v.Tower) {
			t.Fatalf("audio tower %v: read back %+v, wrote %+v", with, got.Audio, a)
		}
	}
	bad := *at
	bad.Entry.Ops = append([]TowerOp{{Code: TOpLinear, In: []uint16{0}, Bind: []Role{RoleADFTCos}}}, bad.Entry.Ops...)
	if err := bad.Valid(); err == nil {
		t.Fatal("an entry of no inputs whose first op reads a value is valid")
	}
}

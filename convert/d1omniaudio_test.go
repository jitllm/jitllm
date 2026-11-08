package convert

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestD1OmniStatesItsAudio reads d1-omni's mmproj: the audio tower is stated
// -- an entry that frames the clip and reads no value, a chained block for
// each of its blocks over the longest clip's rows, a head -- and every role
// its programs bind is a tensor the conversion wrote, at the shape it binds
// it: the front end's tables, the subsampling's convolutions at F32, each
// block's position table for the positions its rows reach, and its
// depthwise taps channel-major.
func TestD1OmniStatesItsAudio(t *testing.T) {
	src := testmodels.Path("d1-omni/mmproj-d1-omni-600M-F16.gguf")
	if _, err := os.Stat(src); err != nil {
		testmodels.Missing(t, "%s (d1-omni's mmproj)", src)
	}
	f, err := gguf.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	a, ts, err := audioOf(f, 12)
	if err != nil {
		t.Fatal(err)
	}
	if a == nil || a.Tower == nil {
		t.Fatal("the mmproj's audio tower is not stated")
	}
	tw := a.Tower
	if tw.Entry.NIn != 0 || tw.Entry.Ops[0].Code != jlm.TOpFrames || len(tw.Chain) != int(a.NLayer) || tw.Head.NIn != 1 {
		t.Fatalf("entry of %d inputs from op %d, %d chained blocks of %d, head of %d inputs",
			tw.Entry.NIn, tw.Entry.Ops[0].Code, len(tw.Chain), a.NLayer, tw.Head.NIn)
	}
	rows := audioRows(a)
	if rows != 376 || tw.Chain[0].Rows != int32(rows) {
		t.Fatalf("blocks over %d rows, the longest clip's %d", tw.Chain[0].Rows, rows)
	}
	shape := map[[2]int32][2]uint64{}
	typ := map[[2]int32]jlm.Type{}
	for _, e := range ts {
		k := [2]int32{int32(e.Role), e.Block}
		shape[k], typ[k] = [2]uint64{e.Dims[0], e.Dims[1]}, e.Type
	}
	d, bins := uint64(a.NEmbd), uint64(a.NFFT/2+1)
	for _, c := range []struct {
		role  jlm.Role
		block int32
		dims  [2]uint64
	}{
		{jlm.RoleADFTCos, jlm.DenseBlock, [2]uint64{uint64(a.NFFT), bins}},
		{jlm.RoleADFTSin, jlm.DenseBlock, [2]uint64{uint64(a.NFFT), bins}},
		{jlm.RoleAMelBank, jlm.DenseBlock, [2]uint64{bins, uint64(a.MelBins)}},
		{jlm.RoleASub0, jlm.DenseBlock, [2]uint64{9, uint64(a.SubChannels)}},
		{jlm.RoleASubPW1, jlm.DenseBlock, [2]uint64{uint64(a.SubChannels), uint64(a.SubChannels)}},
		{jlm.RoleAPos, 12, [2]uint64{d, uint64(2*rows - 1)}},
		{jlm.RoleAPos, 12 + int32(a.NLayer) - 1, [2]uint64{d, uint64(2*rows - 1)}},
		{jlm.RoleADW, 12, [2]uint64{uint64(a.ConvKernel), d}},
	} {
		k := [2]int32{int32(c.role), c.block}
		if shape[k] != c.dims || typ[k] != jlm.TypeF32 {
			t.Errorf("%v of block %d: %v %v, want %v F32", c.role, c.block, shape[k], typ[k], c.dims)
		}
	}
	bound := map[jlm.Role]bool{}
	for _, e := range ts {
		bound[e.Role] = true
	}
	progs := []*jlm.Program{&tw.Entry, &tw.Head}
	for i := range tw.Chain {
		progs = append(progs, &tw.Chain[i].Prog)
	}
	for _, p := range progs {
		for _, op := range p.Ops {
			for _, r := range op.Bind {
				if !bound[r] {
					t.Errorf("a program binds %v, which the conversion did not write", r)
				}
			}
		}
	}
	t.Logf("%d blocks over %d rows, %d entry ops, %d head ops", len(tw.Chain), rows, len(tw.Entry.Ops), len(tw.Head.Ops))
}

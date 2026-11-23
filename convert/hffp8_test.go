package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
)

// stBlob is a safetensors file of the named tensors, in order.
func stBlob(t *testing.T, ts []stTensor) *safetensors.File {
	t.Helper()
	head := map[string]any{}
	var data []byte
	for _, x := range ts {
		head[x.name] = map[string]any{"dtype": x.dtype, "shape": x.shape, "data_offsets": []int{len(data), len(data) + len(x.data)}}
		data = append(data, x.data...)
	}
	h, err := json.Marshal(head)
	if err != nil {
		t.Fatal(err)
	}
	b := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	b = append(append(b, h...), data...)
	f, err := safetensors.OpenAt("fp8.safetensors", bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

type stTensor struct {
	name, dtype string
	shape       []uint64
	data        []byte
}

func fp8F32s(vs ...float32) []byte {
	var b []byte
	for _, v := range vs {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	return b
}

func seqBytes(n, seed int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*37 + seed*11 + 3)
	}
	return b
}

// TestFP8GroupsConvertAsStored holds hf_quant_config.json's two block
// formats to the rows the container holds them in: an NVFP4 bank's codes,
// E4M3 scales and per-sheet amax over the denominator, and an MXFP8 weight's
// values and E8M0 scales, merged into one tensor each and laid out by
// NV4Rows and MX8Rows, every byte as stored. A cast group is passed over;
// an unknown transform, a missing amax and scales that do not cover the
// codes are refused.
func TestFP8GroupsConvertAsStored(t *testing.T) {
	qc := `{"model": {"parameter_groups": [
		{"tensors": ["e.w13"], "weight_transform": "nvfp4_nearest", "weight_block_shape": [1, 16], "weight_scale_format": "amax_denominator"},
		{"tensors": ["m.w"], "weight_transform": "mxfp8_nearest", "weight_block_shape": [32, 32]},
		{"tensors": ["n.w"], "weight_transform": "cast"}]}}`
	dir := hfDir{fstest.MapFS{"hf_quant_config.json": {Data: []byte(qc)}}, "test"}
	groups, err := hfQuantGroupsOf(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || groups["e.w13"] != jlm.TypeNV4 || groups["m.w"] != jlm.TypeMX8 {
		t.Fatalf("groups %v", groups)
	}

	const sheets, rows, k = 2, 3, 48
	codes, scales := seqBytes(sheets*rows*k/2, 1), seqBytes(sheets*rows*k/16, 2)
	for i := range scales {
		scales[i] &= 0x77
	}
	vals, mscales := seqBytes(rows*64, 3), seqBytes(rows*2, 4)
	for i := range vals {
		vals[i] &= 0x7e
	}
	for i := range mscales {
		mscales[i] = 120 + mscales[i]%15
	}
	tensors := func() []stTensor {
		return []stTensor{
			{"e.w13", "U8", []uint64{sheets, rows, k / 2}, codes},
			{"e.w13_scale", "F8_E4M3", []uint64{sheets, rows, k / 16}, scales},
			{"e.w13_amax", "F32", []uint64{sheets}, fp8F32s(5.5, 1344)},
			{"e.w13_outer_scale_denominator", "F32", []uint64{1}, fp8F32s(2688)},
			{"m.w", "F8_E4M3", []uint64{rows, 64}, vals},
			{"m.w_scale", "U8", []uint64{rows, 2}, mscales},
		}
	}
	all := func(ts []stTensor) map[string]hfLoc {
		f := stBlob(t, ts)
		out := map[string]hfLoc{}
		for i := range f.Tensors {
			out[f.Tensors[i].Name] = hfLoc{f: f, t: &f.Tensors[i]}
		}
		return out
	}
	got := all(tensors())
	if err := mergeFP8(got, groups); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("merged to %d tensors, want the two weights alone", len(got))
	}
	nv, err := got["e.w13"].readFP8()
	if err != nil {
		t.Fatal(err)
	}
	want, err := jlm.NV4Rows(codes, scales, []float32{5.5 / 2688, 0.5}, sheets, rows, k)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(nv, want) {
		t.Fatal("the NVFP4 bank's rows are not its planes laid out with amax / denominator a sheet")
	}
	other, err := jlm.NV4Rows(codes, scales, []float32{5.5 / 2688, 5.5 / 2688}, sheets, rows, k)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(other, want) {
		t.Fatal("a sheet's global scale does not reach its rows: the comparison cannot see the amax")
	}
	if s := got["e.w13"].fp8Shape(); s[2] != k {
		t.Fatalf("the NVFP4 bank's shape %v, want %d columns", s, k)
	}
	mx, err := got["m.w"].readFP8()
	if err != nil {
		t.Fatal(err)
	}
	if want, err := jlm.MX8Rows(vals, mscales, rows, 64); err != nil || !bytes.Equal(mx, want) {
		t.Fatalf("the MXFP8 weight's rows are not its planes laid out (%v)", err)
	}

	for _, c := range []struct {
		name, why string
		edit      func([]stTensor) []stTensor
	}{
		{"no amax", "_amax", func(ts []stTensor) []stTensor { return append(ts[:2], ts[3:]...) }},
		{"scales short of the codes", "E4M3 a 16", func(ts []stTensor) []stTensor {
			ts[1] = stTensor{"e.w13_scale", "F8_E4M3", []uint64{sheets, rows, k/16 - 1}, scales[:sheets*rows*(k/16-1)]}
			return ts
		}},
		{"an amax a bank, not a sheet", "an f32 a sheet", func(ts []stTensor) []stTensor {
			ts[2] = stTensor{"e.w13_amax", "F32", []uint64{1}, fp8F32s(5.5)}
			return ts
		}},
		{"MXFP8 scales of another type", "E8M0 a 32", func(ts []stTensor) []stTensor {
			ts[5] = stTensor{"m.w_scale", "F8_E4M3", []uint64{rows, 2}, mscales}
			return ts
		}},
		{"no scale", "_scale", func(ts []stTensor) []stTensor { return ts[:5] }},
	} {
		err := mergeFP8(all(c.edit(tensors())), groups)
		if err == nil || !strings.Contains(err.Error(), c.why) {
			t.Errorf("%s: %v, want a refusal naming %q", c.name, err, c.why)
		}
	}

	bad := hfDir{fstest.MapFS{"hf_quant_config.json": {Data: []byte(strings.Replace(qc, `"mxfp8_nearest"`, `"mxfp6_nearest"`, 1))}}, "test"}
	if _, err := hfQuantGroupsOf(bad); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("an unknown transform: %v, want ErrNotImplemented", err)
	}
	if g, err := hfQuantGroupsOf(hfDir{fstest.MapFS{}, "test"}); g != nil || err != nil {
		t.Errorf("no hf_quant_config.json: %v, %v", g, err)
	}
}

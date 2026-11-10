package convert

import (
	"bytes"
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The GGUF side of MLA's fused up-projection, gated hermetically because the
// two halves fail differently. attn_v_b is a row range and must come through
// byte for byte; attn_k_b is a per-head transpose and must come through as
// the same values in the other order. Where qk_nope equals v_head_dim (every
// released DeepSeek) the two halves are the same shape, so a swap is invisible
// to a dimensional check.
//
// The k half's type is its own axis, hence the table: a quantized source gets
// a packed Q8_0 half (so model.mlaWeights can batch the absorb), a verbatim
// source stays verbatim, and a nope that is not whole Q8_0 blocks falls back
// to f32. An unconditional f32 half is correct and silently loses the batched
// gather, so all three are asserted.
func TestUnfuseMLASplitsTheAbsorbedPair(t *testing.T) {
	for _, tc := range []struct {
		name   string
		src    jlm.Type
		nope   uint64
		wantKB jlm.Type
		// exact is whether the k half must reproduce the source's values to the
		// bit. It does when nothing re-quantized it.
		exact bool
	}{
		{"a quantized source gets a packed k half", jlm.TypeQ8, 64, jlm.TypeQ8, false},
		{"a verbatim source stays verbatim", jlm.TypeF32, 64, jlm.TypeF32, true},
		{"a nope Q8_0 cannot block stays f32", jlm.TypeQ8, 16, jlm.TypeF32, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const (
				nh  = 3   // heads
				vh  = 32  // v_head_dim, deliberately not equal to nope
				lat = 128 // kv_lora_rank
				rot = 16
			)
			nope := tc.nope
			per := nope + vh
			rows := nh * per
			cfg := &jlm.Config{NHead: nh, HeadDim: uint32(nope) + rot, NRot: rot,
				HeadDimV: vh, KVLoraRank: lat}

			// The fixture is encoded by the package's own writer: hand-laid Q8_0
			// blocks once produced a NaN scale, a degenerate oracle (RULE 10).
			vals := make([]float32, rows*lat)
			for r := uint64(0); r < rows; r++ {
				for j := uint64(0); j < lat; j++ {
					// Distinct per (row, column) and comfortably inside f16's
					// range, so no scale saturates.
					vals[r*lat+j] = float32(math.Sin(float64(r)*0.37+float64(j)*0.11)) * 3
				}
			}
			var data []byte
			switch tc.src {
			case jlm.TypeQ8:
				data = quantizeQ8Rows(vals, lat, int(rows))
			case jlm.TypeF32:
				data = append([]byte(nil), f32AsBytes(vals)...)
			default:
				t.Fatalf("fixture cannot encode %v", tc.src)
			}
			src, ok := jlm.SourceType(tc.src)
			if !ok {
				t.Fatalf("%v has no source type", tc.src)
			}
			rowBytes := uint64(len(data)) / rows

			// What the source holds after its own rounding; comparing against this
			// rather than vals measures the transpose, not the fixture's encoder.
			held := make([][]float32, rows)
			for r := uint64(0); r < rows; r++ {
				held[r] = make([]float32, lat)
				if err := quant.Dequant32(src, data[r*rowBytes:(r+1)*rowBytes], held[r]); err != nil {
					t.Fatal(err)
				}
			}

			e := jlm.Tensor{Role: jlm.RoleAttnKVB, Type: tc.src, NDim: 2, Data: data,
				Name: "blk.0.attn_kv_b.weight"}
			e.Dims[0], e.Dims[1] = lat, rows

			out, split, err := unfuseMLA(&e, &Config{Config: cfg})
			if err != nil {
				t.Fatal(err)
			}
			if !split || len(out) != 2 {
				t.Fatalf("split=%v, %d tensor(s) -- this gate proved nothing", split, len(out))
			}
			kb, vb := out[0], out[1]
			if kb.Role != jlm.RoleAttnKB || vb.Role != jlm.RoleAttnVB {
				t.Fatalf("roles are %v and %v", kb.Role, vb.Role)
			}

			// ── shapes: one sheet per head, and the two are not alike ──────
			if kb.NDim != 3 || kb.Dims[0] != nope || kb.Dims[1] != lat || kb.Dims[2] != nh {
				t.Errorf("attn_k_b is %dD %v, want 3D {%d, %d, %d}", kb.NDim, kb.Dims[:3], nope, lat, nh)
			}
			if vb.NDim != 3 || vb.Dims[0] != lat || vb.Dims[1] != vh || vb.Dims[2] != nh {
				t.Errorf("attn_v_b is %dD %v, want 3D {%d, %d, %d}", vb.NDim, vb.Dims[:3], lat, vh, nh)
			}

			// ── the types are the point of this table ──────────────────────
			if kb.Type != tc.wantKB {
				t.Errorf("attn_k_b is %v, want %v", kb.Type, tc.wantKB)
			}
			// A packed k half is the reason: jlm.Packed is the predicate the
			// container consults for the batched absorb.
			if got, want := jlm.Packed(kb.Type), jlm.Packed(tc.wantKB); got != want {
				t.Errorf("jlm.Packed(%v) = %v, want %v", kb.Type, got, want)
			}
			// The v half never changes type: it is a row range.
			if vb.Type != tc.src {
				t.Errorf("attn_v_b is %v, want the source's %v: it is a row range and "+
					"needs no requantization", vb.Type, tc.src)
			}

			// ── attn_v_b: byte for byte, which is what "lossless" means here ─
			for h := uint64(0); h < nh; h++ {
				for r := uint64(0); r < vh; r++ {
					srcRow := h*per + nope + r
					want := data[srcRow*rowBytes : (srcRow+1)*rowBytes]
					at := (h*vh + r) * rowBytes
					if got := vb.Data[at : at+rowBytes]; !bytes.Equal(got, want) {
						t.Fatalf("attn_v_b head %d row %d is not source row %d, byte for byte",
							h, r, srcRow)
					}
				}
			}

			// ── attn_k_b: the source's values, transposed ─────────────────
			ksrc, ok := jlm.SourceType(kb.Type)
			if !ok {
				t.Fatalf("attn_k_b is %v, which has no source type", kb.Type)
			}
			got := make([]float32, nh*lat*nope)
			if err := quant.Dequant32(ksrc, kb.Data, got); err != nil {
				t.Fatalf("reading attn_k_b back as %v: %v", kb.Type, err)
			}
			// RULE 10: a non-finite reference makes every comparison below
			// meaningless.
			for i, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("attn_k_b[%d] is %v: the fixture is degenerate and this "+
						"gate proves nothing", i, v)
				}
			}
			var worst, scale float64
			for h := uint64(0); h < nh; h++ {
				for i := uint64(0); i < nope; i++ {
					row := held[h*per+i]
					for j := uint64(0); j < lat; j++ {
						// W_k[i][j] must land at sheet h, row j, column i.
						g := got[h*lat*nope+j*nope+i]
						if tc.exact && g != row[j] {
							t.Fatalf("attn_k_b head %d [%d][%d] = %v, and W_k^T wants "+
								"W_k[%d][%d] = %v", h, j, i, g, i, j, row[j])
						}
						if d := math.Abs(float64(g) - float64(row[j])); d > worst {
							worst = d
						}
						if a := math.Abs(float64(row[j])); a > scale {
							scale = a
						}
					}
				}
			}
			rel := worst / scale
			t.Logf("%v source -> %v k half: worst |delta| %.3e over a %.3f range (%.3e relative)",
				tc.src, kb.Type, worst, scale, rel)
			// The bound is Q8_0's own step: one round-to-nearest at 127 levels of
			// the block's amax is at most half a step, 1/254 of the row's maximum.
			// Anything larger is a wrong block grouping.
			if !tc.exact && rel > 1.0/254 {
				t.Fatalf("worst relative error %.3e exceeds Q8_0's half-step %.3e: that is "+
					"a layout error, not rounding", rel, 1.0/254)
			}

			// ── and the answer is not the un-transposed one ────────────────
			//
			// Everything above would also pass for a k half in source order if
			// nope equalled lat, so this is checked explicitly rather than relying
			// on the fixture's shape.
			same := true
			for h := uint64(0); h < nh && same; h++ {
				for i := uint64(0); i < nope && same; i++ {
					row := held[h*per+i]
					for j := uint64(0); j < lat; j++ {
						if idx := h*lat*nope + i*lat + j; idx >= uint64(len(got)) ||
							math.Abs(float64(got[idx])-float64(row[j])) > rel*scale+1e-9 {
							same = false
							break
						}
					}
				}
			}
			if same {
				t.Error("attn_k_b holds W_k in SOURCE order: the transpose did not happen, " +
					"and every check above passed anyway")
			}
		})
	}
}

// TestUnfuseMLARefusesAShapeItCannotSplit covers the geometry check, which is
// what stopped the phi3 splitter writing three wrong tensors for qwen3next.
func TestUnfuseMLARefusesAShapeItCannotSplit(t *testing.T) {
	base := func() *jlm.Config {
		return &jlm.Config{NHead: 3, HeadDim: 80, NRot: 16, HeadDimV: 32, KVLoraRank: 128}
	}
	mk := func(k, rows uint64) *jlm.Tensor {
		e := &jlm.Tensor{Role: jlm.RoleAttnKVB, Type: jlm.TypeQ8, NDim: 2,
			Data: make([]byte, 1<<20), Name: "blk.0.attn_kv_b.weight"}
		e.Dims[0], e.Dims[1] = k, rows
		return e
	}
	for _, v := range []struct {
		name string
		e    *jlm.Tensor
		c    *jlm.Config
	}{
		{"wrong latent width", mk(64, 3*(64+32)), base()},
		{"wrong row count", mk(128, 3*64), base()},
		{"no kv_lora_rank", mk(128, 3*(64+32)), &jlm.Config{NHead: 3, HeadDim: 80, NRot: 16}},
	} {
		t.Run(v.name, func(t *testing.T) {
			if _, split, err := unfuseMLA(v.e, &Config{Config: v.c}); err == nil {
				t.Fatalf("accepted (split=%v) -- this gate proved nothing", split)
			} else {
				t.Logf("refused: %v", err)
			}
		})
	}
	// A tensor that is not the fused pair passes straight through.
	e := mk(128, 3*(64+32))
	e.Role = jlm.RoleAttnQ
	if _, split, err := unfuseMLA(e, &Config{Config: base()}); err != nil || split {
		t.Fatalf("attn_q was split=%v err=%v: unfuseMLA must ignore what is not its tensor", split, err)
	}
}

package convert

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestAFusedQKVReadsItsHeadWidth: a phimoe file stating no key_length whose
// attn_qkv holds other rows than embedding/heads gives (four heads of 16 on
// a 64-wide model, 192 rows) is cut at the width its rows hold: 96 rows are
// heads of 8, the statement's q, k and v then 32 rows each. Rows that are
// not one width over the heads and the kv heads twice are refused naming
// the tensor.
func TestAFusedQKVReadsItsHeadWidth(t *testing.T) {
	file := func(rows uint64) *meta.File {
		kv := baseKeys(2)
		kv["expert_count"] = meta.MakeUint(4)
		kv["expert_used_count"] = meta.MakeUint(2)
		f := headerOnly("phimoe", kv)
		ts := append(f.Tensors, meta.Tensor{Name: "blk.0.attn_qkv.weight", Dims: []uint64{64, rows},
			Type: quant.F32, NBytes: 4 * 64 * rows})
		return meta.New(f.Path, f.KV, ts)
	}
	for _, tc := range []struct{ rows, head uint64 }{{192, 16}, {96, 8}} {
		f := file(tc.rows)
		c, err := configOf(f)
		if err != nil {
			t.Fatalf("%d rows: %v", tc.rows, err)
		}
		cut, _, err := statedQKVRows(f, c)
		if err != nil || uint64(c.HeadDim) != tc.head || uint64(c.NRot) != tc.head || cut != tc.rows {
			t.Fatalf("%d rows: heads of %d, rotary %d, cut into %d rows (%v); want heads of %d cutting all of them",
				tc.rows, c.HeadDim, c.NRot, cut, err, tc.head)
		}
	}
	if _, err := configOf(file(100)); err == nil || !strings.Contains(err.Error(), "attn_qkv") || !errors.Is(err, ErrBadInput) {
		t.Fatalf("100 rows converted, or not as a bad input naming attn_qkv: %v", err)
	}
}

// TestAFusedQKVConvertsAsItsParts is Phi-tiny-MoE's first two blocks
// (scripts/fuseqkv.py) twice, q, k and v apart and joined into attn_qkv,
// neither stating attention.key_length: heads of 128 on a 4096-wide model,
// where embedding/heads is 256. The fused file is cut at the width its rows
// hold, so the two containers are one: the same header, config, entries,
// fingerprint and bytes, but for the entries' diagnostic names, which say
// which tensor a part was cut from.
func TestAFusedQKVConvertsAsItsParts(t *testing.T) {
	dir := testmodels.Path("fusedqkv/phitiny")
	var fs [2]*jlm.File
	var raw [2][]byte
	for i, n := range []string{"split.gguf", "fused.gguf"} {
		src := filepath.Join(dir, n)
		if _, err := os.Stat(src); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v -- run scripts/fuseqkv.py on Phi-tiny-MoE-instruct-Q8_0.gguf (RULE 11)", err)
		}
		f, err := gguf.Open(src)
		if err != nil {
			t.Fatal(err)
		}
		c, err := configOf(f)
		f.Close()
		if err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		if c.HeadDim != 128 || c.NRot != 128 {
			t.Fatalf("%s: heads of %d, rotary %d; want 128", n, c.HeadDim, c.NRot)
		}
		dst := filepath.Join(t.TempDir(), n+jlm.Ext)
		if _, err := FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}); err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		if fs[i], err = jlm.Open(dst); err != nil {
			t.Fatal(err)
		}
		defer fs[i].Close()
		if raw[i], err = os.ReadFile(dst); err != nil {
			t.Fatal(err)
		}
	}
	ha, hb := *fs[0].H, *fs[1].H
	if len(raw[0]) != len(raw[1]) {
		t.Fatalf("the containers are %d and %d bytes", len(raw[0]), len(raw[1]))
	}
	ha.TabLen, ha.FPOff, hb.TabLen, hb.FPOff = 0, 0, 0, 0
	if ha != hb {
		t.Fatalf("the headers differ: %+v / %+v", *fs[0].H, *fs[1].H)
	}
	a, b := fs[0].Entries(), fs[1].Entries()
	if len(a) != len(b) {
		t.Fatalf("%d entries and %d", len(a), len(b))
	}
	for i := range a {
		x, y := a[i], b[i]
		x.Name, y.Name = "", ""
		if x != y {
			t.Fatalf("entry %d (%s / %s) differs: %+v / %+v", i, a[i].Name, b[i].Name, x, y)
		}
	}
	h0, h1 := fs[0].H, fs[1].H
	for _, r := range []struct {
		what   string
		a0, a1 uint64
		b0, b1 uint64
	}{
		{"config", h0.CfgOff, h0.TabOff, h1.CfgOff, h1.TabOff},
		{"fingerprint", h0.FPOff, h0.FPOff + h0.FPLen, h1.FPOff, h1.FPOff + h1.FPLen},
		{"weights", h0.DenseOff, uint64(len(raw[0])), h1.DenseOff, uint64(len(raw[1]))},
	} {
		if !bytes.Equal(raw[0][r.a0:r.a1], raw[1][r.b0:r.b1]) {
			t.Fatalf("the containers' %s differs", r.what)
		}
	}
}

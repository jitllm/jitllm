package jlm

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// qkSource is a two-block model whose second block is linear attention, with a
// q/k norm on each block normOn names. Each block carries a 2 MiB matrix so its
// page holds the default read chunk.
func qkSource(flags Flags, normOn func(b int32) bool) *Source {
	const k, rows = 64, 32
	f32 := func(n int) []byte {
		b := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], 0x3F800000)
		}
		return b
	}
	var ts []Tensor
	add := func(r Role, block int32, dims []uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		e := Tensor{Role: r, Block: block, Index: -1, Type: TypeF32, NDim: uint8(len(dims)),
			Data: f32(n), Name: r.String()}
		copy(e.Dims[:], dims)
		ts = append(ts, e)
	}
	add(RoleTokenEmbd, DenseBlock, []uint64{k, rows})
	for b := int32(0); b < 2; b++ {
		add(RoleAttnNorm, b, []uint64{k})
		add(RoleAttnQ, b, []uint64{k, 8192})
		if normOn(b) {
			add(RoleAttnQNorm, b, []uint64{k})
			add(RoleAttnKNorm, b, []uint64{k})
		}
	}
	return &Source{
		Config: &Config{
			Arch: ArchQwen3, NLayer: 2, NEmbd: k, NHead: 1, NKVHead: 1, HeadDim: k, NRot: k,
			NFFN: k, NVocab: rows, NCtx: 64, RMSEps: 1e-6, RopeBase: 10000, EmbdScale: 1, AttnFactor: 1,
			Flags: FlagTiedEmbd | flags, LayerKinds: []LayerKind{LayerFullAttn, LayerLinearAttn},
		},
		Vocab: &Vocab{Kind: VocabBPE, Tokens: NewStrings([]string{"a", "b"}), Scores: []float32{0, 0},
			Kinds: []TokenKind{TokenNormal, TokenNormal}, BOS: 0, EOS: 1, Unk: -1, Pad: -1, Sep: -1, Mask: -1},
		Tensors: ts,
	}
}

// TestQKNormFlagAndTensorsAgree is the gate on q/k norm having ONE derivation
// (QKNormOn): the flag and the tensors must say the same thing on every block,
// at Write and at Open, because the host and the device each read one of the
// two. The consistent model writes; a flag with no tensor, a tensor with no
// flag and a norm on a linear block are each refused by Write; and a container
// whose flag is flipped on disk after a consistent write is refused by Open.
func TestQKNormFlagAndTensorsAgree(t *testing.T) {
	dir := t.TempDir()
	onAttn := func(b int32) bool { return b == 0 }
	ok := filepath.Join(dir, "ok"+Ext)
	if _, err := Write(ok, qkSource(FlagQKNorm, onAttn), Fingerprint{Host: "test"}); err != nil {
		t.Fatalf("the consistent model: %v", err)
	}
	f, err := Open(ok)
	if err != nil {
		t.Fatalf("Open the consistent model: %v", err)
	}
	cfgOff := f.H.CfgOff
	if !f.Config().QKNormAt(0) || f.Config().QKNormAt(1) {
		t.Fatalf("QKNormAt: block 0 %v, linear block 1 %v; want true, false",
			f.Config().QKNormAt(0), f.Config().QKNormAt(1))
	}
	f.Close()

	for _, c := range []struct {
		name   string
		flags  Flags
		normOn func(int32) bool
		want   string
	}{
		{"the flag with no tensor", FlagQKNorm, func(int32) bool { return false }, "has no attn_q_norm"},
		{"a tensor with no flag", 0, onAttn, "does not norm q and k there"},
		{"a norm on a linear block", FlagQKNorm, func(int32) bool { return true }, "block 1 carries"},
	} {
		_, err := Write(filepath.Join(dir, "bad"+Ext), qkSource(c.flags, c.normOn), Fingerprint{Host: "test"})
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Write said %v, want a refusal containing %q", c.name, err, c.want)
		}
	}

	// The reader's half: flip FlagQKNorm in the written config record. Its
	// offset is found by encoding the config both ways, not hardcoded.
	cfg := qkSource(FlagQKNorm, onAttn).Config
	with := encodeConfig(cfg)
	cfg.Flags &^= FlagQKNorm
	without := encodeConfig(cfg)
	at := -1
	for i := range with {
		if with[i] != without[i] {
			at = i
			break
		}
	}
	if at < 0 || len(with) != len(without) {
		t.Fatal("the flag does not change the config record; the flip below would test nothing")
	}
	b, err := os.ReadFile(ok)
	if err != nil {
		t.Fatal(err)
	}
	rec := b[cfgOff : cfgOff+uint64(len(with))]
	if !bytes.Equal(rec, with) {
		t.Fatal("the config record is not where the header says")
	}
	rec[at] = without[at]
	flipped := filepath.Join(dir, "flipped"+Ext)
	if err := os.WriteFile(flipped, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if f, err := Open(flipped); err == nil {
		f.Close()
		t.Fatal("Open read a container carrying q/k norm tensors with FlagQKNorm cleared")
	} else if !strings.Contains(err.Error(), "does not norm q and k there") {
		t.Fatalf("Open refused for another reason: %v", err)
	}
}

package jlm

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The fixtures in testdata/old are oldSource written by the writer of each
// old version, built from the last commit at that version: v26 at f3b71b70^,
// v27 at caf7108b^, v28 at 9409580b^, v29 at c674267f^. Each was a main calling
// jlm.Write(path, oldSource(), Fingerprint{Host: "test"}) with oldSource's
// body as below (prefixed jlm.), gzipped with -9 -n.

// oldSource is the model every historical fixture was written from: a
// two-block mixture whose dense, block and expert tensors are Q4_K and Q5_K
// -- the types v27 repacked -- beside verbatim norms. The same body compiled
// against each old writer, so it uses only what every version from 26 on has.
func oldSource() *Source {
	const k, nExp = 256, 4
	rng := rand.New(rand.NewPCG(26, 30))
	ks := func(rows int, blockB int) []byte {
		b := make([]byte, rows*k/256*blockB)
		for i := 0; i < len(b); i += blockB {
			for j := 4; j < blockB; j++ {
				b[i+j] = byte(rng.Uint32())
			}
			binary.LittleEndian.PutUint16(b[i:], 0x2c00+uint16(rng.IntN(0x400)))
			binary.LittleEndian.PutUint16(b[i+2:], 0x2400+uint16(rng.IntN(0x400)))
		}
		return b
	}
	q4k := func(rows int) []byte { return ks(rows, 144) }
	q5k := func(rows int) []byte { return ks(rows, 176) }
	f32 := func(n int) []byte {
		b := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(1+float32(rng.IntN(100))/1000))
		}
		return b
	}
	var ts []Tensor
	add := func(role Role, block int32, ty Type, b []byte, dims ...uint64) {
		e := Tensor{Role: role, Block: block, Index: -1, Type: ty, NDim: uint8(len(dims)), Data: b, Name: role.String()}
		copy(e.Dims[:], dims)
		ts = append(ts, e)
	}
	add(RoleTokenEmbd, DenseBlock, TypeQ4S, q4k(8), k, 8)
	add(RoleOutputNorm, DenseBlock, TypeF32, f32(k), k)
	for b := int32(0); b < 2; b++ {
		add(RoleAttnNorm, b, TypeF32, f32(k), k)
		add(RoleAttnQ, b, TypeQ5S, q5k(16), k, 16)
		add(RoleAttnK, b, TypeQ4S, q4k(16), k, 16)
		add(RoleExpGateBank, b, TypeQ4S, q4k(16*nExp), k, 16, nExp)
		add(RoleExpDownBank, b, TypeQ5S, q5k(16*nExp), k, 16, nExp)
	}
	return &Source{
		Config: &Config{Arch: ArchQwen3MoE, NLayer: 2, NEmbd: k, NHead: 1, NKVHead: 1,
			HeadDim: k, NRot: k, NFFN: k, NVocab: 8, NCtx: 128, RMSEps: 1e-5, RopeBase: 10000,
			EmbdScale: 1, AttnFactor: 1, NExpert: nExp, NExpertUsed: 2},
		Vocab: &Vocab{Kind: VocabBPE, Tokens: []string{"a", "b", "c", "d", "e", "f", "g", "h"},
			Kinds: make([]TokenKind, 8), BOS: -1, EOS: -1, Unk: -1, Pad: -1, Sep: -1, Mask: -1},
		Tensors: ts,
	}
}

// oldFixture is testdata/old/v<v>.jlm.gz unpacked into a fresh file.
func oldFixture(t *testing.T, v uint32) string {
	t.Helper()
	in, err := os.Open(filepath.Join("testdata", "old", "v"+itoa(uint64(v))+".jlm.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	z, err := gzip.NewReader(in)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(z)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "old"+Ext)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := ReadVersion(p); err != nil || got != v {
		t.Fatalf("the v%d fixture reads as version %d (%v)", v, got, err)
	}
	return p
}

// TestOldWritersUpgradeToAFreshConversion: a container each old writer wrote
// (v26, v27, v28) upgrades through the chain to, byte for byte, what the
// current writer makes of the same model -- so every tensor, and every
// answer a model computes from them, is the fresh conversion's. Each step
// that rewrites is held to the comparison by a violation that only stamps
// its version, and every step by its absence, which must refuse.
func TestOldWritersUpgradeToAFreshConversion(t *testing.T) {
	fresh := filepath.Join(t.TempDir(), "fresh"+Ext)
	if _, err := Write(fresh, oldSource(), Fingerprint{Host: "test"}); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(fresh)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range []uint32{26, 27, 28} {
		t.Run("v"+itoa(uint64(v)), func(t *testing.T) {
			p := oldFixture(t, v)
			if f, err := Open(p); err == nil {
				f.Close()
				t.Fatal("an old container opened: the reader is not version-strict")
			} else if !strings.Contains(err.Error(), "jitllm upgrade") {
				t.Fatalf("the refusal %q does not name jitllm upgrade", err)
			}
			from, err := Upgrade(p)
			if err != nil || from != v {
				t.Fatalf("Upgrade: from %d, %v", from, err)
			}
			got, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("the upgraded container (%d bytes) is not the fresh one (%d bytes)%s",
					len(got), len(want), firstDiff(got, want))
			}
			f, err := Open(p)
			if err != nil {
				t.Fatal(err)
			}
			f.Close()
			if _, err := os.Stat(p + ".upgrade"); !os.IsNotExist(err) {
				t.Fatalf("the upgrade left its scratch file: %v", err)
			}
		})
	}
	// A rewriting step that only stamps its version gives a container that
	// is not the fresh one.
	for _, s := range []uint32{26, 27} {
		t.Run("stamp-only-"+itoa(uint64(s)), func(t *testing.T) {
			saved := migrations[s]
			migrations[s] = func(path string) error { return setVersion(path, s+1) }
			defer func() { migrations[s] = saved }()
			p := oldFixture(t, s)
			if _, err := Upgrade(p); err == nil {
				got, err := os.ReadFile(p)
				if err != nil {
					t.Fatal(err)
				}
				if bytes.Equal(got, want) {
					t.Fatalf("a chain that skipped step %d gave the fresh container", s)
				}
			}
		})
	}
	// A missing step is a refusal naming the convert command, from every
	// version below it.
	for _, s := range []uint32{26, 27, 28, 29} {
		t.Run("missing-"+itoa(uint64(s)), func(t *testing.T) {
			saved := migrations[s]
			delete(migrations, s)
			defer func() { migrations[s] = saved }()
			p := oldFixture(t, 26)
			if _, err := Upgrade(p); err == nil || !strings.Contains(err.Error(), "jitllm convert") {
				t.Fatalf("a chain missing step %d: %v", s, err)
			}
			if v, err := ReadVersion(p); err != nil || v != 26 {
				t.Fatalf("a refused upgrade moved the container to version %d (%v)", v, err)
			}
		})
	}
}

// TestRestream26KeepsEveryField: the v26-to-v27 repack is the stream
// kernels.ScField reads, field for field, for every 6-bit value at every
// sub-block.
func TestRestream26KeepsEveryField(t *testing.T) {
	for sub := 0; sub < 8; sub++ {
		for _, min := range []bool{false, true} {
			for v := uint8(0); v < 64; v++ {
				var ws [3]uint32
				scPut26(&ws, sub, min, v|0xC0)
				if got := kernels.ScValue(ws, sub, min); got != v {
					t.Fatalf("sub-block %d min %v: put %d, read %d", sub, min, v, got)
				}
			}
		}
	}
}

// TestEachStepIsItsVersionsWriter: each migration alone turns the version v
// fixture into exactly what version v+1's own writer made of the same model,
// so a step is gated against its version and not only against the end of
// the chain.
func TestEachStepIsItsVersionsWriter(t *testing.T) {
	for _, v := range []uint32{26, 27, 28} {
		t.Run("v"+itoa(uint64(v)), func(t *testing.T) {
			p, next := oldFixture(t, v), oldFixture(t, v+1)
			want, err := os.ReadFile(next)
			if err != nil {
				t.Fatal(err)
			}
			if err := migrations[v](p); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("v%d migrated (%d bytes) is not v%d's writer (%d bytes)%s",
					v, len(got), v+1, len(want), firstDiff(got, want))
			}
		})
	}
}

package jlm

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestEveryVersionHasItsMigration fails the build that bumps Version without
// the migration from the version before it: a container its source no
// longer exists for would otherwise have no way forward.
func TestEveryVersionHasItsMigration(t *testing.T) {
	if migrations[Version-1] == nil {
		t.Fatalf("Version is %d and there is no migration from %d (migrate.go): every bump adds one",
			Version, Version-1)
	}
	if !Upgradable(26) {
		t.Fatalf("version 26 no longer upgrades to %d: a migration in the chain is missing", Version)
	}
	for v := range migrations {
		if v >= Version {
			t.Errorf("a migration from version %d, which is not older than %d", v, Version)
		}
	}
}

// migrateSource is a small model as a converter hands it over, with or
// without an audio tower; unfolded is a version 29 converter's audio (the
// depthwise bias beside the BatchNorm shift), folded a version 30 one's.
func migrateSource(audio, unfolded bool) *Source {
	const k, rows = 64, 8
	q4 := func(seed byte) []byte {
		b := make([]byte, rows*k/32*18)
		for i := 0; i < len(b); i += 18 {
			binary.LittleEndian.PutUint16(b[i:], 0x3800)
			for j := 2; j < 18; j++ {
				b[i+j] = seed + byte(i+j)
			}
		}
		return b
	}
	vec := func(n int, f func(i int) float32) []byte {
		b := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f(i)))
		}
		return b
	}
	var ts []Tensor
	add := func(role Role, block int32, ty Type, b []byte, dims ...uint64) {
		e := Tensor{Role: role, Block: block, Index: -1, Type: ty, NDim: uint8(len(dims)), Data: b, Name: role.String()}
		copy(e.Dims[:], dims)
		ts = append(ts, e)
	}
	add(RoleTokenEmbd, DenseBlock, TypeQ4, q4(1), k, rows)
	add(RoleAttnQ, 0, TypeQ4, q4(2), k, rows)
	add(RoleAttnNorm, 0, TypeF32, vec(k, func(int) float32 { return 1 }), k)
	src := &Source{
		Config: &Config{Arch: ArchLlama, NLayer: 1, NEmbd: k, NHead: 1, NKVHead: 1, HeadDim: k, NRot: k,
			NFFN: k, NVocab: rows, NCtx: 128, RMSEps: 1e-5, RopeBase: 10000, EmbdScale: 1, AttnFactor: 1,
			Flags: FlagTiedEmbd},
		Vocab: &Vocab{Kind: VocabSPM, Tokens: NewStrings([]string{"<unk>", "a", "b", "c", "d", "e", "f", "g"}),
			Scores: make([]float32, rows), Kinds: make([]TokenKind, rows),
			BOS: 0, EOS: -1, Unk: 0, Pad: -1, Sep: -1, Mask: -1},
	}
	if audio {
		src.Vision = &Vision{NEmbd: k, ProjDim: k, Eps: 1e-5, Projector: ProjD1OmniA,
			Audio: &Audio{NLayer: 1, NEmbd: k, NHead: 1, NFFN: k, ConvKernel: 3, MelBins: 8,
				SubChannels: 4, ProjDim: k, ResWidth: 4, Eps: 1e-5, Projector: ProjD1OmniA,
				SampleRate: 16000, NFFT: 512, Window: 400, Hop: 160}}
		dwb := func(i int) float32 { return 0.1 * float32(i%7) }
		scale := func(i int) float32 { return 1 + 0.01*float32(i) }
		shift := func(i int) float32 { return -0.2 + 0.03*float32(i%5) }
		add(RoleAQ, 0, TypeQ4, q4(3), k, rows)
		add(RoleADW, 0, TypeF32, vec(3*k, func(i int) float32 { return float32(i%3) - 1 }), k, 3)
		add(RoleABNScale, 0, TypeF32, vec(k, scale), k)
		if unfolded {
			add(RoleADWB, 0, TypeF32, vec(k, dwb), k)
			add(RoleABNShift, 0, TypeF32, vec(k, shift), k)
		} else {
			add(RoleABNShift, 0, TypeF32, vec(k, func(i int) float32 { return dwb(i)*scale(i) + shift(i) }), k)
		}
	}
	src.Tensors = ts
	return src
}

// TestUpgradeIsAFreshConversion: a version 29 container upgraded is, byte for
// byte, what writing the same model at the current version gives -- with an
// audio tower (rewritten, its shift folded) and without one (in place). A
// chain that skips the step's work is refused by the comparison.
func TestUpgradeIsAFreshConversion(t *testing.T) {
	fp := Fingerprint{Host: "test"}
	for _, audio := range []bool{false, true} {
		name := map[bool]string{false: "text", true: "audio"}[audio]
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			fresh := filepath.Join(dir, "fresh"+Ext)
			if _, err := Write(fresh, migrateSource(audio, false), fp); err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(fresh)
			if err != nil {
				t.Fatal(err)
			}
			old := func() string {
				p := filepath.Join(t.TempDir(), "old"+Ext)
				if _, err := Write(p, migrateSource(audio, true), fp); err != nil {
					t.Fatal(err)
				}
				if err := setVersion(p, 29); err != nil {
					t.Fatal(err)
				}
				return p
			}
			p := old()
			if f, err := Open(p); err == nil {
				f.Close()
				t.Fatal("a version 29 container opened: the reader is not version-strict")
			} else if !strings.Contains(err.Error(), "jitllm upgrade") {
				t.Fatalf("the refusal %q does not name jitllm upgrade", err)
			}
			from, err := Upgrade(p)
			if err != nil || from != 29 {
				t.Fatalf("Upgrade: from %d, %v", from, err)
			}
			got, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("the upgraded container (%d bytes) is not the fresh one (%d bytes)%s", len(got), len(want), firstDiff(got, want))
			}
			f, err := Open(p)
			if err != nil {
				t.Fatal(err)
			}
			f.Close()
			if from, err := Upgrade(p); err != nil || from != Version {
				t.Fatalf("upgrading a current container: from %d, %v", from, err)
			}
			if !audio {
				return
			}
			// The violation: the step that only stamps the version.
			saved := migrations[29]
			migrations[29] = func(path string) error { return setVersion(path, 30) }
			defer func() { migrations[29] = saved }()
			q := old()
			if _, err := Upgrade(q); err != nil {
				t.Fatal(err)
			}
			bad, err := os.ReadFile(q)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Equal(bad, want) {
				t.Fatal("a chain that skipped the fold gave the fresh container: the gate does not see the step")
			}
		})
	}
}

func firstDiff(a, b []byte) string {
	for i := 0; i < min(len(a), len(b)); i++ {
		if a[i] != b[i] {
			return ", first difference at byte " + itoa(uint64(i))
		}
	}
	return ""
}

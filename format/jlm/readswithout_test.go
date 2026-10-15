package jlm_test

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestPreviousVersionReadsWithoutItsSection holds the one exception to "no
// backward compatibility" (jlm.ReadsWithout): a container of the previous
// revision opens when it carries no audio tower, whose bytes are the only
// ones that changed meaning, and is refused by name when it carries one. The
// same container at the current revision opens either way, and a revision
// before the exception is still refused.
func TestPreviousVersionReadsWithoutItsSection(t *testing.T) {
	const k, rows = 64, 8
	f32 := func(n int) []byte {
		b := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], 0x3F800000)
		}
		return b
	}
	build := func(audio bool) string {
		t.Helper()
		var ts []jlm.Tensor
		add := func(role jlm.Role, block int32, dims ...uint64) {
			n := 1
			for _, d := range dims {
				n *= int(d)
			}
			e := jlm.Tensor{Role: role, Block: block, Index: -1, Type: jlm.TypeF32,
				NDim: uint8(len(dims)), Data: f32(n), Name: role.String()}
			copy(e.Dims[:], dims)
			ts = append(ts, e)
		}
		add(jlm.RoleTokenEmbd, jlm.DenseBlock, k, rows)
		add(jlm.RoleAttnNorm, 0, k)
		add(jlm.RoleAttnQ, 0, k, rows)
		src := &jlm.Source{
			Config: &jlm.Config{Arch: jlm.ArchLlama, NLayer: 1, NEmbd: k, NHead: 1, NKVHead: 1,
				HeadDim: k, NRot: k, NFFN: k, NVocab: rows, NCtx: 128, RMSEps: 1e-5,
				RopeBase: 10000, EmbdScale: 1, AttnFactor: 1, Flags: jlm.FlagTiedEmbd},
			Vocab: &jlm.Vocab{Kind: jlm.VocabSPM, Tokens: []string{"<unk>", "a", "b", "c", "d", "e", "f", "g"},
				Scores: make([]float32, rows), Kinds: make([]jlm.TokenKind, rows),
				BOS: 0, EOS: -1, Unk: 0, Pad: -1, Sep: -1, Mask: -1},
			Tensors: ts,
		}
		if audio {
			src.Vision = &jlm.Vision{NEmbd: k, ProjDim: k, Eps: 1e-5, Projector: jlm.ProjD1OmniA,
				Audio: &jlm.Audio{NLayer: 1, NEmbd: k, NHead: 1, NFFN: k, ConvKernel: 3, MelBins: 8,
					SubChannels: 4, ProjDim: k, ResWidth: 4, Eps: 1e-5, Projector: jlm.ProjD1OmniA,
					SampleRate: 16000, NFFT: 512, Window: 400, Hop: 160}}
		}
		dst := filepath.Join(t.TempDir(), "m"+jlm.Ext)
		if _, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "test"}); err != nil {
			t.Fatalf("jlm.Write: %v", err)
		}
		return dst
	}
	at := func(path string, v uint32) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		binary.LittleEndian.PutUint32(b[8:], v)
		out := filepath.Join(t.TempDir(), "v"+jlm.Ext)
		if err := os.WriteFile(out, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return out
	}
	open := func(path string) error {
		f, err := jlm.Open(path)
		if err == nil {
			f.Close()
		}
		return err
	}
	prev := jlm.ReadsWithout.Version
	if prev != jlm.Version-1 {
		t.Fatalf("the exception names version %d, not the one before %d: retire it at the next bump", prev, jlm.Version)
	}
	text, aud := build(false), build(true)
	for _, p := range []string{text, aud} {
		if err := open(p); err != nil {
			t.Fatalf("a current container did not open: %v", err)
		}
	}
	if err := open(at(text, prev)); err != nil {
		t.Errorf("a version %d container without an audio tower was refused: %v", prev, err)
	}
	f, err := jlm.Open(at(aud, prev))
	if err == nil {
		f.Close()
		t.Fatalf("a version %d container with an audio tower opened: its BatchNorm shift would read unfolded", prev)
	}
	if !strings.Contains(err.Error(), jlm.ReadsWithout.Section) || !strings.Contains(err.Error(), "convert") {
		t.Errorf("refused with %q, which does not name the section or the fix", err)
	}
	if err := open(at(text, prev-1)); err == nil {
		t.Errorf("a version %d container opened: only %d is excepted", prev-1, prev)
	}
}

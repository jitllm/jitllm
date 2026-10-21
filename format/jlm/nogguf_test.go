package jlm_test

import (
	"bytes"
	"encoding/binary"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
	"github.com/jitllm/jitllm/tok"
)

// TestContainerFromNoGGUFAtAll builds a model out of ordinary Go values and
// writes it as a container. Nothing in this file, or in anything it calls,
// knows what a GGUF is.
//
// It is the gate on total decoupling, which the import graph alone cannot
// prove: meta.TestOnlyTheConverterReachesGGUF says jlm does not import gguf,
// not whether a second converter can get in. This walks the safetensors
// reader's path with a synthetic model.
func TestContainerFromNoGGUFAtAll(t *testing.T) {
	const (
		nBlock = 2
		k      = 64
		rows   = 8
	)
	// A Q4_0 block is 18 bytes per 32 elements: an f16 scale then 16 payload
	// bytes. The scale must be finite, or both arms of the comparison below
	// are NaN and the gate passes on a degenerate oracle (RULE 10).
	q4 := func(seed byte) []byte {
		b := make([]byte, rows*k/32*18)
		for i := 0; i < len(b); i += 18 {
			binary.LittleEndian.PutUint16(b[i:], 0x3800) // 0.5
			for j := 2; j < 18; j++ {
				b[i+j] = seed + byte(i+j)
			}
		}
		return b
	}
	norm := func() []byte {
		b := make([]byte, k*4)
		for i := 0; i < k; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], 0x3F800000) // 1.0
		}
		return b
	}

	var tensors []jlm.Tensor
	add := func(role jlm.Role, block int32, ty jlm.Type, dims []uint64, b []byte) {
		e := jlm.Tensor{Role: role, Block: block, Index: -1, Type: ty,
			NDim: uint8(len(dims)), Data: b, Name: role.String()}
		copy(e.Dims[:], dims)
		tensors = append(tensors, e)
	}
	add(jlm.RoleTokenEmbd, jlm.DenseBlock, jlm.TypeQ4, []uint64{k, rows}, q4(1))
	for i := int32(0); i < nBlock; i++ {
		add(jlm.RoleAttnQ, i, jlm.TypeQ4, []uint64{k, rows}, q4(byte(2+i)))
		add(jlm.RoleAttnNorm, i, jlm.TypeF32, []uint64{k}, norm())
	}

	vocab := []string{"<unk>", "▁a", "▁b", "🙂"}
	src := &jlm.Source{
		Config: &jlm.Config{
			Arch: jlm.ArchLlama, NLayer: nBlock, NEmbd: k, NHead: 1, NKVHead: 1,
			HeadDim: k, NRot: k, NFFN: k, NVocab: rows, NCtx: 128,
			RMSEps: 1e-5, RopeBase: 10000, EmbdScale: 1, AttnFactor: 1,
			Flags: jlm.FlagTiedEmbd,
		},
		Vocab: &jlm.Vocab{
			Kind: jlm.VocabSPM, Tokens: jlm.NewStrings(vocab),
			Scores: []float32{0, -1, -2, -3},
			Kinds:  []jlm.TokenKind{jlm.TokenUnknown, jlm.TokenNormal, jlm.TokenNormal, jlm.TokenNormal},
			BOS:    0, EOS: -1, Unk: 0, Pad: -1, Sep: -1, Mask: -1,
			AddBOS: true, AddSpacePrefix: true, ByteFallback: true,
			Templates: []jlm.ChatTemplate{{Body: "{{ bos_token }}"}},
		},
		Tensors: tensors,
	}

	dst := filepath.Join(t.TempDir(), "synthetic"+jlm.Ext)
	h, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "test"})
	if err != nil {
		t.Fatalf("jlm.Write: %v", err)
	}
	if h.NBlocks != nBlock {
		t.Fatalf("wrote %d blocks, want %d", h.NBlocks, nBlock)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if err := c.LoadAll(); err != nil {
		t.Fatal(err)
	}

	// The description survives as typed fields, read with the container's own
	// accessors: no key names, nothing to look up.
	if got := c.Config().Arch; got != jlm.ArchLlama {
		t.Errorf("architecture is %v", got)
	}
	if got := c.Config().NLayer; got != nBlock {
		t.Errorf("NLayer is %d, want %d", got, nBlock)
	}
	if got, ok := c.Vocab().Template(""); !ok || got != "{{ bos_token }}" {
		t.Errorf("chat template is %q (%v)", got, ok)
	}
	v, err := tok.New(c.Vocab())
	if err != nil {
		t.Fatalf("tok.New on a container nothing GGUF ever touched: %v", err)
	}
	if got := v.Size(); got != len(vocab) {
		t.Errorf("vocabulary is %d tokens, want %d", got, len(vocab))
	}

	// And the weights are in the device layout, byte-identical to the packer.
	checked := 0
	for i := range c.Entries() {
		e := &c.Entries()[i]
		qs, d, sc := c.Span(e)
		raw := tensors[i].Data
		q, packed := jlm.Packer(e.Type)
		if !packed {
			if !bytes.Equal(qs[:len(raw)], raw) {
				t.Errorf("%v: verbatim bytes differ", e.Role)
			}
			checked++
			continue
		}
		wq, wd, wsc, err := kernels.PackWeights(q, raw, e.Rows(), e.K())
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range []struct {
			what string
			got  []byte
			want []uint32
		}{{"qs", qs, wq}, {"d", d, wd}, {"sc", sc, wsc}} {
			if n := len(p.want) * 4; n > 0 && !bytes.Equal(p.got[:n], jlm.U32Bytes(p.want)) {
				t.Errorf("%v %s: the container and the packer disagree", e.Role, p.what)
			}
		}
		checked++
	}
	if checked != len(tensors) {
		t.Fatalf("checked %d of %d tensors -- this gate proved nothing", checked, len(tensors))
	}
	t.Logf("%d tensors, %d tokens and a typed config, built and read with no GGUF in the path",
		checked, c.Vocab().Tokens.Len())
}

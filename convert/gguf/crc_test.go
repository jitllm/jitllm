package gguf

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"
)

type tensorCRC struct {
	Model  string `json:"model"`
	Type   string `json:"type"`
	Tensor string `json:"tensor"`
	Elems  uint64 `json:"elems"`
	CRC32  uint32 `json:"crc32"`
}

// TestWholeTensorCRC is the wide half of T0. The windowed goldens are bit-exact
// but cover a few hundred blocks each; this covers every element of the largest
// tensor of each quantization type — 712 million of them across six tensors —
// for the cost of one uint32 per tensor in testdata.
//
// It is CRC32-IEEE because zlib and hash/crc32 agree on it byte for byte, and
// both sides stream it in 1024-block chunks so neither ever holds a tensor in
// memory. A single wrong element anywhere changes the checksum.
func TestWholeTensorCRC(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "tensor_crc.json"))
	if err != nil {
		t.Skipf("no CRC goldens (run scripts/gold.py): %v", err)
	}
	var want []tensorCRC
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	// The goldens name a model by its content ("sha256:<hex>", the Ollama blob
	// it was taken from), so each candidate is identified by its bytes: a
	// symlink into the blob store by the blob's own name, a plain download by
	// hashing it. Keying on the file NAME matched nothing on any host, and
	// every subtest skipped while the gate read green.
	byID := map[string]string{}
	present := 0
	for _, p := range []string{tinyllama(), gemma()} {
		if _, err := os.Stat(p); err != nil {
			continue
		}
		present++
		id, err := contentID(p)
		if err != nil {
			t.Fatal(err)
		}
		byID[id] = p
	}
	if present == 0 {
		t.Skip("MODEL MISSING: neither tinyllama-1.1b-q3_K_M.gguf nor gemma-2b.gguf (set JITLLM_MODELS) -- this gate proved nothing")
	}

	var total uint64
	for _, w := range want {
		t.Run(w.Type, func(t *testing.T) {
			path, ok := byID[w.Model]
			if !ok {
				if present < 2 {
					t.Skipf("the model the golden names, %s, is not on this host", w.Model)
				}
				t.Fatalf("the golden names %s, and neither model on this host has those bytes", w.Model)
			}
			f := open(t, path)
			tn, ok := f.Get(w.Tensor)
			if !ok {
				t.Fatalf("no tensor %q", w.Tensor)
			}
			if got := tn.Elems(); got != w.Elems {
				t.Fatalf("%s has %d elements, golden says %d", w.Tensor, got, w.Elems)
			}

			const chunkBlocks = 1024
			be, bb := tn.Type.BlockElems(), tn.Type.BlockBytes()
			src := f.Bytes(tn)
			buf := make([]float64, chunkBlocks*be)
			out := make([]byte, chunkBlocks*be*4)
			h := crc32.NewIEEE()

			start := time.Now()
			blocks := w.Elems / be
			for done := uint64(0); done < blocks; {
				n := min(uint64(chunkBlocks), blocks-done)
				b := buf[:n*be]
				if err := quant.Dequant(tn.Type, src[done*bb:], b); err != nil {
					t.Fatal(err)
				}
				for i, v := range b {
					binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(float32(v)))
				}
				h.Write(out[:len(b)*4])
				done += n
			}
			if got := h.Sum32(); got != w.CRC32 {
				t.Errorf("%s %s: crc32 %08x, want %08x", w.Type, w.Tensor, got, w.CRC32)
			}
			total += w.Elems
			t.Logf("%-22s %10d elems  crc %08x  %v", w.Tensor, w.Elems, w.CRC32, time.Since(start).Round(time.Millisecond))
		})
	}
	t.Logf("total %d elements verified bit-exact against libggml", total)
}

// contentID is "sha256:<hex>" of the file p names: read off the target's name
// when p resolves into a content-addressed blob store, hashed otherwise.
func contentID(p string) (string, error) {
	r, err := filepath.EvalSymlinks(p)
	if err != nil {
		return "", err
	}
	if b := filepath.Base(r); strings.HasPrefix(b, "sha256:") {
		return b, nil
	}
	f, err := os.Open(r)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

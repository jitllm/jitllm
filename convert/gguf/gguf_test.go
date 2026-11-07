package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The three reference models. They live outside the repo (they are 2.2 GB) --
// tinyllama and gemma in the model directory (JITLLM_MODELS) -- so
// a test skips when its model is absent — but never when it is present and
// wrong. Every literal below was cross-checked against an independent Python
// parser before it was written down; if one of these changes, either the file
// changed or this package has a bug, and both are worth failing over.
func tinyllama() string { return testmodels.Path("tinyllama-1.1b-q3_K_M.gguf") }
func gemma() string     { return testmodels.Path("gemma-2b.gguf") }
func minilm() string    { return testmodels.Path("embed/MiniLM-L6-v2.Q8_0.gguf") }

// stories260K is committed (testdata/models), so this one always runs.
const stories260K = "../../testdata/models/stories260K.gguf"

// stories15M is fetched by internal/testmodels/fetch.sh into the model directory.
func stories15M() string { return testmodels.Path("stories15M-q4_0.gguf") }

func open(t *testing.T, path string) *meta.File {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "model not present: %s (a model-directory file follows JITLLM_MODELS)", path)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open(%s): %v", path, err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

func TestHeaders(t *testing.T) {
	for _, tc := range []struct {
		name                      string
		path                      string
		size, dataStart, tensorBy uint64
		nTensors, nKV             int
		arch                      string
	}{
		{"tinyllama", tinyllama(), 550819200, 1709440, 549109760, 201, 23, "llama"},
		{"gemma", gemma(), 1678447520, 6042528, 1672404992, 164, 21, "gemma"},
		{"minilm", minilm(), 25008064, 755520, 24252528, 101, 24, "bert"},
		{"stories260K", stories260K, 1185376, 14176, 1171200, 48, 19, "llama"},
		{"stories15M", stories15M(), 19077344, 727136, 18350208, 57, 20, "llama"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := open(t, tc.path)
			if f.Version != 3 {
				t.Errorf("Version = %d, want 3", f.Version)
			}
			if f.Alignment != 32 {
				t.Errorf("Alignment = %d, want 32", f.Alignment)
			}
			if f.Size() != tc.size {
				t.Errorf("Size = %d, want %d", f.Size(), tc.size)
			}
			if f.DataStart != tc.dataStart {
				t.Errorf("DataStart = %d, want %d", f.DataStart, tc.dataStart)
			}
			if got := f.TensorBytes(); got != tc.tensorBy {
				t.Errorf("TensorBytes = %d, want %d", got, tc.tensorBy)
			}
			if len(f.Tensors) != tc.nTensors {
				t.Errorf("tensors = %d, want %d", len(f.Tensors), tc.nTensors)
			}
			if len(f.KV) != tc.nKV {
				t.Errorf("kv = %d, want %d", len(f.KV), tc.nKV)
			}
			arch, _ := f.KV["general.architecture"].String()
			if arch != tc.arch {
				t.Errorf("architecture = %q, want %q", arch, tc.arch)
			}
			// The data section must be exactly covered: header + padding + tensors,
			// with only alignment holes. MiniLM has one such 16-byte hole.
			slack := f.Size() - f.DataStart - f.TensorBytes()
			if slack >= f.Alignment*uint64(len(f.Tensors)) {
				t.Errorf("slack %d is too large to be alignment padding", slack)
			}
		})
	}
}

func TestTypeHistogram(t *testing.T) {
	f := open(t, tinyllama())
	want := []meta.TypeCount{
		{Type: quant.Q3_K, Count: 89, Bytes: 290836480},
		{Type: quant.Q4_K, Count: 62, Bytes: 187564032},
		{Type: quant.Q6_K, Count: 1, Bytes: 53760000},
		{Type: quant.Q5_K, Count: 4, Bytes: 16580608},
		{Type: quant.F32, Count: 45, Bytes: 368640},
	}
	got := f.TypeHistogram()
	if len(got) != len(want) {
		t.Fatalf("histogram has %d rows, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestRoleOrder is the reason Layer resolves by name. The two models disagree on
// intra-block tensor order, so any code that indexes a block positionally works
// on one model and silently reads the wrong weights on the other.
func TestRoleOrder(t *testing.T) {
	for _, tc := range []struct {
		name, path string
		order      []string
	}{
		{"tinyllama", tinyllama(), []string{
			"attn_norm.weight", "ffn_down.weight", "ffn_gate.weight", "ffn_up.weight",
			"ffn_norm.weight", "attn_k.weight", "attn_output.weight", "attn_q.weight", "attn_v.weight"}},
		{"gemma", gemma(), []string{
			"attn_output.weight", "attn_k.weight", "attn_v.weight", "attn_q.weight",
			"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "attn_norm.weight", "ffn_norm.weight"}},
		// A third order, from a third converter. Three models, three layouts:
		// positional indexing into a block is wrong on at least two of them.
		{"stories260K", stories260K, []string{
			"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
			"attn_norm.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight", "ffn_norm.weight"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := open(t, tc.path)
			// Every role resolves, and resolves to a distinct tensor.
			seen := map[uint64]string{}
			for _, role := range tc.order {
				tn, ok := f.Layer(0, role)
				if !ok {
					t.Fatalf("Layer(0, %q) not found", role)
				}
				if prev, dup := seen[tn.Off]; dup {
					t.Errorf("role %q and %q resolve to the same offset %d", role, prev, tn.Off)
				}
				seen[tn.Off] = role
			}
			// ...and file order really is the converter's, not execution order.
			var fileOrder []string
			for i := range f.Tensors {
				if n := f.Tensors[i].Name; len(n) > 6 && n[:6] == "blk.0." {
					fileOrder = append(fileOrder, n[6:])
				}
			}
			if len(fileOrder) != len(tc.order) {
				t.Fatalf("block 0 has %d tensors, want %d", len(fileOrder), len(tc.order))
			}
			for i := range fileOrder {
				if fileOrder[i] != tc.order[i] {
					t.Errorf("block 0 file order[%d] = %q, want %q", i, fileOrder[i], tc.order[i])
				}
			}
		})
	}
}

func TestLayerRange(t *testing.T) {
	f := open(t, tinyllama())
	off, size, ok := f.LayerRange(0)
	if !ok {
		t.Fatal("LayerRange(0) not found")
	}
	// Absolute [83629440, 106235264), 22605824 bytes, verified independently.
	if abs := f.DataStart + off; abs != 83629440 {
		t.Errorf("block 0 starts at %d, want 83629440", abs)
	}
	if size != 22605824 {
		t.Errorf("block 0 spans %d bytes, want 22605824", size)
	}
	// Every block is one contiguous run: the sum of its tensors equals its span.
	// This is what makes a per-layer madvise possible, so it is worth asserting.
	nblk, _ := f.KV["llama.block_count"].Uint()
	for l := 0; l < int(nblk); l++ {
		_, span, ok := f.LayerRange(l)
		if !ok {
			t.Fatalf("LayerRange(%d) not found", l)
		}
		var sum uint64
		prefix := "blk." + metaItoa(uint64(l)) + "."
		for i := range f.Tensors {
			if n := f.Tensors[i].Name; len(n) >= len(prefix) && n[:len(prefix)] == prefix {
				sum += f.Tensors[i].NBytes
			}
		}
		if sum != span {
			t.Errorf("block %d: tensors sum to %d but span is %d (%d bytes of hole)", l, sum, span, span-sum)
		}
	}
}

// TestGemmaAbsentKeys pins the gemma-vs-llama metadata differences that a
// config loader must survive. Each of these is absent, not zero, and defaulting
// them wrong produces fluent nonsense rather than an error.
func TestGemmaAbsentKeys(t *testing.T) {
	f := open(t, gemma())
	for _, k := range []string{"gemma.rope.freq_base", "gemma.rope.dimension_count"} {
		if _, ok := f.KV[k]; ok {
			t.Errorf("%s is present; the default-handling path is untested", k)
		}
	}
	if _, ok := f.Get("output.weight"); ok {
		t.Error("gemma has an output.weight; the tied-embedding path is untested")
	}
	// rope.dimension_count is absent, so n_rot has to come from key_length.
	// It happens to equal embd/head_count for gemma-2b (2048/8 == 256), which is
	// exactly why reading it from the wrong place goes unnoticed on this model —
	// assert the source, not the value.
	kl, ok := f.KV["gemma.attention.key_length"].Uint()
	if !ok || kl != 256 {
		t.Errorf("key_length = %d, %v; want 256", kl, ok)
	}
}

// TestAbsentKeyDefaults pins the other half of the default problem: stories15M
// has no attention.head_count_kv at all, which means MHA (kv heads == heads),
// not zero. Reading it as zero divides by zero or silently builds a 0-head
// attention; both are worse than a missing-key error.
func TestAbsentKeyDefaults(t *testing.T) {
	f := open(t, stories15M())
	if _, ok := f.Key("attention.head_count_kv"); ok {
		t.Skip("model now sets head_count_kv; this default is untested")
	}
	heads := f.UintKey("attention.head_count", 0)
	if heads != 6 {
		t.Fatalf("head_count = %d, want 6", heads)
	}
	if got := f.UintKey("attention.head_count_kv", heads); got != heads {
		t.Errorf("head_count_kv defaulted to %d, want %d (MHA)", got, heads)
	}
	if got := f.UintKey("attention.head_count_kv", 0); got != 0 {
		t.Errorf("the default is not being applied: got %d", got)
	}
}

// TestDequantGoldens is T0: bit-exact against llama.cpp's own dequantizer, with
// no tolerance at all. For quant.Q4_0 and quant.Q8_0 the arithmetic is a single product of
// a <=4-bit integer and an f16 scale, which is exact in both float32 and
// float64, so "close enough" would be hiding a bug rather than absorbing noise.
func TestDequantGoldens(t *testing.T) {
	for _, tc := range []struct {
		file string
		typ  quant.Type
	}{
		{"q4_0_gemma", quant.Q4_0},
		{"q8_0_gemma", quant.Q8_0},
		{"q8_0_minilm", quant.Q8_0},
		{"q4_0_stories", quant.Q4_0},
		{"q8_0_stories", quant.Q8_0},
		{"q3_k_tiny", quant.Q3_K},
		{"q4_k_tiny", quant.Q4_K},
		{"q5_k_tiny", quant.Q5_K},
		{"q6_k_tiny", quant.Q6_K},
	} {
		t.Run(tc.file, func(t *testing.T) {
			dir := filepath.Join("..", "..", "testdata", "golden", "dequant")
			in, err := os.ReadFile(filepath.Join(dir, tc.file+".in"))
			if err != nil {
				// Fail, not skip: the fixtures are committed, so their absence
				// is the bug.
				t.Fatalf("no goldens at %s (run scripts/gold.py): %v", dir, err)
			}
			want, err := os.ReadFile(filepath.Join(dir, tc.file+".f32"))
			if err != nil {
				t.Fatal(err)
			}
			n := len(want) / 4
			got := make([]float64, n)
			if err := quant.Dequant(tc.typ, in, got); err != nil {
				t.Fatal(err)
			}
			bad := 0
			for i := range got {
				w := binary.LittleEndian.Uint32(want[i*4:])
				g := math.Float32bits(float32(got[i]))
				if g != w {
					if bad < 5 {
						t.Errorf("[%d] = %v (%#08x), want %v (%#08x)",
							i, float32(got[i]), g, math.Float32frombits(w), w)
					}
					bad++
				}
			}
			if bad != 0 {
				t.Errorf("%d/%d mismatches", bad, n)
			} else {
				t.Logf("dequant: 0 mismatches / %d elements", n)
			}
			// A window of all-zero or constant weights would match a broken
			// dequantizer just as happily. Real weights are neither.
			lo, hi, nz := got[0], got[0], 0
			for _, v := range got {
				lo, hi = math.Min(lo, v), math.Max(hi, v)
				if v != 0 {
					nz++
				}
			}
			if hi-lo < 1e-6 || nz < n/2 {
				t.Errorf("degenerate golden window: range [%g, %g], %d/%d nonzero", lo, hi, nz, n)
			}
		})
	}
}

func TestDequantRejects(t *testing.T) {
	var dst [31]float64 // not a whole block
	if err := quant.Dequant(quant.Q8_0, make([]byte, 1024), dst[:]); err == nil {
		t.Error("quant.Dequant accepted a partial block")
	}
	var ok [32]float64
	if err := quant.Dequant(quant.Q8_0, make([]byte, 33), ok[:]); err == nil {
		t.Error("quant.Dequant accepted a short source buffer")
	}
	if err := quant.Dequant(quant.Type(4), make([]byte, 1024), ok[:]); err == nil {
		t.Error("quant.Dequant accepted a removed ggml type")
	}
}

// FuzzParse: a GGUF is untrusted input. parse must return an error rather than
// panic or read out of bounds, for every byte sequence.
//
// Budget it by execution count, not wall time:
//
//	./scripts/cap 4G -- go test ./convert/gguf -run=XXX -fuzz=FuzzParse -fuzztime=1000000x -parallel=2
//
// With a time budget the coordinator's "execs/sec" freezes at 0 while the
// workers keep running, which reads like a hang and still reports PASS. An exec
// budget either completes the count or it does not.
func FuzzParse(f *testing.F) {
	// Deliberately small seeds: everything is reachable within a few KB, and
	// a large seed makes the mutation engine copy it per execution and stall.
	if b, err := os.ReadFile(tinyllama()); err == nil {
		f.Add(b[:4096])
		f.Add(b[:1024])
		f.Add(b[:256])
		f.Add(b[:64])
		flip := append([]byte(nil), b[:1024]...)
		flip[20] ^= 0x40 // corrupt the kv count
		f.Add(flip)
	}
	f.Add([]byte("GGUF"))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, b []byte) {
		file, err := parse(b, b, "fuzz", func() error { return nil })
		if err != nil {
			return
		}
		// A file that parses must also be internally consistent: every Bytes call
		// must be in range. This catches a validation gap, not just a panic.
		for i := range file.Tensors {
			if got := uint64(len(file.Bytes(&file.Tensors[i]))); got != file.Tensors[i].NBytes {
				t.Fatalf("tensor %q: Bytes gave %d, NBytes is %d", file.Tensors[i].Name, got, file.Tensors[i].NBytes)
			}
		}
	})
}

// metaItoa mirrors meta's unexported one, because this test asserts the key
// names a parser writes and those are meta's business to format.
func metaItoa(v uint64) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

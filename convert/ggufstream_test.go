package convert

import (
	"bytes"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// fileRemote is a local file read the way a Hub file is: by range, counted.
type fileRemote struct {
	f    *os.File
	size int64
	got  atomic.Int64
}

func openFileRemote(t *testing.T, path string) *fileRemote {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	return &fileRemote{f: f, size: fi.Size()}
}

func (r *fileRemote) ReadAt(p []byte, off int64) (int, error) {
	r.got.Add(int64(len(p)))
	return r.f.ReadAt(p, off)
}
func (r *fileRemote) Size() int64  { return r.size }
func (r *fileRemote) Name() string { return r.f.Name() }

// TestStreamedGGUFIsTheConvertedFile holds a streamed conversion to the
// container the same GGUF converts to from disk, byte for byte, across the
// families whose builds read their tensors differently: one that passes every
// tensor through, one whose mixture banks are joined at write time, and a
// vision model whose tower's matrices are re-laid out at build time. It
// also demands the stream read the file about once: a build that read every
// tensor up front, or a writer that read each twice, would show here.
func TestStreamedGGUFIsTheConvertedFile(t *testing.T) {
	if !gguf.Streamable() {
		t.Skip("this platform cannot stream a GGUF, and the apps download instead -- " +
			"run this gate on Linux or macOS")
	}
	cases := []struct{ name, src, mmproj string }{
		{"stories260K", stories260KGGUF, ""},
		{"Qwen3-0.6B", testmodels.Path("Qwen3-0.6B-Q8_0.gguf"), ""},
		{"Qwen3-MOE", testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf"), ""},
		{"SmolVLM", testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf"),
			testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, p := range []string{c.src, c.mmproj} {
				if p == "" {
					continue
				}
				if _, err := os.Stat(p); err != nil {
					testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this case proved nothing", err)
				}
			}
			dir := t.TempDir()
			want := filepath.Join(dir, "disk.jlm")
			if _, err := FromGGUFs(c.src, c.mmproj, want, jlm.Fingerprint{Writer: "gate"}); err != nil {
				t.Fatal(err)
			}
			src := openFileRemote(t, c.src)
			var tower Remote
			var tr *fileRemote
			if c.mmproj != "" {
				tr = openFileRemote(t, c.mmproj)
				tower = tr
			}
			got := filepath.Join(dir, "stream.jlm")
			if _, err := StreamGGUFs(src, tower, got, jlm.Fingerprint{Writer: "gate"}); err != nil {
				t.Fatal(err)
			}
			a, err := os.ReadFile(want)
			if err != nil {
				t.Fatal(err)
			}
			b, err := os.ReadFile(got)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(a, b) {
				t.Fatalf("the streamed container (%d bytes) differs from the one converted from disk (%d)", len(b), len(a))
			}
			size, read := src.size, src.got.Load()
			if tr != nil {
				size, read = size+tr.size, read+tr.got.Load()
			}
			t.Logf("read %d bytes of %d (%.2fx)", read, size, float64(read)/float64(size))
			// The header is read in a prefix and the small tensors once per build, a
			// few MB whatever the model.
			if read < size*9/10 || read > size*11/10+(4<<20) {
				t.Fatalf("the stream read %d bytes of a %d byte source; want about one pass", read, size)
			}
		})
	}
}

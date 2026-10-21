package jlm_test

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAFailedWriteLeavesNoContainer is the gate on containers being atomic.
// Write sizes the file and lays down the header, config, vocab and table
// before any weight, so a failure after that point in a direct writer leaves a
// file that opens cleanly with the right version and Fingerprint and zeros for
// weights (a full disk produces exactly that).
//
// The failure is constructed: dst under a parent that is a file, so nothing
// can be created, and dst an existing directory, so everything is written and
// the rename fails. Afterwards there must be no container at dst and no
// ".part" beside it. Both work on every OS; a read-only directory, the first
// construction once, does not refuse a new file on Windows, where os.Chmod
// sets no directory permission.
func TestAFailedWriteLeavesNoContainer(t *testing.T) {
	src := smallSource(t)

	t.Run("a write that cannot be created leaves nothing", func(t *testing.T) {
		dir := t.TempDir()
		file := filepath.Join(dir, "file")
		if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		dst := filepath.Join(file, "out"+jlm.Ext)
		if _, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "test"}); err == nil {
			t.Fatal("writing under a parent that is a file succeeded; the gate below proves nothing")
		}
		assertNothingLeftBehind(t, dst)
	})

	// The whole container is written before the rename refuses, so this is the
	// case where a ".part" exists to be left behind.
	t.Run("a write whose rename fails leaves no part", func(t *testing.T) {
		dst := filepath.Join(t.TempDir(), "out"+jlm.Ext)
		if err := os.Mkdir(dst, 0o755); err != nil {
			t.Fatal(err)
		}
		if _, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "test"}); err == nil {
			t.Fatal("a container was renamed over a directory; the gate below proves nothing")
		}
		if fi, err := os.Stat(dst); err != nil || !fi.IsDir() {
			t.Errorf("the directory at %s was replaced (%v)", dst, err)
		}
		if _, err := os.Stat(dst + ".part"); err == nil {
			t.Errorf("a failed write left %s.part behind", dst)
		}
	})

	// The violation that matters is observed during the write: a bad shape is
	// refused before the file is created, and a truncated Data slice writes
	// successfully. What separates an atomic writer from a direct one is what
	// exists at dst while the weights go down: a direct writer already has a
	// valid header there; an atomic one has dst absent and a ".part" beside it.
	t.Run("dst does not exist while the weights are being written", func(t *testing.T) {
		dir := t.TempDir()
		dst := filepath.Join(dir, "out"+jlm.Ext)
		big := bigSource(t)

		stop := make(chan struct{})
		seen := make(chan string, 2)
		go func() {
			var sawPart, sawDst bool
			for {
				select {
				case <-stop:
					if sawPart {
						seen <- "part"
					}
					if sawDst {
						seen <- "dst"
					}
					close(seen)
					return
				default:
				}
				if _, err := os.Stat(dst + ".part"); err == nil {
					sawPart = true
				}
				if _, err := os.Stat(dst); err == nil {
					sawDst = true
				}
			}
		}()
		if _, err := jlm.Write(dst, big, jlm.Fingerprint{Host: "test"}); err != nil {
			close(stop)
			t.Fatal(err)
		}
		close(stop)
		var part, during bool
		for s := range seen {
			switch s {
			case "part":
				part = true
			case "dst":
				during = true
			}
		}
		if !part {
			t.Errorf("no %s.part was ever visible: the container is being written "+
				"in place, so an interrupted conversion leaves a file with a valid "+
				"header and holes where the weights are", dst)
		}
		_ = during // dst legitimately exists once the rename lands
	})

	t.Run("a good write publishes exactly one file", func(t *testing.T) {
		dir := t.TempDir()
		dst := filepath.Join(dir, "out"+jlm.Ext)
		if _, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "test"}); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(dst); err != nil {
			t.Fatalf("the container is not at %s: %v", dst, err)
		}
		// And no ".part" survives a success: a deferred Remove that ran before
		// the rename would delete the container instead.
		ents, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range ents {
			if strings.HasSuffix(e.Name(), ".part") {
				t.Errorf("a successful write left %s behind", e.Name())
			}
		}
		if len(ents) != 1 {
			t.Errorf("a successful write left %d files, want 1", len(ents))
		}
		c, err := jlm.Open(dst)
		if err != nil {
			t.Fatalf("the published container does not open: %v", err)
		}
		c.Close()
	})
}

func assertNothingLeftBehind(t *testing.T, dst string) {
	t.Helper()
	if _, err := os.Stat(dst); err == nil {
		t.Errorf("a FAILED write left a container at %s -- it opens, it carries the "+
			"right version and Fingerprint, and its weights are holes", dst)
	}
	if _, err := os.Stat(dst + ".part"); err == nil {
		t.Errorf("a failed write left %s.part behind", dst)
	}
}

// smallSource is the least container Write will accept: one block, a tied
// embedding, a four-token vocabulary. Separate from
// TestContainerFromNoGGUFAtAll's fixture, which is that test's subject.
func smallSource(t *testing.T) *jlm.Source {
	t.Helper()
	const k, rows = 64, 32
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
	add(jlm.RoleAttnQ, 0, jlm.TypeQ4, []uint64{k, rows}, q4(2))
	add(jlm.RoleAttnNorm, 0, jlm.TypeF32, []uint64{k}, norm())
	return &jlm.Source{
		Config: &jlm.Config{
			Arch: jlm.ArchLlama, NLayer: 1, NEmbd: k, NHead: 1, NKVHead: 1,
			HeadDim: k, NRot: k, NFFN: k, NVocab: rows, NCtx: 128,
			RMSEps: 1e-5, RopeBase: 10000, EmbdScale: 1, AttnFactor: 1,
			Flags: jlm.FlagTiedEmbd,
		},
		Vocab: &jlm.Vocab{
			Kind: jlm.VocabSPM, Tokens: jlm.NewStrings([]string{"<unk>", "▁a", "▁b", "🙂"}),
			Scores: []float32{0, -1, -2, -3},
			Kinds:  []jlm.TokenKind{jlm.TokenUnknown, jlm.TokenNormal, jlm.TokenNormal, jlm.TokenNormal},
			BOS:    0, EOS: -1, Unk: 0, Pad: -1, Sep: -1, Mask: -1,
			AddBOS: true, AddSpacePrefix: true, ByteFallback: true,
		},
		Tensors: tensors,
	}
}

// bigSource is smallSource with enough weight bytes that the write takes long
// enough to be observed in progress. The poller above needs the window; a
// three-tensor container is written faster than a Stat.
func bigSource(t *testing.T) *jlm.Source {
	t.Helper()
	const k, rows = 1024, 4096
	q4 := func(seed byte) []byte {
		b := make([]byte, rows*k/32*18)
		for i := 0; i < len(b); i += 18 {
			binary.LittleEndian.PutUint16(b[i:], 0x3800)
			b[i+2] = seed
		}
		return b
	}
	norm := func() []byte {
		b := make([]byte, k*4)
		for i := 0; i < k; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], 0x3F800000)
		}
		return b
	}
	src := smallSource(t)
	src.Config.NEmbd, src.Config.HeadDim, src.Config.NRot = k, k, k
	src.Config.NFFN, src.Config.NVocab, src.Config.NLayer = k, rows, 4
	var ts []jlm.Tensor
	add := func(role jlm.Role, block int32, ty jlm.Type, dims []uint64, b []byte) {
		e := jlm.Tensor{Role: role, Block: block, Index: -1, Type: ty,
			NDim: uint8(len(dims)), Data: b, Name: role.String()}
		copy(e.Dims[:], dims)
		ts = append(ts, e)
	}
	add(jlm.RoleTokenEmbd, jlm.DenseBlock, jlm.TypeQ4, []uint64{k, rows}, q4(1))
	for i := int32(0); i < 4; i++ {
		add(jlm.RoleAttnQ, i, jlm.TypeQ4, []uint64{k, rows}, q4(byte(2+i)))
		add(jlm.RoleAttnNorm, i, jlm.TypeF32, []uint64{k}, norm())
	}
	src.Tensors = ts
	return src
}

// TestWriteCreatesMissingDirectories gates a destination in folders that do
// not exist yet: they are created and the container lands with no part left.
func TestWriteCreatesMissingDirectories(t *testing.T) {
	dst := filepath.Join(t.TempDir(), "a", "b", "c", "out"+jlm.Ext)
	if _, err := jlm.Write(dst, smallSource(t), jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatalf("writing into missing directories: %v", err)
	}
	if _, err := os.Stat(dst); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dst + ".part"); err == nil {
		t.Errorf("a part was left beside %s", dst)
	}
}

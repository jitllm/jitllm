package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Main's recording never changes with the composer: it is kept on disk per
// profile and fixture, keyed by the fixture's size and time and by every
// source file but the composer's and its tier's (the harness's own
// included), so a change to anything main's run reads records it again.
var (
	mainKeyOnce sync.Once
	mainKey     string
)

func mainSourceKey(t *testing.T) string {
	mainKeyOnce.Do(func() {
		root, err := filepath.Abs("../..")
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(root, p)
			if d.IsDir() {
				switch rel {
				case ".git", ".claude", "jit/gpu/compose", "jit/gpu/tier", "models":
					return filepath.SkipDir
				}
				return nil
			}
			harness := strings.HasPrefix(filepath.Base(p), "parity") && filepath.Dir(rel) == "engine/model"
			if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") && !harness {
				return nil
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			h.Write([]byte(rel))
			h.Write(b)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		mainKey = hex.EncodeToString(h.Sum(nil))[:16]
	})
	return mainKey
}

// mainKernelsCached is parityKernels for main's tier, from the cache where
// it holds them.
func mainKernelsCached(t *testing.T, prof recorder.Profile, path string) ([]*ir.Kernel, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	dir, err := os.UserCacheDir()
	if err != nil {
		return parityKernels(t, prof, path, true, nil)
	}
	key := sha256.Sum256([]byte(mainSourceKey(t) + "|" + prof.Name + "|" + filepath.Base(path) + "|" +
		st.ModTime().String() + "|" + strconv.FormatInt(st.Size(), 10)))
	file := filepath.Join(dir, "jitllm-parity", hex.EncodeToString(key[:12])+".gob")
	type entry struct {
		Ks  []*ir.Kernel
		Err string
	}
	if b, err := os.ReadFile(file); err == nil {
		var e entry
		if gob.NewDecoder(bytes.NewReader(b)).Decode(&e) == nil {
			if e.Err != "" {
				return e.Ks, errors.New(e.Err)
			}
			return e.Ks, nil
		}
	}
	ks, err := parityKernels(t, prof, path, true, nil)
	e := entry{Ks: ks}
	if err != nil {
		e.Err = err.Error()
	}
	var buf bytes.Buffer
	if gob.NewEncoder(&buf).Encode(e) == nil && os.MkdirAll(filepath.Dir(file), 0o755) == nil {
		os.WriteFile(file, buf.Bytes(), 0o644)
	}
	return ks, err
}

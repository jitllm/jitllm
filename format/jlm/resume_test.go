package jlm_test

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAnInterruptedWriteResumes is the gate on a conversion that fails hours
// in not starting over. A source that names its origin fails at one tensor;
// the part and its journal must survive, the next Write must load none of the
// tensors the journal vouches for, and the container it publishes must be the
// one an uninterrupted write produces, byte for byte -- which it cannot be if
// the part was truncated or the skipped tensors were never on the disk.
// Another origin, or a part of the wrong size, starts over.
func TestAnInterruptedWriteResumes(t *testing.T) {
	fp := jlm.Fingerprint{Host: "test"}
	dir := t.TempDir()
	ref := filepath.Join(dir, "ref"+jlm.Ext)
	if _, err := jlm.Write(ref, journalled(bigSource(t)), fp); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(ref)
	if err != nil {
		t.Fatal(err)
	}

	// lazy is bigSource with every tensor behind a Load that counts its
	// calls, and tensor failAt's first call failing when fail is set.
	const failAt = 5
	var fail atomic.Bool
	lazy := func(origin string) (*jlm.Source, []atomic.Int64) {
		src := journalled(bigSource(t))
		src.Origin = origin
		loads := make([]atomic.Int64, len(src.Tensors))
		for i := range src.Tensors {
			data := src.Tensors[i].Data
			src.Tensors[i].Data = nil
			src.Tensors[i].Load = func() ([]byte, error) {
				loads[i].Add(1)
				if i == failAt && fail.CompareAndSwap(true, false) {
					return nil, errors.New("the network went away")
				}
				return data, nil
			}
		}
		return src, loads
	}
	if n := len(bigSource(t).Tensors); n <= failAt+1 {
		t.Fatalf("the fixture has %d tensors, too few to fail at %d and resume", n, failAt)
	}

	dst := filepath.Join(dir, "out"+jlm.Ext)
	interrupt := func(origin string) {
		t.Helper()
		src, _ := lazy(origin)
		fail.Store(true)
		if _, err := jlm.Write(dst, src, fp); err == nil {
			t.Fatal("the write survived a failed tensor; the gate below proves nothing")
		}
		if _, err := os.Stat(dst); err == nil {
			t.Fatal("a failed write published a container")
		}
		if _, err := os.Stat(dst + ".part"); err != nil {
			t.Fatalf("a failed write of a named origin removed its part: %v", err)
		}
		if n, ok := jlm.Resumable(dst); !ok || n != failAt {
			t.Fatalf("the journal says %d tensors (%v), want %d", n, ok, failAt)
		}
	}
	finish := func(origin string) []atomic.Int64 {
		t.Helper()
		src, loads := lazy(origin)
		if _, err := jlm.Write(dst, src, fp); err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(dst)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatal("the resumed container is not the uninterrupted one")
		}
		for _, p := range []string{dst + ".part", dst + ".part.resume"} {
			if _, err := os.Stat(p); err == nil {
				t.Errorf("a finished write left %s", filepath.Base(p))
			}
		}
		os.Remove(dst)
		return loads
	}

	t.Run("the same origin resumes at the journal", func(t *testing.T) {
		interrupt("repo@abc")
		loads := finish("repo@abc")
		for i := range loads {
			n := loads[i].Load()
			switch {
			case i < failAt && n != 0:
				t.Errorf("tensor %d, behind the journal, was loaded %d time(s)", i, n)
			case i >= failAt && n != 1:
				t.Errorf("tensor %d was loaded %d time(s), want once", i, n)
			}
		}
	})

	t.Run("another origin starts over", func(t *testing.T) {
		interrupt("repo@abc")
		loads := finish("repo@def")
		for i := range loads {
			if loads[i].Load() != 1 {
				t.Errorf("tensor %d was loaded %d time(s) writing another origin, want once", i, loads[i].Load())
			}
		}
	})

	t.Run("a part of the wrong size starts over", func(t *testing.T) {
		interrupt("repo@abc")
		if err := os.Truncate(dst+".part", 4096); err != nil {
			t.Fatal(err)
		}
		loads := finish("repo@abc")
		if loads[0].Load() != 1 {
			t.Errorf("tensor 0 was loaded %d time(s) over a truncated part, want once", loads[0].Load())
		}
	})
}

// journalled has the write journal after every tensor, so a failure at one
// leaves a journal to resume from.
func journalled(s *jlm.Source) *jlm.Source {
	s.JournalEvery = 1
	return s
}

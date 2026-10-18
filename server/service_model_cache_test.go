package server

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestListingDoesNotReopenUnchangedContainers: a second lookup of an unchanged
// file is answered from the cache, and a file whose size or modification time
// moved is probed again.
//
// VIOLATION SIGNATURE. Key the cache on the path alone and the third call
// reads 1 probe; drop the cache and the second call reads 2.
func TestListingDoesNotReopenUnchangedContainers(t *testing.T) {
	p := filepath.Join(t.TempDir(), "m.jlm")
	if err := os.WriteFile(p, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	var c detailsCache
	probes := 0
	probe := func(string) details { probes++; return details{ctx: int32(probes)} }

	if d := c.get(p, probe); d.ctx != 1 || probes != 1 {
		t.Fatalf("first call: ctx %d, %d probes", d.ctx, probes)
	}
	if d := c.get(p, probe); d.ctx != 1 || probes != 1 {
		t.Fatalf("second call reopened the file: ctx %d, %d probes", d.ctx, probes)
	}
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(p, later, later); err != nil {
		t.Fatal(err)
	}
	if c.get(p, probe); probes != 2 {
		t.Fatalf("a changed mtime was not re-probed: %d probes", probes)
	}
	if err := os.WriteFile(p, []byte("abcdef"), 0o644); err != nil {
		t.Fatal(err)
	}
	if c.get(p, probe); probes != 3 {
		t.Fatalf("a changed size was not re-probed: %d probes", probes)
	}
}

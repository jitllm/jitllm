package gguf

import (
	"github.com/jitllm/jitllm/internal/testmodels"
	"os"
	"sync"
	"testing"
)

// TestWarmIsPerOpen: concurrent Opens with and without the read-ahead pass.
// The choice is an argument of the one Open that made it; held in a package
// variable, one Open's choice is another's, which -race reports here.
func TestWarmIsPerOpen(t *testing.T) {
	if _, err := os.Stat(stories260K); err != nil {
		testmodels.Missing(t, "fixture not present: %v", err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f, err := Open(stories260K, WithoutWarm(i%2 == 0))
			if err != nil {
				t.Error(err)
				return
			}
			f.Close()
		}()
	}
	wg.Wait()
}

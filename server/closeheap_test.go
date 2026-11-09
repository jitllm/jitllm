package server

import (
	"path/filepath"
	"runtime"
	"testing"
)

// TestAClosedEngineGivesItsHeapBack loads the model a server serves, serves
// it, closes the engine and reads the live heap again: what a closed engine
// keeps is heap no later engine of the process can use. The server suite's
// heap bound (TestAServerUnderLoadStaysUnderItsBound) read 112 MiB alone and
// 324 MiB after the suite's earlier tests, every one of which had closed its
// engines.
func TestAClosedEngineGivesItsHeapBack(t *testing.T) {
	path := modelPath(t, serveModel)
	heap := func() uint64 {
		runtime.GC()
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapAlloc
	}
	run := func() {
		e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test", MaxBatchRows: 64}))
		if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "m"}); err != nil {
			e.Close()
			t.Fatal(err)
		}
		for i := 0; i < 4; i++ {
			p := make([]int32, 60)
			for j := range p {
				p[j] = int32(1000 + (i*7919+j*31)%20000)
			}
			generate(t, e, GenerateOptions{ModelID: "m", Prompt: idsPrompt(p), MaxTokens: 8, IgnoreEOS: true})
		}
		e.Close()
	}
	// The first engine of the process pays for what the process keeps once
	// (the runtime's own, the card's driver): the bound is on the next ones.
	run()
	before := heap()
	for i := 0; i < 3; i++ {
		run()
	}
	after := heap()
	t.Logf("live heap %d MiB after one engine, %d MiB after three more, each closed", before>>20, after>>20)
	const slack = 16 << 20
	if after > before+slack {
		t.Errorf("three engines, each closed, left %d MiB of live heap behind (%d -> %d MiB)",
			(after-before)>>20, before>>20, after>>20)
	}
}

//go:build jitllmtest && (amd64 || arm64)

package model

import (
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestPhaseSites names where a fused decode token's phases come from: the
// model-package caller of each, counted per token. A survey, opt-in
// (JITLLM_PHASE_SITES=1): it is how the fused-run work list is read.
func TestPhaseSites(t *testing.T) {
	if os.Getenv("JITLLM_PHASE_SITES") == "" {
		t.Skip("set JITLLM_PHASE_SITES=1")
	}
	name := "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
	if v := os.Getenv("JITLLM_MODEL"); v != "" {
		name = v
	}
	m, err := Open(jlmOf(t, testmodels.Path(name)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := m.NewInstance(64)
	defer s.Close()
	for i := 0; i < 4; i++ {
		if _, err := s.Forward(int32(3 + i)); err != nil {
			t.Fatal(err)
		}
	}
	sites := map[string]int{}
	pcs := make([]uintptr, 32)
	sched.OnPhase = func(total, chunk int) {
		n := runtime.Callers(3, pcs)
		fr := runtime.CallersFrames(pcs[:n])
		var path []string
		for {
			f, more := fr.Next()
			fn := f.Function[strings.LastIndex(f.Function, "/")+1:]
			path = append(path, fn)
			if strings.HasPrefix(fn, "model.") || !more {
				break
			}
		}
		sites[strings.Join(path, " < ")+fmt.Sprintf("  [total %d chunk %d]", total, chunk)]++
	}
	const n = 8
	for i := 0; i < n; i++ {
		if _, err := s.Forward(int32(9 + i)); err != nil {
			t.Fatal(err)
		}
	}
	sched.OnPhase = nil
	var keys []string
	tot := 0
	for k, v := range sites {
		keys = append(keys, k)
		tot += v
	}
	sort.Slice(keys, func(i, j int) bool { return sites[keys[i]] > sites[keys[j]] })
	for _, k := range keys {
		t.Logf("%6.1f/token  %s", float64(sites[k])/n, k)
	}
	t.Logf("%.1f phases/token, %d layers", float64(tot)/n, m.Cfg.NLayer)
}

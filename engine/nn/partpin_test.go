package nn

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestDecodeHonoursAPinnedParticipantCount caches a participant count for a
// model's key and then opens the same model with WithParticipants: decode
// must run at the pin, not at the cache.
//
// Violation (TokenStart applying the cached count over the pin): the pool
// reports the cached count.
func TestDecodeHonoursAPinnedParticipantCount(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	probe := NewJIT(2048, 4096, []quant.Type{quant.Q4_K})
	if probe == nil {
		t.Skip("no generated tier")
	}
	max := probe.pool.Max()
	probe.Close()
	if max < 3 {
		t.Skipf("a %d-worker pool has no count to pin below the cache", max)
	}
	j := NewJIT(2048, 4096, []quant.Type{quant.Q4_K}, WithParticipants(max-1))
	defer j.Close()
	j.AddShape(quant.Q4_K, 2048)
	storeTuned(tuneKey("part", j.keyShapes, j.pool.Max()), max-2)
	if j.tuner == nil {
		t.Fatal("no pack tuner: TokenStart returns before it applies any participant count")
	}
	j.TokenStart()
	j.TokenEnd()
	if got := j.pool.N(); got != max-1 {
		t.Fatalf("decode runs %d participants; pinned %d, cached %d", got, max-1, max-2)
	}
}

package model

import "github.com/jitllm/jitllm/format/jlm"

// enterPager serialises decode on a container whose budget is smaller than its
// pages, and does nothing at all on one that fits. It returns the release, so
// every caller is one deferred line.
//
// Every span a container hands out is a slice of a reused frame, so a second
// session's page-in can take memory the first is still reading; jlm's own lock
// makes that memory-safe but not correct, since model.layer holds spans for a
// whole block's compute. One coarse lock covers every such window (host loop,
// device upload, streamed expert fill, tower) at once, at the cost of
// cross-session parallelism on a paging model.
//
// It is conditional: a model that fits never evicts, so sessions may read spans
// concurrently (TestConcurrentMultiDeviceDoesNotLeak), and the check is one
// atomic load via jlm.CanEvict.
//
// It goes on the public boundary only, because the inner paths nest and
// sync.Mutex is not reentrant: the gated set is exactly Forward, Prefill,
// PrefillMixed, PrefillCached and StepRuns, none of which calls another.
func (m *Model) EnterPager(fr *jlm.Frames) func() {
	if m == nil || m.Container == nil || !fr.CanEvict() {
		return func() {}
	}
	m.pager.Lock()
	// The method value is bound once at construction: m.pager.Unlock written
	// here is a fresh closure every call, one heap object per token on a
	// paging model (TestDecodeDoesNotAllocate, olmoe over a Vulkan budget).
	if m.pagerUnlock != nil {
		return m.pagerUnlock
	}
	return m.pager.Unlock
}

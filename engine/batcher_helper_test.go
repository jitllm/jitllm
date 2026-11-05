package engine

import "github.com/jitllm/jitllm/model"

// testBatcher is a batcher over an instance of its own (in), served by a
// pool of its own, as the gates on continuous batching place and step it.
type testBatcher struct {
	*Batcher
	in   *Instance
	pool *Pool
}

// newTestBatcher batches up to width sequences of up to maxSeq positions
// over a fresh instance of m.
func newTestBatcher(m *model.Model, width, maxSeq int, opts ...BatcherOption) *testBatcher {
	return newTestBatcherOn(m, nil, width, maxSeq, opts...)
}

// newTestBatcherOn is newTestBatcher with the instance made with io.
func newTestBatcherOn(m *model.Model, io []Option, width, maxSeq int, opts ...BatcherOption) *testBatcher {
	in, err := NewInstance(m, maxSeq, io...)
	if err != nil {
		panic(err)
	}
	pool := NewPool()
	if err := pool.Serve(in, width); err != nil {
		panic(err)
	}
	return &testBatcher{Batcher: NewBatcher(m, pool, width, maxSeq, opts...), in: in, pool: pool}
}

// Close retires every sequence and closes the instance.
func (b *testBatcher) Close() {
	b.Batcher.Close()
	b.pool.Unserve(b.in)
	b.in.Close()
}

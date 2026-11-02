package model

// testBatcher is a batcher over an instance of its own (in), served by a
// pool of its own, as the gates on continuous batching place and step it.
type testBatcher struct {
	*Batcher
	in   *Instance
	pool *Pool
}

// newTestBatcher batches up to width sequences of up to maxSeq positions
// over a fresh instance of m.
func newTestBatcher(m *Model, width, maxSeq int, opts ...BatcherOption) *testBatcher {
	in := m.NewInstance(maxSeq)
	pool := NewPool()
	if err := pool.Serve(in, width); err != nil {
		panic(err)
	}
	opts = append(opts, BatchOver(pool))
	return &testBatcher{Batcher: m.NewBatcher(width, maxSeq, opts...), in: in, pool: pool}
}

// Close retires every sequence and closes the instance.
func (b *testBatcher) Close() {
	b.Batcher.Close()
	b.pool.Unserve(b.in)
	b.in.Close()
}

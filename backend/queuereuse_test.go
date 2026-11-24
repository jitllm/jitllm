package backend_test

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/backend"
)

// TestAClosedQueueServesTheNext: a CUDA queue closed after its session leaves
// its drained stream and staging arena for the next NewQueue, which takes
// them, and the bytes a session on the reused queue writes and reads back
// through that staging are its own, not the closed queue's. Against a close
// that destroys the stream the device keeps no idle queue.
func TestAClosedQueueServesTheNext(t *testing.T) {
	gpuLock(t)
	ran := 0
	for _, d := range backend.OpenForTest() {
		defer d.Close()
		qd, ok := d.(backend.Queued)
		if _, cuda := backend.CUDAIdleQueues(d); !ok || !cuda {
			continue
		}
		ran++
		t.Run(d.Name(), func(t *testing.T) {
			const n = 3 << 20
			buf, err := d.Alloc(n)
			if err != nil {
				t.Fatal(err)
			}
			defer buf.Free()
			idle := func() int { k, _ := backend.CUDAIdleQueues(d); return k }
			base := idle()
			for round := range 3 {
				q, err := qd.NewQueue()
				if err != nil {
					t.Fatal(err)
				}
				if round > 0 && idle() != base {
					t.Fatalf("round %d: NewQueue left %d idle queues, want %d: it made a stream beside the idle one", round, idle(), base)
				}
				in := make([]byte, n)
				for j := range in {
					in[j] = byte(j*7 + round*13)
				}
				out := make([]byte, n)
				qd.SessionOn(q, func(s backend.Session) {
					if err = s.Write(buf, in); err == nil {
						err = s.Read(buf, out)
					}
				})
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(in, out) {
					t.Fatalf("round %d: the bytes read back are not the bytes written through the queue's staging", round)
				}
				q.Close()
				if idle() != base+1 {
					t.Fatalf("round %d: a closed, drained queue left %d idle queues, want %d", round, idle(), base+1)
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no CUDA device here runs sessions on queues")
	}
}

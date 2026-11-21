package backend_test

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestACallIntoAnotherCardKeepsTheContext drives the interleaving #314 came
// from on two cards: inside card A's session on a queue of its own, a call into
// card B (B's NewQueue, run on the calling thread), B closed, then a queue made
// for A there. A context stays current on a thread until another is made
// current, so a call into B that kept B's context current would make A's queue
// in B's destroyed context -- the double free inside cuStreamCreate. B's call
// hands A's context back: A's new queue is A's, and a round trip of bytes
// through one of A's buffers on it comes back whole.
func TestACallIntoAnotherCardKeepsTheContext(t *testing.T) {
	gpuLock(t)
	n, err := backend.CUDACount()
	if err != nil || n < 2 {
		t.Skip("two CUDA devices are needed:", n, err)
	}
	open := func(i int) backend.Device {
		d, err := backend.OpenCUDAWith(testdevices.Ordinal(t, "cuda", i), backend.Opts{Admit: testdevices.Admit()})
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	a, b := open(0), open(1)
	defer a.Close()
	defer b.Close()
	qa, qb := a.(backend.Queued), b.(backend.Queued)
	q, err := qa.NewQueue()
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	var inner, made backend.Queue
	var innerErr, madeErr error
	qa.SessionOn(q, func(backend.Session) {
		inner, innerErr = qb.NewQueue()
		b.Close()
		made, madeErr = qa.NewQueue()
	})
	if innerErr != nil || madeErr != nil {
		t.Fatalf("the queues inside A's session: B's %v, A's after B closed %v", innerErr, madeErr)
	}
	defer inner.Close()
	defer made.Close()
	buf, err := a.Alloc(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Free()
	want := bytes.Repeat([]byte{1, 2, 3, 5, 8, 13, 21, 34}, 512)
	got := make([]byte, len(want))
	var wErr, rErr error
	qa.SessionOn(made, func(s backend.Session) {
		wErr = s.Write(buf, want)
		rErr = s.Read(buf, got)
	})
	if wErr != nil || rErr != nil {
		t.Fatalf("A's buffer on the queue made for A inside its session: write %v, read %v", wErr, rErr)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("A's buffer read back other bytes than were written on the queue made for A")
	}
}

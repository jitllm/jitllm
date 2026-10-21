package cuda

import "fmt"

// Stream capture and graph replay: the same launch sequence, issued once and
// then replayed with a single driver call, to cut the per-launch cost of a
// ~450-launch decode token. Whether it pays is measured by `jitllm verify -ab
// graph`, not stated here.
//
// Like llama.cpp (GGML_CUDA_USE_GRAPHS), jitllm does not patch the graph per
// token as the KV cache grows: it pads the KV length so the graph stays
// identical for a run of tokens and re-captures when the padding rolls over.

// captureRelaxed is CU_STREAM_CAPTURE_MODE_RELAXED.
//
// Relaxed, as llama.cpp uses: the stricter modes forbid the capturing thread
// from touching the legacy stream, and nothing is issued outside the capture
// stream while a capture is open anyway.
const captureRelaxed = int32(2)

// Stream is an explicit CUDA stream, created only because a graph cannot be
// captured from the legacy stream. It is a blocking stream (flags 0): the
// legacy stream synchronises with it both ways, so the legacy memcpys of a
// token's position and logits stay ordered around the graph launch. A
// non-blocking stream would silently read the previous token's logits.
type Stream struct{ s CUstream }

// NewStream creates one. It must run on the goroutine that called Open.
func (d *Device) NewStream() (*Stream, error) {
	if cuStreamCreate == nil {
		return nil, fmt.Errorf("cuda: this driver has no stream capture")
	}
	st := &Stream{}
	if err := call(cuStreamCreate(&st.s, 0), "cuStreamCreate"); err != nil {
		return nil, err
	}
	return st, nil
}

// Handle is the stream to launch on. A nil Stream is the legacy stream.
func (s *Stream) Handle() CUstream {
	if s == nil {
		return 0
	}
	return s.s
}

func (s *Stream) Destroy() {
	if s == nil || s.s == 0 {
		return
	}
	cuStreamDestroy(s.s)
	s.s = 0
}

// BeginCapture starts recording. Launches issued on this stream until
// EndCapture are added to a graph and DO NOT RUN, so a caller that captures
// must still replay before it can read a result.
func (s *Stream) BeginCapture() error {
	if !GraphsAvailable() {
		return fmt.Errorf("cuda: this driver has no stream capture")
	}
	return call(cuStreamBeginCapture(s.s, captureRelaxed), "cuStreamBeginCapture")
}

// EndCapture closes the recording and instantiates it.
//
// It must be called even when the capture went wrong: a stream left capturing
// refuses every later launch on it, and the error returned here is the one that
// says what happened.
func (s *Stream) EndCapture() (*Graph, error) {
	g := &Graph{}
	if err := call(cuStreamEndCapture(s.s, &g.g), "cuStreamEndCapture"); err != nil {
		return nil, err
	}
	if g.g == 0 {
		return nil, fmt.Errorf("cuda: capture produced no graph")
	}
	if err := call(cuGraphInstantiate(&g.e, g.g, 0), "cuGraphInstantiate"); err != nil {
		cuGraphDestroy(g.g)
		return nil, err
	}
	return g, nil
}

// Graph is a captured launch sequence in both its forms: the recording and the
// executable the driver built from it. Both are kept because destroying either
// alone leaks the other.
type Graph struct {
	g CUgraph
	e CUgraphExec
}

// Launch replays the whole sequence with one driver call.
//
// One executable may not run twice at once: the driver orders a replay behind
// any previous replay of the same executable, which is safe for one submission
// per token ended by a synchronising readback.
func (g *Graph) Launch(s *Stream) error {
	return call(cuGraphLaunch(g.e, s.Handle()), "cuGraphLaunch")
}

// Destroy releases the executable and the recording. Like every other call
// here, it must run on the goroutine that owns the context.
func (g *Graph) Destroy() {
	if g == nil {
		return
	}
	if g.e != 0 {
		cuGraphExecDestroy(g.e)
		g.e = 0
	}
	if g.g != 0 {
		cuGraphDestroy(g.g)
		g.g = 0
	}
}

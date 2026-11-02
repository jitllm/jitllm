//go:build jitllmfault

package tier

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit"
)

// Fault injection, compiled in only under the `jitllmfault` build tag.
//
// It makes a device fail on demand, which is the only way to exercise the
// recovery path that moves the KV cache to the CPU and finishes on the host. It is
// a tag rather than a runtime flag so that no release binary can be told to
// pretend its GPU broke; the untagged build compiles to `return false`.
//
// It takes a list of call ordinals because the interesting state needs two
// failures: a batched prefill chunk fails, is retried per token, and a later
// retry fails after some rows already went through the device (calls 1 and 3).
//
//	go build -tags jitllmfault ./cmd/jitllm
//	JITLLM_DEVICE_FAIL_AT=5 ./jitllm run ...
//	JITLLM_DEVICE_FAIL_AT=1,3 ./jitllm run ...   <- a partway chunk failure
func faultAt() []int {
	v := os.Getenv("JITLLM_DEVICE_FAIL_AT")
	if v == "" {
		return nil
	}
	var at []int
	for _, f := range strings.Split(v, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n <= 0 {
			return nil
		}
		at = append(at, n)
	}
	// Loud on purpose: a build that can lie about its device should say so.
	fmt.Fprintf(os.Stderr, "jitllm: FAULT INJECTION ARMED (jitllmfault build): the device will fail on call(s) %v\n", at)
	return at
}

// injectedFail reports whether this call should pretend the device failed.
func (g *devTier) injectedFail() bool {
	if len(g.failAt) == 0 {
		return false
	}
	g.calls++
	for _, n := range g.failAt {
		if g.calls == n {
			g.LastErr = "injected device failure (JITLLM_DEVICE_FAIL_AT)"
			g.noteCause(nn.FailFault)
			return true
		}
	}
	return false
}

// faultPanicAt is JITLLM_DEVICE_PANIC_AT: the ordinals of the submissions
// that panic inside the backend's session, as a served step's bug would.
func faultPanicAt() []int {
	v := os.Getenv("JITLLM_DEVICE_PANIC_AT")
	if v == "" {
		return nil
	}
	var at []int
	for _, f := range strings.Split(v, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n <= 0 {
			return nil
		}
		at = append(at, n)
	}
	fmt.Fprintf(os.Stderr, "jitllm: FAULT INJECTION ARMED (jitllmfault build): submission(s) %v will panic\n", at)
	return at
}

// injectedPanic panics where this submission is one panicAt names.
func (g *devTier) injectedPanic() {
	if len(g.panicAt) == 0 {
		return
	}
	g.subCalls++
	for _, n := range g.panicAt {
		if g.subCalls == n {
			panic(fmt.Sprintf("injected panic in submission %d (JITLLM_DEVICE_PANIC_AT)", n))
		}
	}
}

// formFault is a rewrite of the composer's registry: the decomposition of one
// op kind is replaced by one that fails, so every part holding that op fails
// to form, as a kernel the composer cannot build would. Nothing in a kernel is
// touched.
var formFault struct {
	on       bool
	kind     graph.OpKind
	asDevice bool
}

// SetFormFault makes the composer fail to form every part holding an op of
// kind, for the tiers that create their composer after it. asDevice is the
// violation a gate must not survive (RULE 10): the failure is reported as the
// device failing, which moves work. SetFormFault with on false disarms.
func SetFormFault(on bool, kind graph.OpKind, asDevice bool) {
	formFault.on, formFault.kind, formFault.asDevice = on, kind, asDevice
}

var errInjectedForm = errors.New("injected form failure (SetFormFault)")

func (g *devTier) rewriteComposer(r *jit.Registry) {
	if !formFault.on {
		return
	}
	r.Decompositions[formFault.kind] = func(graph.Op, *jit.Context) ([]jit.Prim, error) {
		return nil, errInjectedForm
	}
}

func formFaultedAsDevice() bool { return formFault.on && formFault.asDevice }

// pagedFault names a deliberate error in the paged history's wiring, so a gate
// can run each violation and demand it fails (RULE 10); the kernels' own are
// kernels.SetPagedFault.
//
//	"window"  a windowed layer reads the full descriptors: no window at all
//	"alias"   every row of a batch reads the first row's pages
//	"softwidth" the staged prefill's softmax launched 64 wide whatever its
//	          lanes: two subgroups on one item, where Metal runs the width
//	          it is given
//	"session" every row of a step across sessions reads and writes the
//	          calling session's pages, not its own session's
//	"pos"     a ragged step's second row states half its position: it
//	          writes its key over an earlier one and attends the keys before
//	          it alone (a position off by a constant would not show: the
//	          rotary is relative)
//	"passclip" a prefill pass reads every key, not its own: each key counts
//	          once per pass
//	"winrelease" a windowed layer releases the page its window starts in,
//	          with no slack: the window reads the dummy page
//	"stale"   a page-in uploads the bytes the block's host slices held at
//	          admission, without asking the host for them again: whatever
//	          block the host pager has since put in that frame
//	"passfold" a staged decode over passes folds none after the first:
//	          the finish reads a partial nothing wrote
var pagedFault string

// SetPagedFault arms one violation for the calls after it; "" disarms.
func SetPagedFault(f string) { pagedFault = f }

func pagedFaulted(f string) bool { return pagedFault == f }

// recFault names a deliberate error in a step across sessions' recurrent
// wiring (recpool.go), so the hybrid step gate can run each violation and
// demand it fails (RULE 10):
//
//	"slot"    every row of a step across sessions reads and writes the first
//	          row's seat: sessions sharing one summary
//	"parity"  a session the step brings to the other half of a pool is not
//	          copied there: it reads a half its last step did not write
//	"resize"  a pool rebuilt at another size, or a seat that grows, carries
//	          no session's state
//	"chain"   every row of a ragged step, and of a chunk the composer
//	          scans, is its own run: a prompt chunk's rows each read their
//	          sequence's state from before the step
var recFault string

// SetRecFault arms one violation for the calls after it; "" disarms.
func SetRecFault(f string) { recFault = f }

func recFaulted(f string) bool { return recFault == f }

// rowsFault names a deliberate error in a ragged step's head, so its gate can
// run the violation and demand it fails (RULE 10):
//
//	"headone"  the one-row head launches decode's own kernel, which reads the
//	           wanted row's sums where the second row's scales are
//	"nocap"    the per-row head skips the final softcap: every row's argmax
//	           and logits are the uncapped projection's
//	"headrow"  the composed per-row head projects the first wanted row
//	           alone: every other row's logits are another step's
//	"foldrow"  the composed folded head reads the chunk's first row, not
//	           its last
//	"nohidden" the composed head is the graph that writes the logits
//	           alone: the hidden readback reads normed rows nothing wrote
//	"noswa"    a step hands the device no local rotary table at all
//	"nope"     a ragged step rotates the layers Llama 4 leaves unrotated
//	"notemp"   a ragged step skips Llama 4's attention temperature
//	"nowin"    a ragged step's windowed or chunked layers attend every key
//	"ragtab"   a ragged step's run-table prefill reads its rows' keys through
//	          the decode descriptors, where those rows are empty
//	"piccausal" a composed non-causal block's rows attend causally: each
//	           row's descriptor ends at itself (picDesc)
var rowsFault string

// SetRowsFault arms one violation for the calls after it; "" disarms.
func SetRowsFault(f string) { rowsFault = f }

func rowsFaulted(f string) bool { return rowsFault == f }

// roomFault makes a device short of room in one place, so the gate of the
// path that copes can force it to fire, and names the violation that path
// must not survive (RULE 10):
//
//	"nozeros"         the kept zeros a recurrent seat is reset from cannot
//	                  be allocated: the seat is zeroed from the host
//	"nozeros-nohost"  the same, and the host write is skipped: the reset fails
//	"narrow"          no batched scratch wider than 64 rows fits: RowsWidth
//	                  clamps the Scheduler's steps to it
//	"narrow-lie"      the same, and RowsWidth says the requested width fits
//	"noseat"          no recurrent seat for a batch fits: RowsRoom says so,
//	                  and the blocks move to the CPU before the step
//	"noseat-lie"      the same, and RowsRoom says there is room
//	"nobuild"         a ragged step's twins do not build: RowsRoom returns
//	                  the error and the step is refused by name
//	"nobuild-room"    the same, and RowsRoom reads it as no room, so the
//	                  blocks move to the CPU and the step runs on the host
var roomFault string

// SetRoomFault arms one shortage for the calls after it; "" disarms.
func SetRoomFault(f string) { roomFault = f }

func roomFaulted(f string) bool { return roomFault == f }

// moeFault names a deliberate error in a batched mixture's wiring, so its gate
// can run the violation and demand it fails (RULE 10):
//
//	"floatsrc"  a float bank's down reads the gate's output where the
//	            activated product belongs: same shape, finite, wrong
//	"downsrc"   the same for a quantized bank's down, on the dp4a and the
//	            tensor-core forms alike: its activation is converted from the
//	            gate's output
//	"wout"      a grouped mixture that weights the expert's input (Llama 4)
//	            weights its output instead
var moeFault string

// SetMoEFault arms one violation for the calls after it; "" disarms.
func SetMoEFault(f string) { moeFault = f }

func moeFaulted(f string) bool { return moeFault == f }

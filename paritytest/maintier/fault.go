//go:build jitllmfault

package maintier

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Fault injection, compiled in only under the `jitllmfault` build tag.
//
// It makes a device fail on demand, which is the only way to exercise the
// recovery path that brings the KV cache home and finishes on the host. It is
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
			return true
		}
	}
	return false
}

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
//	"chain"   every row of a ragged step is its own run: a prompt chunk's
//	          rows each read their sequence's state from before the step
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
//	"swatab"   a step's local layers are handed the global rotary table
//	           (csSWA is cs), the shape of the table a rows step once
//	           never passed; it reaches the logits only where the device
//	           uploads the host's tables (Config.RopeTableHost)
//	"noswa"    a step hands the device no local rotary table at all
//	"nope"     a ragged step rotates the layers Llama 4 leaves unrotated
//	"notemp"   a ragged step skips Llama 4's attention temperature
//	"nowin"    a ragged step's windowed or chunked layers attend every key
//	"ragtab"   a ragged step's run-table prefill reads its rows' keys through
//	          the decode descriptors, where those rows are empty
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
//	                  and the blocks come home before the step
//	"noseat-lie"      the same, and RowsRoom says there is room
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

//go:build jitllmfault

package tier

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

// The named violations below are enums (faultids.go) held per tier in
// Config.hook (hooks_on.go), armed with WithFault or GPU.SetFault by family
// and name, and read as g.pagedFaulted(pagedAlias) and the like.
//
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
//	"stream"  a streamed history is read as if resident (pagedstream.go)
//	"passfold" a staged decode over passes folds none after the first:
//	          the finish reads a partial nothing wrote

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
//	"streamcutrows"  a fallback that runs a stream-major chunk in parts cuts
//	          the streams after the first one row on
//	"evictselection"  a block that selects from its whole history may have its
//	          pages sent home, and its context is not charged with it
//	"decodealloc"  every device call allocates a small buffer: the allocation
//	          gates must read it
//	"pieceouter"  a layered chunk goes pieces outer, block groups inner, so
//	          every piece pages the run in again

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
//	"noscratch"       no batched scratch fits a prompt chunk of any width: the
//	                  layers' fallbacks run it a row at a time
//	"noscratch-split" the same, and the width the device reserved is 32 rows:
//	                  the fallback runs the chunk in 32-row parts
//	"noseat"          no recurrent seat for a batch fits: RowsRoom says so,
//	                  and the blocks come home before the step
//	"noseat-lie"      the same, and RowsRoom says there is room

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
//	"prefillwave"     a prompt chunk's expert waves after the first are not
//	                  added to the block's output
//	"prefillunnamed"  a pair whose expert is not in the running wave reads a
//	                  slot's sheets with its own weight
//	"prefillalloc"    a prompt chunk's expert waves allocate on the host
//	"splitrowshost"   a split-rows mixture runs its rows on the host

// peerFault names a deliberate error in the expert card's sheets, so its gate
// can run the violation and demand it fails (RULE 10):
//
//	"wrong"  a sheet copied from the expert card is the next expert's

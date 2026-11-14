//go:build jitllmfault

package gpu

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
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

// Faults is the violations armed on the tiers opened with it (WithFaults),
// shared by their devices. The zero value arms none.
type Faults struct {
	// form is a rewrite of the composer's registry: the decomposition of one
	// op kind is replaced by one that fails, so every part holding that op
	// fails to form, as a kernel the composer cannot build would. Nothing in
	// a kernel is touched.
	form struct {
		on       bool
		kind     graph.OpKind
		asDevice bool
	}
	// moe puts a wrong source under a mixture's expert down bank in a form
	// of several sequences' rows (composer.Step.Seqs 0, a joint step's): it
	// reads the experts' gate output (their up's where they have no gate) in
	// place of their activation. A one-row form and a prompt's of one
	// sequence are untouched, so a session alone -- its prompt and its
	// tokens -- and the same session in a joint step part, as a grouped path
	// wired wrongly would part them.
	moe bool
	// moeSeen counts the down banks moe met: rewired, found no source for,
	// and met in a one-row or one-sequence form it leaves alone. A gate under
	// the fault asserts it rewired one (RULE 10).
	moeSeen [3]atomic.Int64
	// paged, rec, rows and room name one violation each (SetPagedFault,
	// SetRecFault, SetRowsFault, SetRoomFault); "" is none.
	paged, rec, rows, room string
	// block fails the call-th submission that reaches block at, there
	// (SetBlockFault); call 0 arms none. reached counts the submissions that
	// reached it since it was armed.
	block struct {
		at, call int
		reached  atomic.Int64
	}
}

// SetBlockFault fails the call-th submission from now on to reach block at,
// at that block: the blocks before it in the submission have run, it and
// the ones after it have not, as a kernel failing there leaves them. call 0
// disarms.
func (f *Faults) SetBlockFault(at, call int) {
	f.block.at, f.block.call = at, call
	f.block.reached.Store(0)
}

// BlockFaultReached is how many submissions reached the armed block.
func (f *Faults) BlockFaultReached() int64 { return f.block.reached.Load() }

const errInjectedBlock tierError = "injected block failure (SetBlockFault)"

// blockFaultArmed reports a block fault armed (SetBlockFault): the step is
// issued block by block, never replayed, so the fault can land between them.
func (g *devTier) blockFaultArmed() bool { return g.fault != nil && g.fault.block.call > 0 }

// blockFault is the injected failure of block li in this submission, where
// the faults name it (SetBlockFault); nil elsewhere.
func (g *devTier) blockFault(li int) error {
	f := g.fault
	if f == nil || f.block.call == 0 || f.block.at != li {
		return nil
	}
	if f.block.reached.Add(1) != int64(f.block.call) {
		return nil
	}
	g.noteCause(nn.FailFault)
	return errInjectedBlock
}

type tierFaults = Faults

// WithFaults arms f's violations on the tier, read at every call, so a gate
// can arm and disarm them while the tier is open.
func WithFaults(f *Faults) Option { return func(o *openOpts) { o.faults = f } }

// SetFormFault makes the composer fail to form every part holding an op of
// kind, for the devices that create their composer after it. asDevice is
// the violation a gate must not survive (RULE 10): the failure is reported as
// the device failing, which moves work. SetFormFault with on false disarms.
func (f *Faults) SetFormFault(on bool, kind graph.OpKind, asDevice bool) {
	f.form.on, f.form.kind, f.form.asDevice = on, kind, asDevice
}

const errInjectedForm tierError = "injected form failure (SetFormFault)"

// SetMoEFault arms the mixture fault (Faults.moe) for the devices that
// create their composer after it; false disarms.
func (f *Faults) SetMoEFault(on bool) { f.moe = on }

// MoEFaultSeen is what the mixture fault met: the down banks rewired, those
// with no experts' source to read, and those in one-row forms.
func (f *Faults) MoEFaultSeen() (rewired, unsourced, oneRow int64) {
	return f.moeSeen[0].Load(), f.moeSeen[1].Load(), f.moeSeen[2].Load()
}

func (g *devTier) rewriteComposer(r *composer.Registry) {
	if g.fault == nil {
		return
	}
	ff, seen := g.fault.form, &g.fault.moeSeen
	if ff.on {
		r.Decompositions[ff.kind] = func(graph.Op, *composer.Context) ([]composer.Prim, error) {
			return nil, errInjectedForm
		}
	}
	if g.fault.moe {
		d := r.Decompositions[graph.OpExpertMatVec]
		r.Decompositions[graph.OpExpertMatVec] = func(o graph.Op, c *composer.Context) ([]composer.Prim, error) {
			if len(o.Weight) > 0 && o.Weight[0].Role == jlm.RoleExpDownBank {
				switch src := expertsSource(c.Graph, o.In[0]); {
				case c.Step.Rows <= 1 || c.Step.Seqs == 1:
					seen[2].Add(1)
				case src == o.In[0]:
					seen[1].Add(1)
				default:
					seen[0].Add(1)
					o.In = slices.Clone(o.In)
					o.In[0] = src
				}
			}
			return d(o, c)
		}
	}
	// A joint step's form: several sequences' rows (composer.Step.Seqs 0),
	// as the mixture fault's.
	joint := func(c *composer.Context) bool { return c.Step.Rows > 1 && c.Step.Seqs != 1 }
	named := func(o graph.Op, c *composer.Context) string {
		if len(o.Out) == 0 {
			return ""
		}
		return stem(c.Graph.Values[o.Out[0]].Name)
	}
	m := r.Decompositions[graph.OpMap]
	switch v := g.fault.rows; v {
	case "nocap", "notemp":
		r.Decompositions[graph.OpMap] = func(o graph.Op, c *composer.Context) ([]composer.Prim, error) {
			if !joint(c) {
				return m(o, c)
			}
			switch n := named(o, c); {
			case v == "nocap" && n == "capped":
				// The capped logits are the projection's, the cap's input.
				if raw, ok := capInput(c.Graph); ok {
					o.In, o.Attr.Const = []graph.ValueID{raw}, 1
				}
			case v == "notemp" && n == "q.temp":
				o.In, o.Attr.Fn, o.Attr.Const = o.In[:1], graph.MapMul, 1
			}
			return m(o, c)
		}
	case "nowin":
		d := r.Decompositions[graph.OpAttend]
		r.Decompositions[graph.OpAttend] = func(o graph.Op, c *composer.Context) ([]composer.Prim, error) {
			if joint(c) {
				o.Attr.Window, o.Attr.Chunked = 0, false
			}
			return d(o, c)
		}
	case "nope":
		d := r.Decompositions[graph.OpRope]
		r.Decompositions[graph.OpRope] = func(o graph.Op, c *composer.Context) ([]composer.Prim, error) {
			if !joint(c) {
				return d(o, c)
			}
			return m(graph.Op{Kind: graph.OpMap, In: o.In[:1], Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 1}}, c)
		}
	}
}

// capInput is the logits head graph gr caps (graph.Head's softcap): the
// input of its "softcap.arg".
func capInput(gr *graph.Graph) (graph.ValueID, bool) {
	for _, o := range gr.Ops {
		if len(o.Out) == 1 && len(o.In) > 0 && stem(gr.Values[o.Out[0]].Name) == "softcap.arg" {
			return o.In[0], true
		}
	}
	return 0, false
}

// stem is a value's name as its graph's builder was given it: the builder
// names each value apart with "." and its index (graph.Builder.value).
func stem(name string) string {
	if i := strings.LastIndexByte(name, '.'); i >= 0 {
		if _, err := strconv.Atoi(name[i+1:]); err == nil {
			return name[:i]
		}
	}
	return name
}

// expertsSource is the experts' gate output in gr, their up's where they
// have no gate, as wide as v: what the mixture fault reads in place of v.
func expertsSource(gr *graph.Graph, v graph.ValueID) graph.ValueID {
	// The op making v -- the experts' activation -- reads the gate first, or
	// the up projection where there is no gate.
	for _, o := range gr.Ops {
		if len(o.Out) == 0 || o.Out[0] != v {
			continue
		}
		for _, in := range o.In {
			if in != v && gr.Values[in].Width == gr.Values[v].Width {
				return in
			}
		}
	}
	return v
}

func (c *Config) formFaultedAsDevice() bool {
	return c.fault != nil && c.fault.form.on && c.fault.form.asDevice
}

// SetPagedFault names a deliberate error in the paged history's wiring, so a gate
// can run each violation and demand it fails (RULE 10):
//
//	"window"  a windowed layer reads the full descriptors: no window at all
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
//
// SetPagedFault arms one violation for the calls after it; "" disarms.
func (f *Faults) SetPagedFault(v string) { f.paged = v }

func (c *Config) pagedFaulted(f string) bool { return c.fault != nil && c.fault.paged == f }

// SetRecFault names a deliberate error in a step across sessions' recurrent
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
//
// SetRecFault arms one violation for the calls after it; "" disarms.
func (f *Faults) SetRecFault(v string) { f.rec = v }

func (c *Config) recFaulted(f string) bool { return c.fault != nil && c.fault.rec == f }

// SetRowsFault names a deliberate error in a ragged step's head, so its gate can
// run the violation and demand it fails (RULE 10). The four of a joint
// step's features -- nocap, nope, notemp and nowin -- rewrite the composer's
// registry (rewriteComposer) for the forms of several sequences' rows alone,
// as the mixture fault does, so a session alone is untouched:
//
//	"nocap"    the per-row head skips the final softcap: every row's argmax
//	           and logits are the uncapped projection's
//	"headrow"  the composed per-row head projects the first wanted row
//	           alone: every other row's logits are another step's
//	"foldrow"  the composed folded head reads the chunk's first row, not
//	           its last
//	"nohidden" the composed head is the graph that writes the logits
//	           alone: the hidden readback reads normed rows nothing wrote
//	"noswa"    a step hands the device no local rotary table at all
//	"nope"     a ragged step leaves the layers Llama 4 rotates unrotated:
//	           which layers turn is the feature
//	"notemp"   a ragged step skips Llama 4's attention temperature
//	"nowin"    a ragged step's windowed or chunked layers attend every key
//	"piccausal" a composed non-causal block's rows attend causally: each
//	           row's descriptor ends at itself (picDesc)
//	"runcut"   a bidirectional run wider than a submission reads, in its
//	           second phase, only the keys of the call's own rows (runEnd)
//
// SetRowsFault arms one violation for the calls after it; "" disarms.
func (f *Faults) SetRowsFault(v string) { f.rows = v }

func (c *Config) rowsFaulted(f string) bool { return c.fault != nil && c.fault.rows == f }

// SetRoomFault makes a device short of room in one place, so the gate of the
// path that copes can force it to fire, and names the violation that path
// must not survive (RULE 10):
//
//	"nozeros"         the kept zeros a recurrent seat is reset from cannot
//	                  be allocated: the seat is zeroed from the host
//	"nozeros-nohost"  the same, and the host write is skipped: the reset fails
//	"narrow"          no batched scratch wider than 64 rows fits: RowsWidth
//	                  clamps the Scheduler's steps to it
//	"narrow-lie"      the same, and RowsWidth says the requested width fits
//	"noroom"          a step's scratch passes the budget: RowsRoom says so,
//	                  and the blocks move to the CPU before the step
//	"noroom-lie"      the same, and RowsRoom says there is room: the card
//	                  refuses the step
//	"nobuild"         a ragged step's twins do not build: RowsRoom returns
//	                  the error and the step is refused by name
//	"nobuild-room"    the same, and RowsRoom reads it as no room, so the
//	                  blocks move to the CPU and the step runs on the host
//
// SetRoomFault arms one shortage for the calls after it; "" disarms.
func (f *Faults) SetRoomFault(v string) { f.room = v }

func (c *Config) roomFaulted(f string) bool { return c.fault != nil && c.fault.room == f }

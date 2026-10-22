package model

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit"
)

// This package reads no environment. cmd/jitllm reads the JITLLM_* names and
// passes them in as With...() options.
//
// The knobs are per model, not process-wide: a server loading several models
// must not have one State read another model's attention schedule or cache
// width. engine/model/optioniso_test.go is the gate.

// modelOpts is the resolved set of per-model knobs. It is copied onto the Model
// at Open and onto the Tower beside it, so two models cannot collide.
type modelOpts struct {
	// planDev and planGPU are WithPlanDevice's device and the composer's
	// description of it (planrun.go).
	planDev backend.Device
	planGPU jit.Device
	// devSample is WithDeviceSample's setting (devsample.go).
	devSample DeviceSampleMode
	// profile arms the per-op nanosecond counters OpProfile reports. The
	// counters stay process-wide; only the gate is per model.
	profile bool
	// noFuse runs a decode token's host blocks one pool region per op rather
	// than as fused runs; see WithBlockFusion.
	noFuse bool
	// noServingSplit keeps a prefill on the P and E cores beside a decoding
	// session; see WithServingSplit.
	noServingSplit bool
	// noGroups runs a fused run's matvecs a phase each; see WithMatVecGroups.
	noGroups bool
	// noLook turns off the fused runs' weight lookahead; see WithLookahead.
	noLook bool
	// residFold adds the o and down projections into the residual inside
	// their groups; see WithResidualFold.
	residFold bool
	// noShOverlap runs a mixture's shared expert after its routed read
	// rather than behind it; see WithSharedOverlap.
	noShOverlap bool
	// forwardRows is the longest prompt run a token at a time; see
	// WithPrefillForwardRows.
	forwardRows    int
	forwardRowsSet bool
	// hybridSpin is how long the pool's workers spin between a token's
	// host expert calls inside a device's submission; see WithHybridSpin.
	hybridSpin    time.Duration
	hybridSpinSet bool
	// linRowByRow runs a prompt's linear blocks one matvec per row
	// (WithLinearRowByRow).
	linRowByRow bool
	// noStreamTrial keeps an auto-streamed placement without measuring it
	// against the host; see WithStreamTrial.
	noStreamTrial bool
	// hostMoves sends every move between devices through the State's cache
	// and stages nothing ahead (WithDirectMoves).
	hostMoves bool
	// noKVPrewarm leaves a device's KV pools at their first frame after
	// placement (WithKVPrewarm).
	noKVPrewarm bool
	// noTailRows runs a prompt's few-row chunk batched rather than as a
	// ragged step (tailrows.go).
	noTailRows bool
	// noTailPrune carries every prompt row through a KV-sharing tail rather
	// than only the rows something consumes (tailprune.go).
	noTailPrune bool
	// experts is WithExperts' choice.
	experts string
	// trialAA makes the stream trial an A/A self-control (WithStreamTrialAA).
	trialAA bool

	// chatClock is the clock a chat template's strftime_now reads; nil is the
	// wall clock.
	chatClock func() time.Time

	// headPlace allows the output projection onto the device.
	headPlace bool
	// place is WithPlacement's map, nil when the engine decides everything.
	place *Placement
	// elemPool runs the vision tower's elementwise passes on the pool.
	elemPool bool
	// towerGPULayers caps how many tower blocks are offered; negative is all.
	towerGPULayers int
	// towerCLSLast puts a class token at the END of the tower's sequence, with
	// the position table added in row order -- llama.cpp's layout, which pairs
	// the class token's position row with patch 0. It exists only so a test
	// can hold the tower node by node to llama-mtmd-debug; the engine follows
	// the reference's layout (class first).
	towerCLSLast bool
	// growDebug traces -gpu-grow's per-token adoption decisions.
	growDebug bool

	// prefillStream is whether a prompt's host blocks stream through the
	// device for the prompt (WithPrefillStream).
	prefillStream PrefillStream
	// prefillStreamDebug traces the streaming decision
	// (WithPrefillStreamDebug).
	prefillStreamDebug bool

	// The seam tuner's ABBA shape.
	seamWarmup  int
	seamRun     int
	seamRounds  int
	seamVerbose bool

	// attnPair is the attention pairing schedule; see WithAttnPair.
	attnPair int

	// kvPage pins the positions per KV page for every layer that has one, for
	// the A/B that proves paging changes no answer; 0 is kvPageTarget. At
	// P >= maxSeq a layer has one page whose bytes are the flat cache. Only a
	// test sets it (setKVPageForTest).
	kvPage int

	// kvHeadMajor selects the head-major KV layout for States built afterwards.
	kvHeadMajor bool
	// kvType and kvTypeSet force the cache's format, overriding what
	// nn.KVWidthPaysOff decides. Two fields because "not forced" and "forced
	// to f32" are different states.
	kvType    KVType
	kvTypeSet bool
}

// defaultOpts is what the environment variable's absence used to mean.
func defaultOpts() modelOpts {
	return modelOpts{
		headPlace:      true,
		elemPool:       true,
		towerGPULayers: -1,
		seamWarmup:     32,
		seamRun:        24,
		seamRounds:     3,
		attnPair:       attnPairShipped,
	}
}

// WithBlockFusion sets whether a decode token runs each run of consecutive
// resident host blocks as one fused pool region (sched.Pool.Fuse), its ops
// phases of that region, instead of one region per op. On by default: the
// kernels, their chunks and their order are the same, so the answer is the
// same bits (TestFusedBlocksMatchPerOp); off is the other arm.
func WithBlockFusion(on bool) Option { return func(l *loadOpts) { l.opt.noFuse = !on } }

// WithServingSplit sets whether, while another session of the model decodes,
// a session's prefill runs on the E-cores' own pool, leaving the P-cores and
// their fused runs to the decoder. On by default where the host has E-cores a
// thread can be kept on (macOS on Apple Silicon): measured, the decoder beside
// a prefill goes from 11 to 52 tok/s and the prefill beside it loses a quarter
// of its rate (docs/engineering-history/scheduling-and-measurement.md). A
// session with no other decoding is not affected, and nothing changes off
// macOS. Off, a prefill beside a decoder shares the P-cores' crew as before.
func WithServingSplit(on bool) Option { return func(l *loadOpts) { l.opt.noServingSplit = !on } }

// WithMatVecGroups sets whether a fused run's matvecs that read one activation
// run as one region with the activation's quantization folded in
// (nn.MatVecPackedGroup). On by default, the same bits; off is the A/B arm.
func WithMatVecGroups(on bool) Option { return func(l *loadOpts) { l.opt.noGroups = !on } }

// WithLookahead sets whether a fused run's workers that find a phase drained
// hint the next op's weights into the outer caches while the phase's last
// chunks run (nn.JIT.SetAhead). On by default; it writes nothing a kernel
// reads, so the bits are the same either way. Off is the A/B arm.
func WithLookahead(on bool) Option { return func(l *loadOpts) { l.opt.noLook = !on } }

// WithResidualFold sets whether a fused run's o and down groups add their rows
// into the residual as each worker makes them, rather than in an add of its
// own. The same bits either way (TestFusedBlocksMatchPerOp runs both). Off by
// default: on the Xeon its rows against off read 0.9905, 0.9965 and 1.0015
// (Llama-3.2-1B), unresolved, so it is not on until a row shows it gains.
func WithResidualFold(on bool) Option { return func(l *loadOpts) { l.opt.residFold = on } }

// SetResidualFold is WithResidualFold for one State.
func (s *State) SetResidualFold(on bool) { s.residFold = on }

// WithSharedOverlap sets whether a latent mixture's shared experts run while
// its routed experts are read from the container, instead of after. On by
// default: the two read nothing in common and the shared contribution is
// added where it always was, so the answer is the same float sum
// (TestSharedOverlapIsTheSameSum); off is the other arm.
// WithHybridSpin sets how long the host pool's workers spin between the
// routed-expert calls a device makes from inside its submission (hybrid
// experts, nn.LayerWeights.HostExperts). Between two such calls the host waits
// on the card and uses no core, for longer than the pool's own spin budget,
// so the workers park and each call wakes them one by one: on Qwen3.8-Flash-
// Next over a V100 that wake was 140 us of a 545 us call, 48 times a token.
// The default is 2 ms, the measured winner there; 0 keeps the pool's budget.
func WithHybridSpin(d time.Duration) Option {
	return func(l *loadOpts) { l.opt.hybridSpin, l.opt.hybridSpinSet = d, true }
}

// WithPrefillForwardRows runs a prompt of up to n tokens a token at a time,
// through the decode path, rather than as one batched chunk. One token always
// is. A placement whose blocks suspend for host experts runs decode's split
// (the card's cached experts beside the host's) where the batched chunk runs
// every row's experts on the host.
//
// Unset, a model whose device has run host experts takes hostRowsByRow, and
// any other one token: on Qwen3.8-Flash-Next over a V100 a five-token prompt
// took 210-236 ms this way and 474-482 ms as a chunk whose experts all ran
// on the host.
func WithPrefillForwardRows(n int) Option {
	return func(l *loadOpts) { l.opt.forwardRows, l.opt.forwardRowsSet = n, true }
}

func WithSharedOverlap(on bool) Option { return func(l *loadOpts) { l.opt.noShOverlap = !on } }

// WithLinearRowByRow runs a prompt chunk's recurrent blocks (the gated delta
// rule, KDA, Mamba-1 and -2, the short convolution) one row at a time through
// every projection, as decode does, instead of running the projections as
// GEMMs over the chunk around the per-row recurrence (linrows.go). Off by
// default; on is the reference arm the batched form is held to.
func WithLinearRowByRow(on bool) Option { return func(l *loadOpts) { l.opt.linRowByRow = on } }

// WithExperts places the routed experts of every mixture block a device
// takes: "host" (hybrid), "card" (sheets streamed every token), or "" (the
// engine decides; the default). A -placement entry's %host or %card wins for
// its blocks.
func WithExperts(where string) Option { return func(l *loadOpts) { l.opt.experts = where } }

// WithStreamTrialAA turns the stream trial into its own self-control: the
// placement made against itself, through the same migrations, so a paired
// comparison can be checked against the harness's bias and spread first.
func WithStreamTrialAA(on bool) Option { return func(l *loadOpts) { l.opt.trialAA = on } }

// WithDirectMoves sets whether a block moving between two devices
// (State.MoveBlocks, PrepareMove) is prepared on its target while the
// sequence keeps generating and has its history copied device to device. Off,
// every move brings the history home to the State's cache and up again, with
// the target prepared inside the pause: the arm the direct path is measured
// against. On by default.
func WithDirectMoves(on bool) Option { return func(l *loadOpts) { l.opt.hostMoves = !on } }

// WithStreamTrial sets whether a placement that streamed mixture blocks no
// card could hold resident is measured against running them on the host
// (State.initStreamTrial) and the faster kept. On by default.
func WithStreamTrial(on bool) Option { return func(l *loadOpts) { l.opt.noStreamTrial = !on } }

// WithKVPrewarm sets whether a State that places blocks on a device grows the
// device's KV pools ahead, by one prompt chunk, into the room placement left
// (nn.KVPrewarmer), so its first prompt does not grow them inside its own
// time. On by default; off is the A/B arm.
func WithKVPrewarm(on bool) Option { return func(l *loadOpts) { l.opt.noKVPrewarm = !on } }

// WithTailPrune sets whether a prompt's final run of KV-sharing blocks, which
// write no history, carries only the rows something consumes: none of an
// intermediate chunk's, the last row of the final chunk's (tailprune.go). On by
// default; off is the A/B arm.
func WithTailPrune(on bool) Option { return func(l *loadOpts) { l.opt.noTailPrune = !on } }

// WithTailRows sets whether a prompt chunk of at most four rows, every block
// and the head on one device that runs ragged rows, runs as one ragged step
// of its rows (decode's matvec over them) rather than as a batched chunk at
// the device's grain (tailrows.go). On by default; off is the A/B arm.
func WithTailRows(on bool) Option { return func(l *loadOpts) { l.opt.noTailRows = !on } }

// WithProfile arms the per-op nanosecond counters OpProfile reports.
func WithProfile(on bool) Option { return func(l *loadOpts) { l.opt.profile = on } }

// WithChatClock is the clock a chat template reads the date from
// (strftime_now; Llama 3's templates print it in the system turn). The wall
// clock by default. Pinning it makes a rendered prompt a function of its
// messages alone: a golden compares across days, and a server's cached prefix
// survives midnight.
func WithChatClock(now func() time.Time) Option { return func(l *loadOpts) { l.opt.chatClock = now } }

// WithHeadPlacement decides whether the output projection may be placed on the
// device. Off leaves it on the host, which is the other arm of the placement
// comparison and the way to bisect a divergence to it.
//
// It is a load option rather than an -ab arm because the choice is made once at
// prepDevice, before the first token.
func WithHeadPlacement(on bool) Option { return func(l *loadOpts) { l.opt.headPlace = on } }

// Place is where one block, or the output projection, runs.
type Place struct {
	// On is "host", or a device as the -devices entry that opened it names it:
	// "cuda:0", "vulkan:1", "metal", "gpu:0". A device opened by all or auto
	// has no name a placement can use.
	On string
	// Pin keeps it there. Seam moves, relocation, the seam tuner and
	// -gpu-grow leave a pinned block where it is, the device pager never
	// evicts it, and a device budget that would is refused. Unpinned, On is
	// only where the block starts.
	Pin bool
	// Stream runs the block on device On with its weights streaming through
	// the device's slots rather than resident: paged in when the block runs
	// and a victim when another streamed block needs the room, so a device
	// smaller than the blocks it is given still runs every one of them. Its
	// attention history stays on the device, since nothing can re-read it.
	// A block placed without it is resident on the device, or runs on the
	// host when it does not fit. On must name a device. With Pin, the block
	// stays on that device and keeps streaming.
	Stream bool
	// Experts places a mixture block's routed experts off the card: "host"
	// runs them on the host's kernels while the rest of the block runs on
	// device On (hybrid), "card" streams their sheets to it every token. ""
	// leaves it to the engine: resident when the block fits, otherwise
	// streamed with the experts on the host and measured against the host
	// (WithStreamTrial). Written "N=DEV%host" or "N=DEV%card".
	Experts string
}

// Placement is an explicit map of where blocks run, for the blocks it names;
// the engine places the rest as it would without it.
type Placement struct {
	Blocks map[int]Place
	// Rest places every block Blocks does not name; nil leaves them to the
	// engine.
	Rest *Place
	// Head places the output projection; nil leaves it to the engine.
	Head *Place
	// Strict makes it all or nothing: a named block or head that cannot go
	// where it is named fails State.SetDevice and nothing is placed. Best
	// effort (false) puts such a block where the engine would have, and the
	// decline is reported. Naming a device that is not attached is an error
	// either way.
	Strict bool
}

// Streams reports whether the placement streams any block through a device
// (Place.Stream): what a caller sizing the packed arena a page-in reads from
// asks.
func (p *Placement) Streams() bool {
	if p == nil {
		return false
	}
	if p.Rest != nil && p.Rest.Stream {
		return true
	}
	for _, pl := range p.Blocks {
		if pl.Stream {
			return true
		}
	}
	return false
}

// WithPlacement places blocks and the head where the map says whenever a
// device is attached (State.SetDevice).
func WithPlacement(p Placement) Option {
	blocks := make(map[int]Place, len(p.Blocks))
	for k, v := range p.Blocks {
		blocks[k] = v
	}
	p.Blocks = blocks
	if p.Head != nil {
		h := *p.Head
		p.Head = &h
	}
	if p.Rest != nil {
		r := *p.Rest
		p.Rest = &r
	}
	return func(l *loadOpts) { l.opt.place = &p }
}

// WithKVHeadMajor selects the head-major KV layout for States this model
// creates afterwards. A running State cannot change layout, because its cache
// already holds a history in the old one. Off by default until an end-to-end
// A/B charges the scattered write (see kvLayout).
func WithKVHeadMajor(on bool) Option { return func(l *loadOpts) { l.opt.kvHeadMajor = on } }

// WithKVF16 forces the cache width, overriding what nn.KVWidthPaysOff decides:
// WithKVType(KVF16) or WithKVType(KVF32).
func WithKVF16(on bool) Option {
	t := KVF32
	if on {
		t = KVF16
	}
	return WithKVType(t)
}

// WithKVType forces the KV cache's format for States this model creates
// afterwards (kvtype.go has the formats and what each refuses).
func WithKVType(t KVType) Option {
	return func(l *loadOpts) { l.opt.kvType, l.opt.kvTypeSet = t, true }
}

// WithAttnPair selects the attention pairing schedule: 0 off, 3 (the default)
// shares both the K walk and the V walk between each pair of query heads, 1 and
// 2 select the halves.
//
// It lets a single-binary board row price the pairing. Values outside 0..3 are
// ignored: call sites mask with pair&1 / pair&2, so -1 would pair everything
// while reading as "off".
func WithAttnPair(n int) Option {
	return func(l *loadOpts) {
		if n >= 0 && n <= 3 {
			l.opt.attnPair = n
		}
	}
}

// WithTowerTiling runs the vision tower's elementwise passes on the pool rather
// than on one core. On is the default.
//
// It exists so the A/B is one binary; two builds differ by as much as the
// effect.
func WithTowerTiling(on bool) Option { return func(l *loadOpts) { l.opt.elemPool = on } }

// WithTowerGPULayers caps how many tower blocks are offered to the device.
// Negative means "offer them all", which is the default.
//
// It is the bisection knob: placing one block separates "this block is wrong"
// from "twelve nearly-right blocks compound".
func WithTowerGPULayers(n int) Option { return func(l *loadOpts) { l.opt.towerGPULayers = n } }

// WithGrowDebug traces -gpu-grow's per-token adoption decisions to stderr.
func WithGrowDebug(on bool) Option { return func(l *loadOpts) { l.opt.growDebug = on } }

// WithSeamTuneSchedule sets the seam tuner's ABBA shape: tokens skipped before
// it starts, tokens in one timed run, and quads before it decides. A
// non-positive value keeps the default (32, 24, 3). Verbose traces each duel.
//
// Deciding costs about warmup + rounds*4*perRun tokens and four migrations.
func WithSeamTuneSchedule(warmup, perRun, rounds int, verbose bool) Option {
	return func(l *loadOpts) {
		if warmup > 0 {
			l.opt.seamWarmup = warmup
		}
		if perRun > 0 {
			l.opt.seamRun = perRun
		}
		if rounds > 0 {
			l.opt.seamRounds = rounds
		}
		l.opt.seamVerbose = verbose
	}
}

// ParsePlacement reads a placement written as comma-separated SEL=WHERE entries,
// the form -placement takes:
//
//	0=host,1-7=cuda:0,8=host!,9-15=vulkan:1~,head=vulkan:1
//	*=cuda:0~
//
// SEL is a block, an inclusive range of blocks, "head", or "*" for every block
// no other entry names. WHERE is "host" or a device as -devices names it; a
// trailing "!" pins that entry (Place.Pin) and a trailing "~" streams it
// through the device (Place.Stream), in either order.
func ParsePlacement(spec string, strict bool) (Placement, error) {
	p := Placement{Blocks: map[int]Place{}, Strict: strict}
	for _, raw := range strings.Split(spec, ",") {
		e := strings.TrimSpace(raw)
		if e == "" {
			continue
		}
		sel, where, ok := strings.Cut(e, "=")
		if !ok || strings.TrimSpace(where) == "" {
			return Placement{}, fmt.Errorf("model: placement %q: %q is not SEL=WHERE", spec, e)
		}
		where = strings.TrimSpace(where)
		var pl Place
		if w, ex, ok := strings.Cut(where, "%"); ok {
			if ex != "host" && ex != "card" {
				return Placement{}, fmt.Errorf("model: placement %q: %q places the experts on %q; "+
					"the choices are %%host and %%card", spec, e, ex)
			}
			where, pl.Experts = strings.TrimSpace(w), ex
		}
		for {
			switch {
			case strings.HasSuffix(where, "!"):
				pl.Pin, where = true, strings.TrimSuffix(where, "!")
				continue
			case strings.HasSuffix(where, "~"):
				pl.Stream, where = true, strings.TrimSuffix(where, "~")
				continue
			}
			break
		}
		pl.On = where
		if pl.On != "host" {
			pl.On = nn.DeviceName(pl.On)
		} else if pl.Experts != "" {
			return Placement{}, fmt.Errorf("model: placement %q: %q places a block's experts beside "+
				"a block that is on the host already", spec, e)
		} else if pl.Stream {
			return Placement{}, fmt.Errorf("model: placement %q: %q streams a block through the host, "+
				"which has no slots to stream through", spec, e)
		}
		sel = strings.TrimSpace(sel)
		if sel == "head" {
			if pl.Stream {
				return Placement{}, fmt.Errorf("model: placement %q: the head is read every token "+
					"and does not stream", spec)
			}
			h := pl
			p.Head = &h
			continue
		}
		if sel == "*" {
			if p.Rest != nil {
				return Placement{}, fmt.Errorf("model: placement %q names * twice", spec)
			}
			r := pl
			p.Rest = &r
			continue
		}
		lo, hi := sel, sel
		if a, b, ok := strings.Cut(sel, "-"); ok {
			lo, hi = a, b
		}
		a, err1 := strconv.Atoi(strings.TrimSpace(lo))
		b, err2 := strconv.Atoi(strings.TrimSpace(hi))
		if err1 != nil || err2 != nil || a < 0 || b < a {
			return Placement{}, fmt.Errorf("model: placement %q: %q is not a block, a range N-M or head", spec, sel)
		}
		for li := a; li <= b; li++ {
			if _, dup := p.Blocks[li]; dup {
				return Placement{}, fmt.Errorf("model: placement %q names block %d twice", spec, li)
			}
			p.Blocks[li] = pl
		}
	}
	return p, nil
}

// hybridSpinFor is WithHybridSpin's duration with its default.
func (o *modelOpts) hybridSpinFor() time.Duration {
	if o.hybridSpinSet {
		return o.hybridSpin
	}
	return 2 * time.Millisecond
}

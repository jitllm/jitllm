// Package engine attaches the shared engine (common/engine) to the window: a
// Front over app.Shell and a State over the Store's signals. The engine itself
// knows no front end; this is the window's half of that seam.
package engine

import (
	core "github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/common/hardware"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/screen"
)

// Engine is the shared engine, as the window holds it.
type Engine = core.Engine

// The shared engine and the hardware probe are what the screens drive.
var (
	_ screen.Engine  = (*Engine)(nil)
	_ screen.Machine = hardware.Host{}
	_ core.Front     = front{}
)

// New starts the engine worker for sh. Close stops it.
func New(sh *app.Shell) *Engine { return core.New(front{sh}, StateOf(sh.Store)) }

// front is the shell, plus what the engine asks of the screens and the
// transcript.
type front struct{ *app.Shell }

func (f front) OfferConvert(path string)   { screen.OfferConvert(f.Shell, path) }
func (f front) OfferReconvert(path string) { screen.OfferReconvert(f.Shell, path) }

func (f front) SetTurn(i int, t session.Turn) { f.Store.SetTurn(i, t) }

func (f front) SetThinkOpen(i int, open bool) { f.Store.SetReplyThinkOpen(i, open) }

// StateOf is the engine's view of st: the same signals, so a write from the
// worker is a signal write, which the Store's concurrency contract allows.
func StateOf(st *app.Store) core.State {
	return core.State{
		Status:  st.Status,
		Problem: st.Problem,

		ModelPath:    st.ModelPath,
		Models:       st.Models,
		Active:       st.Active,
		ModelSummary: st.ModelSummary,
		Loaded:       st.Loaded,
		Busy:         st.Busy,
		Streaming:    st.Streaming,
		Loads:        st.Loads,

		Revision:     st.Revision,
		ShowThinking: st.ShowThinking,
		Stream:       st.Stream,
		StreamThink:  st.StreamThink,
		Vision:       st.Vision,
		Chat:         st.Chat,
		ChatCapable:  st.ChatCapable,

		PromptTokS:  st.PromptTokS,
		DecodeTokS:  st.DecodeTokS,
		BytesPerTok: st.BytesPerTok,
		GBs:         st.GBs,

		DeviceSpec:   st.DeviceSpec,
		Placement:    st.Placement,
		MaxSeq:       st.MaxSeq,
		DeviceReport: st.DeviceReport,
		BlockMap:     st.BlockMap,
		Pager:        st.Pager,
		Alloc:        st.Alloc,

		KVCache: st.KVCache,
	}
}

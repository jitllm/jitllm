package engine

import "github.com/jitllm/jitllm/common/session"

// Front is the front end the engine reports to. Every method is safe from any
// goroutine; Post is the way onto the front end's own goroutine, and the
// engine writes its State values from inside a Post wherever the order of the
// writes matters.
type Front interface {
	// Post runs fn on the front end's goroutine, after what is already queued.
	Post(fn func())
	// SetStatus writes the status line.
	SetStatus(msg string)
	// Alert raises a modal alert.
	Alert(title, message string)
	// Report shows p as the session's problem, giving it a fresh Seq.
	Report(p session.Problem)
	// GoSettings shows the settings, the fix for a device setting that does
	// not fit the machine.
	GoSettings()
	// OfferConvert and OfferReconvert show the converter with path chosen:
	// the fix for a GGUF handed to the loader and for an outdated container.
	OfferConvert(path string)
	OfferReconvert(path string)
	// SetTurn replaces transcript turn i; an index the transcript does not
	// have is ignored.
	SetTurn(i int, t session.Turn)
	// SetThinkOpen opens or folds turn i's reasoning.
	SetThinkOpen(i int, open bool)
}

// State is the shared state the engine reads and writes. The front end owns
// every value; the engine reads the settings (DeviceSpec, MaxSeq, KVCache,
// Loaded, ShowThinking) and publishes everything else.
type State struct {
	Status  session.Value[string]
	Problem session.Value[session.Problem]

	ModelPath    session.Value[string]
	Models       session.Value[[]session.LoadedModel]
	Active       session.Value[string]
	ModelSummary session.Value[string]
	Loaded       session.Value[bool]
	Busy         session.Value[bool]
	Streaming    session.Value[bool]
	// Loads is the list [session.LoadingOf] and its siblings work on.
	Loads session.Value[[]session.Loading]

	// Revision bumps when a turn's content changes without the count moving.
	Revision     session.Value[int]
	ShowThinking session.Value[bool]
	Stream       session.Value[string]
	StreamThink  session.Value[string]
	Vision       session.Value[bool]
	Chat         session.Value[bool]
	ChatCapable  session.Value[bool]

	PromptTokS  session.Value[float64]
	DecodeTokS  session.Value[float64]
	BytesPerTok session.Value[float64]
	GBs         session.Value[float64]

	DeviceSpec   session.Value[string]
	Placement    session.Value[string]
	MaxSeq       session.Value[int]
	DeviceReport session.Value[string]
	// BlockMap is one session.Placement per block.
	BlockMap session.Value[[]byte]
	Pager    session.Value[session.PagerStat]
	Alloc    session.Value[session.Allocation]

	// KVCache is whether prompts are cached on disk between runs.
	KVCache session.Value[bool]
}

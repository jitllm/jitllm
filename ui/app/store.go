package app

import (
	"sync"

	"github.com/gogpu/ui/state"

	"github.com/jitllm/jitllm/common/catalog"
	"github.com/jitllm/jitllm/common/crash"
	core "github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/common/session"
)

// Store is every piece of state the screens share.
//
// Every field is a signal, and that is the whole concurrency contract:
// state.Signal is safe to write from any goroutine, so the engine worker
// writes signals and widgets read them on the UI goroutine. Calling a widget
// mutator (SetValue, SetText, SetItems) from the worker is a data race.
//
// The one non-signal is the transcript, a slice behind an RWMutex with Turns
// as the count signal that drives redraw.
type Store struct {
	// --- navigation and chrome ---
	Tab    state.Signal[int]
	Status state.Signal[string]
	Dialog state.Signal[DialogReq]
	// Problem is the session's error banner; see [Problem].
	Problem state.Signal[Problem]
	// Crash is a crash report to show over every screen; the zero value
	// shows nothing. See crash.go.
	Crash state.Signal[crash.Report]

	// --- the loaded model ---
	ModelPath state.Signal[string]
	// Models is every open model and Active the path of the one running.
	// Priority is the toggle that gives the running one the memory.
	Models       state.Signal[[]LoadedModel]
	Active       state.Signal[string]
	Priority     state.Signal[bool]
	ModelSummary state.Signal[string]
	Loaded       state.Signal[bool]
	Busy         state.Signal[bool]
	Streaming    state.Signal[bool]

	// --- the conversation ---
	// Turns is the transcript length and is what listview.ItemCountSignal
	// binds to. Read the turns themselves with [Store.Turn].
	Turns state.Signal[int]
	// Revision bumps when a turn's content changes without the count moving.
	// The transcript watches it and calls listview.Widget.InvalidateData.
	Revision state.Signal[int]
	// ShowThinking is the default a turn's reasoning opens at, and the
	// "Reasoning" chip writes it. A turn that has been clicked follows its
	// own signal from then on; see [Store.ThinkOpen].
	ShowThinking state.Signal[bool]
	// Stream is the in-flight assistant bubble. The decode loop coalesces
	// writes to <= 30 Hz; see the engine's flush rule.
	Stream state.Signal[string]
	// StreamThink is the in-flight reasoning, which grows while Stream is
	// still empty: a reasoning model has no answer yet at that point, and
	// showing the monologue in the answer's place says it does.
	StreamThink state.Signal[string]
	Draft       state.Signal[string]
	// Attach is the pictures waiting to go with the next message, as paths.
	Attach state.Signal[[]string]
	// Vision is true while the loaded model carries a vision tower. Without
	// one there is nothing to send a picture to, and the attach control says
	// so by being disabled rather than by failing at Send.
	Vision state.Signal[bool]
	// Chat is true for templated chat turns, false for raw completion.
	// Completion is the default because every board row and golden is a
	// completion of a bare prompt.
	Chat state.Signal[bool]
	// ChatCapable is false for a base model, which has no chat template. The
	// control is disabled rather than allowed to fail once per turn.
	ChatCapable state.Signal[bool]
	System      state.Signal[string]

	// --- telemetry ---
	PromptTokS  state.Signal[float64]
	DecodeTokS  state.Signal[float64]
	BytesPerTok state.Signal[float64]
	GBs         state.Signal[float64]
	MemWall     state.Signal[float64]

	// --- long-running work ---
	Progress      state.Signal[float64]
	ProgressLabel state.Signal[string]

	// --- placement ---
	DeviceSpec state.Signal[string]
	// Placement is the -placement map for the next load.
	Placement state.Signal[string]
	// Sampling is how the next reply is drawn from the logits, and MaxSeq the
	// context window the next load allocates.
	Sampling      state.Signal[Sampling]
	MaxSeq        state.Signal[int]
	DeviceReport  state.Signal[string]
	DeclineReport state.Signal[string]
	// BlockMap is one byte per block; see widgets.Placement.
	BlockMap state.Signal[[]byte]
	Pager    state.Signal[PagerStat]
	// Alloc is where the loaded model IS: bytes per device against that
	// device's allowance, and bytes on the host against the page budget.
	Alloc state.Signal[Allocation]
	// SeamTarget is where the relocation control is set, as a fraction of
	// the model's blocks: the slider's maximum is fixed at construction and
	// the model changes underneath it.
	SeamTarget state.Signal[float32]

	// --- catalog ---
	// Catalog is the Models tab's list: containers, the files the app loads.
	Catalog     state.Signal[[]catalog.Entry]
	CatalogRows state.Signal[int]
	CatalogSel  state.Signal[int]
	// Sources is the Convert tab's list: the GGUF and safetensors files the
	// same scan found, which are converter input and never loaded.
	Sources     state.Signal[[]catalog.Entry]
	SourcesRows state.Signal[int]
	SourcesSel  state.Signal[int]

	// --- machine ---
	Machine state.Signal[MachineReport]

	// --- chats ---
	// Chats is the conversation list, newest first, and ChatSel the one on
	// screen; its turns are the transcript above. See chats.go.
	Chats   state.Signal[[]ChatInfo]
	ChatSel state.Signal[int]
	// Shown moves every time a chat is put on screen, so a transcript built
	// for the last one rebuilds even at the same turn count.
	Shown state.Signal[int]

	// --- layout ---
	// Loads is every model being opened or switched to, in the order asked.
	// The engine works through them one at a time, first to last.
	Loads state.Signal[[]Loading]
	// ShowTuning is whether the sampling panel is open beside the chat.
	ShowTuning state.Signal[bool]

	// --- settings ---
	// KVCache is whether prompts are cached on disk between runs.
	KVCache state.Signal[bool]
	// API is whether the app serves jitllm's API, on APIAddr (ServeAPI).
	API     state.Signal[bool]
	APIAddr state.Signal[string]
	// APIServing is the address the API listens on now, empty while it is
	// off or was refused: the setting is a wish, this is what happened.
	APIServing state.Signal[string]
	// Engine is which engine the window drives: its own or a jitllmd.
	Engine state.Signal[core.Where]
	// EngineMode and ServiceAddr are the engine setting (core.Mode*): what
	// Settings applies, not what runs (Engine says that).
	EngineMode  state.Signal[string]
	ServiceAddr state.Signal[string]
	// ServiceKey is the API key a jitllmd at ServiceAddr wants, APIKey the one
	// the app's own API requires; blank is none. ShowKeys unmasks both fields
	// and is not persisted.
	ServiceKey state.Signal[string]
	APIKey     state.Signal[string]
	ShowKeys   state.Signal[bool]
	// ModelDirs is the folders models are found in; the first is where
	// downloads go. Write it through Shell.SetModelDirs, which keeps Cfg too.
	ModelDirs state.Signal[[]string]

	mu    sync.RWMutex
	turns []Turn
	// chats is every conversation, its current one the turns above.
	chats session.ChatList
	// thinkOpen is one expansion signal per turn index, created on demand.
	// It lives here because widgets.Column rebuilds every row when the turn
	// count moves; keyed by index, the state outlives the widget.
	thinkOpen map[int]state.Signal[bool]
}

// NewStore builds the shared state with its defaults.
func NewStore() *Store {
	return &Store{
		Tab:     state.NewSignal(0),
		Status:  state.NewSignal("ready"),
		Dialog:  state.NewSignal(DialogReq{}),
		Problem: state.NewSignal(Problem{}),
		Crash:   state.NewSignal(crash.Report{}),

		ModelPath:    state.NewSignal(""),
		Models:       state.NewSignal([]LoadedModel(nil)),
		Active:       state.NewSignal(""),
		Priority:     state.NewSignal(false),
		ModelSummary: state.NewSignal("no model loaded"),
		Loaded:       state.NewSignal(false),
		Busy:         state.NewSignal(false),
		Streaming:    state.NewSignal(false),

		Turns:       state.NewSignal(0),
		Stream:      state.NewSignal(""),
		StreamThink: state.NewSignal(""),
		Draft:       state.NewSignal(""),
		Attach:      state.NewSignal([]string(nil)),
		Vision:      state.NewSignal(false),
		Chat:        state.NewSignal(false),
		System:      state.NewSignal(""),

		PromptTokS:  state.NewSignal(0.0),
		DecodeTokS:  state.NewSignal(0.0),
		BytesPerTok: state.NewSignal(0.0),
		GBs:         state.NewSignal(0.0),
		MemWall:     state.NewSignal(0.0),

		Progress:      state.NewSignal(0.0),
		ProgressLabel: state.NewSignal(""),

		DeviceSpec:    state.NewSignal("auto"),
		Placement:     state.NewSignal(""),
		Sampling:      state.NewSignal(DefaultConfig().Sampling),
		MaxSeq:        state.NewSignal(DefaultConfig().MaxSeq),
		DeviceReport:  state.NewSignal(""),
		DeclineReport: state.NewSignal(""),
		BlockMap:      state.NewSignal([]byte(nil)),
		ChatCapable:   state.NewSignal(true),
		Revision:      state.NewSignal(0),
		ShowThinking:  state.NewSignal(false),
		Pager:         state.NewSignal(PagerStat{}),
		Alloc:         state.NewSignal(Allocation{}),
		SeamTarget:    state.NewSignal(float32(0)),

		Catalog:     state.NewSignal([]catalog.Entry(nil)),
		CatalogRows: state.NewSignal(0),
		CatalogSel:  state.NewSignal(-1),
		Sources:     state.NewSignal([]catalog.Entry(nil)),
		SourcesRows: state.NewSignal(0),
		SourcesSel:  state.NewSignal(-1),

		Machine: state.NewSignal(MachineReport{}),

		ShowTuning:  state.NewSignal(true),
		Loads:       state.NewSignal([]Loading(nil)),
		KVCache:     state.NewSignal(true),
		API:         state.NewSignal(false),
		APIAddr:     state.NewSignal(""),
		APIServing:  state.NewSignal(""),
		Engine:      state.NewSignal(core.Where{}),
		EngineMode:  state.NewSignal(""),
		ServiceAddr: state.NewSignal(""),
		ServiceKey:  state.NewSignal(""),
		APIKey:      state.NewSignal(""),
		ShowKeys:    state.NewSignal(false),
		ModelDirs:   state.NewSignal([]string(nil)),

		Chats:   state.NewSignal([]ChatInfo{{}}),
		ChatSel: state.NewSignal(0),
		Shown:   state.NewSignal(0),
		chats:   session.ChatList{Chats: []Chat{{}}},
	}
}

// AppendTurn adds a turn and publishes the new count. Safe from any goroutine.
func (s *Store) AppendTurn(t Turn) int {
	s.mu.Lock()
	s.turns = append(s.turns, t)
	n := len(s.turns)
	named := t.Role == RoleUser && session.ChatTitle(s.turns[:n-1]) == ""
	s.mu.Unlock()
	s.Turns.Set(n)
	if named {
		// The first message names the chat in the list.
		s.publishChats()
	}
	return n - 1
}

// SetTurn replaces one turn in place -- how a streamed assistant bubble is
// finalised once its text is complete. It does NOT publish a new count,
// because the count did not change; call [Store.Touch] if a repaint is needed.
//
// A reply goes to the chat that asked for it: while another chat is on screen,
// the turn is written into that chat where it is parked.
func (s *Store) SetTurn(i int, t Turn) {
	s.mu.Lock()
	if s.chats.SetWritingTurn(i, t) {
		s.mu.Unlock()
		s.publishChats()
		return
	}
	if i >= 0 && i < len(s.turns) {
		s.turns[i] = t
	}
	s.mu.Unlock()
}

// Turn reads one turn. It returns the zero Turn for an index the transcript
// does not have, which is what a listview building a row mid-append sees.
func (s *Store) Turn(i int) Turn {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if i < 0 || i >= len(s.turns) {
		return Turn{}
	}
	return s.turns[i]
}

// AllTurns copies the transcript. Use it for "copy the conversation", never in
// a per-row build function.
func (s *Store) AllTurns() []Turn {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Turn, len(s.turns))
	copy(out, s.turns)
	return out
}

// TruncateTurns drops every turn from n on, for a reply generated again from
// the message before it. Safe from any goroutine.
func (s *Store) TruncateTurns(n int) {
	s.mu.Lock()
	if n < 0 {
		n = 0
	}
	if n < len(s.turns) {
		s.turns = s.turns[:n]
	}
	k := len(s.turns)
	s.mu.Unlock()
	s.Turns.Set(k)
}

// ClearTurns empties the transcript.
func (s *Store) ClearTurns() {
	s.mu.Lock()
	s.turns = nil
	s.thinkOpen = nil
	s.mu.Unlock()
	s.Turns.Set(0)
	s.Stream.Set("")
	s.StreamThink.Set("")
}

// ClearConversation is the session's Clear: the transcript and everything that
// only made sense beside it -- the unsent draft, pictures waiting to go, the
// last reply's rates and the status line reporting it. The pager line stays:
// it counts the loaded model's pager since load, which a Clear does not touch.
func (s *Store) ClearConversation() {
	s.ClearTurns()
	s.Draft.Set("")
	s.Attach.Set(nil)
	s.PromptTokS.Set(0)
	s.DecodeTokS.Set(0)
	s.BytesPerTok.Set(0)
	s.GBs.Set(0)
	s.Status.Set("conversation cleared")
}

// Touch marks the transcript dirty: the turns are the same in number but one
// of them has new content. It bumps Revision rather than flapping Turns, which
// would fire every Turns watcher twice and rebuild every row.
func (s *Store) Touch() { s.Revision.Set(s.Revision.Get() + 1) }

// ThinkOpen is turn i's reasoning-disclosure state, created on first ask at
// whatever ShowThinking currently says.
//
// It must be a writable signal: collapsible.ExpandedReadonlySignal is a one-way
// bind, so a disclosure bound that way renders but cannot be toggled.
// ExpandedSignal is the two-way form.
func (s *Store) ThinkOpen(i int) state.Signal[bool] {
	s.mu.Lock()
	defer s.mu.Unlock()
	if sig, ok := s.thinkOpen[i]; ok {
		return sig
	}
	if s.thinkOpen == nil {
		s.thinkOpen = map[int]state.Signal[bool]{}
	}
	sig := state.NewSignal(s.ShowThinking.Get())
	s.thinkOpen[i] = sig
	return sig
}

// SetReplyThinkOpen opens or folds the reasoning of the reply in flight, turn
// i of the chat that asked for it; nothing while another chat is on screen.
func (s *Store) SetReplyThinkOpen(i int, open bool) {
	if s.ReplyShown() {
		s.ThinkOpen(i).Set(open)
	}
}

// MarkReply records the chat on screen as the one the next reply is written
// into. Call it as the reply's turn is appended.
func (s *Store) MarkReply() {
	s.mu.Lock()
	s.chats.MarkWriting()
	s.mu.Unlock()
}

// ReplyShown says whether the stream belongs to the chat on screen: no reply
// has been written into another chat. Only then does the transcript's live
// bubble read Stream.
func (s *Store) ReplyShown() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, writing := s.chats.Writing()
	return !writing || s.chats.WritingShown()
}

// SetAllThinkOpen is the "Reasoning" chip: it moves the default and every
// turn that already has a state of its own.
func (s *Store) SetAllThinkOpen(open bool) {
	s.ShowThinking.Set(open)
	s.mu.RLock()
	sigs := make([]state.Signal[bool], 0, len(s.thinkOpen))
	for _, sig := range s.thinkOpen {
		sigs = append(sigs, sig)
	}
	s.mu.RUnlock()
	// Outside the lock: Set notifies subscribers synchronously, and a
	// subscriber may read the store.
	for _, sig := range sigs {
		sig.Set(open)
	}
}

// SetCatalog publishes a catalog and its row count together, so a datatable
// bound to the count can never read past the slice.
func (s *Store) SetCatalog(entries []catalog.Entry) {
	s.Catalog.Set(entries)
	s.CatalogRows.Set(len(entries))
	if s.CatalogSel.Get() >= len(entries) {
		s.CatalogSel.Set(-1)
	}
}

// SetSources is SetCatalog for the Convert tab's list.
func (s *Store) SetSources(entries []catalog.Entry) {
	s.Sources.Set(entries)
	s.SourcesRows.Set(len(entries))
	if s.SourcesSel.Get() >= len(entries) {
		s.SourcesSel.Set(-1)
	}
}

// SelectedSource returns the selected Convert row, or false when nothing is
// selected.
func (s *Store) SelectedSource() (catalog.Entry, bool) {
	i := s.SourcesSel.Get()
	all := s.Sources.Get()
	if i < 0 || i >= len(all) {
		return catalog.Entry{}, false
	}
	return all[i], true
}

// SelectedEntry returns the selected catalog row, or false when nothing is
// selected.
func (s *Store) SelectedEntry() (catalog.Entry, bool) {
	i := s.CatalogSel.Get()
	all := s.Catalog.Get()
	if i < 0 || i >= len(all) {
		return catalog.Entry{}, false
	}
	return all[i], true
}

// LoadingOf is the load of path in progress or waiting, if there is one.
func (s *Store) LoadingOf(path string) (Loading, bool) { return session.LoadingOf(s.Loads, path) }

// Loading reports whether any model is being opened or switched to.
func (s *Store) Loading() bool { return len(s.Loads.Get()) > 0 }

// QueueLoad adds a load to the end of the list. UI goroutine only.
func (s *Store) QueueLoad(l Loading) { session.QueueLoad(s.Loads, l) }

// SetLoadStage says what the load of path is doing now. UI goroutine only.
func (s *Store) SetLoadStage(path, stage string) { session.SetLoadStage(s.Loads, path, stage) }

// EndLoad removes the first load of path, done or failed. UI goroutine only.
func (s *Store) EndLoad(path string) { session.EndLoad(s.Loads, path) }

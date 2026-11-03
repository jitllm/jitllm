package app

import (
	"strings"
	"testing"
)

func TestTabOrderMatchesTheNames(t *testing.T) {
	if len(TabNames()) != TabSettings+1 {
		t.Fatalf("TabNames() has %d entries; the constants name %d", len(TabNames()), TabSettings+1)
	}
	for i, want := range map[int]string{TabSession: "Chat", TabModels: "Models", TabDownload: "Discover", TabConvert: "Convert", TabMachine: "Machine", TabSettings: "Settings"} {
		if TabNames()[i] != want {
			t.Errorf("TabNames()[%d] = %q, want %q", i, TabNames()[i], want)
		}
	}
}

func TestStoreTranscript(t *testing.T) {
	s := NewStore()
	if s.Turns.Get() != 0 {
		t.Fatal("a fresh store has no turns")
	}
	i := s.AppendTurn(Turn{Role: RoleUser, Text: "hello"})
	if i != 0 || s.Turns.Get() != 1 {
		t.Fatalf("AppendTurn = %d, Turns = %d; want 0, 1", i, s.Turns.Get())
	}
	j := s.AppendTurn(Turn{Role: RoleAssistant})
	s.SetTurn(j, Turn{Role: RoleAssistant, Text: "hi", Tokens: 2})
	if got := s.Turn(j); got.Text != "hi" || got.Tokens != 2 {
		t.Fatalf("SetTurn did not replace in place: %+v", got)
	}
	if s.Turns.Get() != 2 {
		t.Error("SetTurn must not change the count")
	}
	if got := s.Turn(99); got.Text != "" {
		t.Error("an out-of-range turn must read as the zero Turn, not panic")
	}
	s.ClearTurns()
	if s.Turns.Get() != 0 || s.Stream.Get() != "" {
		t.Error("ClearTurns must empty the transcript and the in-flight bubble")
	}
}

func TestSetCatalogDropsAStaleSelection(t *testing.T) {
	s := NewStore()
	s.CatalogSel.Set(5)
	s.SetCatalog(nil)
	if s.CatalogRows.Get() != 0 {
		t.Error("the row count must follow the catalog")
	}
	if s.CatalogSel.Get() != -1 {
		t.Error("a selection past the end of a new catalog must be dropped")
	}
	if _, ok := s.SelectedEntry(); ok {
		t.Error("SelectedEntry must report no selection")
	}
}

func TestDefaultConfigUsesTheDefaultModelDir(t *testing.T) {
	c := DefaultConfig(Env{})
	if len(c.ModelDirs) != 1 || c.ModelDirs[0] != DefaultConfig(Env{}).DefaultModelDir() {
		t.Fatalf("ModelDirs = %v, want the default model folder", c.ModelDirs)
	}
	if c.AddModelDir(DefaultConfig(Env{}).DefaultModelDir()) {
		t.Error("AddModelDir must not duplicate a directory")
	}
	if !c.AddModelDir("/somewhere/else") {
		t.Error("AddModelDir must add a new directory")
	}
}

// Clear must reset what only made sense beside the transcript, and leave the
// model's own state alone: the draft, attachments, stats and status line.
func TestClearConversationResetsWhatDescribedTheConversation(t *testing.T) {
	s := NewStore()
	s.AppendTurn(Turn{Role: RoleUser, Text: "my name is Zorblax"})
	s.Stream.Set("partial")
	s.Draft.Set("unsent")
	s.Attach.Set([]string{"/pic.png"})
	s.PromptTokS.Set(61.07)
	s.DecodeTokS.Set(37.28)
	s.BytesPerTok.Set(816e6)
	s.GBs.Set(30.4)
	s.Status.Set("17 token(s) at 37.28 tok/s")
	pager := PagerStat{Frames: 16, PageIns: 30, BytesRead: 598 << 20}
	s.Pager.Set(pager)

	s.ClearConversation()

	if s.Turns.Get() != 0 || s.Stream.Get() != "" {
		t.Error("the transcript or the live bubble survived")
	}
	if s.Draft.Get() != "" {
		t.Errorf("the draft survived: %q", s.Draft.Get())
	}
	if n := len(s.Attach.Get()); n != 0 {
		t.Errorf("%d attached picture(s) survived and would go with the next message", n)
	}
	if s.PromptTokS.Get() != 0 || s.DecodeTokS.Get() != 0 || s.BytesPerTok.Get() != 0 || s.GBs.Get() != 0 {
		t.Error("the rate line still describes the cleared reply")
	}
	if strings.Contains(s.Status.Get(), "tok/s") {
		t.Errorf("the status line still reports the cleared reply: %q", s.Status.Get())
	}
	if s.Pager.Get() != pager {
		t.Error("the pager readout was reset, but it describes the loaded model, which Clear does not touch")
	}
}

// A dialog request must reach the Store on the UI goroutine, whoever raised it:
// widgets.DialogHost reacts on the setter's goroutine. Fails against Alert
// setting the signal directly, where the request is visible before the drain.
func TestAlertIsPublishedFromTheUIGoroutine(t *testing.T) {
	sh := NewShell(nil, nil)
	done := make(chan struct{})
	go func() { sh.Alert("Convert failed", "no such file"); close(done) }()
	<-done
	if sh.Store.Dialog.Get().Seq != 0 {
		t.Fatal("the request was published on the worker, before the UI queue ran")
	}
	sh.DrainQueue(0)
	if d := sh.Store.Dialog.Get(); d.Seq == 0 || d.Title != "Convert failed" {
		t.Fatalf("draining the UI queue did not publish the request: %+v", d)
	}
}

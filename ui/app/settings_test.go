package app

import "testing"

// Every setting a person adjusts by hand must survive a restart. The gate is
// the round trip, not the field: it drives NewShell (Config -> Store) and
// Shell.Close (Store -> Config), which a tagged-but-unwired field fails.
func TestEverySettingSurvivesARestart(t *testing.T) {
	cfg := DefaultConfig(Env{})
	sh := NewShell(nil, cfg)

	// What a session of adjusting things leaves behind.
	sh.Store.ShowTuning.Set(false)
	sh.Store.Chat.Set(false)
	sh.Store.System.Set("be terse")
	sh.Store.MaxSeq.Set(8192)
	s := sh.Store.Sampling.Get()
	s.Temp, s.MinP = 0.31, 0.07
	sh.Store.Sampling.Set(s)

	sh.Close()

	// The next launch.
	next := NewShell(nil, cfg)
	if next.Store.ShowTuning.Get() {
		t.Error("the sampling panel came back open after it was closed")
	}
	if next.Store.Chat.Get() {
		t.Error("Chat came back on")
	}
	if got := next.Store.System.Get(); got != "be terse" {
		t.Errorf("System came back %q", got)
	}
	if got := next.Store.MaxSeq.Get(); got != 8192 {
		t.Errorf("MaxSeq came back %d", got)
	}
	if got := next.Store.Sampling.Get(); got.Temp != 0.31 || got.MinP != 0.07 {
		t.Errorf("Sampling came back %+v", got)
	}
}

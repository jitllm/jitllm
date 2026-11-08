package scores

import (
	"math"
	"path/filepath"
	"testing"
)

func TestBoardRanksAndSummarises(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scores.musq")
	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	runs := []Run{
		{Mode: ModeAI, Who: "laya", Score: 100, AvgMs: 10},
		{Mode: ModeAI, Who: "laya", Score: 300, AvgMs: 30},
		{Mode: ModeAI, Who: "laya", Score: 200, AvgMs: 20},
		{Mode: ModeAI, Who: "lev", Score: 900, AvgMs: 400},
		{Mode: ModeHuman, Who: "human", Score: 500},
		{Mode: ModeGhosts, Who: "lev", Score: 50},
	}
	for _, r := range runs {
		if err := s.Add(r); err != nil {
			t.Fatal(err)
		}
	}
	// Reopened: the board is what is on disk, not what is in memory.
	s.Close()
	if s, err = Open(path); err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	b, err := s.Board()
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 4 {
		t.Fatalf("%d rows, want 4: %+v", len(b), b)
	}
	order := []string{"lev", "You", "laya", "You vs lev ghosts"}
	for i, w := range order {
		if b[i].Label() != w {
			t.Fatalf("row %d is %q, want %q (%+v)", i, b[i].Label(), w, b)
		}
	}
	laya := b[2]
	// 100, 200, 300: mean 200, sample sd 100, SE 100/sqrt(3).
	if laya.Runs != 3 || laya.Best != 300 || laya.Mean != 200 || math.Abs(laya.SE-100/math.Sqrt(3)) > 1e-9 || laya.AvgMs != 20 {
		t.Fatalf("laya's row: %+v", laya)
	}
	if b[0].SE != 0 || b[0].Runs != 1 {
		t.Fatalf("a single run has a standard error: %+v", b[0])
	}
	hs, err := s.HighScore()
	if err != nil || hs != 900 {
		t.Fatalf("high score %d, %v", hs, err)
	}
}

func TestAnEmptyBoard(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "x", "scores.musq"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	b, err := s.Board()
	if err != nil || len(b) != 0 {
		t.Fatalf("empty board: %+v, %v", b, err)
	}
	if hs, err := s.HighScore(); err != nil || hs != 0 {
		t.Fatalf("empty high score %d, %v", hs, err)
	}
}

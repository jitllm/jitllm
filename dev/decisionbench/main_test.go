package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The harness counts what it says it counts: a scripted server answers each
// word from a fixed table, and the row the harness prints from
// testdata/centipedes.tsv is held to the counts worked out by hand. Speed is
// not checked; it is the main session's to measure.

// scripted answers P(yes) per word, and a choice of "centipede" when P >= 0.5.
func scripted(t *testing.T, pyes map[string]float64) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/systemone" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var req struct {
			State     string                     `json:"state"`
			Questions map[string]json.RawMessage `json:"questions"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("bad request: %v", err)
			return
		}
		p, ok := pyes[req.State]
		if !ok {
			t.Errorf("the harness asked about %q, which the file does not hold", req.State)
		}
		var q struct {
			Type string `json:"type"`
		}
		json.Unmarshal(req.Questions["q"], &q)
		if q.Type == "noul" {
			fmt.Fprintf(w, `{"model":"m","answers":{"q":{"type":"noul","noul":%g}},"usage":{"input_tokens":3,"output_tokens":0}}`, p)
			return
		}
		choice := "other"
		if p >= 0.5 {
			choice = "centipede"
		}
		fmt.Fprintf(w, `{"model":"m","answers":{"q":{"type":"choice","choice":%q,"confidence":1,"probabilities":{}}},"usage":{"input_tokens":3,"output_tokens":0}}`, choice)
	}))
}

// The table: four centipedes (two caught), four other words (one wrongly picked).
func pyes() map[string]float64 {
	return map[string]float64{
		"Scolopendra": 0.9, "Lithobius": 0.7, "Geophilus": 0.2, "Scutigera": 0.4,
		"teapot": 0.1, "Lumbricus": 0.6, "granite": 0.0, "Julus": 0.3,
	}
}

func items(t *testing.T) []item {
	f, err := os.Open(filepath.Join("testdata", "centipedes.tsv"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	its, err := readItems(f)
	if err != nil {
		t.Fatal(err)
	}
	if len(its) != len(pyes()) {
		t.Fatalf("read %d words, the table has %d", len(its), len(pyes()))
	}
	return its
}

func TestHarnessCountsANoul(t *testing.T) {
	srv := scripted(t, pyes())
	defer srv.Close()
	c, err := newConfig(srv.URL, "m", `{"type": "noul", "instructions": "Is this the name of a centipede?"}`, "", 0.5, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	r, err := run(c, items(t), srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	// Right: Scolopendra, Lithobius (caught), teapot, granite, Julus
	// (rejected); wrong: Geophilus, Scutigera (missed), Lumbricus (picked).
	want := result{Answered: 8, Scored: 8, InWindow: 8, Positives: 4, Correct: 5, Caught: 2, WrongPicks: 1, Exhausted: true}
	r.P50 = 0
	if r != want {
		t.Errorf("counted %+v, want %+v", r, want)
	}
	row := format(r, time.Minute)
	for _, s := range []string{"accuracy 0.625 (5/8)", "caught 2/4", "wrong picks 1", "words in 1m0s: 8", "the file ended"} {
		if !strings.Contains(row, s) {
			t.Errorf("the row %q does not say %q", row, s)
		}
	}
}

func TestHarnessCountsAChoice(t *testing.T) {
	srv := scripted(t, pyes())
	defer srv.Close()
	c, err := newConfig(srv.URL, "m", `{"type": "choice", "criteria": {"centipede": null, "other": null}}`, "centipede", 0.5, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	its := items(t)
	for i := range its {
		if its[i].label == "1" {
			its[i].label = "centipede"
		} else {
			its[i].label = "other"
		}
	}
	r, err := run(c, its, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	r.P50 = 0
	want := result{Answered: 8, Scored: 8, InWindow: 8, Positives: 4, Correct: 5, Caught: 2, WrongPicks: 1, Exhausted: true}
	if r != want {
		t.Errorf("counted %+v, want %+v", r, want)
	}
}

// TestHarnessStopsAtTheWindow: a window that has closed answers nothing more.
func TestHarnessStopsAtTheWindow(t *testing.T) {
	srv := scripted(t, pyes())
	defer srv.Close()
	c, err := newConfig(srv.URL, "m", `{"type": "noul"}`, "", 0.5, 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := run(c, items(t), srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	if r.Answered != 0 || r.Exhausted {
		t.Errorf("a closed window answered %d words (exhausted %v)", r.Answered, r.Exhausted)
	}
}

func TestHarnessRefusesABadLabel(t *testing.T) {
	srv := scripted(t, map[string]float64{"x": 1})
	defer srv.Close()
	c, err := newConfig(srv.URL, "m", `{"type": "noul"}`, "", 0.5, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := run(c, []item{{"x", "maybe"}}, srv.Client()); err == nil {
		t.Error("a label that is not a yes/no was counted")
	}
	if _, err := newConfig(srv.URL, "m", `{"type": "choice"}`, "", 0.5, time.Minute); err == nil {
		t.Error("a choice with no positive key was accepted")
	}
}

// A cycling run fills the window from a short file and still counts the file once.
func TestHarnessCyclesAShortFile(t *testing.T) {
	srv := scripted(t, pyes())
	defer srv.Close()
	c, err := newConfig(srv.URL, "m", `{"type": "noul"}`, "", 0.5, 300*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	c.cycle = true
	r, err := run(c, items(t), srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	if r.Answered <= 8 || r.Scored != 8 || r.Correct != 5 || r.Caught != 2 || r.WrongPicks != 1 || r.Exhausted {
		t.Errorf("counted %+v: want more than 8 answered, the file scored once (5 right, 2 caught, 1 wrong)", r)
	}
}

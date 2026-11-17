// Command decisionbench reproduces the user's decision-model table against any
// server speaking TypeSafe's POST /v1/systemone -- jitllm's, `lev serve`
// or any other -- so every engine in the table is measured by one harness
// (docs/design/decision-models.md, section 5).
//
// The workload is a stream of words, one request each: the state is the word
// and the question asks whether it is a positive (a noul, by default "Is this
// the name of a centipede?"), or picks among a choice's keys. Per word it
// records the wall time from request to answer and whether the answer was
// right. It reports:
//
//	per-word p50   the median wall time of one request
//	words in 32 s  requests answered within the window, serially, from the start
//	accuracy       answers equal to their label, over the words answered
//	caught         positives answered positive
//	wrong picks    negatives answered positive
//
// The labelled file is one word per line: "word<TAB>label", the label 1/0,
// yes/no or true/false for a noul, or a choice key. Lines starting with # are
// comments.
//
// It measures; it does not gate. Speed numbers come from the main session,
// serially, under scripts/cap (AGENTS.md RULE 1-3).
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// item is one labelled word.
type item struct {
	word, label string
}

// readItems reads the labelled file.
func readItems(r io.Reader) ([]item, error) {
	var out []item
	sc := bufio.NewScanner(r)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimRight(sc.Text(), "\r")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		word, label, ok := strings.Cut(line, "\t")
		if !ok || word == "" || label == "" {
			return nil, fmt.Errorf("line %d: want word<TAB>label, got %q", n, line)
		}
		out = append(out, item{word: word, label: strings.TrimSpace(label)})
	}
	return out, sc.Err()
}

// positive reads a noul label.
func positive(label string) (bool, error) {
	switch strings.ToLower(label) {
	case "1", "yes", "true":
		return true, nil
	case "0", "no", "false":
		return false, nil
	}
	return false, fmt.Errorf("label %q is not a yes/no", label)
}

// config is one run.
type config struct {
	url, model string
	question   json.RawMessage // one TypeSafe question
	noul       bool
	positive   string  // the choice key that counts as a pick
	threshold  float64 // P(yes) at or above which a noul answers yes
	window     time.Duration
	cycle      bool // keep asking the file again until the window ends
}

// result is a run's table row.
type result struct {
	Answered, Scored, InWindow, Positives, Correct, Caught, WrongPicks int
	P50                                                                time.Duration
	Exhausted                                                          bool // the file ended inside the window
}

// run asks every word in turn, stopping at the end of the window.
func run(c config, items []item, client *http.Client) (result, error) {
	var r result
	var lat []time.Duration
	start := time.Now()
	for i := 0; ; i++ {
		if time.Since(start) >= c.window || (i >= len(items) && !c.cycle) {
			break
		}
		it := items[i%len(items)]
		first := i < len(items) // accuracy, caught and wrong picks count the file once
		body, err := json.Marshal(map[string]any{
			"model":     c.model,
			"state":     it.word,
			"questions": map[string]json.RawMessage{"q": c.question},
		})
		if err != nil {
			return r, err
		}
		t := time.Now()
		resp, err := client.Post(c.url+"/v1/systemone", "application/json", bytes.NewReader(body))
		if err != nil {
			return r, err
		}
		raw, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		took := time.Since(t)
		if err != nil {
			return r, err
		}
		if resp.StatusCode != http.StatusOK {
			return r, fmt.Errorf("%q: status %d: %s", it.word, resp.StatusCode, raw)
		}
		var ans struct {
			Answers map[string]struct {
				Noul   *float64 `json:"noul"`
				Choice string   `json:"choice"`
			} `json:"answers"`
		}
		if err := json.Unmarshal(raw, &ans); err != nil {
			return r, fmt.Errorf("%q: %w", it.word, err)
		}
		a, ok := ans.Answers["q"]
		if !ok {
			return r, fmt.Errorf("%q: the response has no answer: %s", it.word, raw)
		}
		lat = append(lat, took)
		r.Answered++
		if time.Since(start) <= c.window {
			r.InWindow++
		}
		var want, got bool
		if c.noul {
			if a.Noul == nil {
				return r, fmt.Errorf("%q: a noul question came back without noul: %s", it.word, raw)
			}
			if want, err = positive(it.label); err != nil {
				return r, err
			}
			got = *a.Noul >= c.threshold
			if want == got && first {
				r.Correct++
			}
		} else {
			want, got = it.label == c.positive, a.Choice == c.positive
			if a.Choice == it.label && first {
				r.Correct++
			}
		}
		if !first {
			continue
		}
		r.Scored++
		switch {
		case want:
			r.Positives++
			if got {
				r.Caught++
			}
		case got:
			r.WrongPicks++
		}
	}
	r.Exhausted = !c.cycle && r.Answered == len(items) && time.Since(start) < c.window
	if len(lat) > 0 {
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		r.P50 = lat[(len(lat)-1)/2]
	}
	return r, nil
}

func main() {
	url := flag.String("url", "http://localhost:4242", "the server's base URL")
	model := flag.String("model", "", "the model field of each request")
	file := flag.String("file", "", "the labelled words, word<TAB>label per line")
	question := flag.String("question", `{"type": "noul", "instructions": "Is this the name of a centipede?"}`,
		"the TypeSafe question asked of every word, as JSON")
	pos := flag.String("positive", "", "for a choice question, the key that counts as a pick")
	threshold := flag.Float64("threshold", 0.5, "for a noul, the P(yes) that answers yes")
	cycle := flag.Bool("cycle", false, "ask the file again until the window ends, so a short file still fills it; accuracy counts the first pass")
	raw := flag.Bool("raw", false, "print p50 in microseconds, words in the window, scored, correct, caught, positives and wrong picks, tab separated")
	window := flag.Duration("window", 32*time.Second, "the window words are counted in")
	flag.Parse()
	if *file == "" {
		fmt.Fprintln(os.Stderr, "usage: decisionbench -file words.tsv [-url URL] [-model NAME] [-question JSON] [-positive KEY]")
		os.Exit(2)
	}
	if err := mainErr(*url, *model, *file, *question, *pos, *threshold, *window, *cycle, *raw); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func mainErr(url, model, file, question, pos string, threshold float64, window time.Duration, cycle, raw bool) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}
	items, err := readItems(f)
	f.Close()
	if err != nil {
		return err
	}
	c, err := newConfig(url, model, question, pos, threshold, window)
	if err != nil {
		return err
	}
	c.cycle = cycle
	r, err := run(c, items, &http.Client{Timeout: 5 * time.Minute})
	if err != nil {
		return err
	}
	if raw {
		fmt.Printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", r.P50.Microseconds(), r.InWindow, r.Scored, r.Correct, r.Caught, r.Positives, r.WrongPicks)
		return nil
	}
	fmt.Println(format(r, window))
	return nil
}

// newConfig checks the question and what counts as a pick.
func newConfig(url, model, question, pos string, threshold float64, window time.Duration) (config, error) {
	var q struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(question), &q); err != nil {
		return config{}, fmt.Errorf("-question: %w", err)
	}
	c := config{url: strings.TrimRight(url, "/"), model: model, question: json.RawMessage(question),
		threshold: threshold, window: window, positive: pos}
	switch q.Type {
	case "noul":
		c.noul = true
	case "choice":
		if pos == "" {
			return config{}, errors.New("a choice question needs -positive, the key that counts as a pick")
		}
	default:
		return config{}, fmt.Errorf("-question is a %q; the table asks a noul or a choice", q.Type)
	}
	return c, nil
}

// format is the table row.
func format(r result, window time.Duration) string {
	acc := 0.0
	if r.Scored > 0 {
		acc = float64(r.Correct) / float64(r.Scored)
	}
	s := fmt.Sprintf("per-word p50 %v | words in %v: %d | accuracy %.3f (%d/%d) | caught %d/%d | wrong picks %d",
		r.P50.Round(time.Microsecond), window, r.InWindow, acc, r.Correct, r.Scored, r.Caught, r.Positives, r.WrongPicks)
	if r.Exhausted {
		s += " | the file ended inside the window: the words count is the file, not the rate"
	}
	return s
}

package screen_test

import (
	"strings"
	"testing"
	"unicode"

	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/screen"
)

// helpLimit is the most characters one piece of text on the Machine page may
// carry: about two lines at the page's 1440 px and 1000 px widths.
const helpLimit = 240

// acronyms() are the all-capital words that are names rather than shouting.
func acronyms() map[string]bool {
	return map[string]bool{
		"CPU": true, "GPU": true, "GGUF": true, "CUDA": true, "RAM": true, "ISA": true, "SMT": true,
	}
}

// notYetPlain() are help lines the page draws that are known not to be plain
// yet, matched by prefix and skipped; an entry that no longer matches anything
// fails.
func notYetPlain() []string {
	return []string{}
}

// The Machine page's help text must be plain and short. It walks what the
// screen builds rather than a list of constants, so a later inline note is
// held to the same rule; text that appears only after a hardware probe is not
// reached.
func TestMachineHelpIsPlainAndShort(t *testing.T) {
	sh := app.NewShell(nil, nil)
	// A loaded model puts the header's reason line on screen, and also keeps
	// StartProbe from reading the real machine inside a test.
	sh.Store.Loaded.Set(true)

	var texts []string
	collectText(screen.Devices(sh), &texts)

	// Selection check: the header line and the last section's content were
	// both reached, so an empty walk cannot pass.
	var header, last bool
	for _, s := range texts {
		header = header || strings.HasPrefix(s, "Re-probe is off")
		last = last || s == "pager"
	}
	if !header || !last {
		t.Fatalf("the walk missed part of the page (header line %v, Placement section %v); it saw:\n  %s",
			header, last, strings.Join(texts, "\n  "))
	}

	used := make([]bool, len(notYetPlain()))
	for _, s := range texts {
		if i := exempt(s); i >= 0 {
			used[i] = true
			continue
		}
		if n := len([]rune(s)); n > helpLimit {
			t.Errorf("%d characters, over the %d a glance can take:\n  %q", n, helpLimit, s)
		}
		if w := shouted(s); len(w) > 0 {
			t.Errorf("shouts %v -- say it in sentence case:\n  %q", w, s)
		}
	}
	for i, ok := range used {
		if !ok {
			t.Errorf("exemption %q matches nothing on the page any more; delete it", notYetPlain()[i])
		}
	}
}

// collectText gathers every non-empty text a widget tree shows.
func collectText(w widget.Widget, out *[]string) {
	if w == nil {
		return
	}
	if c, ok := w.(interface{ Content() string }); ok {
		if s := strings.TrimSpace(c.Content()); s != "" {
			*out = append(*out, s)
		}
	}
	for _, ch := range w.Children() {
		collectText(ch, out)
	}
}

func exempt(s string) int {
	for i, p := range notYetPlain() {
		if strings.HasPrefix(s, p) {
			return i
		}
	}
	return -1
}

// shouted is every word of s written in capitals for emphasis: two or more
// letters, all of them upper case, and not an acronym.
func shouted(s string) []string {
	var out []string
	words := strings.FieldsFunc(s, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
	for _, w := range words {
		letters, upper := 0, 0
		for _, r := range w {
			if unicode.IsLetter(r) {
				letters++
				if unicode.IsUpper(r) {
					upper++
				}
			}
		}
		if letters >= 2 && upper == letters && !acronyms()[w] {
			out = append(out, w)
		}
	}
	return out
}

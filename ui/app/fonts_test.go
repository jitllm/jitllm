package app

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gogpu/gg/text"
)

const (
	latinA   = 'A'
	arabicAl = 'ا' // ARABIC LETTER ALEF
	cjkHan   = '中' // CJK UNIFIED IDEOGRAPH-4E2D
)

// The chain must answer per rune, with a family that can actually draw it.
// The gate asserts coverage, not identity, so it does not bake the host's font
// set into the test.
func TestTheChainAnswersWithAFamilyThatCoversTheRune(t *testing.T) {
	var fonts Fonts
	if len(fonts.Load()) == 0 {
		t.Skip("no candidate font on this box -- this gate proved nothing")
	}
	t.Logf("chain: %v", fonts.chain)

	for _, r := range []rune{latinA, arabicAl, cjkHan} {
		fam := fonts.FamilyFor(r, "")
		if fam == "" {
			t.Logf("U+%04X: nothing in the chain covers it (the embedded default draws it)", r)
			continue
		}
		if !fonts.FamilyCovers(fam, r) {
			t.Errorf("FamilyFor(U+%04X) named %q, which has no glyph for it: that is "+
				"the .notdef box, chosen deliberately", r, fam)
		}
	}
}

// A rune the current run can already draw must stay in that run, or every
// space would split a non-head-family sentence into per-word runs.
func TestAFamilyKeepsARuneItCanAlreadyDraw(t *testing.T) {
	var fonts Fonts
	if len(fonts.Load()) < 2 {
		t.Skip("need at least two families on this box to tell a chain from a pick")
	}
	other := ""
	for _, f := range fonts.chain {
		if fonts.FamilyCovers(f, ' ') {
			other = f
			break
		}
	}
	if other == "" {
		t.Skip("no second family covering a space")
	}
	if got := fonts.FamilyFor(' ', other); got != other {
		t.Errorf("a space inside a %q run was handed to %q: the run breaks at every space", other, got)
	}
	if got := fonts.FamilyFor(' ', ""); got != "" {
		t.Errorf("with no run in progress a space went to %q, want Inter (the empty default)", got)
	}
}

// Prose is Inter, and data is JetBrains Mono, whatever leads the fallback
// chain: the chain is for runes Inter cannot draw.
func TestProseIsInterAndDataIsMono(t *testing.T) {
	var fonts Fonts
	fonts.Load()
	if got := fonts.FamilyFor(latinA, ""); got != "" {
		t.Errorf("Latin prose went to %q, want Inter", got)
	}
	if !fonts.FamilyCovers(MonoFamily, latinA) {
		t.Skipf("%s is not on this box -- the mono half proved nothing", MonoFamily)
	}
	if got := fonts.MonoFor(latinA, ""); got != MonoFamily {
		t.Errorf("data went to %q, want %s", got, MonoFamily)
	}
	// A rune Inter lacks still reaches a family that has it.
	if interCovers(cjkHan) {
		t.Fatal("setup: Inter covers Han, pick a rune it lacks")
	}
	if fam := fonts.FamilyFor(cjkHan, ""); fam != "" && !fonts.FamilyCovers(fam, cjkHan) {
		t.Errorf("Han went to %q, which cannot draw it", fam)
	}
}

// interCover is generated from the toolkit's Inter; a toolkit upgrade that
// changes the font must regenerate it (go run ./app/internal/gencover).
func TestInterCoverMatchesTheEmbeddedFont(t *testing.T) {
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/gogpu/ui").Output()
	if err != nil {
		t.Skipf("cannot locate gogpu/ui: %v -- this gate proved nothing", err)
	}
	data, err := os.ReadFile(filepath.Join(strings.TrimSpace(string(out)), "internal", "render", "fonts", "Inter-Regular.ttf"))
	if err != nil {
		t.Fatal(err)
	}
	src, err := text.NewFontSource(data)
	if err != nil {
		t.Fatal(err)
	}
	p := src.Parsed()
	for r := rune(0x20); r <= 0x2FFFF; r++ {
		if want := p.GlyphIndex(r) != 0; interCovers(r) != want {
			t.Fatalf("U+%04X: table says %v, Inter says %v -- run go run ./app/internal/gencover > app/intercover.go",
				r, !want, want)
		}
	}
}

// An unregistered family covers nothing, and a chain that failed to load leaves
// every rune on the embedded default rather than on a family that draws boxes.
func TestAnUnknownFamilyCoversNothing(t *testing.T) {
	var fonts Fonts
	if fonts.FamilyCovers("No Such Family", latinA) {
		t.Error("an unregistered family reported coverage")
	}
	fonts = Fonts{}
	if got := fonts.FamilyFor(latinA, ""); got != "" {
		t.Errorf("with no chain loaded FamilyFor returned %q, want the empty default", got)
	}
}

// A font file that is not a font must not reach the registry: a family that
// registers and then fails to parse draws nothing at all.
func TestAnUnparseableFileIsNotRegistered(t *testing.T) {
	var fonts Fonts
	f, err := os.CreateTemp(t.TempDir(), "notafont*.ttf")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("this is not a font"); err != nil {
		t.Fatal(err)
	}
	f.Close()

	fonts.candidates = []fontCandidate{{"Bogus", f.Name()}}
	if got := fonts.Load(); len(got) != 0 {
		t.Errorf("an unparseable file was registered as %v", got)
	}
}

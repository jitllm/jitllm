package app

import (
	"os"
	"path/filepath"

	"github.com/gogpu/gg/text"
	"github.com/gogpu/ui/plugin"

	"github.com/jitllm/jitllm/ui/widgets"
)

// Fonts is the app's font chain, in preference order: the first family that
// has a glyph for a rune is the one that draws it. It is empty until Load runs,
// and an empty chain means the embedded default -- so a box with no fonts
// renders exactly as before rather than not at all.
type Fonts struct {
	chain []string
	// parsed holds each registered family's parsed font, for asking whether it
	// has a glyph.
	parsed map[string]text.ParsedFont
	// candidates replaces the default list; a test sets it.
	candidates []fontCandidate
}

// Chain is the families Load registered, in order.
func (f *Fonts) Chain() []string { return f.chain }

// Prose is the face for running text and labels: Inter, then the chain. Its
// advance is Inter's measured mean with 6% to spare, so a line heavy in wide
// letters still fits the width it was broken for.
func (f *Fonts) Prose() widgets.Face {
	return widgets.Face{For: f.FamilyFor, Advance: interAdvance * 1.06, BoldAdvance: interBoldAdvance * 1.06}
}

// Mono is the face for data: JetBrains Mono's advance is exactly 0.6 em.
func (f *Fonts) Mono() widgets.Face { return widgets.Face{For: f.MonoFor, Advance: 0.6} }

// MonoFamily is the face for data: rates, sizes, paths and code, where
// aligned digits help. Prose is set in the toolkit's embedded Inter.
const MonoFamily = "JetBrainsMono Nerd Font"

// fontCandidates are registered in order, and the order is the fallback chain
// behind Inter.
//
// The chain is not a list of scripts: [FamilyFor] asks each family in turn
// whether it has a glyph for the rune and takes the first that does. Adding
// coverage is adding a file to this list, never code. The Nerd Font comes
// first for its programming glyphs; it adds no script coverage.
func fontCandidates() []fontCandidate {
	return []fontCandidate{
		{MonoFamily, home(".local/share/fonts/JetBrainsMonoNerdFont-Regular.ttf")},
		{"DejaVu Sans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"},
		{"FreeSerif", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"},
		{"Noto Sans CJK", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"},
	}
}

type fontCandidate struct{ family, file string }

func home(rel string) string {
	h, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(h, rel)
}

// Load registers every candidate that is present and parses, and returns
// the resulting chain.
//
// It registers through plugin.NewDefaultPluginContext, the framework's public
// route to the global font registry. It parses with gg/text.NewFontSource, the
// parser render.FontRegistry uses, and skips a font that parser rejects: a
// family that registers but fails to resolve draws nothing at all.
func (f *Fonts) Load() []string {
	ctx := plugin.NewDefaultPluginContext()
	if ctx == nil || ctx.Assets == nil {
		return nil
	}
	f.chain, f.parsed = nil, map[string]text.ParsedFont{}
	candidates := f.candidates
	if candidates == nil {
		candidates = fontCandidates()
	}
	for _, c := range candidates {
		if c.file == "" {
			continue
		}
		data, err := os.ReadFile(c.file)
		if err != nil {
			continue
		}
		src, err := text.NewFontSource(data)
		if err != nil {
			continue
		}
		p := src.Parsed()
		if p == nil {
			continue
		}
		if err := ctx.Assets.LoadFont(c.family, data); err != nil {
			continue
		}
		f.chain = append(f.chain, c.family)
		f.parsed[c.family] = p
	}
	return f.chain
}

// FamilyCovers reports whether family has a glyph for r. A family this app did
// not register answers false: it cannot be asked, and guessing yes is how a
// .notdef box gets drawn. GlyphIndex returning 0 is exactly the box.
func (f *Fonts) FamilyCovers(family string, r rune) bool {
	if f == nil {
		return false
	}
	p, ok := f.parsed[family]
	return ok && p.GlyphIndex(r) != 0
}

// FamilyFor returns the family that should draw r in prose: "" -- the
// toolkit's embedded Inter, which has both weights -- when Inter has the glyph,
// otherwise the first family in the chain that does, otherwise "" again, the
// same thing the widget would have drawn with no chain at all.
//
// prefer is the family of the run the caller is already in: a rune it can
// draw stays in it, so spaces do not split a sentence into per-word runs.
func (f *Fonts) FamilyFor(r rune, prefer string) string {
	if prefer != "" && f.FamilyCovers(prefer, r) {
		return prefer
	}
	if interCovers(r) {
		return ""
	}
	return f.fallback(r)
}

// MonoFor is [FamilyFor] for data: JetBrains Mono when it has the glyph, then
// the same chain.
func (f *Fonts) MonoFor(r rune, prefer string) string {
	if prefer != "" && f.FamilyCovers(prefer, r) {
		return prefer
	}
	if f.FamilyCovers(MonoFamily, r) {
		return MonoFamily
	}
	if interCovers(r) {
		return ""
	}
	return f.fallback(r)
}

func (f *Fonts) fallback(r rune) string {
	if f == nil {
		return ""
	}
	for _, family := range f.chain {
		if f.FamilyCovers(family, r) {
			return family
		}
	}
	return ""
}

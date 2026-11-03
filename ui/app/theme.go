package app

import (
	"image"
	"image/color"

	"github.com/gogpu/ui/theme/material3"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/widgets"
)

// The palettes are the website's (website/public/themes.css): Tokyo Night when
// dark, the default, and Rosé Pine Dawn when light. Each is mapped onto every
// Material 3 role, so the toolkit's painters inherit it with no per-widget
// colour. The seed is only how material3 is constructed; it decides nothing.
const seed = 0x9ECE6A

func newTheme(dark bool) *material3.Theme {
	if dark {
		t := material3.NewDark(widget.Hex(seed))
		t.Colors = tokyoNight()
		t.Shape = shapes()
		return t
	}
	t := material3.New(widget.Hex(seed))
	t.Colors = rosePineDawn()
	t.Shape = shapes()
	return t
}

// shapes() are tighter than Material's: a desktop tool, not a phone.
func shapes() material3.ShapeScale {
	return material3.ShapeScale{None: 0, ExtraSmall: 4, Small: 6, Medium: 8, Large: 12, ExtraLarge: 16, Full: 9999}
}

func tokyoNight() material3.ColorScheme {
	return material3.ColorScheme{
		Primary: widget.Hex(0x9ECE6A), OnPrimary: widget.Hex(0x1A1B26),
		PrimaryContainer: widget.Hex(0x2E3B26), OnPrimaryContainer: widget.Hex(0xC3E3A4),
		Secondary: widget.Hex(0x7AA2F7), OnSecondary: widget.Hex(0x1A1B26),
		SecondaryContainer: widget.Hex(0x283457), OnSecondaryContainer: widget.Hex(0xC0D1FB),
		Tertiary: widget.Hex(0xBB9AF7), OnTertiary: widget.Hex(0x1A1B26),
		TertiaryContainer: widget.Hex(0x382E55), OnTertiaryContainer: widget.Hex(0xE0D0FB),
		Error: widget.Hex(0xF7768E), OnError: widget.Hex(0x1A1B26),
		ErrorContainer: widget.Hex(0x47242E), OnErrorContainer: widget.Hex(0xFFC2CD),
		Surface: widget.Hex(0x1A1B26), OnSurface: widget.Hex(0xC0CAF5),
		SurfaceVariant: widget.Hex(0x24283B), OnSurfaceVariant: widget.Hex(0x8B93B8),
		SurfaceContainerLowest: widget.Hex(0x16161E), SurfaceContainerLow: widget.Hex(0x1C1D29),
		SurfaceContainer: widget.Hex(0x1F2230), SurfaceContainerHigh: widget.Hex(0x24283B),
		SurfaceContainerHighest: widget.Hex(0x2A2F45),
		Background:              widget.Hex(0x1A1B26), OnBackground: widget.Hex(0xC0CAF5),
		Outline: widget.Hex(0x414868), OutlineVariant: widget.Hex(0x2A2E42),
		InverseSurface: widget.Hex(0xC0CAF5), InverseOnSurface: widget.Hex(0x1A1B26),
		InversePrimary: widget.Hex(0x587A3B),
	}
}

// rosePineDawn()'s primary is pine, not the website's foam: white on foam is
// 3.3:1, too faint for a button label, and pine is the same palette's teal.
func rosePineDawn() material3.ColorScheme {
	return material3.ColorScheme{
		Primary: widget.Hex(0x286983), OnPrimary: widget.Hex(0xFFFFFF),
		PrimaryContainer: widget.Hex(0xD3E5EA), OnPrimaryContainer: widget.Hex(0x1C4A5C),
		Secondary: widget.Hex(0x907AA9), OnSecondary: widget.Hex(0xFFFFFF),
		SecondaryContainer: widget.Hex(0xE7E0F0), OnSecondaryContainer: widget.Hex(0x4D3E63),
		Tertiary: widget.Hex(0xD7827E), OnTertiary: widget.Hex(0xFFFFFF),
		TertiaryContainer: widget.Hex(0xF6E0DD), OnTertiaryContainer: widget.Hex(0x7A3B37),
		Error: widget.Hex(0xB4637A), OnError: widget.Hex(0xFFFFFF),
		ErrorContainer: widget.Hex(0xF4DFE5), OnErrorContainer: widget.Hex(0x6B2A3C),
		Surface: widget.Hex(0xFAF4ED), OnSurface: widget.Hex(0x575279),
		SurfaceVariant: widget.Hex(0xF2E9E1), OnSurfaceVariant: widget.Hex(0x797593),
		SurfaceContainerLowest: widget.Hex(0xFFFAF3), SurfaceContainerLow: widget.Hex(0xF7F0E9),
		SurfaceContainer: widget.Hex(0xF2E9E1), SurfaceContainerHigh: widget.Hex(0xEBE3DC),
		SurfaceContainerHighest: widget.Hex(0xDFDAD9),
		Background:              widget.Hex(0xFAF4ED), OnBackground: widget.Hex(0x575279),
		Outline: widget.Hex(0xCECACD), OutlineVariant: widget.Hex(0xDFDAD9),
		InverseSurface: widget.Hex(0x575279), InverseOnSurface: widget.Hex(0xFAF4ED),
		InversePrimary: widget.Hex(0x9CCFD8),
	}
}

// Warn is the palette's orange, for the attention history: Tokyo Night's
// orange and Rosé Pine's gold. Material has no role for it.
func warn(dark bool) widget.Color {
	if dark {
		return widget.Hex(0xFF9E64)
	}
	return widget.Hex(0xEA9D34)
}

// SetDark swaps the theme and rebuilds the root. It is the one case that
// rebuilds (painters are values captured by the widgets), which is why every
// [ScreenFunc] must be callable twice.
//
// Call it on the UI goroutine (from a click handler, or through [Shell.Post]).
func (s *Shell) SetDark(dark bool) {
	if s.M3 != nil && s.M3.IsDark() == dark {
		return
	}
	s.M3 = newTheme(dark)
	s.P = NewPainters(s.M3, s.Fonts)
	s.Cfg.Light = !dark
	if s.UI != nil {
		s.UI.SetTheme(s.M3.AsTheme())
		s.UI.SetRoot(s.Build())
	}
}

// The mark's bands, crest to dim, as the website's themes give them (--f-crest
// .. --f-dim in the website's themes.css): bright on the dark theme,
// inverted to dark ink on the light one.

// markBands is the mark's palette for a theme.
func markBands(dark bool) [5]widget.Color {
	src := lightMark()
	if dark {
		src = darkMark()
	}
	var out [5]widget.Color
	for i, h := range src {
		out[i] = widget.Hex(h)
	}
	return out
}

// Icon is the window icon: the website's favicon, the "j" on its dark square.
func Icon() image.Image { return IconAt(184) }

// IconAt is the same mark drawn size pixels square, for the icon files a
// package carries (cmd/pack): the Windows resource and the macOS .icns.
func IconAt(size int) image.Image {
	var bands [5]color.Color
	for i, h := range darkMark() {
		bands[i] = color.RGBA{R: uint8(h >> 16), G: uint8(h >> 8), B: uint8(h), A: 0xFF}
	}
	return widgets.LogoIcon(size, color.RGBA{R: 0x0E, G: 0x0E, B: 0x14, A: 0xFF}, bands)
}

func darkMark() [5]uint32 {
	return [5]uint32{0xDAECC6, 0xBBDD97, 0x9ECE6A, 0x678549, 0x39482E}
}

func lightMark() [5]uint32 {
	return [5]uint32{0x2F5157, 0x43737C, 0x56949F, 0x8BAFB4, 0xB7C6C5}
}

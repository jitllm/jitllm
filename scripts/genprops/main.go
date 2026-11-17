// Command genprops generates internal/uniprop/tables.go, the code points every
// Unicode property escape of an ECMA-262 pattern names (\p{...}), from the
// Unicode Character Database of the version it is given.
//
// It generates rather than reading Go's unicode package for the reason
// scripts/genunicode gives: those tables ship with the toolchain, so what a
// pattern matches would change under a go upgrade. The version is the one
// tok's tables carry (tok.UnicodeVersion), so a grammar and the tokenizer read
// one Unicode.
//
// Usage: genprops <ucd dir> <out.go>, the directory holding the version's
// UnicodeData.txt, Scripts.txt, ScriptExtensions.txt, PropertyAliases.txt,
// PropertyValueAliases.txt, PropList.txt, DerivedCoreProperties.txt,
// DerivedNormalizationProps.txt, emoji/emoji-data.txt and
// extracted/DerivedBinaryProperties.txt (unicode.org/Public/<version>/ucd).
package main

import (
	"bufio"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"maps"
	mathbits "math/bits"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const maxRune = 0x10FFFF

// binaries are ECMA-262's binary Unicode properties (table-binary-unicode-properties),
// each by its canonical name; Any, ASCII and Assigned are the spec's own.
func binaries() []string {
	return []string{"ASCII", "ASCII_Hex_Digit", "Alphabetic", "Any", "Assigned", "Bidi_Control", "Bidi_Mirrored",
		"Case_Ignorable", "Cased", "Changes_When_Casefolded", "Changes_When_Casemapped", "Changes_When_Lowercased",
		"Changes_When_NFKC_Casefolded", "Changes_When_Titlecased", "Changes_When_Uppercased", "Dash",
		"Default_Ignorable_Code_Point", "Deprecated", "Diacritic", "Emoji", "Emoji_Component", "Emoji_Modifier",
		"Emoji_Modifier_Base", "Emoji_Presentation", "Extended_Pictographic", "Extender", "Grapheme_Base",
		"Grapheme_Extend", "Hex_Digit", "IDS_Binary_Operator", "IDS_Trinary_Operator", "ID_Continue", "ID_Start",
		"Ideographic", "Join_Control", "Logical_Order_Exception", "Lowercase", "Math", "Noncharacter_Code_Point",
		"Pattern_Syntax", "Pattern_White_Space", "Quotation_Mark", "Radical", "Regional_Indicator",
		"Sentence_Terminal", "Soft_Dotted", "Terminal_Punctuation", "Unified_Ideograph", "Uppercase",
		"Variation_Selector", "White_Space", "XID_Continue", "XID_Start"}
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: genprops <ucd dir> <out.go>")
		os.Exit(2)
	}
	dir := os.Args[1]
	in := func(name string) string { return filepath.Join(dir, name) }
	version := versionOf(in("Scripts.txt"))

	// Each code point's general category: UnicodeData.txt, its First/Last
	// pairs a range, every code point it does not list Cn.
	gc := make([]string, maxRune+1)
	for i := range gc {
		gc[i] = "Cn"
	}
	var first int
	lines(in("UnicodeData.txt"), func(f []string) {
		cp := hexOf(f[0])
		switch {
		case strings.HasSuffix(f[1], ", First>"):
			first = cp
		case strings.HasSuffix(f[1], ", Last>"):
			for c := first; c <= cp; c++ {
				gc[c] = f[2]
			}
		default:
			gc[cp] = f[2]
		}
	}, ";")
	sets := map[string]bits{}
	member := func(key string) bits {
		if sets[key] == nil {
			sets[key] = make(bits, maxRune/64+1)
		}
		return sets[key]
	}
	for c, v := range gc {
		member("gc=" + v).set(c)
	}

	// The aliases: a property value's every name, and the groups a
	// general category's major class and LC are (the comment after each).
	propAlias := map[string]string{}
	lines(in("PropertyAliases.txt"), func(f []string) {
		for _, a := range f {
			propAlias[a] = f[1]
		}
	}, ";")
	valAlias := map[string]map[string]string{"gc": {}, "sc": {}}
	groups := map[string][]string{}
	group := regexp.MustCompile(`#\s*([A-Za-z]{2}(?:\s*\|\s*[A-Za-z]{2})+)`)
	rawLines(in("PropertyValueAliases.txt"), func(line string) {
		body, comment, _ := strings.Cut(line, "#")
		f := fields(body, ";")
		if len(f) < 3 || valAlias[f[0]] == nil {
			return
		}
		canon := f[1]
		if f[0] == "sc" {
			canon = f[2]
		}
		for _, a := range f[1:] {
			valAlias[f[0]][a] = canon
		}
		if m := group.FindStringSubmatch("#" + comment); f[0] == "gc" && m != nil {
			for _, g := range strings.Split(m[1], "|") {
				groups[canon] = append(groups[canon], strings.TrimSpace(g))
			}
		}
	})
	for g, members := range groups {
		set := member("gc=" + g)
		for _, m := range members {
			for i, w := range member("gc=" + m) {
				set[i] |= w
			}
		}
	}

	// Scripts: each code point's, Unknown where Scripts.txt lists none; and
	// its extensions, its script alone where ScriptExtensions.txt lists none.
	sc := make([]string, maxRune+1)
	for i := range sc {
		sc[i] = "Unknown"
	}
	lines(in("Scripts.txt"), func(f []string) {
		lo, hi := rangeOf(f[0])
		for c := lo; c <= hi; c++ {
			sc[c] = f[1]
		}
	}, ";")
	scx := make([][]string, maxRune+1)
	lines(in("ScriptExtensions.txt"), func(f []string) {
		lo, hi := rangeOf(f[0])
		var names []string
		for _, s := range strings.Fields(f[1]) {
			n, ok := valAlias["sc"][s]
			if !ok {
				fail("ScriptExtensions.txt names script %q, which PropertyValueAliases.txt does not", s)
			}
			names = append(names, n)
		}
		for c := lo; c <= hi; c++ {
			scx[c] = names
		}
	}, ";")
	for c := range sc {
		member("sc=" + sc[c]).set(c)
		if scx[c] == nil {
			member("scx=" + sc[c]).set(c)
		}
		for _, n := range scx[c] {
			member("scx=" + n).set(c)
		}
	}

	// The binary properties: each file's "range ; name" lines, the
	// normalisation file's CWKCF among its others.
	for _, name := range []string{"PropList.txt", "DerivedCoreProperties.txt", "DerivedNormalizationProps.txt",
		"emoji/emoji-data.txt", "extracted/DerivedBinaryProperties.txt"} {
		lines(in(name), func(f []string) {
			if len(f) != 2 || !slices.Contains(binaries(), f[1]) {
				return
			}
			lo, hi := rangeOf(f[0])
			set := member(f[1])
			for c := lo; c <= hi; c++ {
				set.set(c)
			}
		}, ";")
	}
	member("ASCII")
	for c := range maxRune + 1 {
		member("Any").set(c)
		if c < 0x80 {
			member("ASCII").set(c)
		}
		if gc[c] != "Cn" {
			member("Assigned").set(c)
		}
	}
	for _, b := range binaries() {
		if !slices.ContainsFunc(sets[b], func(w uint64) bool { return w != 0 }) {
			fail("binary property %s holds no code point in these files", b)
		}
	}

	// Each set against the total its file states after its ranges, the
	// general categories' from DerivedGeneralCategory.txt.
	stated := map[string]int{}
	totals(in("extracted/DerivedGeneralCategory.txt"), func(n string) string { return "gc=" + n }, stated)
	totals(in("Scripts.txt"), func(n string) string { return "sc=" + n }, stated)
	for _, name := range []string{"PropList.txt", "DerivedCoreProperties.txt", "DerivedNormalizationProps.txt",
		"emoji/emoji-data.txt", "extracted/DerivedBinaryProperties.txt"} {
		totals(in(name), func(n string) string { return n }, stated)
	}
	checked := 0
	for k, n := range stated {
		set, ok := sets[k]
		if !ok {
			continue
		}
		if got := set.count(); got != n {
			fail("%s holds %d code points, its file states %d", k, got, n)
		}
		checked++
	}
	fmt.Fprintf(os.Stderr, "genprops: Unicode %s, %d sets, %d held to their files' totals\n", version, len(sets), checked)

	// A name or value an escape may spell: every alias, onto its key.
	keys := map[string]string{}
	for a, v := range valAlias["gc"] {
		keys["gc|"+a] = "gc=" + v
	}
	for a, v := range valAlias["sc"] {
		keys["sc|"+a] = "sc=" + v
		keys["scx|"+a] = "scx=" + v
	}
	for a, p := range propAlias {
		if slices.Contains(binaries(), p) {
			keys["|"+a] = p
		}
	}
	for _, b := range binaries() {
		keys["|"+b] = b
	}
	// A lone name is a general category before a binary property (ECMA-262's
	// LoneUnicodePropertyNameOrValue).
	for a, v := range valAlias["gc"] {
		keys["|"+a] = "gc=" + v
	}
	for _, k := range keys {
		member(k)
	}

	var b strings.Builder
	fmt.Fprintf(&b, `// Code generated by scripts/genprops. DO NOT EDIT.
//
// Source: the Unicode Character Database, %s (unicode.org/Public/%s/ucd).

package uniprop

// Version is the Unicode version a property escape reads.
const Version = %q

// Ranges is the code points key holds (Key's), as unsigned varint
// pairs in base64: each span's distance past the last one's end and its
// length less one; "" for a key that names nothing.
func Ranges(key string) string {
	switch key {
`, version, version, version)
	for _, k := range slices.Sorted(maps.Keys(sets)) {
		fmt.Fprintf(&b, "\tcase %q:\n\t\treturn %q\n", k, encode(sets[k]))
	}
	b.WriteString(`	}
	return ""
}

// Key is the key the property of an escape names: kind is gc, sc or scx
// for a name=value escape, or "" for a lone name -- a general category or a
// binary property -- and name the value or the lone name as spelled, any of
// its aliases. "" where it names none.
func Key(kind, name string) string {
	switch kind + "|" + name {
`)
	for _, k := range slices.Sorted(maps.Keys(keys)) {
		fmt.Fprintf(&b, "\tcase %q:\n\t\treturn %q\n", k, keys[k])
	}
	b.WriteString(`	}
	return ""
}

// KindOf is the kind a name=value escape's name spells, "" for none.
func KindOf(name string) string {
	switch name {
`)
	for _, k := range []string{"gc", "sc", "scx"} {
		var names []string
		for a, p := range propAlias {
			if p == map[string]string{"gc": "General_Category", "sc": "Script", "scx": "Script_Extensions"}[k] {
				names = append(names, strconv.Quote(a))
			}
		}
		slices.Sort(names)
		fmt.Fprintf(&b, "\tcase %s:\n\t\treturn %q\n", strings.Join(names, ", "), k)
	}
	b.WriteString(`	}
	return ""
}
`)
	if err := os.WriteFile(os.Args[2], []byte(b.String()), 0o644); err != nil {
		fail("%v", err)
	}
}

// bits is a set of code points, bit c%64 of word c/64.
type bits []uint64

func (b bits) set(c int)      { b[c/64] |= 1 << (c % 64) }
func (b bits) has(c int) bool { return b[c/64]&(1<<(c%64)) != 0 }

func (b bits) count() int {
	n := 0
	for _, w := range b {
		n += mathbits.OnesCount64(w)
	}
	return n
}

// totals is each block's stated size in a UCD file, the "# Total code
// points: N" (or the emoji file's "# Total elements: N") after its ranges,
// keyed by key of the name its last data line gives.
func totals(path string, key func(string) string, into map[string]int) {
	total := regexp.MustCompile(`^# Total (?:code points|elements): (\d+)`)
	last := ""
	rawLines(path, func(line string) {
		if m := total.FindStringSubmatch(line); m != nil && last != "" {
			n, err := strconv.Atoi(m[1])
			if err != nil {
				fail("%s: %v", path, err)
			}
			into[key(last)] = n
			return
		}
		body, _, _ := strings.Cut(line, "#")
		if f := fields(body, ";"); len(f) >= 2 && f[0] != "" {
			last = f[1]
		}
	})
}

// encode is set's spans as Ranges states them.
func encode(set bits) string {
	var out []byte
	next := 0
	for c := 0; c <= maxRune; {
		if !set.has(c) {
			c++
			continue
		}
		lo := c
		for c <= maxRune && set.has(c) {
			c++
		}
		out = binary.AppendUvarint(out, uint64(lo-next))
		out = binary.AppendUvarint(out, uint64(c-1-lo))
		next = c
	}
	return base64.StdEncoding.EncodeToString(out)
}

// versionOf is the version a UCD file's first line names (# Scripts-15.1.0.txt).
func versionOf(path string) string {
	f, err := os.Open(path)
	if err != nil {
		fail("%v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Scan()
	m := regexp.MustCompile(`-(\d+\.\d+\.\d+)\.txt`).FindStringSubmatch(sc.Text())
	if m == nil {
		fail("%s states no version on its first line", path)
	}
	return m[1]
}

// lines calls f with each data line's fields, split at sep and trimmed, its
// comment dropped.
func lines(path string, f func([]string), sep string) {
	rawLines(path, func(line string) {
		body, _, _ := strings.Cut(line, "#")
		if strings.TrimSpace(body) == "" {
			return
		}
		f(fields(body, sep))
	})
}

func rawLines(path string, f func(string)) {
	file, err := os.Open(path)
	if err != nil {
		fail("%v", err)
	}
	defer file.Close()
	sc := bufio.NewScanner(file)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		f(sc.Text())
	}
	if err := sc.Err(); err != nil {
		fail("%s: %v", path, err)
	}
}

func fields(s, sep string) []string {
	f := strings.Split(s, sep)
	for i := range f {
		f[i] = strings.TrimSpace(f[i])
	}
	return f
}

func rangeOf(s string) (int, int) {
	lo, hi, ok := strings.Cut(s, "..")
	if !ok {
		hi = lo
	}
	return hexOf(lo), hexOf(hi)
}

func hexOf(s string) int {
	v, err := strconv.ParseUint(strings.TrimSpace(s), 16, 32)
	if err != nil || v > maxRune {
		fail("code point %q", s)
	}
	return int(v)
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "genprops: "+format+"\n", a...)
	os.Exit(1)
}

package tok

// The splitters each older stage kind ran before every stage ran through the
// pattern engine. They are the reference the engine's reading of those kinds
// (pretok.Op.AsPattern) is held to, in TestSplitPatternsMatchTheStages, and
// nothing outside the tests runs them.

import "github.com/jitllm/jitllm/tok/pretok"

// applyOp runs one stage over s through the splitter written for its kind,
// handing its pieces to emit: the reference the pattern engine is held to
// (TestSplitPatternsMatchTheStages).
func applyOp(op pretok.Op, s string, sc *runes, emit func(string)) {
	switch op.Kind {
	case pretok.OpSplit:
		splitLlama3Each(s, op.Split, sc, emit)
	case pretok.OpDigits:
		splitDigitsOpEach(s, op.Individual, sc, emit)
	case pretok.OpPunctuation:
		splitPunctuationEach(s, op.Behavior, sc, emit)
	case pretok.OpDigitTriples:
		splitDigitTriplesEach(s, emit)
	case pretok.OpPunctDefault:
		splitRunsEach(s, isPunctDefault, sc, emit)
	case pretok.OpDigitGroups:
		splitDigitGroupsEach(s, op.Split.DigitGroup, sc, emit)
	case pretok.OpKanaHanRuns:
		splitRunsEach(s, isKanaHan, sc, emit)
	case pretok.OpHunyuanSplit:
		splitHunyuanEach(s, false, sc, emit)
	case pretok.OpSparkSplit:
		splitHunyuanEach(s, true, sc, emit)
	case pretok.OpByteLevel:
		if op.UseRegex {
			splitGPT2Each(s, sc, emit)
			return
		}
		emit(s)
	default:
		emit(s)
	}
}

// splitDigitsOpEach isolates digits: individual_digits=true gives one piece per
// digit, false groups each run.
func splitDigitsOpEach(s string, individual bool, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	for i := 0; i < len(rs); {
		if !ucIsNumber(rs[i]) {
			j := i
			for j < len(rs) && !ucIsNumber(rs[j]) {
				j++
			}
			emit(sc.piece(s, i, j))
			i = j
			continue
		}
		if individual {
			emit(sc.piece(s, i, i+1))
			i++
			continue
		}
		j := i
		for j < len(rs) && ucIsNumber(rs[j]) {
			j++
		}
		emit(sc.piece(s, i, j))
		i = j
	}
}

// splitDigitTriplesEach isolates each run of three ASCII digits, scanning left
// to right without overlap -- the regex "[0-9][0-9][0-9]" under HuggingFace's
// Split with behavior Isolated. What lies between matches is emitted whole.
// ASCII only, as the class says: a byte test needs no rune decode.
func splitDigitTriplesEach(s string, emit func(string)) {
	isD := func(b byte) bool { return b >= '0' && b <= '9' }
	start := 0
	for i := 0; i+3 <= len(s); {
		if isD(s[i]) && isD(s[i+1]) && isD(s[i+2]) {
			if i > start {
				emit(s[start:i])
			}
			emit(s[i : i+3])
			i += 3
			start = i
			continue
		}
		i++
	}
	if start < len(s) {
		emit(s[start:])
	}
}

// isPunctDefault is "[\p{P}\$\+<=>\^~\|]" -- \p{P} and the eight named
// ASCII symbols, and not the backtick HuggingFace's is_punc includes.
func isPunctDefault(r rune) bool {
	switch r {
	case '$', '+', '<', '=', '>', '^', '~', '|':
		return true
	}
	return ucFlags(r)&ucFlagPunctuation != 0
}

// splitRunsEach emits maximal runs of in(r) and of !in(r), in order.
func splitRunsEach(s string, in func(rune) bool, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	for i := 0; i < len(rs); {
		j, want := i, in(rs[i])
		for j < len(rs) && in(rs[j]) == want {
			j++
		}
		emit(sc.piece(s, i, j))
		i = j
	}
}

// splitPunctuationEach isolates punctuation. "Contiguous" keeps a run together;
// anything else isolates each mark.
//
// Punctuation is HuggingFace's is_punc: is_ascii_punctuation || \p{P}. The
// ASCII symbols $+<=>^`|~ are \p{S}, not \p{P}, and HF still isolates them.
// BERT's wordPiece stage reads the same class (wpm.go).
func splitPunctuationEach(s string, behavior string, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	isPunct := func(r rune) bool {
		fl := ucFlags(r)
		return fl&ucFlagPunctuation != 0 || (r < 0x7F && fl&ucFlagSymbol != 0)
	}
	for i := 0; i < len(rs); {
		if !isPunct(rs[i]) {
			j := i
			for j < len(rs) && !isPunct(rs[j]) {
				j++
			}
			emit(sc.piece(s, i, j))
			i = j
			continue
		}
		if behavior != "Contiguous" {
			emit(sc.piece(s, i, i+1))
			i++
			continue
		}
		j := i
		for j < len(rs) && isPunct(rs[j]) {
			j++
		}
		emit(sc.piece(s, i, j))
		i = j
	}
}

// Hunyuan's pre-tokenizer stages (pretok.OpDigitGroups, OpKanaHanRuns,
// OpHunyuanSplit), each a HuggingFace Split with behavior Isolated: what a
// pattern matches is a piece, and so is every stretch between two matches.

// splitDigitGroupsEach isolates \p{N} in groups of n, left to right: the
// regex \p{N}{1,n} scans greedily without overlap, so a run of five digits at
// n=3 is "123" then "45". Everything else is emitted in the stretches between.
func splitDigitGroupsEach(s string, n int, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	for i := 0; i < len(rs); {
		if !ucIsNumber(rs[i]) {
			j := i
			for j < len(rs) && !ucIsNumber(rs[j]) {
				j++
			}
			emit(sc.piece(s, i, j))
			i = j
			continue
		}
		j := i
		for j < len(rs) && j-i < n && ucIsNumber(rs[j]) {
			j++
		}
		emit(sc.piece(s, i, j))
		i = j
	}
}

// isKanaHan is the class [一-龥぀-ゟ゠-ヿ]: the
// ranges as written, not \p{Han}.
func isKanaHan(r rune) bool {
	return r >= 0x4e00 && r <= 0x9fa5 || r >= 0x3040 && r <= 0x309f || r >= 0x30a0 && r <= 0x30ff
}

// isHunyuanMark is the first alternative's class, the ASCII punctuation and
// symbols written out: [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~].
func isHunyuanMark(r rune) bool {
	return r >= '!' && r <= '/' || r >= ':' && r <= '@' || r >= '[' && r <= '`' || r >= '{' && r <= '~'
}

func isASCIILetter(r rune) bool { return r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' }

// splitHunyuanEach is pretok.OpHunyuanSplit: at each position the first of
// the six alternatives that matches, as a backtracking regex takes them, and
// the characters no alternative matches kept as pieces of their own. spark
// is pretok.OpSparkSplit: no newline tail on the punctuation run, and a
// newline a piece of its own in place of \s*[\r\n]+.
func splitHunyuanEach(s string, spark bool, sc *runes, emit func(string)) {
	sc.load(s)
	rs := sc.rs
	n := len(rs)
	fl := func(i int) uint16 { return ucFlags(rs[i]) }
	letter := func(i int) bool { return i < n && fl(i)&ucFlagLetter != 0 }
	lm := func(i int) bool { return i < n && fl(i)&(ucFlagLetter|ucFlagAccentMark) != 0 }
	ps := func(i int) bool { return i < n && fl(i)&(ucFlagPunctuation|ucFlagSymbol) != 0 }
	nl := func(i int) bool { return i < n && (rs[i] == '\r' || rs[i] == '\n') }
	space := func(i int) bool { return i < n && ucIsSpace(rs[i]) }
	run := func(i int, in func(int) bool) int {
		for in(i) {
			i++
		}
		return i
	}
	// The punctuation run's tail: [\r\n]*, or nothing for Spark.
	tail := nl
	if spark {
		tail = func(int) bool { return false }
	}
	at := func(i int) int {
		// [ASCII mark][A-Za-z]+
		if isHunyuanMark(rs[i]) && i+1 < n && isASCIILetter(rs[i+1]) {
			return run(i+1, func(j int) bool { return j < n && isASCIILetter(rs[j]) })
		}
		// [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+: the prefix first, then without.
		if !nl(i) && !letter(i) && !ps(i) && lm(i+1) {
			return run(i+1, lm)
		}
		if lm(i) {
			return run(i, lm)
		}
		// ` ?[\p{P}\p{S}]+[\r\n]*`; Spark's has no tail.
		if rs[i] == ' ' && ps(i+1) {
			return run(run(i+1, ps), tail)
		}
		if ps(i) {
			return run(run(i, ps), tail)
		}
		// Spark's [\r\n]: one character, before any whitespace run.
		if spark && nl(i) {
			return i + 1
		}
		if !space(i) {
			return i
		}
		w := run(i, space)
		if spark {
			// \s+(?!\S) then \s+, with no newline alternative in between.
			if w == n || w-i < 2 {
				return w
			}
			return w - 1
		}
		// \s*[\r\n]+ backtracks the greedy \s* to the last newline in the run.
		for k := w - 1; k >= i; k-- {
			if nl(k) {
				return k + 1
			}
		}
		// \s+(?!\S): the whole run at the end of the text, all but its last
		// character before anything else; then \s+.
		if w == n || w-i >= 2 {
			if w == n {
				return w
			}
			return w - 1
		}
		return w
	}
	prev := 0
	for i := 0; i < n; {
		end := at(i)
		if end == i {
			i++
			continue
		}
		if i > prev {
			emit(sc.piece(s, prev, i))
		}
		emit(sc.piece(s, i, end))
		i, prev = end, end
	}
	if prev < n {
		emit(sc.piece(s, prev, n))
	}
}

// splitLlama3 is the parameterised Split stage at llama3's settings, the
// pattern llama3, qwen2 and qwen3 share:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//	| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Hand-rolled because `\s+(?!\S)` is a negative lookahead RE2 cannot run.
// It tries the pattern's alternatives in order at each position, as a
// backtracking engine reads the pattern, where the pattern alone does not
// settle a case.
func splitLlama3(text string, digitGroup int) []string {
	return splitLlama3Params(text, llama3Params(digitGroup))
}

// splitLlama3Params is the Split stage. Every pattern in the wild is this
// skeleton with different parameters; see pretok.SplitParams.
func splitLlama3Params(text string, sp pretok.SplitParams) []string {
	return collect(func(emit func(string)) { splitLlama3Each(text, sp, &runes{}, emit) })
}

// splitLlama3Each is the Split stage, handing each piece to emit as it is found.
func splitLlama3Each(text string, sp pretok.SplitParams, sc *runes, emit func(string)) {
	digitGroup := sp.DigitGroup
	if digitGroup < 1 {
		digitGroup = 1
	}
	sc.load(text)
	rs := sc.rs
	n := len(rs)
	prev := 0
	add := func(end int) {
		if end > prev {
			emit(sc.piece(text, prev, end))
		}
		prev = end
	}
	isLetter := func(i int) bool {
		if i < 0 || i >= n {
			return false
		}
		// qwen35's pattern is [\p{L}\p{M}]+ -- letters AND combining marks --
		// where llama3's is \p{L}+. A mark joining the letter run or starting a
		// new piece is a different tokenization of the same text.
		if sp.LetterMarks && ucFlags(rs[i])&ucFlagAccentMark != 0 {
			return true
		}
		return ucIsLetter(rs[i])
	}
	isNumber := func(i int) bool { return i >= 0 && i < n && ucIsNumber(rs[i]) }
	isSpace := func(i int) bool { return i >= 0 && i < n && ucIsSpace(rs[i]) }
	// isHan is \p{Han}, and under HanRuns it is subtracted from the letter
	// classes rather than merely taken first -- see SplitParams.HanRuns for why
	// the two halves are one flag.
	isHan := func(i int) bool { return sp.HanRuns && i >= 0 && i < n && ucIsHan(rs[i]) }
	// Han is \p{Lo}, a letter to every predicate below and to the upper
	// class, so the subtraction is wrapped into the classes once rather than
	// patched at each call site.
	isLetterX := func(i int) bool { return isLetter(i) && !isHan(i) }
	// plain is "in range and none of letter/number/whitespace", which is the
	// [^\s\p{L}\p{N}] class plus a bounds check.
	// It uses the unsubtracted isLetter on purpose: those classes are plain
	// \p{L} in the pattern, so a Han codepoint is never punctuation.
	plain := func(i int) bool { return i >= 0 && i < n && !isLetter(i) && !isNumber(i) && !isSpace(i) }

	// contractAt matches (?i:'s|'t|'re|'ve|'m|'ll|'d) at p and returns where it
	// ends, or p. SuffixContract decides where it may appear.
	contractAt := func(p int) int {
		if !sp.Contractions || p >= n || rs[p] != '\'' || p+1 >= n {
			return p
		}
		c1 := ucFold(rs[p+1])
		if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
			return p + 2
		}
		if p+2 < n {
			c2 := ucFold(rs[p+2])
			if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
				return p + 3
			}
		}
		return p
	}
	// caseRun matches o200k's two letter alternatives at s, in order:
	//
	//	[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]* [\p{Ll}\p{Lm}\p{Lo}\p{M}]+
	//	[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+ [\p{Ll}\p{Lm}\p{Lo}\p{M}]*
	//
	// The backtrack is required: Lm, Lo and M are in both classes, so a
	// greedy upper run can swallow what the first alternative's mandatory
	// lower run needs. Without it every CJK word takes the second alternative.
	upperAt := func(i int) bool { return i < n && ucUpperClass(rs[i]) && !isHan(i) }
	lowerAt := func(i int) bool { return i < n && ucLowerClass(rs[i]) && !isHan(i) }
	caseRun := func(s int) int {
		u := s
		for upperAt(u) {
			u++
		}
		for k := u; k >= s; k-- {
			if lowerAt(k) {
				e := k
				for lowerAt(e) {
					e++
				}
				return e
			}
		}
		if u > s {
			e := u
			for lowerAt(e) {
				e++
			}
			return e
		}
		return s
	}

	for pos := 0; pos < n; {
		// [\p{Han}]+ -- kimi-k2's first alternative, and first here for the same
		// reason: a greedy alternation tries it before the letter runs, and the
		// letter runs have had Han subtracted so they cannot take it back.
		if isHan(pos) {
			for isHan(pos) {
				pos++
			}
			add(pos)
			continue
		}
		// o200k: [^\r\n\p{L}\p{N}]? (case runs) (?i:'s|'t|...)?
		//
		// The optional leading character is tried first and given back if
		// no letters follow, as a greedy `?` does.
		if sp.CaseRuns {
			end := pos
			if rs[pos] != '\r' && rs[pos] != '\n' && !isLetter(pos) && !isNumber(pos) {
				if e := caseRun(pos + 1); e > pos+1 {
					end = e
				}
			}
			if end == pos {
				end = caseRun(pos)
			}
			if end > pos {
				if sp.SuffixContract {
					end = contractAt(end)
				}
				pos = end
				add(pos)
				continue
			}
		}
		// (?i:'s|'t|'re|'ve|'m|'ll|'d), as its own leading alternative. Under
		// SuffixContract it is not one: a lone "'t" with no letters before it is
		// punctuation there, not a contraction.
		if sp.Contractions && !sp.SuffixContract && rs[pos] == '\'' && pos+1 < n {
			c1 := ucFold(rs[pos+1])
			if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
				pos += 2
				add(pos)
				continue
			}
			if pos+2 < n {
				c2 := ucFold(rs[pos+2])
				if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
					pos += 3
					add(pos)
					continue
				}
			}
		}
		// [^\r\n\p{L}\p{N}]?\p{L}+ -- an optional single non-letter, then letters.
		if !sp.CaseRuns && rs[pos] != '\r' && rs[pos] != '\n' && !isNumber(pos) {
			if isLetterX(pos) || isLetterX(pos+1) {
				pos++
				for isLetterX(pos) {
					pos++
				}
				add(pos)
				continue
			}
		}
		// llama3 is \p{N}{1,3} and qwen2 a bare \p{N}; see digitGroupFor.
		if isNumber(pos) {
			ini := pos
			for isNumber(pos) {
				pos++
				if pos-ini >= digitGroup {
					add(pos)
					ini = pos
				}
			}
			add(pos)
			continue
		}
		// ' ?[^\s\p{L}\p{N}]+[\r\n]*'
		probe := pos
		if rs[pos] == ' ' {
			probe = pos + 1
		}
		if plain(probe) {
			if rs[pos] == ' ' {
				pos++
			}
			for plain(pos) {
				pos++
			}
			for pos < n && !sp.PunctNoNewline && (rs[pos] == '\r' || rs[pos] == '\n' || (sp.PunctSlash && rs[pos] == '/')) {
				pos++
			}
			add(pos)
			continue
		}
		// The three whitespace alternatives, which differ only in where they stop.
		nws, lastNL := 0, 0
		for isSpace(pos + nws) {
			if rs[pos+nws] == '\r' || rs[pos+nws] == '\n' {
				lastNL = pos + nws + 1
			}
			nws++
		}
		switch {
		case lastNL > 0: // \s*[\r\n]+
			pos = lastNL
		case nws > 1 && pos+nws < n: // \s+(?!\S): give the last space back
			pos += nws - 1
		case nws > 0: // \s+
			pos += nws
		default:
			pos++
		}
		add(pos)
	}
}

// splitGPT2Each implements the original GPT-2 pattern, the one the gpt-2, mpt,
// olmo and jais tokenizers state, handing each piece to emit as it is found:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Unlike splitLlama3 the contractions are case-sensitive, numbers run
// unbounded, and a newline is plain whitespace. Hand-rolled for the same
// negative lookahead.
func splitGPT2Each(text string, sc *runes, emit func(string)) {
	sc.load(text)
	rs := sc.rs
	n := len(rs)
	prev := 0
	add := func(end int) {
		if end > prev {
			emit(sc.piece(text, prev, end))
		}
		prev = end
	}
	isLetter := func(i int) bool { return i >= 0 && i < n && ucIsLetter(rs[i]) }
	isNumber := func(i int) bool { return i >= 0 && i < n && ucIsNumber(rs[i]) }
	isSpace := func(i int) bool { return i >= 0 && i < n && ucIsSpace(rs[i]) }
	plain := func(i int) bool { return i >= 0 && i < n && !isLetter(i) && !isNumber(i) && !isSpace(i) }

	for pos := 0; pos < n; {
		// 's|'t|'re|'ve|'m|'ll|'d -- no case folding, unlike llama3's.
		if rs[pos] == '\'' && pos+1 < n {
			c1 := rs[pos+1]
			if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
				pos += 2
				add(pos)
				continue
			}
			if pos+2 < n {
				c2 := rs[pos+2]
				if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
					pos += 3
					add(pos)
					continue
				}
			}
		}
		probe := pos
		if rs[pos] == ' ' {
			probe = pos + 1
		}
		// ' ?\p{L}+', then ' ?\p{N}+', then ' ?[^\s\p{L}\p{N}]+'
		var cls func(int) bool
		switch {
		case isLetter(probe):
			cls = isLetter
		case isNumber(probe):
			cls = isNumber
		case plain(probe):
			cls = plain
		}
		if cls != nil {
			if rs[pos] == ' ' {
				pos++
			}
			for cls(pos) {
				pos++
			}
			add(pos)
			continue
		}
		nws := 0
		for isSpace(pos + nws) {
			nws++
		}
		switch {
		case nws > 1 && pos+nws < n: // \s+(?!\S): the last space starts the next word
			pos += nws - 1
		case nws > 0:
			pos += nws
		default:
			pos++
		}
		add(pos)
	}
}

// ucUpperClass and ucLowerClass are o200k's two letter classes -- the shape the
// gpt-4o pre-tokenizer splits a word into and the reason ucUpper is generated.
//
//	ucUpperClass  [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
//	ucLowerClass  [\p{Ll}\p{Lm}\p{Lo}\p{M}]
//
// The general category is not in the flag table (it folds Lu/Ll/Lt/Lm/Lo into
// one bit). The two case
// mappings in the same artefact separate the classes exactly:
//
//	Lu, Lt   have a lowercase mapping        -> upper class only
//	Ll       has an uppercase mapping only   -> lower class only
//	Lm, Lo   are letters with neither        -> both classes
//	M        is ucFlagAccentMark             -> both classes
//
// Lt is the case that makes the test "has a lowercase mapping" rather than
// "has no uppercase mapping": U+01C5 ǅ carries both, and it belongs with the
// uppercase letters.
func ucUpperClass(r rune) bool {
	if ucFlags(r)&ucFlagAccentMark != 0 {
		return true
	}
	if !ucIsLetter(r) {
		return false
	}
	if _, down := ucMap(ucLower, r); down {
		return true // Lu or Lt
	}
	_, up := ucMap(ucUpper, r)
	return !up // Lm/Lo have neither mapping; Ll has only the upward one
}

func ucLowerClass(r rune) bool {
	if ucFlags(r)&ucFlagAccentMark != 0 {
		return true
	}
	if !ucIsLetter(r) {
		return false
	}
	_, down := ucMap(ucLower, r)
	return !down // everything but Lu and Lt
}

// ucIsHan is \p{Han}, the script, from the pinned Unicode 15.1.0 table. A
// script cuts across categories, so it is a separate table (from the UCD; see
// scripts/genunicode). A linear scan suffices: only the kimi-k2 splitter asks,
// and Latin text exits on the first range.
func ucIsHan(r rune) bool {
	if r < 0x2E80 {
		return false // below every Han range; the whole of ASCII
	}
	u := uint32(r)
	for o := 0; o < len(ucHan); o += 6 {
		if u < ucWord(ucHan, o) {
			return false
		}
		if u <= ucWord(ucHan, o+3) {
			return true
		}
	}
	return false
}

// ucIsLetter is \p{L}: every letter subcategory (Lu, Ll, Lt, Lm, Lo) in one
// bit, which is all the pre-tokenizer patterns ask for.
func ucIsLetter(r rune) bool { return ucFlags(r)&ucFlagLetter != 0 }

// ucIsNumber is \p{N} -- decimal, letter and other numbers, as the regex class
// means.
func ucIsNumber(r rune) bool { return ucFlags(r)&ucFlagNumber != 0 }

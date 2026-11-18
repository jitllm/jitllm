package grammar

import (
	"encoding/json"
	"fmt"
	"math/big"
	"strconv"
	"strings"
)

// maxBoundDigits is how many digits a bound may take on either side of its
// point. The rules are written a digit at a time, so a request's bound is
// held to a ceiling as its text is (maxSchemaText); this one is past the
// widest a float64 writes, 309 integral digits and 5e-324's 324 fraction
// digits.
const maxBoundDigits = 1024

// boundKeys are the keywords that bound a number.
func boundKeys() []string {
	return []string{"minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"}
}

// hasBounds reports whether schema s bounds a number.
func hasBounds(s map[string]any) bool {
	for _, k := range boundKeys() {
		if _, ok := s[k]; ok {
			return true
		}
	}
	return false
}

// intBounds is the integers schema s admits, lo through hi: the tightest of
// its bounds on each side, nil on a side it does not bound. ok is false
// where no integer lies between them.
func intBounds(s map[string]any) (lo, hi *big.Int, ok bool, err error) {
	one := big.NewInt(1)
	for _, k := range boundKeys() {
		v, has := s[k]
		if !has {
			continue
		}
		n, isNum := v.(json.Number)
		if !isNum {
			return nil, nil, false, fmt.Errorf("grammar: %s must be a number", k)
		}
		if _, err := fixedOf(k, string(n)); err != nil {
			return nil, nil, false, err
		}
		r, isRat := new(big.Rat).SetString(string(n))
		if !isRat {
			return nil, nil, false, fmt.Errorf("grammar: %s is %q, not a number", k, n)
		}
		q, m := new(big.Int).DivMod(r.Num(), r.Denom(), new(big.Int))
		ceil := new(big.Int).Set(q)
		if m.Sign() != 0 {
			ceil.Add(ceil, one)
		}
		switch k {
		case "minimum":
			if lo == nil || ceil.Cmp(lo) > 0 {
				lo = ceil
			}
		case "exclusiveMinimum":
			if b := q.Add(q, one); lo == nil || b.Cmp(lo) > 0 {
				lo = b
			}
		case "maximum":
			if hi == nil || q.Cmp(hi) < 0 {
				hi = q
			}
		case "exclusiveMaximum":
			if b := ceil.Sub(ceil, one); hi == nil || b.Cmp(hi) < 0 {
				hi = b
			}
		}
	}
	return lo, hi, lo == nil || hi == nil || lo.Cmp(hi) <= 0, nil
}

// fixed is a bound's value written out with no exponent: its sign, its
// integral part with no leading zero ("0" for none) and its fraction with no
// trailing zero. Zero has no sign.
type fixed struct {
	neg         bool
	whole, frac string
}

func (f fixed) String() string {
	s := f.whole
	if f.frac != "" {
		s += "." + f.frac
	}
	if f.neg {
		s = "-" + s
	}
	return s
}

func (f fixed) zero() bool { return f.whole == "0" && f.frac == "" }

// fixedOf is keyword k's JSON number n written out, from its significant
// digits and power of ten (decimal).
func fixedOf(k, n string) (fixed, error) {
	c, ok := decimal(n)
	if !ok {
		return fixed{}, fmt.Errorf("grammar: %s is %q, not a number", k, n)
	}
	if c == "0" {
		return fixed{whole: "0"}, nil
	}
	var f fixed
	if strings.HasPrefix(c, "-") {
		f.neg, c = true, c[1:]
	}
	sig, e, _ := strings.Cut(c, "e")
	exp, err := strconv.ParseInt(e, 10, 64)
	if err != nil {
		return fixed{}, fmt.Errorf("grammar: %s is %q, not a number", k, n)
	}
	point := int64(len(sig)) + exp
	if point > maxBoundDigits || int64(len(sig))-point > maxBoundDigits {
		return fixed{}, ErrUnsupported{fmt.Sprintf("a bound past %d digits either side of its point", maxBoundDigits)}
	}
	switch p := int(point); {
	case p >= len(sig):
		f.whole = sig + strings.Repeat("0", p-len(sig))
	case p <= 0:
		f.whole, f.frac = "0", strings.Repeat("0", -p)+sig
	default:
		f.whole, f.frac = sig[:p], sig[p:]
	}
	return f, nil
}

// cmpFixed is -1, 0 or 1 as a is below, at or above b. Fractions with no
// trailing zero compare as their text does.
func cmpFixed(a, b fixed) int {
	if a.neg != b.neg {
		if a.neg {
			return -1
		}
		return 1
	}
	c := len(a.whole) - len(b.whole)
	if c == 0 {
		c = strings.Compare(a.whole, b.whole)
	}
	if c == 0 {
		c = strings.Compare(a.frac, b.frac)
	}
	c = min(max(c, -1), 1)
	if a.neg {
		return -c
	}
	return c
}

// bound is one side of a number's range: its value, and whether the value
// itself is outside (an exclusive bound).
type bound struct {
	v      fixed
	strict bool
}

// empty reports whether no value lies from lo to hi.
func empty(lo, hi bound) bool {
	c := cmpFixed(lo.v, hi.v)
	return c > 0 || c == 0 && (lo.strict || hi.strict)
}

// numberBounds is the range schema s admits a number in: the tightest of its
// bounds on each side, nil on a side it does not bound.
func numberBounds(s map[string]any) (lo, hi *bound, err error) {
	for _, k := range boundKeys() {
		v, has := s[k]
		if !has {
			continue
		}
		n, isNum := v.(json.Number)
		if !isNum {
			return nil, nil, fmt.Errorf("grammar: %s must be a number", k)
		}
		d, err := fixedOf(k, string(n))
		if err != nil {
			return nil, nil, err
		}
		cur, up := &hi, -1
		if strings.HasSuffix(k, "inimum") {
			cur, up = &lo, 1
		}
		strict := strings.HasPrefix(k, "exclusive")
		if *cur != nil {
			if c := up * cmpFixed(d, (*cur).v); c < 0 || c == 0 && !strict {
				continue
			}
		}
		*cur = &bound{v: d, strict: strict}
	}
	return lo, hi, nil
}

// numberRange is the rule text for the numbers from lo to hi, a nil side
// unbounded, as RFC 8259's number writes them with no exponent: a minus
// before a negative, no leading zero, a fraction of any length, and 0 as 0
// or -0. lo to hi holds a value (empty is false).
func numberRange(lo, hi *bound) string {
	zero := bound{v: fixed{whole: "0"}}
	abs := func(b *bound) *bound {
		return &bound{v: fixed{whole: b.v.whole, frac: b.v.frac}, strict: b.strict}
	}
	var alts []string
	if hi == nil || !hi.v.neg {
		from := zero
		if lo != nil && !lo.v.neg {
			from = *lo
		}
		if m := magnitudes(from, hi); m != "" {
			alts = append(alts, "( "+m+" )")
		}
	}
	if lo == nil || lo.v.neg || lo.v.zero() {
		from := zero
		if hi != nil && (hi.v.neg || hi.v.zero()) {
			from = *abs(hi)
		}
		var to *bound
		if lo != nil {
			to = abs(lo)
		}
		if m := magnitudes(from, to); m != "" {
			alts = append(alts, `"-" ( `+m+" )")
		}
	}
	return "( " + strings.Join(alts, " | ") + " )"
}

// magnitudes is the rule text for the numerals with no sign and no exponent
// from lo to hi, hi nil for no end, "" for none: lo's integral part followed
// by a fraction from lo's, the integral parts between followed by any
// fraction, and hi's followed by a fraction up to hi's.
func magnitudes(lo bound, hi *bound) string {
	if hi != nil && empty(lo, *hi) {
		return ""
	}
	var alts []string
	whole := func(w string, eps bool, ne string) {
		if f, ok := fraction(eps, ne); ok {
			alts = append(alts, `"`+w+`"`+f)
		}
	}
	if hi != nil && lo.v.whole == hi.v.whole {
		eps, ne := fracBetween(lo.v.frac, lo.strict, hi.v.frac, hi.strict)
		whole(lo.v.whole, eps, ne)
		return strings.Join(alts, " | ")
	}
	eps, ne := fracAtLeast(lo.v.frac, lo.strict)
	whole(lo.v.whole, eps, ne)
	a, _ := new(big.Int).SetString(lo.v.whole, 10)
	a.Add(a, big.NewInt(1))
	var b *big.Int
	if hi != nil {
		b, _ = new(big.Int).SetString(hi.v.whole, 10)
		b.Sub(b, big.NewInt(1))
	}
	if b == nil || a.Cmp(b) <= 0 {
		alts = append(alts, `( `+numerals(a, b)+` ) ( "." [0-9]+ )?`)
	}
	if hi != nil {
		eps, ne := fracAtMost(hi.v.frac, hi.strict)
		whole(hi.v.whole, eps, ne)
	}
	return strings.Join(alts, " | ")
}

// fraction is the rule text after an integral part whose fraction's digits D
// are held to a bound: none where D may be empty (eps), and ne, the rule
// text for the nonempty D, after a point. ok is false where neither is.
func fraction(eps bool, ne string) (string, bool) {
	switch {
	case eps && ne != "":
		return ` ( "." ( ` + ne + ` ) )?`, true
	case eps:
		return "", true
	case ne != "":
		return ` "." ( ` + ne + ` )`, true
	}
	return "", false
}

// led is the rule text for the digits beginning with c, the rest held as eps
// and ne say, "" where none are.
func led(c byte, eps bool, ne string) string {
	q := `"` + string(c) + `"`
	switch {
	case eps && ne != "":
		return q + " ( " + ne + " )?"
	case eps:
		return q
	case ne != "":
		return q + " ( " + ne + " )"
	}
	return ""
}

func alternatives(alts ...string) string {
	var out []string
	for _, a := range alts {
		if a != "" {
			out = append(out, a)
		}
	}
	return strings.Join(out, " | ")
}

// fracAtLeast is the digits D of a fraction 0.D from 0.f up, 0.f itself
// excluded where strict: eps where D may be empty, ne the rule text for the
// nonempty D, "" for none.
func fracAtLeast(f string, strict bool) (eps bool, ne string) {
	if f == "" {
		if strict {
			return false, `"0"* [1-9] [0-9]*`
		}
		return true, `[0-9]+`
	}
	above := ""
	if f[0] < '9' {
		above = digitClass(f[0]+1, '9') + " [0-9]*"
	}
	eps, ne = fracAtLeast(f[1:], strict)
	return false, alternatives(above, led(f[0], eps, ne))
}

// fracAtMost is the digits D of a fraction 0.D up to 0.f, 0.f itself
// excluded where strict.
func fracAtMost(f string, strict bool) (eps bool, ne string) {
	if f == "" {
		if strict {
			return false, ""
		}
		return true, `"0"+`
	}
	below := ""
	if f[0] > '0' {
		below = digitClass('0', f[0]-1) + " [0-9]*"
	}
	eps, ne = fracAtMost(f[1:], strict)
	return true, alternatives(below, led(f[0], eps, ne))
}

// fracBetween is the digits D of a fraction 0.D from 0.lf to 0.hf, ls and hs
// excluding each, 0.lf no more than 0.hf: their common first digits, then the
// first they differ in split three ways as sameLength splits it. An empty lf
// is 0, its digits zeros.
func fracBetween(lf string, ls bool, hf string, hs bool) (eps bool, ne string) {
	if hf == "" {
		if ls || hs {
			return false, ""
		}
		return true, `"0"+`
	}
	eps = lf == "" && !ls
	a, lr := byte('0'), ""
	if lf != "" {
		a, lr = lf[0], lf[1:]
	}
	b, hr := hf[0], hf[1:]
	if a == b {
		e, n := fracBetween(lr, ls, hr, hs)
		return eps, led(a, e, n)
	}
	mid := ""
	if a+1 < b {
		mid = digitClass(a+1, b-1) + " [0-9]*"
	}
	e, n := fracAtLeast(lr, ls)
	from := led(a, e, n)
	e, n = fracAtMost(hr, hs)
	return eps, alternatives(from, mid, led(b, e, n))
}

// intRange is the rule text for the integers lo through hi, a nil side
// unbounded, as RFC 8259's int writes them: no leading zero, a minus before
// a negative, and 0 as 0 or -0.
func intRange(lo, hi *big.Int) string {
	var alts []string
	if hi == nil || hi.Sign() >= 0 {
		a := lo
		if a == nil || a.Sign() < 0 {
			a = new(big.Int)
		}
		alts = append(alts, numerals(a, hi))
	}
	if lo == nil || lo.Sign() < 0 {
		a := big.NewInt(1)
		if hi != nil && hi.Sign() < 0 {
			a = new(big.Int).Neg(hi)
		}
		var b *big.Int
		if lo != nil {
			b = new(big.Int).Neg(lo)
		}
		alts = append(alts, `"-" ( `+numerals(a, b)+` )`)
	}
	if (lo == nil || lo.Sign() <= 0) && (hi == nil || hi.Sign() >= 0) {
		alts = append(alts, `"-0"`)
	}
	return "( " + strings.Join(alts, " | ") + " )"
}

// numerals is the rule text for the decimal numerals a through b, 0 <= a <=
// b, b nil for no end, none written with a leading zero: one alternative a
// length, and past a's length any longer numeral where b is nil.
func numerals(a, b *big.Int) string {
	as := a.String()
	if b == nil {
		return sameLength(as, strings.Repeat("9", len(as))) + fmt.Sprintf(" | [1-9] [0-9]{%d,}", len(as))
	}
	bs := b.String()
	var alts []string
	for n := len(as); n <= len(bs); n++ {
		x, y := as, bs
		if n > len(as) {
			x = "1" + strings.Repeat("0", n-1)
		}
		if n < len(bs) {
			y = strings.Repeat("9", n)
		}
		alts = append(alts, sameLength(x, y))
	}
	return strings.Join(alts, " | ")
}

// sameLength is the rule text for the numerals x through y of one length,
// x <= y: their common prefix, then the first digit they differ in split
// three ways -- x's digit followed by x's tail or more, the digits between
// followed by anything, y's digit followed by y's tail or less -- a side
// whose tail is all zeros or all nines joining the middle.
func sameLength(x, y string) string {
	i := 0
	for i < len(x) && x[i] == y[i] {
		i++
	}
	if i == len(x) {
		return `"` + x + `"`
	}
	pre := ""
	if i > 0 {
		pre = `"` + x[:i] + `" `
	}
	r := len(x) - i - 1
	lo, hi := x[i], y[i]
	if r == 0 {
		return pre + digitClass(lo, hi)
	}
	nines, zeros := strings.Repeat("9", r), strings.Repeat("0", r)
	var alts []string
	if x[i+1:] != zeros {
		alts = append(alts, fmt.Sprintf(`"%c" ( %s )`, lo, sameLength(x[i+1:], nines)))
		lo++
	}
	top := ""
	if y[i+1:] != nines {
		top = fmt.Sprintf(`"%c" ( %s )`, hi, sameLength(zeros, y[i+1:]))
		hi--
	}
	if lo <= hi {
		alts = append(alts, fmt.Sprintf("%s [0-9]{%d}", digitClass(lo, hi), r))
	}
	if top != "" {
		alts = append(alts, top)
	}
	if len(alts) == 1 {
		return pre + alts[0]
	}
	return pre + "( " + strings.Join(alts, " | ") + " )"
}

// digitClass is the digits a through b.
func digitClass(a, b byte) string {
	if a == b {
		return `"` + string(a) + `"`
	}
	return fmt.Sprintf("[%c-%c]", a, b)
}

// formatString is the body of a string of format f, between its quotes:
// RFC 3339's full-date, full-time and date-time, which JSON Schema 2020-12
// names date, time and date-time, and RFC 4122's uuid. ABNF's quoted text
// is case-insensitive, so T and Z are written either way (RFC 3339, 5.6). A
// date is a real one: a month's own days, February's 29th in a leap year of
// the Gregorian calendar. ok is false for every other format, which 2020-12
// reads as an annotation.
func (c *converter) formatString(f string) (body string, ok bool) {
	date := func() string {
		day31 := `( "0" [1-9] | [1-2] [0-9] | "3" [01] )`
		day30 := `( "0" [1-9] | [1-2] [0-9] | "30" )`
		day28 := `( "0" [1-9] | "1" [0-9] | "2" [0-8] )`
		leap := `( [0-9] [0-9] ( "0" [48] | [2468] [048] | [13579] [26] ) | ( [02468] [048] | [13579] [26] ) "00" )`
		return c.add("date", fmt.Sprintf(`[0-9]{4} "-" ( ( "0" [13578] | "1" [02] ) "-" %s | ( "0" [469] | "11" ) "-" %s | "02" "-" %s ) | %s "-02-29"`,
			day31, day30, day28, leap))
	}
	clock := func() string {
		h := `( [01] [0-9] | "2" [0-3] )`
		return c.add("time", fmt.Sprintf(`%s ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]+ )? ( [Zz] | [-+] %s ":" [0-5] [0-9] )`, h, h))
	}
	switch f {
	case "date":
		return date(), true
	case "time":
		return clock(), true
	case "date-time":
		return c.add("date-time", date()+` [Tt] `+clock()), true
	case "uuid":
		hex := "[0-9a-fA-F]"
		return c.add("uuid", fmt.Sprintf(`%s{8} "-" %s{4} "-" %s{4} "-" %s{4} "-" %s{12}`, hex, hex, hex, hex, hex)), true
	}
	return "", false
}

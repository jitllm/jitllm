package grammar

import (
	"encoding/json"
	"fmt"
	"math/big"
	"strings"
)

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

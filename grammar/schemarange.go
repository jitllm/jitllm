package grammar

import (
	"encoding/json"
	"fmt"
	"math/big"
	"strings"
)

// intCap is how far an integer range reaches on a side no keyword bounds:
// the sixteen digits the integer primitive writes.
const intCap = 9_999_999_999_999_999

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
// its bounds on each side, intCap's magnitude where it states none. ok is
// false where no integer lies between them.
func intBounds(s map[string]any) (lo, hi *big.Int, ok bool, err error) {
	lo, hi = big.NewInt(-intCap), big.NewInt(intCap)
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
			if ceil.Cmp(lo) > 0 {
				lo = ceil
			}
		case "exclusiveMinimum":
			if b := q.Add(q, one); b.Cmp(lo) > 0 {
				lo = b
			}
		case "maximum":
			if q.Cmp(hi) < 0 {
				hi = q
			}
		case "exclusiveMaximum":
			if b := ceil.Sub(ceil, one); b.Cmp(hi) < 0 {
				hi = b
			}
		}
	}
	return lo, hi, lo.Cmp(hi) <= 0, nil
}

// intRange is GBNF for the integers lo through hi as JSON writes them: no
// leading zero, a minus before a negative.
func intRange(lo, hi *big.Int) string {
	var alts []string
	if hi.Sign() >= 0 {
		a := lo
		if a.Sign() < 0 {
			a = new(big.Int)
		}
		alts = append(alts, numerals(a.String(), hi.String()))
	}
	if lo.Sign() < 0 {
		a := new(big.Int).Neg(hi)
		if a.Sign() <= 0 {
			a = big.NewInt(1)
		}
		alts = append(alts, `"-" ( `+numerals(a.String(), new(big.Int).Neg(lo).String())+` )`)
	}
	return strings.Join(alts, " | ")
}

// numerals is GBNF for the decimal numerals a through b, a <= b, neither
// written with a leading zero: one alternative a length.
func numerals(a, b string) string {
	var alts []string
	for n := len(a); n <= len(b); n++ {
		x, y := a, b
		if n > len(a) {
			x = "1" + strings.Repeat("0", n-1)
		}
		if n < len(b) {
			y = strings.Repeat("9", n)
		}
		alts = append(alts, sameLength(x, y))
	}
	return strings.Join(alts, " | ")
}

// sameLength is GBNF for the numerals x through y of one length, x <= y:
// their common prefix, then the first digit they differ in split three ways
// -- x's digit followed by x's tail or more, the digits between followed by
// anything, y's digit followed by y's tail or less -- a side whose tail is
// all zeros or all nines joining the middle.
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

// formatString is the body of a string of format f, between its quotes, for
// the formats llama.cpp's json-schema-to-grammar builds: date, time,
// date-time and uuid. Its dates are real ones (a month's own days, February
// to the 28th), where llama.cpp lets any month run to the 31st. ok is false
// for every other format, which JSON Schema 2020-12 reads as an annotation.
func (c *converter) formatString(f string) (body string, ok bool) {
	date := func() string {
		day31 := `( "0" [1-9] | [1-2] [0-9] | "3" [01] )`
		day30 := `( "0" [1-9] | [1-2] [0-9] | "30" )`
		day28 := `( "0" [1-9] | "1" [0-9] | "2" [0-8] )`
		return c.add("date", fmt.Sprintf(`[0-9]{4} "-" ( ( "0" [13578] | "1" [02] ) "-" %s | ( "0" [469] | "11" ) "-" %s | "02" "-" %s )`,
			day31, day30, day28))
	}
	clock := func() string {
		h := `( [01] [0-9] | "2" [0-3] )`
		return c.add("time", fmt.Sprintf(`%s ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) %s ":" [0-5] [0-9] )`, h, h))
	}
	switch f {
	case "date":
		return date(), true
	case "time":
		return clock(), true
	case "date-time":
		return c.add("date-time", date()+` "T" `+clock()), true
	case "uuid":
		hex := "[0-9a-fA-F]"
		return c.add("uuid", fmt.Sprintf(`%s{8} "-" %s{4} "-" %s{4} "-" %s{4} "-" %s{12}`, hex, hex, hex, hex, hex)), true
	}
	return "", false
}

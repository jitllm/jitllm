package paritytest

import "strings"

// firstDiff is the first line two kernels' instruction texts part at, with
// the line from each.
func firstDiff(a, b string) string {
	al, bl := strings.Split(a, "\n"), strings.Split(b, "\n")
	for i := 1; i < min(len(al), len(bl)); i++ { // line 0 is the name
		if al[i] != bl[i] {
			return "main:     " + al[i] + "\ncomposer: " + bl[i]
		}
	}
	return "one ends first"
}

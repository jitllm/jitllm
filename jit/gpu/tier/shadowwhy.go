package tier

// shadowWhy names what about the step, or this place in it, keeps a slice
// from being shadowed, "" for nothing.
func shadowWhy(ragged, persist, skips bool, here string) string {
	switch {
	case ragged:
		return "a ragged step"
	case persist:
		return "the persistent block holds its launches"
	case skips:
		return "a bisection probe skips phases"
	}
	return here
}

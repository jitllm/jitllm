package tier

// shadowWhy names what about the step, or this place in it, keeps a slice
// from being shadowed, "" for nothing.
func shadowWhy(R int, ragged, persist, skips bool, here string) string {
	switch {
	case R != 1 || ragged:
		return "not a one-row step"
	case persist:
		return "the persistent block holds its launches"
	case skips:
		return "a bisection probe skips phases"
	}
	return here
}

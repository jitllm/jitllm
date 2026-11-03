package model

// modelError is a sentinel error that is a constant, so a package-level
// sentinel needs no variable. Two are equal when their text is.
type modelError string

func (e modelError) Error() string { return string(e) }

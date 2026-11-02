//go:build !jitllmbench

package kernels

// benchKnob is the bench build's alternative emissions (benchknob.go); here
// every knob is 0, the shipping form, and each branch on one compiles away.
func benchKnob(string) int { return 0 }

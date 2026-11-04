package sched

// seq is the CPU ids 0..n-1, at least one.
func seq(n int) []int {
	if n < 1 {
		n = 1
	}
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

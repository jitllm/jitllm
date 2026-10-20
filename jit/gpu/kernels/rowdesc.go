package kernels

// The words of a paged row's descriptor, in order: what the tier writes for
// each row of a step and a kernel reading a paged history addresses it by.
const (
	PRowTab   = iota // where the row's table starts in the page table
	PRowStart        // first key position attended (window or chunk start)
	PRowEnd          // one past the last key position attended
	PRowWrite        // the position this row's new K/V is written at
	PRowWords
)

package offheap

// ErrNoMemory marks a mapping or commit the host refused for want of memory:
// what was asked does not fit now, which is a limit of the host, not a fault
// of the request or of the engine.
const ErrNoMemory noMemory = "offheap: the host has no memory for the mapping"

type noMemory string

func (e noMemory) Error() string { return string(e) }

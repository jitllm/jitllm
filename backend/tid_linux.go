package backend

import "syscall"

// threadID is the calling OS thread, which is how cudaDev.do tells a call from
// inside a Session (the owner's locked thread) from one outside it.
func threadID() int64 { return int64(syscall.Gettid()) }

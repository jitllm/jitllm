//go:build unix

package main

import (
	"os"
	"runtime"
	"syscall"
	"testing"

	"golang.org/x/sys/unix"
)

// Sending the log to a file must not close descriptors 1 and 2. The *os.File
// that os.Stderr held closes its descriptor when collected, and the runtime
// writes a fatal error's traceback to descriptor 2 -- after the close, to
// whatever file was opened next under that number, or nowhere. Against the
// violation (the old files dropped) descriptor 2 is closed after two
// collections.
func TestTheLogFileLeavesTheStandardDescriptorsOpen(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	// The test's own descriptors, restored after: the test binary's output
	// goes through them.
	saved := make([]int, 3)
	for fd := 1; fd <= 2; fd++ {
		d, err := syscall.Dup(fd)
		if err != nil {
			t.Fatal(err)
		}
		saved[fd] = d
	}
	t.Cleanup(func() {
		for fd := 1; fd <= 2; fd++ {
			if err := unix.Dup2(saved[fd], fd); err != nil {
				t.Error(err)
			}
			syscall.Close(saved[fd])
		}
		os.Stdout, os.Stderr = os.NewFile(1, "/dev/stdout"), os.NewFile(2, "/dev/stderr")
	})

	before := os.Stderr
	var held stdio
	logToFileWithoutAConsole(&held, true)
	if os.Stderr == before {
		t.Fatal("the log did not go to a file")
	}
	// Dropped by the test too, so only what the code keeps holds the old files.
	before = nil
	for range 3 {
		runtime.GC()
	}
	for fd := 1; fd <= 2; fd++ {
		var st syscall.Stat_t
		if err := syscall.Fstat(fd, &st); err != nil {
			t.Errorf("descriptor %d was closed: %v", fd, err)
		}
	}
	runtime.KeepAlive(&held)
}

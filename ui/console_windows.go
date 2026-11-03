package main

import (
	"log"
	"os"
	"syscall"
)

// attachParentConsole points standard output and error at the console of the
// process that started this one, when it has one: jitllm-ui is linked as a GUI
// program (-H=windowsgui), so Windows gives it no console, and run from cmd or
// PowerShell its output would otherwise vanish. Started from Explorer there is
// no parent console and it reports false.
func attachParentConsole(held *stdio) bool {
	const attachParentProcess = ^uintptr(0) // ATTACH_PARENT_PROCESS, (DWORD)-1
	attach := syscall.NewLazyDLL("kernel32.dll").NewProc("AttachConsole")
	if r, _, _ := attach.Call(attachParentProcess); r == 0 {
		return false
	}
	con, err := os.OpenFile("CONOUT$", os.O_WRONLY, 0)
	if err != nil {
		return false
	}
	held.replace(con)
	log.SetOutput(con) // package log took the old os.Stderr at init
	return true
}

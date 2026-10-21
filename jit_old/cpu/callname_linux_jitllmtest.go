//go:build jitllmtest && linux

package cpu

func codeName(c *Code) string { return c.Name }

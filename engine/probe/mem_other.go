//go:build !linux

package probe

func hugeBuffer(n int) ([]byte, func(), error) {
	return make([]byte, n), func() {}, nil
}

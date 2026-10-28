package backend

import "fmt"

// View is n bytes of b from byte off: the same device memory, bound where a
// launch binds a buffer, at a pointer plus the offset on CUDA, a
// descriptor's offset on Vulkan (a multiple of the device's
// minStorageBufferOffsetAlignment, an offset that is not refused by name)
// and setBuffer's offset on Metal. Its Write, WriteAt and Read address the
// view's bytes. b owns the memory and must outlive the view, whose Free
// leaves it. A backend that binds no offset refuses by name.
func View(b Buf, off, n int) (Buf, error) {
	switch v := b.(type) {
	case *cudaBuf:
		w, err := v.b.View(off, n)
		if err != nil {
			return nil, err
		}
		return &cudaBuf{b: w, dev: v.dev}, nil
	case *vkBuf:
		w, err := v.b.View(off, n)
		if err != nil {
			return nil, err
		}
		return &vkBuf{b: w}, nil
	case *mtlBuf:
		w, err := v.b.View(off, n)
		if err != nil {
			return nil, err
		}
		return &mtlBuf{b: w, c: v.c}, nil
	}
	return nil, fmt.Errorf("backend: a %T bound at an offset: views are not implemented on its backend", b)
}

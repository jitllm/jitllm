package jlm

import "fmt"

// PageOf is the text page block li of the text's block space reads its
// weights from: WeightsOf's, or its own.
func (c *Config) PageOf(li int) int {
	if li >= 0 && li < len(c.WeightsOf) {
		return int(c.WeightsOf[li])
	}
	return li
}

// weightsOfValid refuses a statement of the pages blocks read that is not
// one per block, or where a page is first read by a block other than the
// block of its own index past the pages before it: the pages are the
// blocks' in order, each taken by the first block reading it, so a page
// no block reads, or one read only ahead of the page before it, is a
// statement of nothing.
func (c *Config) weightsOfValid() error {
	if len(c.WeightsOf) == 0 {
		return nil
	}
	if blocks, _ := c.statedBlocks(); len(c.WeightsOf) != blocks {
		return fmt.Errorf("jlm: %d blocks' weight pages for %d blocks", len(c.WeightsOf), blocks)
	}
	next := uint32(0)
	for li, p := range c.WeightsOf {
		switch {
		case p == next:
			next++
		case p > next:
			return fmt.Errorf("jlm: block %d reads page %d before any block reads page %d", li, p, next)
		}
	}
	return nil
}

// TextPages is how many pages of the text page array the blocks read
// (NBlocks), and TextSpace how many blocks the text's block space has:
// more than its pages where blocks share them (Config.WeightsOf). The
// tower's blocks follow the text's in the block space.
func (f *File) TextPages() int { return int(f.H.NBlocks) }
func (f *File) TextSpace() int {
	if n := len(f.cfg.WeightsOf); n != 0 {
		return n
	}
	return int(f.H.NBlocks)
}

// PageOf is the page block li of the block space reads its weights from: a
// text block's (Config.PageOf), the tower's own page past the text's.
func (f *File) PageOf(li int) int {
	if n := f.TextSpace(); li >= n {
		return int(f.H.NBlocks) + li - n
	}
	return f.cfg.PageOf(li)
}

// weightPagesValid holds the pages the blocks read to the page array.
func (f *File) weightPagesValid() error {
	if len(f.cfg.WeightsOf) == 0 {
		return nil
	}
	n := uint32(0)
	for _, p := range f.cfg.WeightsOf {
		n = max(n, p+1)
	}
	if n != f.H.NBlocks {
		return fmt.Errorf("jlm: the blocks read %d pages of a page array of %d", n, f.H.NBlocks)
	}
	return nil
}

// windowsValid refuses blocks' windows that are not one a block, or that
// sit beside a period stating them too, or one wider than the model's
// widest window.
func (c *Config) windowsValid() error {
	if len(c.Windows) == 0 {
		return nil
	}
	if len(c.Windows) != int(c.NLayer) {
		return fmt.Errorf("jlm: %d blocks' windows for %d blocks", len(c.Windows), c.NLayer)
	}
	if c.SWAPeriod != 0 {
		return fmt.Errorf("jlm: the blocks' windows stated beside a period of %d", c.SWAPeriod)
	}
	for b, w := range c.Windows {
		if w > c.SWAWindow {
			return fmt.Errorf("jlm: block %d's window of %d is past the model's %d", b, w, c.SWAWindow)
		}
	}
	return nil
}

package tier

// EmbedRows is nn.EmbedDevice's gather of token rows on the card. The
// composer forms no gather yet, so the host looks the rows up.
func (g *devTier) EmbedRows(ids []int32, dst []float32) bool { return false }

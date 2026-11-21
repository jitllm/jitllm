package composer

// PooledKeep is pooledOracle for a gate outside the package: the positions
// a pooled selection of blocks of b, k kept, keeps over n positions of keys,
// departing from the definition by fault (a pooledFault).
func PooledKeep(b, k, nrot int, neox bool, hi, di, off int, keys [][]float32, iq, w []float32, n, fault int) []bool {
	return pooledOracle(pooledCase{Hi: hi, Di: di, Off: off, B: b, K: k, NRot: nrot, Neox: neox}, keys, iq, w, n, pooledFault(fault))
}

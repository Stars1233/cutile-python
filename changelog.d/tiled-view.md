- Add `Array.tiled_view(tile_shape, padding_mode=...)` to create a tiled view of an array
  with a fixed tile shape and padding mode.
- The `TiledView` object exposes properties `dtype`, `tile_shape`.
  It supports `num_tiles`, `load` and `store` methods for tile access.

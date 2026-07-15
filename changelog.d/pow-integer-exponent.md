- On TileIR 13.4+, lowered ``x ** y`` (and ``ct.pow(x, y)``) to ``FPowI`` when
  ``x`` is an unrestricted float and ``y`` is an integer whose value range fits
  in a signed 32-bit int, keeping the exponent an integer instead of promoting
  it to float. Otherwise and on old targets, it promotes the exponent to float
  and emits ``FPowF``.

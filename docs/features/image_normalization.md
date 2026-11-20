# Device image normalization

When `--mm-device-do-normalize` is enabled, supported vision models send raw
`uint8` patches to the HPU. `HPUFusedMMInputNorm` normalizes those patches in
FP32 and casts the result to the vision tower's dtype.

The HPU operation preserves the Torchvision image processor's order of
operations: rescaling is folded into the channel mean and standard deviation,
then each pixel is normalized by subtraction and division. Using a single
affine multiply/add instead can change BF16 rounding near the channel mean,
which can change vision embeddings and generated tokens.

Floating-point patch inputs retain the upstream native affine behavior.
`--no-mm-device-do-normalize` continues to select processor-side normalization.

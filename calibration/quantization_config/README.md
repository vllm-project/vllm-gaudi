# Supported JSON Config File Options

The following table summarizes the options for the JSON config file:

| Attribute            | Description | Values |
|----------------------|-------------|--------|
| **Mode**             | The mode to run INC with. | - **MEASURE** – Measure statistics of all modules and emit the results to `dump_stats_path`.<br>- **QUANTIZE** *(default)* – Quantize and run the model according to the provided measurements. |
| **Observer**         | The observer to measure the statistics. | - **maxabs** *(default)*<br>- **save** – Saves all tensors to files. |
| **Allowlist**        | List of `nn.Module` names or types to quantize. Empty list means all supported modules are quantized by default. See *supported-modules*. | Default: empty list |
| **Blocklist**        | List of `nn.Module` names or types **not** to quantize. | Default: empty list |
| **dump_stats_path**  | Path to save and load measurements. Directory structure is created up to the last `/`; the string after the last `/` is used as a prefix for measurement files. | Default: `stats` |
| **scale_method**     | Method for calculating the scale from measurements. | - `unit_scale` *(default)* – Always use scale of 1.<br>- `maxabs_arbitrary` – Stretch/compress maxabs to full-scale of FP8.<br>- `maxabs_hw` – Stretch/compress maxabs to full-scale of FP8, then replace with HW-accelerated scale based on `device_for_scales`.<br>- `maxabs_pow2` – Same as above but rounded to power of 2.<br>- `maxabs_hw_opt_weight` – Weight scale chosen for minimal MSE among HW accelerated scales; activations use `maxabs_hw`.<br>- `act_maxabs_pow2_weights_pcs_opt_pow2` – Per-channel weights use `maxabs_hw_opt_weight`; activations use `maxabs_pow2`.<br>- `act_maxabs_hw_weights_pcs_maxabs_pow2` – Per-channel weights use `maxabs_pow2`; activations use `maxabs_hw`.<br>- `act_maxabs_pcs_pow2_weight_maxabs_pts_pow2_hw` – **Dynamic quant only**: per-tensor weights use `maxabs_hw`; activations use per-token `maxabs_pow2`. |
| **measure_exclude**  | Tensor types to exclude from measurement. | - `NONE` – Measure all tensors.<br>- `OUTPUT` *(default)* – Skip output tensors. |
| **scale_format**     | Format of scales passed to custom PyTorch ops. | - `const` – Scales passed as tensors.<br>- `scalar` *(default)* – Scales passed as scalar values for compile-time & throughput optimizations. |
| **device_for_scales**| Exponent-bias values for converting FP32/BF16 to FP8-143. | - `GAUDI3` – Expanded exponent-bias range (0–63).<br>- `GAUDI2` – 4 possible exponent biases (3, 7, 11, 15), default is 7. |
| **dynamic_quantization** | Enables dynamic FP8 quantization with per-token scales. Only supported with `act_maxabs_pcs_pow2_weight_maxabs_pts_pow2_hw`. | - `true` – Enable.<br>- `false` *(default)* – Disable. |

---

## Configuring Backoff Factors

Maxabs-based scaling methods support backoff factors `input_backoff` and `weight_backoff` to leave margin when converting inputs and weights to FP8.

For example, if an activation has a larger absolute value than observed in calibration, the maxabs value is scaled to:

```
input_backoff * FP8_143_FULLSCALE
```

Similarly, for weights:

```
weight_backoff * FP8_143_FULLSCALE
```

Defaults:
- `input_backoff = 0.25`
- `weight_backoff = 0.5`

To change these values, add the following to the quantization configuration JSON file:

```json
"scale_params": {"input_backoff": <INPUT_BACKOFF>, "weight_backoff": <WEIGHT_BACKOFF>}
```

---

## Compile Time and Throughput Optimization

Setting `"scale_format": "scalar"` enables:

- Faster compile time for FP8 inference by reducing the number of compiled recipes.
- Less host-side overhead when launching FP8 ops, improving throughput in host-bound cases (e.g., small batch sizes).

> **Note:**
> - Compile time improvement depends on model properties such as recipe count and scale distribution.
> - Not applicable to PCQ.

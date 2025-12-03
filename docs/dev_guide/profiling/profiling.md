# Profiling

There are a few approaches available for measuring and analyzing vLLM performance, each offering different levels of detail and suited to specific use cases. This document outlines these approaches to help evaluate execution time, identify performance bottlenecks, and analyze both host and device behavior during inference.
The following table lists the available methods of collecting performance traces. Each linked method is described in detail in a separate section.

| Profiling method                                     | Category                  | Detail level  | Use case                                                                                         |
|------------------------------------------------------|---------------------------|----------------|---------------------------------------------------------------------------------------------------|
| [End-to-end profiling](e2e-profiling.md)                      | Comprehensive profiling   | High           | Capturing all profiling data across host, Python, and device.                                    |
| [High-level profiling](high-level-profiling.md)        | High-level profiling      | Low            | Debugging prompt/decode structure, batch sizes, and scheduling patterns.                         |
| [PyTorch profiling via asynchronous server](pytorch-profiling-async.md) | Server-based profiling    | Medium         | Measuring latency, host gaps, and server response timing.                                        |
| [PyTorch profiling via script](pytorch-profiling-script.md)  | Script-based profiling    | Medium         | Profiling within test scripts.                                                                   |
| [Profiling specific prompt or decode execution](profiling-prompt-decode.md) | Device-level profiling | Medium/High    | Capturing a general execution flow without graph details (no shapes, ops). Optionally, analyzing fused ops, node names, graph structures, and timing. |

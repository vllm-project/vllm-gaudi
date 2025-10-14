# CI Failures

## CI

For all PRs that are created in vllm-gaudi repository all checks in CI are required:
- pre-commit & DCO
- HPU tests
- HPU Gaudi tests

### Pre-commit & DCO
To install run:

```pre-commit install```

This way all of your commits should be correctly formated and signed-off. If you need to manually sign off your commits, remember to use ```git commit -s``` to pass DCO.

### HPU tests
HPU tests consist of several unit tests:
- pre merge tests
- unit tests
- perf test
- feature tests
- e2e tests

All of the above tests are mandatory. Those tests operate in fast fail mode, meaning if one test fails, all of the others won't be triggered.

### HPU Gaudi tests
Additional Gaudi tests are expectd to pass, but aren't mandatory. Those tests are being run on internal Jenkins system, so results are internal only. Those tests can be run by CODEOWNERs and TESTOWNERs only.

## Docs Pull Requests
All PRs that do not interfere in code, like docstring changes or README updates can be merged without HPU tests and Gaudi tests. It is still required to pass pre-commit check.

## Hourly Checks and Tests
On vllm-gaudi repository hourly tests can be found in ```Hourly Commit Check and Tests``` under ```Actions``` tab. This tab also allows developers to manually trigger hourly tests on selected branch.

If the last hourly test is failing it means that vllm-gaudi main branch doesn't work with upstream newest main commit. To find last good commit check [last good commit](https://github.com/vllm-project/vllm-gaudi/blob/vllm/last-good-commit-for-vllm-gaudi/VLLM_STABLE_COMMIT).

Failing hourly checks will be fixed by developers as soon as possible.

## Troubleshooting
### Unreleated failures
Sometimes there may be some issues that are unreleated to your specific changes in code. Often causeb by connection problems. In this case failed checks should be reruned. Those errors are:
- ```Error response from daemon: No such container```
- ```ValueError: Unsupported device: the device type is 7.```
- ```[Device not found] Device acquire failed.```

### Accuracy and functionality issues
Accuracy issues can be tracked in HPU Gaudi tests with gsm8k runs. If any check fails with accuracy - too low accuracy compare to the one measured, or functionality issues, the **PR can't be merged** until solved.

### Pre-commit failures
To run pre-commit test manually run:

```pre-commit run --show-diff-on-failure --color=always --all-files --hook-stage manual```

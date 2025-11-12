# CI Failures

If you want to contribute to the vLLM Hardware Plugin for Intel® Gaudi® project, familiarize with our process to understand how automated testing and integration are handled.

## CI Validation

Pull requests created in the [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) repository require the following CI checks:

- Pre-commit and DCO
- HPU tests
- HPU Intel® Gaudi® tests

### Pre-Commit and DCO

We use the `pre-commit` hook to ensure each commit meets certain standards before it is added to the repository. To install it, run this command:

```bash
pre-commit install
```

This ensures that all of your commits are correctly formatted and signed off. If you need to manually sign off your commits, use `git commit -s` to pass DCO.

### HPU Tests

HPU tests consist of the following tests:

- Pre-merge tests
- Unit tests
- Performance test
- Feature tests
- E2E tests

All of the these tests are mandatory and operate in fast fail mode, meaning if one test fails, next tests will not be triggered.

### HPU Intel® Gaudi® Tests

Additional Intel® Gaudi® tests are expected to pass, but are not mandatory. Only code owners and test owners can run these tests on the internal Jenkins system. The results are internal.

## Documentation Validation

All pull requests that do not interfere with code, such as documentation changes or readme updates, can be merged without HPU tests and Intel® Gaudi® tests. However, they need to pass the pre-commit check.

## Hourly Checks and Tests
In the [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) repository hourly tests are available in `Hourly Commit Check and Tests` under the `Actions` tab. This tab also allows developers to manually trigger hourly tests on a selected branch.

If the last hourly test fails, it indicates that the `main` branch is not compatible with the latest commit from the upstream `main` branch. To find the last good commit, see [last good commit](https://github.com/vllm-project/vllm-gaudi/tree/vllm/last-good-commit-for-vllm-gaudi/VLLM_STABLE_COMMIT). Failing hourly checks are fixed by developers as soon as possible.

## Troubleshooting

Follow these instructions to fix common CI failure issues.

### Unrelated Failures

Sometimes you may see issues that are unrelated to your code changes, often caused by temporary connection problems. These issues trigger errors, such as the following:

`Error response from daemon: No such container`

`ValueError: Unsupported device: the device type is 7.`

`[Device not found] Device acquire failed.`

If you see such errors, simply rerun the failed checks.

### Accuracy and Functionality Issues

Accuracy issues can be tracked in HPU Intel® Gaudi® tests with the `gsm8k` runs. If any functionality or accuracy issues are detected in a pull request, such as too low accuracy in comparison to the one measured, it can not be merged until the issue is resolved.

### Pre-Commit Failures

To manually run pre-commit test, use the following command:

```bash
pre-commit run --show-diff-on-failure --color=always --all-files --hook-stage manual
```

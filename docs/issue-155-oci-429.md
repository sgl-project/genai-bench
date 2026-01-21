# OCI 429 Metrics Drop (Issue 155) - Handoff README

Purpose
This doc summarizes the 429 metrics drop issue for OCI users and outlines a
minimal fix so another agent can implement it without repeating investigation.

Problem Summary
When OCI backends return HTTP errors (especially 429), the benchmark does not
record the failure metrics. The request exceptions are converted to a
UserResponse but the error response never goes through BaseUser.collect_metrics,
so the aggregated metrics and Locust stats under-report failures.

Symptoms
- Runs against OCI backends show warning logs or ServiceError exceptions.
- Aggregated metrics show fewer errors than expected (429s missing).
- In a simple repro, OCICohereUser/OCIGenAIUser return status_code=500 on
  ServiceError and do not emit metrics events.

Root Cause
OCI SDK raises ServiceError for non-2xx responses. In both OCI user classes,
send_request wraps the request in try/except, but the except path returns a
UserResponse without calling collect_metrics.

Investigation Notes
- OCI SDK raises ServiceError before code reaches the response.status branch.
  (See oci/base_client.py: request is checked and raise_service_error() is
  called on non-2xx.)
- OCICohereUser.send_request and OCIGenAIUser.send_request catch Exception and
  return UserResponse without collect_metrics.
- Other users (OpenAI, Together, Azure, GCP, AWS) always call collect_metrics.
  Cohere (non-OCI) does record failures but maps all RequestException errors to
  status_code=500, which is a separate accuracy issue.

Repro (Unit-Level, No OCI Server Needed)
Use ServiceError to simulate a 429 in OCICohereUser:

```python
from oci.exceptions import ServiceError
from genai_bench.user.oci_cohere_user import OCICohereUser

class DummyUser(OCICohereUser):
    def collect_metrics(self, *args, **kwargs):
        print("collect_metrics called")

user = DummyUser.__new__(DummyUser)
def raise_429():
    raise ServiceError(429, "rate limit", "Rate limit", "dummy", None)

resp = user.send_request(
    make_request=raise_429,
    endpoint="chat",
    payload=None,
    parse_strategy=None,
    num_prefill_tokens=0,
)
print("status_code:", resp.status_code)
```

Expected (current behavior): no "collect_metrics called", status_code=500.

Implemented Fix
The exception handler now calls collect_metrics and preserves the real status
code when available using getattr().

Approach:
1) Single Exception handler catches all exceptions (including ServiceError)
2) Use getattr(e, "status", 500) to extract status code from ServiceError objects
3) Build UserResponse with the extracted status code and error_message=str(e)
4) Call collect_metrics(metrics_response, endpoint) before returning

Implementation:
```python
try:
    ...
except Exception as e:
    metrics_response = UserResponse(
        status_code=getattr(e, "status", 500),
        error_message=str(e),
        num_prefill_tokens=num_prefill_tokens or 0,
    )
    self.collect_metrics(metrics_response, endpoint)
    return metrics_response
```

This approach is cleaner than having separate ServiceError and Exception handlers
since getattr() safely handles both cases:
- ServiceError objects have .status attribute → returns actual status (e.g., 429)
- Other exceptions don't have .status → returns default 500

Involved Code Files
- genai_bench/user/oci_cohere_user.py
- genai_bench/user/oci_genai_user.py

Implemented Tests
Added comprehensive unit tests that verify exception handling and metrics collection.

Files updated:
- tests/user/test_oci_cohere_user.py - Added test_send_request_collects_metrics_on_exception
- tests/user/test_oci_genai_user.py - Added test_send_request_collects_metrics_on_exception

Test coverage:
1. ServiceError with status attribute → verifies status_code=429 is preserved
2. Generic Exception without status → verifies status_code=500 default is used
3. Both cases verify collect_metrics is called with proper UserResponse

All tests pass (34 tests across both files)

Optional Follow-Ups (Not Required for Fix)
- Cohere (non-OCI) maps HTTP errors to status_code=500 due to raise_for_status.
  Consider preserving the actual HTTP status in the future for accuracy.


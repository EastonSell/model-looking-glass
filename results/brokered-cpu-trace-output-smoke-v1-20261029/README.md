# Two-client CPU trace and same-forward output smoke

This bounded development smoke exercised the opt-in CPU brokered trace and
same-forward output delivery added in
[PR #634](https://github.com/EastonSell/model-looking-glass/pull/634), merged
at `7ad7196e4704cfa0e367fd8635dd20dc09412238`. It used the pinned
`distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`
and two already observed inputs, `[50256, 103, 203, 303]` and
`[50256, 104, 204, 304]`. The harness calls the loader once and supplies one
resident model object to a serial broker; the saved evidence is not an
independently instrumented count of all model loads.

Each of two distinct clients received the full `[1, 4, 50257]` float32 logits
from its traced forward, along with layer-0 attention and MLP captures at
token position 3. The four 3,072-byte trace vectors and the two 804,112-byte
logit outputs each match their corresponding direct-hook reference bytes.
Those two reference forwards are separate model invocations. The output and
trace receipts bind each client's team, request, job, generation, transfer and
run identities. This evidence checks one small serial workload.

The owner reported both clients exited zero, the second queued behind the
first, the server and service closed, and cooperative host claims drained. A
separately retained direct-process handle recorded owner exit zero; the public
observation binds the hashes of the private original process records while
omitting their paths and operating-system process IDs. The independently
reviewed saved-byte report is `independent-review.json`. Run
`python -B audit_public.py .` to check this public projection; the supplied
`independent-reader.py` also checks private originals when they are available.

Source hashes are file readbacks around a trusted, quiescent checkout; they do
not attest the interpreter or dependencies that ran. Process and queue facts
are cooperative local observations and do not establish cleanup of an entire
process tree. This is not a GPU, batching, throughput, or general capacity
result. The earlier trace-only smoke and its incomplete first attempt are
documented separately in
[PR #647](https://github.com/EastonSell/model-looking-glass/pull/647).


The complete public projection is stored as base64 text parts for transport.
Run `python verify_archive.py` to verify the parts and archive, or
`python verify_archive.py --extract <new-directory>` to reconstruct them.
Then run `python <new-directory>/audit_public.py <new-directory>` to check the
public trace, same-forward output and reference bytes. The index binds every part
and archive member.

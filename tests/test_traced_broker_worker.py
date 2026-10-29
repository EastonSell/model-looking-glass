"""CPU-only, fail-closed trace-plan binding for one broker job."""
import unittest

import torch
from torch import nn

from looking_glass.broker import Broker
from looking_glass.capture_dispatcher import CaptureDispatcher
from looking_glass.resident_worker import InvalidRequest, ResidentWorker
from looking_glass.trace_schema import TraceEvent, validate_trace
from looking_glass.traced_broker_worker import (
    BoundTracePlan,
    BrokerJobIdentity,
    SampleTrace,
    SiteTrace,
    TraceLimits,
    TraceSpec,
    bind_trace_plan,
)


class TracePlanTests(unittest.TestCase):
    def setUp(self):
        self.identity = BrokerJobIdentity("a" * 32, 7, "team-a", "client-request")
        self.samples = (SampleTrace("sample-0", (0, 2)), SampleTrace("sample-1", (1,)))
        self.sites = (SiteTrace(0, "attn"), SiteTrace(1, "mlp"))
        self.spec = TraceSpec("run-a", self.samples, self.sites, 4, 12, 7, 8, 256)
        self.limits = TraceLimits(4, 8, 3, 4, 16, 1024, 8)

    def bind(self, identity=None, spec=None, limits=None, **kwargs):
        return bind_trace_plan(identity or self.identity, spec or self.spec,
                               limits or self.limits, batch_size=2,
                               sequence_length=3, layer_count=2, **kwargs)

    def test_one_job_owns_all_rows_tokens_and_schema_identities(self):
        plan = self.bind()
        self.assertEqual(plan.worker_request_id, "lg-" + "a" * 32 + "-7")
        self.assertEqual(plan.identity, self.identity)
        self.assertEqual(plan.sites, self.sites)
        request = plan.capture_request()
        dispatcher = CaptureDispatcher((request,), batch_size=2, sequence_length=3,
                                       max_events=8, max_trace_bytes=256,
                                       sequence_starts=plan.sequence_starts())
        source = torch.tensor([[[11.], [12.], [13.]], [[21.], [22.], [23.]]])
        def forward():
            dispatcher.record("transformer.h.0.attn", layer_index=0,
                              module_kind="attn", output=source, output_path=(0,))
        _, events = dispatcher.run(forward)
        self.assertEqual([e.job_id for e in events], [plan.worker_request_id] * 2)
        self.assertEqual([e.request_id for e in events], ["client-request"] * 2)
        self.assertEqual([e.run_id for e in events], ["run-a"] * 2)
        self.assertEqual([e.sample_id for e in events], ["sample-0", "sample-1"])
        self.assertEqual([e.batch_position for e in events], [0, 1])
        self.assertEqual([e.sequence for e in events], [12, 13])
        self.assertEqual([e.to_trace_event().identity.job_id for e in events],
                         [plan.worker_request_id] * 2)
        round_trip = [TraceEvent.from_json(e.to_trace_event().to_json()) for e in events]
        self.assertEqual([e.generation.generation_step for e in round_trip], [7, 7])
        self.assertEqual([e.generation.stage for e in round_trip], ["unknown"] * 2)
        self.assertEqual([e.generation.cache_position for e in round_trip], [None] * 2)
        self.assertEqual([e.output_path for e in round_trip], [(0,)] * 2)
        self.assertEqual([e.forward_index for e in round_trip], [4, 4])
        self.assertEqual([e.token_positions for e in round_trip], [(0, 2), (1,)])
        self.assertTrue(torch.equal(events[0].value, torch.tensor([[11.], [13.]])))
        self.assertTrue(torch.equal(events[1].value, torch.tensor([[22.]])))

    def test_mutating_original_specs_or_returned_copies_cannot_change_plan(self):
        plan = self.bind()
        object.__setattr__(self.identity, "request_id", "forged")
        object.__setattr__(self.samples[0], "sample_id", "forged-sample")
        object.__setattr__(self.samples[0], "token_positions", (1,))
        object.__setattr__(self.sites[0], "module_kind", "mlp")
        object.__setattr__(self.spec, "run_id", "forged-run")
        object.__setattr__(self.limits, "max_trace_bytes", 1)
        exposed = plan.capture_request()
        object.__setattr__(exposed.samples[0], "batch_position", 1)
        starts = plan.sequence_starts()
        starts["run-a"] = 90
        fresh = plan.capture_request()
        self.assertEqual(plan.identity.request_id, "client-request")
        self.assertEqual((fresh.run_id, fresh.job_id, fresh.request_id),
                         ("run-a", "lg-" + "a" * 32 + "-7", "client-request"))
        self.assertEqual([(s.batch_position, s.sample_id, s.token_positions) for s in fresh.samples],
                         [(0, "sample-0", (0, 2)), (1, "sample-1", (1,))])
        self.assertEqual(plan.sites, (SiteTrace(0, "attn"), SiteTrace(1, "mlp")))
        self.assertEqual(plan.sequence_starts(), {"run-a": 12})

    def test_plan_cannot_be_built_or_mutated_without_validation(self):
        with self.assertRaises(TypeError):
            BoundTracePlan(("a" * 32, 7, "team", "request"), "run", (), (),
                           0, 0, None, 1, 1)
        plan = self.bind()
        with self.assertRaises(AttributeError):
            plan._identity = ("b" * 32, 8, "other", "other")

    def test_broker_scoped_job_ids_do_not_allocate_cross_plan_sequences(self):
        first = self.bind()
        second = self.bind(identity=BrokerJobIdentity("b" * 32, 7, "team-a", "client-request"))
        self.assertNotEqual(first.capture_request().job_id, second.capture_request().job_id)
        self.assertEqual(first.capture_request().request_id, second.capture_request().request_id)
        # Distinct job IDs do not make the schema's (run_id, sequence) unique.
        self.assertEqual(first.sequence_starts(), second.sequence_starts())
        def first_event(plan):
            dispatcher = CaptureDispatcher((plan.capture_request(),), batch_size=2,
                                           sequence_length=3, max_events=8,
                                           max_trace_bytes=256,
                                           sequence_starts=plan.sequence_starts())
            def forward():
                dispatcher.record("transformer.h.0.attn", layer_index=0,
                                  module_kind="attn", output=torch.ones(2, 3, 1),
                                  output_path=(0,))
            _, events = dispatcher.run(forward)
            return events[0].to_trace_event()
        with self.assertRaisesRegex(ValueError, "duplicate trace event identity"):
            validate_trace([first_event(first), first_event(second)])

    def test_oversized_selections_are_rejected_before_inspection(self):
        invalid_samples = (object(),) * 1000
        invalid_sites = (object(),) * 1000
        for samples, sites in ((invalid_samples, self.sites), (self.samples, invalid_sites)):
            spec = TraceSpec("run", samples, sites, 0, 0, None, 8, 256)
            with self.assertRaisesRegex(ValueError, "every row requires a sample"):
                self.bind(spec=spec)

    def test_invalid_identity_selections_and_caps_fail_before_execution(self):
        bad_identities = [BrokerJobIdentity("x" * 32, 7, "team-a", "request"),
                          BrokerJobIdentity("a" * 32, True, "team-a", "request"),
                          BrokerJobIdentity("a" * 32, 7, "", "request")]
        for identity in bad_identities:
            with self.subTest(identity=identity):
                with self.assertRaises(ValueError):
                    self.bind(identity=identity)
        bad_samples = [
            (SampleTrace("same", (0,)), SampleTrace("same", (1,))),
            (SampleTrace("a", (-1,)), SampleTrace("b", (1,))),
            (SampleTrace("a", (True,)), SampleTrace("b", (1,))),
            (SampleTrace("a", (0, 0)), SampleTrace("b", (1,))),
            (SampleTrace("a", (3,)), SampleTrace("b", (1,))),
            (SampleTrace("a", (0,)),),
        ]
        bad_sites = [(SiteTrace(0, "attn"), SiteTrace(0, "attn")),
                     (SiteTrace(2, "attn"),), (SiteTrace(0, "arbitrary"),)]
        specs = [TraceSpec("run-a", samples, self.sites, 4, 12, None, 8, 256)
                 for samples in bad_samples]
        specs += [TraceSpec("run-a", self.samples, sites, 4, 12, None, 8, 256)
                  for sites in bad_sites]
        specs += [TraceSpec("run-a", self.samples, self.sites, 4, 12, None, 3, 256),
                  TraceSpec("run-a", self.samples, self.sites, 4, 12, None, 8, 1025)]
        for spec in specs:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    self.bind(spec=spec)

    def test_gpu_and_ipc_are_rejected_at_plan_boundary(self):
        with self.assertRaises(ValueError):
            self.bind(device="cuda:0")
        with self.assertRaises(ValueError):
            self.bind(transport="ipc")

    def test_current_forward_only_broker_rejects_trace_plan(self):
        plan = self.bind()
        worker = ResidentWorker("fixture", nn.Identity().eval(), max_batch_size=2,
                                max_sequence_length=3)
        with Broker(worker, max_pending=1, max_pending_per_team=1) as broker:
            with self.assertRaises((InvalidRequest, TypeError, ValueError)):
                broker.submit("team-a", plan)


if __name__ == "__main__":
    unittest.main()

"""Actual training counters and phase clocks, tested with tiny CPU fixtures only."""
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
resume=load('resume_for_metrics','tests/test_laya_kit_resume.py')
kit=resume.kit

@unittest.skipUnless(resume.HAVE_TORCH_STACK,'local Torch/Laya stack unavailable')
class TrainingMeasurements(unittest.TestCase):
    def setUp(self):
        self.ns=resume.fresh_script_ns();self.ns['TRAIN_METHOD']='supervised'
    def loop(self,**extra):
        torch=resume.torch; model=resume.TinyDecisionModel(); items=resume.make_items(5,seed=4)
        opt=torch.optim.SGD(model.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        clock=types.SimpleNamespace(value=0.)
        observed={'tokens':0,'sequences':0}
        def forward(*args):
            clock.value+=2
            observed['tokens']+=int(args[1].sum());observed['sequences']+=len(args[0])
            return model(*args)
        kwargs=dict(device=torch.device('cpu'),epochs=2,micro_batch=2,grad_accum=2,group_size=4,sigma_start=.4,sigma_end=.1,
            rank=0,world_size=1,use_scaler=False,scaler=None,autocast_device='cpu',autocast_dtype=None,autocast_enabled=False,
            max_steps=0,mem_log_fn=None,log_prefix='fixture')
        def callback(seconds):
            def run(*args):clock.value+=seconds
            return run
        kwargs.update(epoch_end_fn=callback(7),checkpoint_fn=callback(11));kwargs.update(extra)
        with mock.patch.object(self.ns['time'],'perf_counter',lambda:clock.value),mock.patch('builtins.print'):
            result=self.ns['run_training_loop'](forward,list(items),list(model.parameters()),opt,sched,types.SimpleNamespace(pad_token_id=0),**kwargs)
        return result,items,observed
    def test_counts_use_unpadded_sequences_and_exclude_measured_callbacks_from_rates(self):
        result,items,observed=self.loop()
        m=result['measurements']
        self.assertEqual(m['processed_sequences'],10)
        self.assertEqual(m['processed_nonpadding_tokens'],sum(len(x['ids']) for x in items)*2)
        self.assertEqual(m['processed_nonpadding_tokens'],observed['tokens'])
        self.assertEqual(m['training_elapsed_s'],12)
        self.assertEqual(m['loop_elapsed_s'],48)
        self.assertEqual(m['dev_callback_s'],14);self.assertEqual(m['checkpoint_write_s'],22)
        self.assertEqual(m['tokens_per_second'],observed['tokens']/12)
        self.assertEqual(m['sequences_per_second'],10/12)
        self.assertEqual(m['scope'],'current_invocation_rank_local_training_loop')
    def test_partial_leg_and_rdrop_count_actual_presentations_without_global_multiplier(self):
        result,_,observed=self.loop(max_steps=2,rdrop_alpha=.2,rank=1,world_size=2)
        m=result['measurements']
        self.assertTrue(result['stopped_early']);self.assertEqual(m['processed_sequences'],4)
        self.assertEqual(m['forward_nonpadding_token_presentations'],observed['tokens'])
        self.assertEqual(m['processed_nonpadding_tokens']*2,observed['tokens'])
        self.assertEqual(m['rank'],1);self.assertEqual(m['world_size'],2)
    def test_resumed_leg_does_not_count_prior_epoch_examples(self):
        result,items,observed=self.loop(start_epoch=1,start_global_step=3)
        m=result['measurements']
        self.assertEqual(m['processed_sequences'],5)
        self.assertEqual(m['processed_nonpadding_tokens'],sum(len(x['ids']) for x in items))
        self.assertEqual(m['processed_nonpadding_tokens'],observed['tokens'])
        self.assertEqual(result['global_step'],6)

    def test_missing_mps_peak_api_is_unknown_despite_allocator_snapshot(self):
        fake=types.SimpleNamespace(mps=types.SimpleNamespace(driver_allocated_memory=lambda:999999))
        original=self.ns['torch'];self.ns['torch']=fake
        try:
            self.assertEqual(self.ns['accelerator_peak_memory'](types.SimpleNamespace(type='mps'),reset=True)['status'],'UNKNOWN')
            m=self.ns['accelerator_peak_memory'](types.SimpleNamespace(type='mps'))
            self.assertIsNone(m['allocated_bytes']);self.assertEqual(m['reason'],'UNSUPPORTED_PEAK_API')
        finally:self.ns['torch']=original
    def test_supported_peak_api_resets_then_reads_actual_value(self):
        resets=[];original=self.ns['torch'];self.ns['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda device:resets.append(device),max_memory_allocated=lambda device:12345))
        device=types.SimpleNamespace(type='cuda')
        try:
            self.assertEqual(self.ns['accelerator_peak_memory'](device,reset=True)['status'],'MEASURING')
            self.assertEqual(self.ns['accelerator_peak_memory'](device)['allocated_bytes'],12345)
            self.assertEqual(resets,[device])
        finally:self.ns['torch']=original
    def test_process_rss_units_are_separate_from_device_memory(self):
        import resource
        with mock.patch.object(resource,'getrusage',return_value=types.SimpleNamespace(ru_maxrss=123)),mock.patch.object(sys,'platform','darwin'):
            m=self.ns['process_peak_rss']();self.assertEqual(m['bytes'],123);self.assertEqual(m['scope'],'current_process_lifetime')
        with mock.patch.object(resource,'getrusage',return_value=types.SimpleNamespace(ru_maxrss=123)),mock.patch.object(sys,'platform','linux'):
            self.assertEqual(self.ns['process_peak_rss']()['bytes'],123*1024)

    def test_completed_resume_preserves_previous_metadata_and_records_only_new_leg(self):
        fixture=resume.MainLocalResume('test_default_argv_passes_four_epochs_to_loop_and_schedule_and_writes_no_resume');fixture.setUp()
        try:
            out=fixture.tmp/'measured';fixture.run_local(fixture.argv(out,epochs='1',keep='keep'))
            previous=json.loads((out/'local_training_run.json').read_text())
            state=kit.read_resume_state(out)
            fixture.run_local(fixture.argv(out,epochs='1',resume=state['epoch_dir'],keep='keep'))
            current=json.loads((out/'local_training_run.json').read_text())
            self.assertEqual(current['measurements']['processed_sequences'],0)
            self.assertIsNone(current['measurements']['sequences_per_second'])
            self.assertGreater(current['resume_checkpoint_restore_s'],0)
            self.assertGreater(current['resume_rng_restore_s'],0)
            self.assertEqual(current['previous_invocations'],[{k:v for k,v in previous.items() if k!='previous_invocations'}])
        finally:fixture.tearDown()

if __name__=='__main__':unittest.main()

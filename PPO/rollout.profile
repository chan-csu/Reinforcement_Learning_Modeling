         939759 function calls (937641 primitive calls) in 0.847 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.847    0.847 /Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/PPO/flux_explorer.py:401(rollout)
      6/5    0.000    0.000    0.730    0.146 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/_private/client_mode_hook.py:98(wrapper)
        1    0.000    0.000    0.730    0.730 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:1761(get)
        1    0.000    0.000    0.730    0.730 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:338(get_objects)
        1    0.471    0.471    0.471    0.471 {method 'get_objects' of 'ray._raylet.CoreWorker' objects}
        1    0.000    0.000    0.259    0.259 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:329(deserialize_objects)
        1    0.000    0.000    0.259    0.259 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:329(deserialize_objects)
        4    0.000    0.000    0.259    0.065 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:230(_deserialize_object)
        4    0.000    0.000    0.259    0.065 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:188(_deserialize_msgpack_data)
        4    0.000    0.000    0.259    0.065 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:176(_deserialize_pickle5_data)
        4    0.006    0.001    0.253    0.063 {built-in method _pickle.loads}
     8000    0.003    0.000    0.216    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:217(_load_from_bytes)
     8000    0.013    0.000    0.213    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:607(load)
     8000    0.021    0.000    0.166    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:733(_legacy_load)
        4    0.000    0.000    0.094    0.023 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/remote_function.py:109(_remote_proxy)
        4    0.000    0.000    0.094    0.023 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/util/tracing/tracing_helper.py:290(_invocation_remote_span)
        4    0.000    0.000    0.093    0.023 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/remote_function.py:171(_remote)
        4    0.000    0.000    0.093    0.023 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/remote_function.py:281(invocation)
        4    0.002    0.000    0.093    0.023 {method 'submit_task' of 'ray._raylet.CoreWorker' objects}
        8    0.000    0.000    0.091    0.011 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:409(serialize)
        4    0.000    0.000    0.091    0.023 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:369(_serialize_to_msgpack)
        4    0.001    0.000    0.091    0.023 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:351(_serialize_to_pickle5)
        4    0.000    0.000    0.090    0.022 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle_fast.py:59(dumps)
        4    0.000    0.000    0.090    0.022 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle_fast.py:618(dump)
   2108/4    0.016    0.000    0.090    0.022 {function CloudPickler.dump at 0x15f3881f0}
     8000    0.011    0.000    0.052    0.000 {method 'load' of '_pickle.Unpickler' objects}
     2104    0.001    0.000    0.044    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:625(__reduce__)
     2104    0.002    0.000    0.042    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:333(save)
     2104    0.006    0.000    0.037    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:384(_legacy_save)
     8000    0.017    0.000    0.036    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:853(persistent_load)
     8000    0.031    0.000    0.032    0.000 {built-in method builtins.__build_class__}
    32000    0.031    0.000    0.031    0.000 {built-in method _pickle.load}
     8000    0.003    0.000    0.026    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_utils.py:137(_rebuild_tensor_v2)
     8000    0.005    0.000    0.023    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_utils.py:131(_rebuild_tensor)
     8000    0.012    0.000    0.017    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:46(_is_zipfile)
     8008    0.017    0.000    0.017    0.000 {built-in method torch.tensor}
    14728    0.008    0.000    0.017    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:395(persistent_id)
    10104    0.004    0.000    0.016    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:228(_open_file_like)
     2104    0.001    0.000    0.015    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_tensor.py:175(__reduce_ex__)
     2104    0.006    0.000    0.015    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_tensor.py:207(_reduce_ex_internal)
        2    0.012    0.006    0.014    0.007 /Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/PPO/flux_explorer.py:352(compute_rtgs)
     8000    0.012    0.000    0.012    0.000 {method '_set_from_file' of 'torch._C.StorageBase' objects}
    10104    0.005    0.000    0.010    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:280(_should_read_directly)
     8000    0.004    0.000    0.008    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:218(__init__)
    13664    0.006    0.000    0.008    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/collections/__init__.py:935(__getitem__)
    20208    0.007    0.000    0.008    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_utils.py:553(_element_size)
    12208    0.006    0.000    0.007    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:362(__init__)
    16000    0.005    0.000    0.007    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:296(_check_seekable)
   134257    0.007    0.000    0.007    0.000 {built-in method builtins.isinstance}
    13756    0.005    0.000    0.007    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle_fast.py:664(reducer_override)
     8000    0.003    0.000    0.006    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/numpy/core/numeric.py:1855(_frombuffer)
     8000    0.003    0.000    0.006    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:740(find_class)
        4    0.005    0.001    0.005    0.001 {ray._raylet.unpack_pickle5_buffers}
    46450    0.003    0.000    0.005    0.000 {built-in method builtins.len}
    12208    0.004    0.000    0.005    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:295(__new__)
     2104    0.002    0.000    0.004    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_tensor.py:196(storage)
     8000    0.004    0.000    0.004    0.000 {method 'set_' of 'torch._C._TensorBase' objects}
        2    0.003    0.002    0.004    0.002 {built-in method numpy.array}
     8416    0.004    0.000    0.004    0.000 {built-in method _pickle.dump}
    10104    0.004    0.000    0.004    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:272(_is_compressed_file)
    10104    0.002    0.000    0.004    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:193(_is_path)
     2104    0.000    0.000    0.004    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:616(size)
     8000    0.002    0.000    0.003    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:983(__init__)
        8    0.000    0.000    0.003    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/glpk_interface.py:593(__getstate__)
        8    0.000    0.000    0.003    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/glpk_interface.py:671(_glpk_representation)
    16000    0.002    0.000    0.003    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:949(_maybe_decode_ascii)
     8000    0.002    0.000    0.003    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:173(default_restore_location)
     2104    0.001    0.000    0.003    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:444(__len__)
     4000    0.002    0.000    0.002    0.000 {method 'insert' of 'list' objects}
     8000    0.002    0.000    0.002    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_utils.py:159(_validate_loaded_sparse_tensors)
     8000    0.002    0.000    0.002    0.000 {built-in method numpy.frombuffer}
     8000    0.002    0.000    0.002    0.000 {method 'reshape' of 'numpy.ndarray' objects}
    32000    0.002    0.000    0.002    0.000 {method 'read' of '_io.BytesIO' objects}
    13660    0.002    0.000    0.002    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/collections/__init__.py:932(__missing__)
     2104    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:543(element_size)
     2104    0.001    0.000    0.001    0.000 {method '_write_file' of 'torch._C.StorageBase' objects}
    18104    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:433(_untyped)
    24000    0.001    0.000    0.001    0.000 {method 'seek' of '_io.BytesIO' objects}
    32072    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        8    0.000    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/util.py:285(__init__)
     8000    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:738(UnpicklerWrapper)
     8000    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:840(_get_dtype_from_pickle_storage_type)
     4000    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_tensor.py:753(__array__)
     2104    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:164(location_tag)
        8    0.000    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:513(NamedTemporaryFile)
    10104    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:199(__init__)
    12624    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/__init__.py:301(is_storage)
    24000    0.001    0.000    0.001    0.000 {method 'tell' of '_io.BytesIO' objects}
    10104    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:314(_check_dill_version)
        8    0.000    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:239(_mkstemp_inner)
    12212    0.001    0.000    0.001    0.000 {built-in method __new__ of type object at 0x1023571b8}
    10104    0.001    0.000    0.001    0.000 {method 'fileno' of '_io._IOBase' objects}
        8    0.000    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/swiglpk/swiglpk.py:784(glp_write_prob)
     8023    0.001    0.000    0.001    0.000 {built-in method builtins.hasattr}
     8000    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:127(_cpu_deserialize)
    15936    0.001    0.000    0.001    0.000 {built-in method builtins.issubclass}
        8    0.001    0.000    0.001    0.000 {built-in method swiglpk._swiglpk.glp_write_prob}
    10104    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:202(__enter__)
     2104    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:117(_cpu_tag)
        8    0.001    0.000    0.001    0.000 {built-in method posix.open}
     8000    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:961(_get_restore_location)
     2104    0.000    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/_namedtensor_internals.py:10(check_serializing_named_tensor)
      544    0.000    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/nn/modules/module.py:1194(__getattr__)
    10104    0.001    0.000    0.001    0.000 {method 'keys' of 'dict' objects}
     8000    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:24(__init__)
     4000    0.001    0.000    0.001    0.000 {method 'numpy' of 'torch._C._TensorBase' objects}
     2104    0.000    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:224(__exit__)
     8000    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/typing.py:1353(cast)
     8000    0.001    0.000    0.001    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/serialization.py:205(__exit__)
     6312    0.000    0.000    0.000    0.000 {method 'data_ptr' of 'torch._C.StorageBase' objects}
      512    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/nn/parameter.py:63(__reduce_ex__)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/util.py:295(__exit__)
        8    0.000    0.000    0.000    0.000 {built-in method posix.remove}
        4    0.000    0.000    0.000    0.000 {ray._raylet.split_buffer}
       96    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle.py:249(_should_pickle_by_reference)
     2104    0.000    0.000    0.000    0.000 {built-in method builtins.sorted}
     2104    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/typing.py:269(inner)
       16    0.000    0.000    0.000    0.000 {built-in method io.open}
     8000    0.000    0.000    0.000    0.000 {method 'clear' of 'list' objects}
       76    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle_fast.py:411(_class_reduce)
     2104    0.000    0.000    0.000    0.000 {method 'size' of 'torch._C._TensorBase' objects}
     2104    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/storage.py:619(pickle_storage_type)
     2104    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/utils/hooks.py:54(warn_if_has_hooks)
     2104    0.000    0.000    0.000    0.000 {method 'stride' of 'torch._C._TensorBase' objects}
     6104    0.000    0.000    0.000    0.000 {built-in method torch._C._has_torch_function_unary}
     2104    0.000    0.000    0.000    0.000 {method 'data_ptr' of 'torch._C._TensorBase' objects}
     2104    0.000    0.000    0.000    0.000 {method 'has_names' of 'torch._C._TensorBase' objects}
       96    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle.py:285(_lookup_module_and_qualname)
     4208    0.000    0.000    0.000    0.000 {method 'flush' of '_io.BytesIO' objects}
     2104    0.000    0.000    0.000    0.000 {method '_storage' of 'torch._C._TensorBase' objects}
     2412    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}
     2104    0.000    0.000    0.000    0.000 {method 'nbytes' of 'torch._C.StorageBase' objects}
     2104    0.000    0.000    0.000    0.000 {method 'storage_offset' of 'torch._C._TensorBase' objects}
     12/8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/msgpack/__init__.py:32(packb)
     2108    0.000    0.000    0.000    0.000 {method 'getvalue' of '_io.BytesIO' objects}
        4    0.000    0.000    0.000    0.000 {ray._raylet.dumps}
      544    0.000    0.000    0.000    0.000 {method 'format' of 'str' objects}
       32    0.000    0.000    0.000    0.000 {built-in method builtins.next}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:145(__next__)
       20    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle_fast.py:568(_function_reduce)
        8    0.000    0.000    0.000    0.000 {method 'read' of '_io.TextIOWrapper' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:148(<listcomp>)
       96    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle.py:187(_is_registered_pickle_by_value)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/_private/signature.py:81(flatten_args)
     10/5    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/_private/client_mode_hook.py:110(client_mode_should_convert)
       96    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/pickle.py:322(_getattribute)
       64    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/random.py:343(choice)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/posixpath.py:373(abspath)
        4    0.000    0.000    0.000    0.000 {built-in method loads}
       26    0.000    0.000    0.000    0.000 {method 'extend' of 'list' objects}
     12/8    0.000    0.000    0.000    0.000 {method 'pack' of 'msgpack._cmsgpack.Packer' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:496(close)
      128    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/cobra/core/object.py:101(__getstate__)
       16    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:430(close)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/glpk_interface.py:395(__getstate__)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/posixpath.py:334(normpath)
       64    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/random.py:237(_randbelow_with_getrandbits)
       96    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle.py:201(_whichmodule)
       56    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/cobra/core/species.py:60(__getstate__)
        8    0.000    0.000    0.000    0.000 {method 'close' of '_io.TextIOWrapper' objects}
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/inspect.py:3038(bind)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/posixpath.py:71(join)
        5    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/_collections_abc.py:760(get)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/inspect.py:2907(_bind)
       16    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/_bootlocale.py:33(getpreferredencoding)
        2    0.000    0.000    0.000    0.000 <__array_function__ internals>:177(sum)
        9    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:212(get_serialization_context)
      208    0.000    0.000    0.000    0.000 {method 'rsplit' of 'str' objects}
        5    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/os.py:674(__getitem__)
        2    0.000    0.000    0.000    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
        2    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/numpy/core/fromnumeric.py:2160(sum)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/util/placement_group.py:353(configure_placement_group_based_on_context)
       12    0.000    0.000    0.000    0.000 {method '__exit__' of '_io._IOBase' objects}
       32    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/cobra/core/dictlist.py:313(__reduce__)
      188    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
       17    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:164(current_job_id)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/interface.py:1463(update)
        2    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/numpy/core/fromnumeric.py:69(_wrapreduction)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/inspect.py:2779(__init__)
      140    0.000    0.000    0.000    0.000 {method 'copy' of 'dict' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:106(_sanitize_params)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/contextlib.py:261(helper)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:200(current_session_and_job)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/util.py:303(__getattr__)
       16    0.000    0.000    0.000    0.000 {built-in method _locale.nl_langinfo}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/codecs.py:319(decode)
        2    0.000    0.000    0.000    0.000 {method 'reduce' of 'numpy.ufunc' objects}
      104    0.000    0.000    0.000    0.000 {method 'split' of 'str' objects}
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/contextlib.py:86(__init__)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/cloudpickle/cloudpickle_fast.py:652(__init__)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/posixpath.py:60(isabs)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/glpk_interface.py:446(_get_feasibility)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:458(__init__)
       17    0.000    0.000    0.000    0.000 {method 'get_current_job_id' of 'ray._raylet.CoreWorker' objects}
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/util/placement_group.py:37(empty)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:134(rng)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/msgpack/ext.py:24(__new__)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/codecs.py:309(__init__)
       32    0.000    0.000    0.000    0.000 {method 'startswith' of 'str' objects}
        5    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/os.py:754(encode)
      116    0.000    0.000    0.000    0.000 {method 'getrandbits' of '_random.Random' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/cobra/core/model.py:77(__getstate__)
        5    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:1409(is_initialized)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/_private/utils.py:307(resources_from_ray_options)
        8    0.000    0.000    0.000    0.000 {built-in method _codecs.utf_8_decode}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:85(_infer_return_type)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/contextlib.py:114(__enter__)
       16    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/posixpath.py:41(_get_sep)
        4    0.000    0.000    0.000    0.000 {method 'pop' of 'list' objects}
       32    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/cobra/core/dictlist.py:321(__getstate__)
       16    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/torch/optim/optimizer.py:61(__getstate__)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/contextlib.py:123(__exit__)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:152(get_and_clear_contained_object_refs)
        4    0.000    0.000    0.000    0.000 <string>:1(<lambda>)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:145(set_out_of_band_serialization)
       16    0.000    0.000    0.000    0.000 {method 'join' of 'str' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/glpk_interface.py:473(verbosity)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:196(should_capture_child_tasks_in_placement_group)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:440(__del__)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/util/placement_group.py:286(check_placement_group_index)
       64    0.000    0.000    0.000    0.000 {method 'bit_length' of 'int' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/codecs.py:186(__init__)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:415(__init__)
       32    0.000    0.000    0.000    0.000 {built-in method posix.fspath}
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:390(_python_serializer)
        4    0.000    0.000    0.000    0.000 {built-in method nil}
        5    0.000    0.000    0.000    0.000 {method 'encode' of 'str' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/glpk_interface.py:464(presolve)
        8    0.000    0.000    0.000    0.000 {method 'split' of 'bytes' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/interface.py:1258(status)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:142(set_in_band_serialization)
       12    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/inspect.py:2558(kind)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/codecs.py:260(__init__)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/util/placement_group.py:45(is_empty)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:280(gettempdir)
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:176(current_task_id)
        5    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:234(check_connected)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/util/placement_group.py:41(__init__)
        8    0.000    0.000    0.000    0.000 {built-in method posix.getpid}
       10    0.000    0.000    0.000    0.000 {method '__exit__' of '_thread.RLock' objects}
       12    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/inspect.py:2546(name)
        8    0.000    0.000    0.000    0.000 {method 'pop' of 'dict' objects}
        8    0.000    0.000    0.000    0.000 {built-in method sys.audit}
        4    0.000    0.000    0.000    0.000 {method 'should_capture_child_tasks_in_placement_group' of 'ray._raylet.CoreWorker' objects}
        8    0.000    0.000    0.000    0.000 {method 'endswith' of 'str' objects}
       10    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/worker.py:149(connected)
        4    0.000    0.000    0.000    0.000 {ray._raylet._temporarily_disable_gc}
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/inspect.py:2638(__init__)
        1    0.000    0.000    0.000    0.000 {method 'get_current_task_id' of 'ray._raylet.CoreWorker' objects}
        2    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/numpy/core/fromnumeric.py:70(<dictcomp>)
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/util.py:292(__enter__)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/serialization.py:198(_python_deserializer)
       12    0.000    0.000    0.000    0.000 {method 'is_nil' of 'ray._raylet.PlacementGroupID' objects}
        4    0.000    0.000    0.000    0.000 {method 'values' of 'mappingproxy' objects}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/tempfile.py:225(_get_candidate_names)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/inspect.py:2550(default)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/util/tracing/tracing_helper.py:112(is_tracing_enabled)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/utils.py:72(parse_runtime_env)
        8    0.000    0.000    0.000    0.000 {built-in method builtins.iter}
        8    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/optlang/glpk_interface.py:482(timeout)
       10    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/numpy/core/__init__.py:144(_DType_reduce)
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/PPO/flux_explorer.py:403(<dictcomp>)
        4    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/inspect.py:2863(parameters)
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/_private/profiling.py:19(profile)
        1    0.000    0.000    0.000    0.000 {method 'current_actor_is_asyncio' of 'ray._raylet.CoreWorker' objects}
        2    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/numpy/core/fromnumeric.py:2155(_sum_dispatcher)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/PPO/flux_explorer.py:407(<dictcomp>)
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/PPO/flux_explorer.py:404(<dictcomp>)
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/PPO/flux_explorer.py:405(<dictcomp>)
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/PPO/flux_explorer.py:406(<dictcomp>)
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/_private/profiling.py:11(__exit__)
        1    0.000    0.000    0.000    0.000 /Users/parsaghadermarzi/.pyenv/versions/miniforge3-4.10.3-10/lib/python3.9/site-packages/ray/_private/profiling.py:8(__enter__)



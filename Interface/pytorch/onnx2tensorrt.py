#!/usr/bin/env python
# coding: utf-8
"""
module: given onnx model, convert to tensorRT model for deployment
"""

from __future__ import division  # ensure accurate division (auto-convert to float point)

import logging


logger = logging.getLogger(__name__)

__all__ = [
    'onnx2trt_infer',
    'convert_validate_save',
]


def flatten_dict(
        obj:'Mapping[Any, Any]',
        out:'Optional[Mapping[Any, Any]]'=None)->'Mapping[Any, Any]':
    r"""make 'dict'-like `obj` flatten"""

    assert isinstance(obj, dict), "'dict' required"

    if out is None:
        out = type(obj)()
    for key, value in obj.items():
        if isinstance(value, dict):
            flatten_dict(value, out)
        else:
            assert key not in out, 'key conflicted'
            out[key] = value
    return out


def onnx2trt_infer(
        onnx_model_filename:str, input_values:'Sequence[np.ndarray]',
        batch_size:int=1,
        workspace_size:int=(1024 * 1024 * 16),
        )->'Sequence[np.ndarray]':
    r"""infer model with 'onnx_tensorrt' backend"""

    import onnx
    import onnx.optimizer as optimizer
    import onnx_tensorrt.backend as backend

    from onnx.utils import polish_model

    model = onnx.load(onnx_model_filename)
    passes = optimizer.get_available_passes()
    passes = list(filter(lambda name: not name.startswith('split_'), passes))
    logger.debug('optimizations to perform in ONNX:\n\t%s', passes)
    model = optimizer.optimize(model, passes=passes)
    model = polish_model(model)
    onnx.save(model, onnx_model_filename.rpartition('.onnx')[0] + '.optimized.onnx')
    engine = backend.prepare(
            model, device='CUDA',
            max_batch_size=batch_size, max_workspace_size=workspace_size,
            )
    return engine.run(input_values)


def convert_validate_save(
        onnx_model_filename:str,
        golden_data_filename:'Optional[str]'='',
        atol:float=1e-3, rtol:float=1e-3,
        batch_size:int=1, #
        debug:bool=False,
        **kwargs)->bool:
    r"""
        inference model in 'tensorrt'
        validate with given golden data
        save if accuracy passed
    """

    import numpy as np
    import pycuda.autoinit # noqa: just import, no code check
    import pycuda.driver as cuda
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.VERBOSE if debug else trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)

    logger.info('loading ONNX model: %s ...', onnx_model_filename)
    with open(onnx_model_filename, 'rb') as fp:
        onnx_model_proto_str = fp.read()
    success = parser.parse(onnx_model_proto_str)
    if not success:
        logger.error('model parsing failed:')
        for idx_error in range(parser.num_errors):
            logger.error('\t%s', parser.get_error(idx_error))
        return False
    logger.info('model parsing passed')

    workspace_size = kwargs.pop('workspace_size', 1024 * 1024 * 16) # default to 1024*1024*16
    fp16_mode = kwargs.pop('fp16_mode', builder.platform_has_fast_fp16)
    int8_mode = kwargs.pop('int8_mode', builder.platform_has_fast_int8)

    builder.debug_sync = debug
    builder.fp16_mode = fp16_mode
    builder.max_batch_size = batch_size
    builder.max_workspace_size = workspace_size
    builder.refittable = False
    builder.strict_type_constraints = True

    logger.info('using batch_size: %d', builder.max_batch_size)
    logger.info('I/O type-shape info:')
    if int8_mode:
        default_range = (-127, +127)
        builder.int8_mode = int8_mode
        for layer in network:
            for idx_out in range(layer.num_outputs):
                var_out = layer.get_output(idx_out)
                var_out.set_dynamic_range(-127, +127)
        dynamic_ranges = kwargs.pop('io_dynamic_ranges', dict())
        for idx_inp in range(network.num_inputs):
            var_inp = network.get_input(idx_inp)
            dr_lo, dr_hi = dynamic_ranges.get(var_inp.name, default_range)
            var_inp.set_dynamic_range(dr_lo, dr_hi)
            logger.info('\t input %d (%12s): %s%s in [%d, %d]',
                        idx_inp, var_inp.name, var_inp.dtype, var_inp.shape,
                        dr_lo, dr_hi)
        for idx_out in range(network.num_outputs):
            var_out = network.get_output(idx_out)
            dr_lo, dr_hi = dynamic_ranges.get(var_out.name, default_range)
            var_out.set_dynamic_range(dr_lo, dr_hi)
            logger.info('\toutput %d (%12s): %s%s in [%d, %d]',
                        idx_out, var_out.name, var_out.dtype, var_out.shape,
                        dr_lo, dr_hi)
        # TODO: int8 calibrate
    else:
        for idx_inp in range(network.num_inputs):
            var_inp = network.get_input(idx_inp)
            logger.info('\t input %d (%12s): %s%s',
                        idx_inp, var_inp.name, var_inp.dtype, var_inp.shape)
        for idx_out in range(network.num_outputs):
            var_out = network.get_output(idx_out)
            logger.info('\toutput %d (%12s): %s%s',
                        idx_out, var_out.name, var_out.dtype, var_out.shape)

    # not exposed
#    builder.getNbDLACores() > 0
#    builder.allowGPUFallback(True)
#    builder.setDefaultDeviceType(kDLA)
#    builder.setDLACore(1)

    engine = builder.build_cuda_engine(network)
    if engine is None:
        logger.info('engine building failed')
        return False
    logger.info('engine building passed')

#    globals().update(locals())

    if golden_data_filename:
        logger.info('using golden data %s', golden_data_filename)
        if golden_data_filename.endswith('.npz'):
            test_data = np.load(
                    golden_data_filename,
                    encoding='bytes', allow_pickle=True,
                    )
            input_data = test_data['inputs'].tolist()
            output_data = test_data['outputs'].tolist()
        else:
            test_data = np.load(
                    golden_data_filename,
                    encoding='bytes', allow_pickle=True,
                    ).tolist()
            input_data = test_data['inputs']
            output_data = test_data['outputs']

        input_data = flatten_dict(input_data)
        output_data = flatten_dict(output_data)
#        input_names = input_data.keys()
        output_names = output_data.keys()
        logger.info('with %d inputs and %d outputs',
                    len(input_data), len(output_data))

        input_device_data = {
                name: cuda.to_device(value)
                for name, value in input_data.items()
                }
        output_device_data = {
                name: cuda.mem_alloc(value.nbytes)
                for name, value in output_data.items()
                }
        output_host_data = {
                name: cuda.pagelocked_empty_like(value)
                for name, value in output_data.items()
                }
        logger.info('data transfered to device')

        profiler = trt.Profiler()
        with engine.create_execution_context() as context:
            if debug:
                context.profiler = profiler
            stream = cuda.Stream()

#            for name in input_names:
#                cuda.memcpy_htod_async(
#                        input_data[name], input_device_data[name],
#                                       stream=stream)

            device_data = list(input_device_data.values()) + list(output_device_data.values())
            success = context.execute_async(
                    batch_size,
                    bindings=list(map(int, device_data)),
                    stream_handle=stream.handle, input_consumed=None)
            if not success:
                logger.error('execution failed')
                return False

            for name in output_names:
                cuda.memcpy_dtoh_async(
                        output_host_data[name], output_device_data[name],
                        stream=stream,
                        )

            stream.synchronize()

        logger.info('execution passed')

#        output_host_data[name] = onnx2trt_inference(
#                onnx_model_filename, list(input_data.values()),
#                batch_size, workspace_size)[0]

    # validate
    passed = True
    if golden_data_filename:
        for name in output_names:
            pr = output_host_data[name]
            gt = output_data[name]
            logger.info('testing on output %s ...', name)
            try:
                np.testing.assert_allclose(
                        pr, gt,
                        rtol=rtol, atol=atol,
                        equal_nan=False, verbose=True,
                        )
            except AssertionError as e:
                passed = False
                logger.error('failed: %s\n', e)
        logger.info('accuracy %spassed', '' if passed else 'not ')

    globals().update(locals())

    if passed:
        trt_engine_filename = onnx_model_filename[:-len('.onnx')] + '.bin' # or .trt
        with open(trt_engine_filename, 'wb') as fp:
            fp.write(engine.serialize())
        logger.info('engine saved to %s', trt_engine_filename)

    return passed


def main():
    r"""main entry"""

    import argparse, ast

    def parse_keyvalue(
            s:str,
            delimiter:str='='):
        r"""parse key-value assignment where value is a Python expression"""

        s = s.replace(delimiter, '=', 1)
        m = ast.parse(s)  # compile `s` into a AST/code object that can be run by `eval`/`exec`
        assert len(m.body) == 1
        a = m.body[0]
        assert len(a.targets) == 1
        key = a.targets[0].id
        value = a.value
        value = ast.literal_eval(value)  # eval with safety check: allow only python literal structure, but not arbitrary code
        return (key, value)

    parser = argparse.ArgumentParser(
            description='onnx2tensorrt',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
    parser.add_argument(
            'model', nargs=1,
            help='path to model.onnx',
            )
    parser.add_argument(
            'params', nargs='*', type=parse_keyvalue,
            help='extra parameters',
            )
    parser.add_argument(
            '--debug', '-d', action='store_true',
            help='enable debug logging and checking',
            )
    parser.add_argument(
            '--test_data', '-t', type=str, default='',
            help='I/O golden data for validation, e.g. test.npy, test.npz',
            )
    parser.add_argument(
            '--atol', '-p', type=float, default=1e-3,
            help='assertion absolute tolerance for validation',
            )
    parser.add_argument(
            '--rtol', type=float, default=1e-2,
            help='assertion relative tolerance for validation',
            )
    parser.add_argument(
            '--batch_size', '-b', type=int, default=1,
            help='max batch size of model',
            )
    args = parser.parse_args()

    logging_format = '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s'
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format=logging_format, level=logging_level)

    debug = args.debug
    onnx_model_filename = args.model[0]
    golden_data_filename = args.test_data
    atol, rtol = args.atol, args.rtol
    batch_size = args.batch_size
    kwargs = dict()
    kwargs.update(args.params)

    convert_validate_save(
            onnx_model_filename, golden_data_filename,
            atol=atol, rtol=rtol,
            batch_size=batch_size,
            debug=debug,
            **kwargs)

    # access global symbol table (a dict) & update it with local symbol table (add/update variable)
    globals().update(locals())


if __name__ == '__main__':
    print(__file__)
    main()

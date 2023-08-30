from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, OptimizationConfig, ORTOptimizer
import argparse

def convert(model_path, save_dir, optimize_for_gpu=False):
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path, export=True)
    optimizer = ORTOptimizer.from_pretrained(model)
    optimization_config = OptimizationConfig(
        optimization_level=99,
        enable_transformers_specific_optimizations=True,
        optimize_for_gpu=optimize_for_gpu
    )
    optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of the huggingface model directory')
    parser.add_argument('--save_dir', type=str, help='path of the output directory')
    parser.add_argument('--optimize_for_gpu', action='store_true', help='whether to optimize for gpu')

    args = parser.parse_args()

    convert(args.model_path, args.save_dir, args.optimize_for_gpu)

"""Command-line entry point for the variant post-filtering pipeline."""

import argparse
import logging
import os

from .config import PipelineConfig
from .pipeline import run_apply_pipeline, run_train_pipeline


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Variant Quality Scoring Pipeline")
    parser.add_argument("--config", "-c", default="pipeline_config.yaml",
                        help="Path to YAML/JSON config file")
    parser.add_argument("--mode", choices=["train", "apply"],
                        help="Override pipeline mode")
    parser.add_argument("--output-dir", help="Override output directory base name")
    parser.add_argument("--train-vcf", help="Override training VCF path")
    parser.add_argument("--truth-vcf", help="Override truth VCF path")
    parser.add_argument("--apply-vcf", help="Override apply VCF path")
    parser.add_argument("--trained-model-dir",
                        help="Override path to training output dir (apply mode)")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config) if os.path.exists(args.config) else PipelineConfig()
    if not os.path.exists(args.config):
        # Can't log yet — print directly so the warning isn't silently lost
        print(f"[WARNING] Config file {args.config} not found, using defaults", flush=True)

    if args.mode:
        config.set('mode', args.mode)
    if args.output_dir:
        config.set('paths.output_dir', args.output_dir)
    if args.train_vcf:
        config.set('paths.train_input_vcf', args.train_vcf)
    if args.truth_vcf:
        config.set('paths.truth_vcf', args.truth_vcf)
    if args.apply_vcf:
        config.set('paths.apply_input_vcf', args.apply_vcf)
    if args.trained_model_dir:
        config.set('paths.trained_model_dir', args.trained_model_dir)

    # ── Create output directory ONCE here ──────────────────────────────────────
    # run_train_pipeline / run_apply_pipeline receive this dir and do NOT call
    # initialize_paths() again.  Previously each pipeline called initialize_paths()
    # independently, producing a second timestamped directory whose path differed
    # from the one used for the log file → log file ended up in the wrong folder.
    output_dir = config.initialize_paths()

    # ── Configure logging IMMEDIATELY after output_dir exists ──────────────────
    # All subsequent log calls — including those inside the pipeline — go here.
    log_file = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
    )

    # Save final resolved config next to the log
    config.to_yaml(os.path.join(output_dir, "final_config.yaml"))
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Pipeline mode: {config._raw_get('mode')}")

    try:
        mode = config._raw_get('mode')
        if mode == 'train':
            run_train_pipeline(config, output_dir)
        elif mode == 'apply':
            run_apply_pipeline(config, output_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

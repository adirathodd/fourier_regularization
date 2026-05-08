import os
import argparse
from config import Config
from training import train
from utils import (
    get_device,
    create_dataset,
    create_model,
    print_dataset_info,
    evaluate_model,
    filter_data_by_operation,
    load_model,
    plot_training_curves,
    resolve_checkpoint,
    extract_checkpoint_accuracies,
)
from analysis.rnn import ablation as rnn_ablation
from analysis.rnn import combined_ablation as rnn_combined_ablation
from analysis.rnn import fourier as rnn_fourier
from analysis.rnn import svd as rnn_svd
from analysis.rnn import trig as rnn_trig
from analysis.transformer import ablation as transformer_ablation
from analysis.transformer import combined_ablation as transformer_combined_ablation
from analysis.transformer import fourier as transformer_fourier
from analysis.transformer import svd as transformer_svd
from analysis.transformer import trig as transformer_trig
from analysis.variable_length_metrics import (
    cross_length_generalization_matrix,
    operation_pattern_failure_taxonomy,
    per_length_accuracy_breakdown,
    token_position_accuracy,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and/or analyze an RNN/Transformer on modular arithmetic (a op b) mod p.'
    )
    parser.add_argument('-p', '--path',       default='.', help='Directory containing config.yaml')
    parser.add_argument('-o', '--operation',  default=['addition'], nargs='+',
                        choices=['addition', 'subtraction', 'multiplication', 'division'],
                        help='Modular arithmetic operation(s)')
    parser.add_argument('-m', '--model-type', default='rnn', choices=['rnn', 'transformer'])
    parser.add_argument('--device',           default=None, help='e.g. cuda, cuda:0, cpu')
    parser.add_argument('-t', '--train',      action='store_true')
    parser.add_argument('-a', '--analysis',   action='store_true')
    parser.add_argument('--verbose',          action='store_true')
    args = parser.parse_args()
    args.operation.sort()

    config, args.path = Config.resolve(
        args.path, args.operation, args.model_type,
        analysis_only=args.analysis and not args.train,
    )

    require_cuda = bool(args.device and args.device.startswith('cuda'))
    device = get_device(args.device, require_cuda=require_cuda)
    print(f'Using device: {device}')

    dataset = create_dataset(config.operations, config.data.to_dict(), device)
    model = create_model(config.model.to_dict(), device=device)

    ops_str = ', '.join(op.capitalize() for op in config.operations)
    print(f'\nCreated Modular {ops_str} {config.model_type.upper()} Model and Dataset:\n')
    print_dataset_info(dataset)

    if args.verbose:
        print("\nModel:")
        print(model)

        print('\nConfiguration:')
        print(config)

    if args.train:
        final_checkpoint = train(model, dataset, config.training)
        plot_training_curves(final_checkpoint, args.path)

    if args.analysis:
        is_variable_length = bool(getattr(dataset, "is_variable_length", False))
        if is_variable_length:
            print(
                "Variable-length analysis mode: running embedding/unembedding Fourier + "
                "model-level SVD spectrum only. Skipping hidden-state and ablation analyses."
            )
        print('\n=== Fourier Spectrum Analysis Report ===\n')

        ckpt_path = resolve_checkpoint(args.path)
        model, checkpoint = load_model(ckpt_path, config.model.to_dict(), device=str(device))
        dataset = dataset.to_device(str(device))
        analysis_cache = {}
        
        if config.model_type == 'transformer':
            fourier_analysis = transformer_fourier
            svd_analysis = transformer_svd
            ablation_analysis = transformer_ablation
            combined_ablation_analysis = transformer_combined_ablation
            trig_analysis = transformer_trig
        else:
            fourier_analysis = rnn_fourier
            svd_analysis = rnn_svd
            ablation_analysis = rnn_ablation
            combined_ablation_analysis = rnn_combined_ablation
            trig_analysis = rnn_trig

        train_acc, test_acc, epoch = extract_checkpoint_accuracies(checkpoint)

        os.makedirs(os.path.join(args.path, 'figures'), exist_ok=True)

        print(f"Epoch: {epoch if epoch is not None else 'N/A'}")
        print('1. Model performance:')
        train_acc_str = f'{train_acc}%' if train_acc is not None else 'N/A'
        test_acc_str = f'{test_acc}%' if test_acc is not None else 'N/A'
        if is_variable_length:
            print(
                f'Accuracy on entire dataset: N/A (variable-length dataset)'
                f' \n\t Train set accuracy: {train_acc_str}'
                f' \n\t Test set accuracy: {test_acc_str}'
            )
            print('Per-operation accuracy: skipped for variable-length analysis mode')
        else:
            print(
                f'Accuracy on entire dataset: {evaluate_model(model, dataset.dataset, dataset.labels)}%'
                f' \n\t Train set accuracy: {train_acc_str}'
                f' \n\t Test set accuracy: {test_acc_str}'
            )
            print('Per-operation accuracy:')
            for op in config.operations:
                op_all_data, op_all_labels = filter_data_by_operation(dataset, op, split='all', target='final')
                op_train_data, op_train_labels = filter_data_by_operation(dataset, op, split='train', target='final')
                op_test_data, op_test_labels = filter_data_by_operation(dataset, op, split='test', target='final')

                op_all_acc = evaluate_model(model, op_all_data, op_all_labels)
                op_train_acc = evaluate_model(model, op_train_data, op_train_labels)
                op_test_acc = evaluate_model(model, op_test_data, op_test_labels)

                print(
                    f' \n\t {op.capitalize()} - total: {op_all_acc}%'
                    f' \n\t {op.capitalize()} - train: {op_train_acc}%'
                    f' \n\t {op.capitalize()} - test: {op_test_acc}%'
                )

        if is_variable_length:
            print('\n1.1 Per-length accuracy breakdown:')
            per_len = per_length_accuracy_breakdown(
                model, dataset, training_target=config.training.training_target
            )
            for nterms in sorted(per_len.keys()):
                row = per_len[nterms]
                if config.training.training_target == 'seq_cot':
                    print(
                        f' \n\t nterms={nterms} | final: {row["final_accuracy"]:.2f}%'
                        f' | prefix: {row["prefix_accuracy"]:.2f}% | N={row["num_samples"]}'
                    )
                else:
                    print(
                        f' \n\t nterms={nterms} | final: {row["final_accuracy"]:.2f}%'
                        f' | N={row["num_samples"]}'
                    )

            print('\n1.2 Cross-length generalization matrix (sampled extrapolation):')
            cross_len = cross_length_generalization_matrix(
                model,
                dataset,
                training_target=config.training.training_target,
                num_samples_extrapolation=2048,
                num_extrapolation_lengths=1,
                seed=int(config.data.data_seed),
                save_dir=args.path,
            )
            if cross_len is not None:
                for nterms in cross_len['eval_lengths']:
                    row = cross_len['results'][nterms]
                    src = row.get('source', 'unknown')
                    if config.training.training_target == 'seq_cot':
                        print(
                            f' \n\t nterms={nterms} | final: {row["final_accuracy"]:.2f}%'
                            f' | prefix: {row["prefix_accuracy"]:.2f}% | N={row["num_samples"]} | {src}'
                        )
                    else:
                        print(
                            f' \n\t nterms={nterms} | final: {row["final_accuracy"]:.2f}%'
                            f' | N={row["num_samples"]} | {src}'
                        )
                print(
                    f"\n\t saved: {args.path}/figures/cross_length_generalization_matrix.png"
                )

            print('\n1.3 Token-position accuracy (seq_cot only):')
            if config.training.training_target == 'seq_cot':
                pos_acc = token_position_accuracy(model, dataset)
                aggregate = pos_acc['aggregate']
                if aggregate:
                    for pos in sorted(aggregate.keys()):
                        row = aggregate[pos]
                        print(
                            f' \n\t position {pos}: {row["accuracy"]:.2f}%'
                            f' ({row["correct"]}/{row["total"]})'
                        )
                else:
                    print(' \n\t no supervised positions found')
            else:
                print(' \n\t skipped (training_target != seq_cot)')

            print('\n1.4 Operation-pattern failure taxonomy:')
            taxonomy = operation_pattern_failure_taxonomy(
                model,
                dataset,
                training_target=config.training.training_target,
                top_k=12,
            )
            top_rows = taxonomy['top_failures']
            if top_rows:
                for row in top_rows:
                    print(
                        f' \n\t {row["pattern_str"]}: error={row["error_rate"]:.2f}%'
                        f' | acc={row["accuracy"]:.2f}% | N={row["total"]}'
                    )
            else:
                print(' \n\t no operation patterns found')

        op_elbows = {}
        for op in config.operations:
            op_elbows[op] = fourier_analysis.compute_ip_elbow(
                checkpoint, dataset, op,
                norm_threshold=config.analysis.fourier_norm_threshold,
                fourier_reg_mode=config.training.fourier_reg_mode,
                analysis_cache=analysis_cache,
            )

        for op in config.operations:
            ip_elbow = op_elbows[op]
            print(f'\n=== {op.upper()} ===')

            print('2. Generating fourier coefficient analysis plots')
            fourier_analysis.fourier_spectrum_analysis_plotting(
                model, checkpoint, dataset, args.path, op,
                fourier_reg_mode=config.training.fourier_reg_mode,
                norm_threshold=config.analysis.fourier_norm_threshold,
                freq_threshold_top_fraction=config.analysis.freq_threshold_top_fraction,
                freq_threshold_min_components=config.analysis.freq_threshold_min_components,
                freq_threshold_fixed=config.analysis.freq_threshold_fixed,
                analysis_cache=analysis_cache,
            )
            print()

            if is_variable_length:
                print('3. Skipping SVD ablation analysis (variable-length mode)')
                print()
                print('4. Skipping Fourier component ablation analysis (variable-length mode)')
                print()
                print('5. Skipping trigonometric identity verification (variable-length mode)')
                print()
            else:
                print('3. Performing SVD ablation analysis')
                try:
                    ablation_analysis.svd_ablation_analysis(
                        config.model.to_dict(), checkpoint, dataset, op, ip_elbow,
                        fourier_reg_mode=config.training.fourier_reg_mode,
                        analysis_cache=analysis_cache,
                    )
                except Exception as e:
                    print(f'Exception: {e}')
                print()

                print('4. Performing fourier component ablation analysis')
                try:
                    ablation_analysis.fourier_ablation_analysis(
                        config.model.to_dict(), checkpoint, dataset, op, ip_elbow,
                        fourier_reg_mode=config.training.fourier_reg_mode,
                        analysis_cache=analysis_cache,
                    )
                except Exception as e:
                    print(f'Exception: {e}')
                print()

                if op in ('addition', 'subtraction'):
                    print('5. Trigonometric Identity Verification:')
                    try:
                        if op == 'addition':
                            trig_analysis.verify_trigonometric_identity(
                                model,
                                checkpoint,
                                dataset,
                                freq_threshold_top_fraction=config.analysis.freq_threshold_top_fraction,
                                freq_threshold_min_components=config.analysis.freq_threshold_min_components,
                                freq_threshold_fixed=config.analysis.freq_threshold_fixed,
                                analysis_cache=analysis_cache,
                            )
                        else:
                            trig_analysis.verify_trigonometric_identity_subtraction(
                                model,
                                checkpoint,
                                dataset,
                                freq_threshold_top_fraction=config.analysis.freq_threshold_top_fraction,
                                freq_threshold_min_components=config.analysis.freq_threshold_min_components,
                                freq_threshold_fixed=config.analysis.freq_threshold_fixed,
                                analysis_cache=analysis_cache,
                            )
                    except Exception as e:
                        print(f'Exception: {e}')
                    print()

        print('\n6. Generating Weight SVD spectrum plots (whole model)')
        try:
            svd_analysis.svd_spectrum_analysis_plotting_model(
                model,
                checkpoint,
                dataset,
                args.path,
                op_elbows,
                config.operations,
                analysis_cache=analysis_cache,
            )
        except Exception as e:
            print(f'Exception: {e}')
        print()

        if is_variable_length:
            print('7. Skipping combined-ip-elbow embedding/unembedding ablation analysis (variable-length mode)')
            print()
            print('8. Skipping top-k ablation accuracy curve (variable-length mode)')
            print()
        else:
            print('7. Performing combined-ip-elbow embedding/unembedding ablation analysis')
            try:
                combined_ablation_analysis.combined_ip_svd_ablation_analysis(
                    config.model.to_dict(),
                    checkpoint,
                    dataset,
                    op_elbows,
                    config.operations,
                    analysis_cache=analysis_cache,
                )
            except Exception as e:
                print(f'Exception: {e}')
            print()

            print('8. Plotting per-operation accuracy vs kept top-k singular components')
            try:
                combined_ablation_analysis.combined_ip_svd_ablation_accuracy_curve(
                    config.model.to_dict(),
                    checkpoint,
                    dataset,
                    args.path,
                    op_elbows,
                    config.operations,
                    analysis_cache=analysis_cache,
                )
            except Exception as e:
                print(f'Exception: {e}')
            print()

        print('=== Analysis Complete ===')

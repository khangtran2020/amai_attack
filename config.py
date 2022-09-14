import argparse


def add_general_group(group):
    group.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    group.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    group.add_argument("--save_path", type=str, default="results/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=1, help="seed value")
    group.add_argument("--mode", type=str, default='train', help="Mode of running")
    group.add_argument("--train_mode", type=str, default='clean', help="Mode of training [clean, dp, target]")
    group.add_argument("--robustness", type=str, default='false', help="with or withour robustness inference")


def add_data_group(group):
    group.add_argument('--dataset', type=str, default='celebA', help="used dataset")
    group.add_argument('--use_embedding', type=bool, default=True, help="Use embedding for LDP or not ")
    group.add_argument('--data_path', type=str, default='../../Datasets/CelebA/embeddings/',
                       help="the directory used to save dataset")
    group.add_argument('--num_feature', type=int, default=512, help="number of target")
    group.add_argument('--num_target', type=int, default=1, help="number of target")
    group.add_argument('--target', type=str, default='0',
                       help="The targets must be separate by a - (For example: 1-16-24-10)")
    group.add_argument('--train_multiplier', type=int, default=25, help="number of multiplier")
    group.add_argument('--valid_multiplier', type=int, default=20, help="number of multiplier")
    group.add_argument('--train_index', type=int, default=162770, help="number of multiplier")
    group.add_argument('--valid_index', type=int, default=182637, help="number of multiplier")


def add_model_group(group):
    group.add_argument("--lr", type=float, default=0.01, help="learning rate")
    group.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    group.add_argument("--embed_dim", type=int, default=512, help="embedding dim")
    group.add_argument('--batch_size', type=int, default=200000)
    group.add_argument('--train_verbose', action='store_true', help="print training details")
    group.add_argument('--log_every', type=int, default=1, help='print every x epoch')
    group.add_argument('--eval_every', type=int, default=5, help='evaluate every x epoch')
    group.add_argument('--model_save_path', type=str, default='results/model/')
    group.add_argument("--num_steps", type=int, default=2000)
    group.add_argument("--early_stopping", type=int, default=100)
    group.add_argument("--test_ver", type=int, default=1)


def add_defense_group(group):
    group.add_argument('--epsilon', type=float, default=1.0, help="epsilon")
    group.add_argument('--min_epsilon', type=float, default=0.1, help="epsilon")
    group.add_argument('--max_epsilon', type=float, default=20.0, help="epsilon")
    group.add_argument('--fix_epsilon', type=float, default=1.0, help="epsilon")
    group.add_argument('--sens', type=float, default=1.0, help="sensitivity")
    group.add_argument('--num_draws', type=int, default=1000, help="sensitivity")
    group.add_argument('--eval_mode', type=str, default='eps', help="eps/alpha")
    group.add_argument('--alpha', type=float, default=0.05, help="confidence rate")
    group.add_argument('--num_test_set', type=int, default=100, help="number of generated test set")
    group.add_argument('--num_test_point', type=int, default=1, help="number of test point in one test set")
    group.add_argument('--sample_target_rate', type=float, default=0.1, help="rate of target")


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    general_group = parser.add_argument_group(title="General configuration")
    defense_group = parser.add_argument_group(title="Defense configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    add_defense_group(defense_group)
    return parser.parse_args()

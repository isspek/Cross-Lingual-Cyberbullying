RANDOM_SEED = 1234


def data_args(parser):
    parser.add_argument('--lang', type=str, help='Language', choices=['en', 'es'])
    parser.add_argument('--data', type=str, help='Path to dataset')
    return parser


RANDOM_SEED = 1234


def data_args(parser):
    parser.add_argument('--lang', type=str, help='Language', choices=['en', 'es', 'en_es', 'es_en'])
    parser.add_argument('--data', type=str, help='Path to dataset')
    return parser


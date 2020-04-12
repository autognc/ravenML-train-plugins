from rmltfbbox.validation.stats import BoundingBoxEvaluator
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    evaluator = BoundingBoxEvaluator.load_from_dump(args.dump_path)
    evaluator.calculate_default_and_save(args.output_path)
    evaluator.dump(args.output_path + '/results.pickle')


if __name__ == '__main__':
    main()

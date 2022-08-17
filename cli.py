from transformers import HfArgumentParser

from prompteval import PromptEvalConfig, PromptEval

def main():
    parser = HfArgumentParser(PromptEvalConfig)
    config, = parser.parse_args_into_dataclasses()

    prompteval = PromptEval(config)

    prompteval.run()

if __name__ == "__main__":
    main()


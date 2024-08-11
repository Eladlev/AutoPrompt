import argparse
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Tuple

import pandas as pd

from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml, override_config

# Constants

ANNOTATION_COLUMN_NAMES = ["label", "annotation", "answer"]
DEFAULT_BASIC_CONFIG_PATH = str(
    Path(__file__).parent / "config" / "config_auto_sync_small.yml")

RLHF_ANNOTATOR_CONFIG = {
    "method": "argilla",
    "config": {
        "api_url": "http://0.0.0.0:6900",
        "api_key": 'admin.apikey',
        "workspace": "admin",
        "time_interval": 5,
    }
}

RLAIF_ANNOTATOR_CONFIG = {
    "method": "llm",
    "config": {
        "llm": {
            "type": "Azure",
            "name": "ug-dev-east-us2-gpt-4-32k",
        },
        "instruction": "",  # to be filled
        "num_workers": 5,
        "prompt": "prompts/predictor_completion/prediction.prompt",
        "mini_batch_size": 1,
        "mode": "annotation",
    }
}


def opposite_label(label: str) -> str:
    if label == "yes":
        return "no"
    return "yes"


def parse_args() -> argparse.Namespace:
    # General Training Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--prompt",
        default="",
        required=True,
        type=str,
        help="Prompt to use as initial.",
    )
    parser.add_argument(
        "--task_description",
        default="",
        required=True,
        type=str,
        help="Describing the task",
    )

    parser.add_argument(
        "--max_usage",
        default=1.0,
        required=False,
        type=float,
        help="Maximum usage of the annotator in USD",
    )

    parser.add_argument(
        "-c",
        "--basic_config_path",
        default=DEFAULT_BASIC_CONFIG_PATH,
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--batch_config_path",
        default="",
        type=str,
        help="Batch classification configuration file path",
    )

    parser.add_argument(
        "--output_path",
        default="dump",
        required=False,
        type=str,
        help="Output to save checkpoints",
    )
    parser.add_argument(
        "--num_steps", default=40, type=int, help="Number of iterations"
    )

    # section to override easily configuration parameters
    parser.add_argument(
        "-d",
        "--dataset_path",
        default="",
        type=str,
        help="Dataset path to use",
    )

    parser.add_argument(
        "--use_rlhf",
        action="store_true",
        help="Use RLHF annotator",
    )
    parser.add_argument(
        "--rlaif_instructions",
        default="",
        type=str,
        help="instructions for the AI annotation (for example: 'Assess whether the User would like to finish the interaction. Answer Yes if it does and No otherwise.')",
    )

    # continue from previous checkpoint
    parser.add_argument(
        "--continue_from_checkpoint",
        action="store_true",
        help="Continue from previous checkpoint",
    )

    opt = parser.parse_args()
    return opt


def get_annotator_config(use_rlhf: bool = False,
                         rlaif_instructions: str = "", ) -> dict:
    if use_rlhf:
        return RLHF_ANNOTATOR_CONFIG
    assert rlaif_instructions, "RLAIF instructions must be provided"
    RLAIF_ANNOTATOR_CONFIG['config']['instruction'] = rlaif_instructions
    return RLAIF_ANNOTATOR_CONFIG


def prepare_dataset(dataset_path: str, default_dataset_config: dict,
                    output_path: str) -> dict:
    dataset_path = Path(dataset_path)
    name = dataset_path.stem
    initial_dataset = str(dataset_path.absolute())
    if not dataset_path.exists():
        raise ValueError(f"Dataset path {initial_dataset} does not exist.")
    # df = pd.read_csv(initial_dataset, dtype={"conversation": str})
    df = pd.read_csv(
        initial_dataset,
    ).rename(columns=lambda x: x.strip())
    # rename columns to small letters
    columns = [col.lower() for col in df.columns]
    for _optional_label_header in ANNOTATION_COLUMN_NAMES:
        if _optional_label_header in columns:
            label_header = _optional_label_header
            label_schema = df[_optional_label_header].unique().tolist()
            break
    else:
        raise ValueError(
            f"No {'/'.join(ANNOTATION_COLUMN_NAMES)} column found in the dataset.")
    df.rename(
        columns={
            "conversation": "text",
            label_header: "annotation",
            "index": "id"
        },
        inplace=True,
        errors='ignore',
    )
    if "id" not in df.columns:
        df.insert(0, "id", value=range(len(df))).astype(int, copy=False)
    # df.set_index("id", inplace=True)
    if "prediction" not in df.columns:
        # apply the 'opposite_label' on the whole column
        df["prediction"] = df["annotation"].apply(opposite_label)
    if "metadata" not in df.columns:
        df["metadata"] = None
    if "score" not in df.columns:
        df["score"] = 0
    if "batch_id" not in df.columns:
        df["batch_id"] = int(0)

    new_dataset_path = Path(output_path) / f"{name}_processed.csv"
    df.to_csv(new_dataset_path, index=False)
    new_dataset_config = {
        "name": name,
        "initial_dataset": new_dataset_path,
        "label_schema": label_schema,
    }
    default_dataset_config.update(new_dataset_config)
    return default_dataset_config


def main(
    prompt: str,
    task_description: str,
    max_usage: float,
    use_rlhf: bool = False,
    rlaif_instructions: str = "",
    dataset_path: str = "",
    basic_config_path: str = DEFAULT_BASIC_CONFIG_PATH,
    batch_config_path: str = "",
    num_steps: int = 40,
    output_path: str = "",
    continue_from_checkpoint: bool = False,
) -> Tuple[float, str]:
    """
    Main function to run the pipeline
    @param prompt: Prompt to use as initial, for example: "Does this movie review contain a spoiler? answer Yes or No"
    @param task_description: Describing the task, for example: "Assistant is an expert classifier that will classify a movie review, and let the user know if it contains a spoiler for the reviewed movie or not."
    @param use_rlhf: Use RLHF annotator
    @param rlaif_instructions: instructions for the AI annotation (for example: 'Assess whether the User would like to finish the interaction. Answer Yes if it does and No otherwise.')
    @param dataset_path: Dataset path to use
    @param basic_config_path: Configuration file path
    @param batch_config_path: Batch configuration file path
    @param num_steps: Number of iterations
    @param output_path: Output to save checkpoints
    @param load_path: In case of loading from checkpoint
    @return: Tuple[float, str]: final score and final_prompt
    """
    if not batch_config_path:
        # load the basic configuration using load_yaml
        config_params = load_yaml(basic_config_path)
    else:
        # override the basic configuration with the batch configuration
        config_params = override_config(
            batch_config_path, config_file=basic_config_path
        )

    if (not continue_from_checkpoint) and Path(output_path).exists() and len(
        list(Path(output_path).iterdir())) > 0:
        # add timestamp suffix to the output_path
        time_stamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        output_path = Path(output_path).parent / (Path(output_path).name + time_stamp)
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = str(output_path)
        print(f"Output path already exists and is not empty. Saving to {output_path}")

    if dataset_path:
        config_params['dataset'] = prepare_dataset(
            dataset_path,
            default_dataset_config=config_params['dataset'],
            output_path=output_path,
        )
    config_params["annotator"] = get_annotator_config(
        use_rlhf, rlaif_instructions,
    )
    config_params["stop_criteria"]["max_usage"] = max_usage

    print("Configurations:")
    pprint(config_params, indent=4)

    # Initializing the pipeline
    pipeline = OptimizationPipeline(
        config_params, task_description, initial_prompt=prompt, output_path=output_path
    )

    if continue_from_checkpoint:
        pipeline.load_state(output_path)
    print("##################################Running the pipeline...")
    best_prompt = pipeline.run_pipeline(num_steps)
    final_prompt, final_score = best_prompt["prompt"], best_prompt["score"]
    print(
        "\033[92m" + "Calibrated prompt score:", str(final_score) + "\033[0m"
    )
    print("\033[92m" + "Calibrated prompt:", final_prompt + "\033[0m")
    return final_prompt, final_score


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))

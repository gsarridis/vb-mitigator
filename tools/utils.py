import os
import sys
import subprocess
import torch
import logging
import numpy as np
import random
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
import torch
from torch.optim import Optimizer


class AdamP(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        p_norm=0.5,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= p_norm:
            raise ValueError("Invalid p_norm value: {}".format(p_norm))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            p_norm=p_norm,
        )
        super(AdamP, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamP does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]
                p_norm = group["p_norm"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # The core change: using p_norm to control steepness sensitivity
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (
                        max_exp_avg_sq.pow(p_norm) / (bias_correction2**p_norm)
                    ).add_(group["eps"])
                else:
                    denom = (
                        exp_avg_sq.pow(p_norm) / (bias_correction2**p_norm)
                    ).add_(group["eps"])

                step_size = group["lr"] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class ConsistencyOptimizer(Optimizer):
    """
    A custom PyTorch optimizer that modulates the learning rate based on the
    directional consistency of gradients.

    For each parameter, it maintains a running average of the gradient direction.
    The effective learning rate is then scaled by the cosine similarity between
    the current gradient and its historical average. This promotes faster learning
    of features with stable, consistent gradients, and slower learning of
    unstable or "spurious" features.

    Args:
        params (iterable): An iterable of parameters to optimize or dicts defining
                           parameter groups.
        lr (float, optional): The base learning rate.
        consistency_decay (float, optional): The decay rate for the gradient
                                             direction moving average.
    """

    def __init__(self, params, lr=required, consistency_decay=0.9):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= consistency_decay < 1.0:
            raise ValueError(
                "Invalid consistency decay rate: {}".format(consistency_decay)
            )

        defaults = dict(lr=lr, consistency_decay=consistency_decay)
        super(ConsistencyOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ConsistencyOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization for the consistency average
                if "consistency_average" not in state:
                    state["consistency_average"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                consistency_average = state["consistency_average"]
                consistency_decay = group["consistency_decay"]

                # Update the running average of the gradient direction
                consistency_average.mul_(consistency_decay).add_(
                    grad, alpha=1 - consistency_decay
                )

                # Compute the cosine similarity between the current gradient and the average
                # This serves as our "consistency score"
                # Use a small epsilon to avoid division by zero
                # The cosine similarity is naturally between -1 and 1. We want a score
                # from 0 to 1, where 1 means high consistency.
                cos_sim = F.cosine_similarity(
                    grad.flatten(), consistency_average.flatten(), dim=0, eps=1e-8
                )

                # Clamp the score to be non-negative.
                consistency_score = torch.clamp(cos_sim, min=0.0)

                # Modulate the learning rate based on the consistency score
                # The effective learning rate is base_lr * consistency_score
                effective_lr = group["lr"] * (1 - consistency_score)

                # Perform the parameter update using the modulated learning rate
                p.add_(-effective_lr * grad)

        return loss


# this function guarantees reproductivity
# other packages also support seed options, you can add to this function
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_msg(msg, mode="INFO", logger=None):
    """
    Logs a message with a specific mode and color.

    Args:
        msg (str): The message to be logged.
        mode (str, optional): The mode of the log message. Defaults to "INFO".
            Available modes are:
            - "INFO": Informational messages (default, color code 36).
            - "TRAIN": Training messages (color code 32).
            - "EVAL": Evaluation messages (color code 31).
        logger (logging.Logger, optional): The logger to use for logging the message. If None, the message will be printed with ANSI color codes.

    Returns:
        None
    """
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    if logger:
        if mode == "INFO":
            logger.info(msg)
        elif mode == "TRAIN":
            logger.info("\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg))
        elif mode == "EVAL":
            logger.info("\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg))
    else:
        msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
        print(msg)


def save_checkpoint(obj, path):
    """
    Save a checkpoint object to a specified file path.

    Args:
        obj (Any): The object to be saved, typically a model state dictionary or other checkpoint data.
        path (str): The file path where the object will be saved.

    Returns:
        None
    """
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    """
    Load a checkpoint from a given file path.

    Args:
        path (str): The file path to the checkpoint file.

    Returns:
        dict: The loaded checkpoint data.

    Example:
        checkpoint = load_checkpoint("/path/to/checkpoint.pth")
    """
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")


def load_ollama_docker(llm_name):
    """
    Load and run an Ollama Docker container with the specified LLM.

    Args:
        llm_name (str): The name of the LLM to run inside the Docker container.

    Returns:
        None
    """
    # Command to run the Docker container
    # Check if the container exists
    try:
        # Check if the container exists and is running
        existing_containers = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=ollama"],
            capture_output=True,
            text=True,
        )

        if existing_containers.stdout.strip():  # Container is running
            print("Container 'ollama' is already running.")
        else:
            # Check if the container exists (stopped)
            existing_containers_stopped = subprocess.run(
                ["docker", "ps", "-aq", "-f", "name=ollama"],
                capture_output=True,
                text=True,
            )

            if (
                existing_containers_stopped.stdout.strip()
            ):  # Container exists but is stopped
                print("Container 'ollama' exists but is not running. Starting it...")
                subprocess.run(["docker", "start", "ollama"])
            else:  # Container does not exist, create it
                print("Container 'ollama' does not exist. Creating it...")
                run_docker_command = f"docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama"
                subprocess.run(run_docker_command, shell=True)
        # At this point, the container should be running
        print("Executing LLM in the running container...")
        exec_docker_command = f"docker exec -it ollama ollama run {llm_name}"
        subprocess.run(exec_docker_command, shell=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    return


def load_ollama(llm_name):
    """
    Ensures that the Ollama tool is installed, starts the Ollama server, and pulls the specified LLM model.

    Parameters:
    llm_name (str): The name of the LLM model to pull.

    This function performs the following steps:
    1. Checks if the Ollama tool is installed. If not, it installs Ollama using a shell script.
    2. Starts the Ollama server.
    3. Pulls the specified LLM model using the Ollama tool.

    If any step fails, an appropriate error message is printed and the function exits.

    Returns:
    None
    """
    if not os.path.exists("/usr/local/bin/ollama"):
        print("Ollama not found, installing...")
        try:
            # Use subprocess.run instead of os.system for better error handling
            subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                check=True,
            )
            print("Ollama installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during Ollama installation: {e}")
            return  # Exit if installation fails

    # Serve the Ollama model
    print("Starting Ollama server...")
    try:
        subprocess.Popen("ollama serve", shell=True)
        print("Ollama server started.")
    except Exception as e:
        print(f"Error starting Ollama server: {e}")
        return  # Exit if server fails to start

    # Pull the specified LLM model
    print(f"Pulling model: {llm_name}...")
    try:
        subprocess.run(f"ollama pull {llm_name}", shell=True, check=True)
        print(f"Model '{llm_name}' pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model '{llm_name}': {e}")
    return

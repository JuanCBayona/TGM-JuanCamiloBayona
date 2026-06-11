from pathlib import Path

import time

import cv2
import numpy as np
import pandas as pd

from PIL import Image

from tqdm import tqdm

from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
    mean_squared_error
)

from io_utils import (
    get_relative_image_map
)


class ImageMetrics:

    @staticmethod
    def load_rgb(
        image_path
    ):

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        return np.array(
            image
        )

    @staticmethod
    def compare_pair(
        original_path,
        generated_path
    ):

        original = (
            ImageMetrics.load_rgb(
                original_path
            )
        )

        generated = (
            ImageMetrics.load_rgb(
                generated_path
            )
        )

        if (
            original.shape
            != generated.shape
        ):

            print(
                "WARNING: "
                f"shape mismatch "
                f"{original.shape} "
                f"vs "
                f"{generated.shape}"
            )

            generated = cv2.resize(
                generated,
                (
                    original.shape[1],
                    original.shape[0]
                ),
                interpolation=cv2.INTER_LINEAR
            )

        mse = float(
            mean_squared_error(
                original,
                generated
            )
        )

        if mse == 0:

            psnr = float(
                "inf"
            )

        else:

            psnr = float(
                peak_signal_noise_ratio(
                    original,
                    generated,
                    data_range=255
                )
            )

        ssim = float(
            structural_similarity(
                original,
                generated,
                channel_axis=2,
                data_range=255
            )
        )

        return {
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim
        }

    @staticmethod
    def evaluate(
        originals_dir,
        objective_dir
    ):

        originals = (
            get_relative_image_map(
                originals_dir
            )
        )

        objective = (
            get_relative_image_map(
                objective_dir
            )
        )

        pair_results = []

        total_images = len(
            originals
        )

        print(
            f"\nComputing conditional metrics "
            f"for {total_images:,} image pairs..."
        )

        start_time = time.time()

        for name in tqdm(
            sorted(
                originals.keys()
            ),
            desc=(
                "Computing "
                "conditional metrics"
            ),
            unit="image"
        ):

            metrics = (
                ImageMetrics.compare_pair(
                    originals[name],
                    objective[name]
                )
            )

            metrics["image"] = name

            metrics["original_path"] = str(
                originals[name]
            )

            metrics["generated_path"] = str(
                objective[name]
            )

            pair_results.append(
                metrics
            )

        elapsed = (
            time.time()
            - start_time
        )

        images_per_second = (
            total_images
            / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"\nConditional metrics "
            f"completed in "
            f"{elapsed:.2f} seconds"
        )

        print(
            f"Processing speed: "
            f"{images_per_second:.2f} "
            f"images/sec"
        )

        df = pd.DataFrame(
            pair_results
        )

        summary = {

            "MSE_mean":
            float(
                df["mse"].mean()
            ),

            "PSNR_mean":
            float(
                df["psnr"].mean()
            ),

            "SSIM_mean":
            float(
                df["ssim"].mean()
            )
        }

        return {

            "pair_results":
            pair_results,

            "summary":
            summary
        }

    @staticmethod
    def get_worst_pairs(
        pair_results,
        top_k=8
    ):

        sorted_pairs = sorted(
            pair_results,
            key=lambda x:
            x["ssim"]
        )

        return sorted_pairs[
            :top_k
        ]

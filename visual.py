import argparse
import os
from pathlib import Path

from PIL import Image
import torch
from facenet_pytorch import MTCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Align full-face images with facenet-pytorch MTCNN.")
    parser.add_argument("--face-dir", type=str, default="../facedata/lfw-deepfunneled/lfw-deepfunneled")
    parser.add_argument("--reference-dir", type=str, default="../periocular_filtered")
    parser.add_argument("--output-dir", type=str, default="aligned_faces_facenet_160")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def select_device(requested_device):
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available.")
    if requested_device == "mps":
        raise RuntimeError("MTCNN preprocessing is not reliable on MPS for this project. Use cpu instead.")
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("cpu")


def common_identities(face_root, reference_root):
    face_people = {path.name for path in Path(face_root).iterdir() if path.is_dir()}
    reference_people = {path.name for path in Path(reference_root).iterdir() if path.is_dir()}
    return sorted(face_people & reference_people)


def matched_files(face_dir, reference_dir):
    return sorted(
        file_name
        for file_name in os.listdir(reference_dir)
        if (Path(face_dir) / file_name).exists()
    )


def save_resized_fallback(image, output_path, image_size):
    image.resize((image_size, image_size), Image.Resampling.BILINEAR).save(output_path)


def main():
    args = parse_args()
    device = select_device(args.device)

    face_root = Path(args.face_dir)
    reference_root = Path(args.reference_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    mtcnn = MTCNN(
        image_size=args.image_size,
        margin=args.margin,
        post_process=False,
        select_largest=True,
        device=device,
    )

    identities = common_identities(face_root, reference_root)
    processed = 0
    detected = 0
    fallback = 0

    for person in identities:
        face_dir = face_root / person
        reference_dir = reference_root / person
        output_dir = output_root / person
        output_dir.mkdir(parents=True, exist_ok=True)

        for file_name in matched_files(face_dir, reference_dir):
            input_path = face_dir / file_name
            output_path = output_dir / file_name

            image = Image.open(input_path).convert("RGB")
            face_tensor, probability = mtcnn(image, save_path=str(output_path), return_prob=True)
            processed += 1

            if face_tensor is not None and probability is not None:
                detected += 1
            else:
                fallback += 1
                save_resized_fallback(image, output_path, args.image_size)

    print(f"Using device: {device}")
    print(f"Identities: {len(identities)}")
    print(f"Processed: {processed}")
    print(f"MTCNN detections: {detected}")
    print(f"Fallback resized: {fallback}")
    print(f"Aligned dataset written to: {output_root}")


if __name__ == "__main__":
    main()

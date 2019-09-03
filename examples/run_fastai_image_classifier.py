from htx.adapters.fast_ai import serve
from pathlib import Path
from fastai.vision import ImageDataBunch, get_transforms, models, cnn_learner, accuracy


def fastai_image_classifier(image_dir, filenames, labels, output_dir):
    """
    This script provides FastAI-compatible training for the input labeled images
    :param image_dir: directory with images
    :param filenames: image filenames
    :param labels: image labels
    :param output_dir: output directory where results will be exported
    :return: fastai.basic_train.Learner object
    """
    tfms = get_transforms()
    data = ImageDataBunch.from_lists(
        Path(image_dir),
        filenames,
        labels=labels,
        ds_tfms=tfms,
        size=224,
        bs=4
    )
    learn = cnn_learner(data, models.resnet18, metrics=accuracy, path=output_dir)
    learn.fit_one_cycle(10)
    return learn


if __name__ == "__main__":
    from run_fastai_image_classifier import fastai_image_classifier
    serve(learner_script=fastai_image_classifier, port=16118)

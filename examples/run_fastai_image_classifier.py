from htx.adapters.fast_ai import serve
from pathlib import Path
from fastai.vision import ImageDataBunch, get_transforms, models, cnn_learner, accuracy


def fastai_image_classifier(image_dir, filenames, labels, output_dir):
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
    serve(learner_script=fastai_image_classifier, port=16121)

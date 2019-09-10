from os.path import join, basename, dirname
import uuid
import zipfile
import glob
import logging
import json
from subprocess import Popen
import os

import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np

import torch

from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy,
                                      list_paths, download_if_needed,
                                      sync_from_dir, sync_to_dir, str_to_file,
                                      json_to_file)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.utils.misc import terminate_at_exit
from rastervision.data import ObjectDetectionLabels

log = logging.getLogger(__name__)


def make_debug_chips(data, class_map, tmp_dir, train_uri, max_count=30):
    """Save debug chips for a fastai DataBunch.

    This saves a plot for each example in the training and validation sets into
    train-debug-chips.zip and valid-debug-chips.zip under the train_uri. This
    is useful for making sure we are feeding correct data into the model.

    Args:
        data: fastai DataBunch for a semantic segmentation dataset
        class_map: (rv.ClassMap) class map used to map class ids to colors
        tmp_dir: (str) path to temp directory
        train_uri: (str) URI of root of training output
        max_count: (int) maximum number of chips to generate. If None,
            generates all of them.
    """

    def _make_debug_chips(split):
        debug_chips_dir = join(tmp_dir, '{}-debug-chips'.format(split))
        zip_path = join(tmp_dir, '{}-debug-chips.zip'.format(split))
        zip_uri = join(train_uri, '{}-debug-chips.zip'.format(split))
        make_dir(debug_chips_dir)
        ds = data.train_ds if split == 'train' else data.valid_ds
        for i, (x, y) in enumerate(ds):
            if i >= max_count:
                break

            x.show(y=y)
            plt.savefig(
                join(debug_chips_dir, '{}.png'.format(i)), figsize=(5, 5))
            plt.close()

        zipdir(debug_chips_dir, zip_path)
        upload_or_copy(zip_path, zip_uri)

    _make_debug_chips('train')
    _make_debug_chips('val')


class PyTorchObjectDetection(Backend):
    """Chip classification backend using PyTorch and fastai."""

    def __init__(self, task_config, backend_opts, train_opts):
        """Constructor.

        Args:
            task_config: (ChipClassificationConfig)
            backend_opts: (simple_backend_config.BackendOptions)
            train_opts: (pytorch_chip_classification_config.TrainOptions)
        """
        self.task_config = task_config
        self.backend_opts = backend_opts
        self.train_opts = train_opts
        self.inf_learner = None

        torch_cache_dir = '/opt/data/torch-cache'
        os.environ['TORCH_HOME'] = torch_cache_dir

    def log_options(self):
        log.info('backend_opts:\n' +
                 json.dumps(self.backend_opts.__dict__, indent=4))
        log.info('train_opts:\n' +
                 json.dumps(self.train_opts.__dict__, indent=4))

    def process_scene_data(self, scene, data, tmp_dir):
        """Process each scene's training data.

        This writes {scene_id}/{scene_id}-{ind}.png and
        {scene_id}/{scene_id}-labels.json in COCO format.

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            backend-specific data-structures consumed by backend's
            process_sceneset_results
        """
        scene_dir = join(tmp_dir, str(scene.id))
        labels_path = join(scene_dir, '{}-labels.json'.format(scene.id))

        make_dir(scene_dir)
        images = []
        annotations = []
        categories = [{'id': item.id, 'name': item.name}
                      for item in self.task_config.class_map.get_items()]

        for im_ind, (chip, window, labels) in enumerate(data):
            im_id = '{}-{}'.format(scene.id, im_ind)
            fn = '{}.png'.format(im_id)
            chip_path = join(scene_dir, fn)
            save_img(chip, chip_path)
            images.append({
                'file_name': fn,
                'id': im_id,
                'height': chip.shape[0],
                'width': chip.shape[1]
            })

            npboxes = labels.get_npboxes()
            npboxes = ObjectDetectionLabels.global_to_local(npboxes, window)
            for box_ind, (box, class_id) in enumerate(
                    zip(npboxes, labels.get_class_ids())):
                bbox = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
                bbox = [int(i) for i in bbox]
                annotations.append({
                    'id': '{}-{}'.format(im_id, box_ind),
                    'image_id': im_id,
                    'bbox': bbox,
                    'category_id': int(class_id)
                })

        coco_dict = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        json_to_file(coco_dict, labels_path)

        return scene_dir

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, process the result set.

        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip
        containing:
        train/{scene_id}-{ind}.png
        train/{scene_id}-labels.json
        val/{scene_id}-{ind}.png
        val/{scene_id}-labels.json

        Args:
            training_results: dependent on the ml_backend's process_scene_data
            validation_results: dependent on the ml_backend's
                process_scene_data
        """
        self.print_options()

        group = str(uuid.uuid4())
        group_uri = join(self.backend_opts.chip_uri, '{}.zip'.format(group))
        group_path = get_local_path(group_uri, tmp_dir)
        make_dir(group_path, use_dirname=True)

        with zipfile.ZipFile(group_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            def _write_zip(results, split):
                for scene_dir in results:
                    scene_paths = glob.glob(join(scene_dir, '*'))
                    for p in scene_paths:
                        zipf.write(p, join(split, basename(p)))
            _write_zip(training_results, 'train')
            _write_zip(validation_results, 'valid')

        upload_or_copy(group_path, group_uri)

    def train(self, tmp_dir):
        """Train a model.

        This downloads any previous output saved to the train_uri,
        starts training (or resumes from a checkpoint), periodically
        syncs contents of train_dir to train_uri and after training finishes.

        Args:
            tmp_dir: (str) path to temp directory
        """
        self.log_options()

        # Sync output of previous training run from cloud.
        train_uri = self.backend_opts.train_uri
        train_dir = get_local_path(train_uri, tmp_dir)
        make_dir(train_dir)
        sync_from_dir(train_uri, train_dir)

        # Get zip file for each group, and unzip them into chip_dir.
        chip_dir = join(tmp_dir, 'chips')
        make_dir(chip_dir)
        for zip_uri in list_paths(self.backend_opts.chip_uri, 'zip'):
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(chip_dir)

        if self.train_opts.debug:
            make_debug_chips(data, class_map, tmp_dir, train_uri)

        # Since model is exported every epoch, we need some other way to
        # show that training is finished.
        str_to_file('done!', self.backend_opts.train_done_uri)

        # Sync output to cloud.
        sync_to_dir(train_dir, self.backend_opts.train_uri)

    def load_model(self, tmp_dir):
        pass

    def predict(self, chips, windows, tmp_dir):
        pass
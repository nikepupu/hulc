from importlib.resources import path
from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple

from importlib_metadata import files
import torch
import numpy as np
import json
from hulc.datasets.arnold_base_dataset import ArnoldBaseDataset
from hulc.datasets.utils.episode_utils import lookup_naming_pattern
from PIL import Image

logger = logging.getLogger(__name__)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class ArnoldDiskDataset(ArnoldBaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 0,
        save_format: str = "npz",
        pretrain: bool = False,
        data_split: list = None,
        validation: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if data_split is None:
            raise NotImplementedError("Please provide the specified split files list!")

        
        self.files = sorted([Path(self.abs_datasets_dir).parent / filename for filename in data_split])
        # ArnoldBaseDataset is abstract, length is defined here
        self.validation = validation
        self.length = len(list(filter(lambda x: x.is_dir, self.files)))
        
        
        # self._load_episode(0)
        # self.save_format = save_format
        # if self.save_format == "pkl":
        #     self.load_file = load_pkl
        # elif self.save_format == "npz":
        #     self.load_file = load_npz
        # else:
        #     raise NotImplementedError
        # self.pretrain = pretrain
        self.skip_frames = skip_frames
        self.episode_lookup = self._build_file_indices_lang(self.abs_datasets_dir)
        # self._load_episode(200, 10)
        # if self.with_lang:
        #     self.episode_lookup, self.lang_lookup, self.lang_ann = self._build_file_indices_lang(self.abs_datasets_dir)
        # else:
        #     self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        # self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)

        


    # def _get_episode_name(self, file_idx: int) -> Path:
    #     """
    #     Convert file idx to file path.

    #     Args:
    #         file_idx: index of starting frame.

    #     Returns:
    #         Path to file.
    #     """
    #     return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _load_episode(self, idx, window_size) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_file, start_index, steps_to_next_episode = self.episode_lookup[idx]
        end_index = start_index + window_size
        
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        traj = self.files[start_file]
        # print("keys: ", keys)
        def check_multiple(name):
            patterns = str(name).split('/')[-1].split('.')[0]
            if int(patterns) % self.skip_frames == 0:
                # print(name, patterns)
                return True
            else:
                return False

        episodes = {}
        for camera in range(4):
            folder = traj / 'depth_images' / "Camera{0}".format(camera) 
            imgs = sorted(list(filter(check_multiple, list(folder.glob("*.png")))))
            loaded_imgs = np.stack([np.asarray(Image.open(img)) for img in imgs][start_index:end_index])
            episodes['depth'+str(camera)] = loaded_imgs

        assert loaded_imgs[0].shape == (256, 256)

        for camera in range(4):
            folder = traj / 'raw_images' / "Camera{0}".format(camera) 
            imgs = sorted(list(filter(check_multiple, list(folder.glob("*.png")))))
            loaded_imgs = np.stack([np.asarray(Image.open(img)) for img in imgs][start_index:end_index])
            episodes['rgb'+str(camera)] = loaded_imgs
        
        folder = traj / 'trajectory'
        files = sorted(list(filter(check_multiple, list(folder.glob("*.json")))))
        
        loaded_jsons = []
        # print("files: ", files)
        for idx, file in enumerate(files):
            with open(file) as f:
                data = json.load(f)
            loaded_jsons.append(data)

        loaded_jsons = loaded_jsons[start_index:end_index]
        # if len(loaded_jsons) == 39:
        #     print("hello")
        robot_pos = np.stack([ np.array(data['robot_state'])[:7, 0] for data in loaded_jsons ])
        episodes['robot_pos'] = robot_pos
        episodes['actions'] = robot_pos
        episodes["original_action"] = np.stack([ np.array(data['applied_actions']["joint_positions"][:7]) for data in loaded_jsons ])
        # print("robot_pos: ", robot_pos.shape)
        robot_vel = np.stack([ np.array(data['robot_state'])[:, 1] for data in loaded_jsons ])
        episodes['robot_vel'] = robot_vel

        robot_effort = np.stack([ np.array(data['robot_state'])[:, 2] for data in loaded_jsons ])
        episodes['robot_effort'] = robot_effort
        
        robot_gripper = np.stack([ np.array(data['modified_actions']['joint_positions'])[8]*2-1 for data in loaded_jsons ])
        episodes['robot_gripper'] = robot_gripper

        episodes['actions'] = np.concatenate((episodes['actions'], episodes['robot_gripper']))

        robot_obs = np.stack([self.get_robot_obs(data) for data in loaded_jsons ])
        episodes['robot_obs'] = robot_obs


        langauge_file = traj / 'mission.json'
        # with open(langauge_file) as f:
            # annotation = json.load(f)['language_annotation'][0]
        
        episodes['language'] = torch.load(traj/ 'language_embedding.pt')[0]
        
        
        # episodes['embedding'] = torch.load(traj/ 'language_embedding.pt')[0]

        return episodes
        # start_idx = self.episode_lookup[idx]
        # end_idx = start_idx + window_size
        # keys = list(chain(*self.observation_space.values()))
        # keys.remove("language")
        # keys.append("scene_obs")
        # episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        # episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        # if self.with_lang:
        #     episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]

    def get_robot_obs(self, data):
        # https://arxiv.org/pdf/2112.03227.pdf
        ee_pos = data['ee_pose'][0]
        ee_rot = data['ee_pose'][1]
        # print('ee_pos')
        # print(ee_pos)
        # print('ee_rot')
        # print(ee_rot)
        gripper_width =  data['robot_state'][8][0] + data['robot_state'][7][0]
        joint_positions = [ item[0] for item in  data['robot_state']]
      
        gripper_action = int(data['applied_actions']['joint_positions'][8] > 3.9)
        if gripper_action == 0:
            gripper_action = -1
        robot_obs = ee_pos +  ee_rot +  [gripper_width] + joint_positions+ [gripper_action]
        
        robot_obs =  np.array(robot_obs)
        
        return robot_obs

    def _build_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()
        def check_multiple(name):
            patterns = str(name).split('/')[-1].split('.')[0]
            if int(patterns) % self.skip_frames == 0:
                # print(name, patterns)
                return True
            else:
                return False

        episode_lookup = []
   
        for i in range(len(self.files)):
            
            traj = self.files[i]
            folder = traj / 'trajectory'
            json_files = sorted(list(filter(check_multiple, list(folder.glob("*.json")))))
            for idx in range(0, len(json_files) - self.min_window_size):
                episode_lookup.append((i,idx, len(json_files)-idx-1 )) # i: file index, idx frame start index, steps to next episode
                
            
        
        # print("keys: ", keys)
        
        
                
        # try:
        #     print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
        #     lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        # except Exception:
        #     print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
        #     lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        # ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        # lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        # lang_lookup = []
        # for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
        #     if self.pretrain:
        #         start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
        #     assert end_idx >= self.max_window_size
        #     cnt = 0
        #     for idx in range(start_idx, end_idx + 1 - self.min_window_size):
        #         if cnt % self.skip_frames == 0:
        #             lang_lookup.append(i)
        #             episode_lookup.append(idx)
        #         cnt += 1

        return np.array(episode_lookup)

    # def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
    #     """
    #     This method builds the mapping from index to file_name used for loading the episodes of the non language
    #     dataset.

    #     Args:
    #         abs_datasets_dir: Absolute path of the directory containing the dataset.

    #     Returns:
    #         episode_lookup: Mapping from training example index to episode (file) index.
    #     """
    #     assert abs_datasets_dir.is_dir()

    #     episode_lookup = []

    #     ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
    #     logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
    #     for start_idx, end_idx in ep_start_end_ids:
    #         assert end_idx > self.max_window_size
    #         for idx in range(start_idx, end_idx + 1 - self.min_window_size):
    #             episode_lookup.append(idx)
    #     return np.array(episode_lookup)

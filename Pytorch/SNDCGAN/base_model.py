import sys
import os
from datetime import datetime

VERSION_INFO = \
"""----------------------------
| BaseModel
| Version: 1.0.0
| Available feature
|   * Snapshot
| Modified date: 2018.08.29.
----------------------------"""
print(VERSION_INFO)


def remove_flag(dir):
    dirList = os.listdir(dir)
    targetDir = [dir for dir in dirList if "[Latest]" in dir]
    for target in targetDir:
        os.rename(os.path.join(dir, target), os.path.join(dir, target.replace("[Latest]", "")))


class BaseModel(object):
    def __init__(self, name, base_dir, is_test=False, external_dir_name=None, format="{name}_%y%m%d_%H%M%S%f",
                 splited=False, checkpoint_dir="ckpt", result_dir="result", log_dir="logs", snapshot_dir="snapshot"):
        self.model_name = name
        self.base_dir = base_dir
        self.format = format
        self.splited = splited
        self.checkpoint_dir_prefix = checkpoint_dir
        self.result_dir_prefix = result_dir
        self.log_dir_prefix = log_dir
        self.snapshot_dir_prefix = snapshot_dir

        self.is_test = is_test
        self.external_dir = external_dir_name

        self.executable = sys.executable

    def connect_paths(self):
        if self.is_test:
            if self.splited:
                self.checkpoint_dir = os.path.join(self.base_dir, self.checkpoint_dir_prefix, self.model_name, self.external_dir)
                self.result_dir = os.path.join(self.base_dir, self.result_dir_prefix, self.model_name, self.external_dir)
                self.log_dir = os.path.join(self.base_dir, self.log_dir_prefix, self.model_name, self.external_dir)
                self.snapshot_dir = os.path.join(self.base_dir, self.snapshot_dir_prefix, self.model_name, self.external_dir)
            else:
                self.checkpoint_dir = os.path.join(self.base_dir, self.model_name, self.external_dir, self.checkpoint_dir_prefix)
                self.result_dir = os.path.join(self.base_dir, self.model_name, self.external_dir, self.result_dir_prefix)
                self.log_dir = os.path.join(self.base_dir, self.model_name, self.external_dir, self.log_dir_prefix)
                self.snapshot_dir = os.path.join(self.base_dir, self.model_name, self.external_dir, self.snapshot_dir_prefix)

        else:
            subModelName = datetime.now().strftime(self.format.format(name=self.model_name))
            if self.splited:
                if os.path.exists(os.path.join(self.base_dir, self.checkpoint_dir_prefix)):
                    pass
                else:
                    os.mkdir(os.path.join(self.base_dir, self.checkpoint_dir_prefix))

                if os.path.exists(os.path.join(self.base_dir, self.result_dir_prefix)):
                    pass
                else:
                    os.mkdir(os.path.join(self.base_dir, self.result_dir_prefix))

                if os.path.exists(os.path.join(self.base_dir, self.log_dir_prefix)):
                    pass
                else:
                    os.mkdir(os.path.join(self.base_dir, self.snapshot_dir_prefix))

                if os.path.exists(os.path.join(self.base_dir, self.snapshot_dir_prefix)):
                    pass
                else:
                    os.mkdir(os.path.join(self.base_dir, self.snapshot_dir_prefix))

                CKPT_DIR = os.path.join(self.base_dir, self.checkpoint_dir_prefix, self.model_name)
                if os.path.exists(CKPT_DIR):
                    remove_flag(CKPT_DIR)
                    os.mkdir(os.path.join(CKPT_DIR, "[Latest]"+subModelName))
                    self.checkpoint_dir = os.path.join(CKPT_DIR, subModelName)
                else:
                    remove_flag(CKPT_DIR)
                    os.mkdir(CKPT_DIR)
                    os.mkdir(os.path.join(CKPT_DIR, "[Latest]"+subModelName))
                    self.checkpoint_dir = os.path.join(CKPT_DIR, "[Latest]"+subModelName)

                RESULT_DIR = os.path.join(self.base_dir, self.result_dir_prefix, self.model_name)
                if os.path.exists(RESULT_DIR):
                    remove_flag(RESULT_DIR)
                    os.mkdir(os.path.join(RESULT_DIR, "[Latest]"+subModelName))
                    self.result_dir = os.path.join(RESULT_DIR, "[Latest]"+subModelName)
                else:
                    remove_flag(RESULT_DIR)
                    os.mkdir(RESULT_DIR)
                    os.mkdir(os.path.join(RESULT_DIR, "[Latest]"+subModelName))
                    self.result_dir = os.path.join(RESULT_DIR, "[Latest]"+subModelName)

                LOG_DIR = os.path.join(self.base_dir, self.log_dir_prefix, self.model_name)
                if os.path.exists(LOG_DIR):
                    remove_flag(LOG_DIR)
                    os.mkdir(os.path.join(LOG_DIR, "[Latest]"+subModelName))
                    self.log_dir = os.path.join(LOG_DIR, "[Latest]"+subModelName)
                else:
                    remove_flag(LOG_DIR)
                    os.mkdir(LOG_DIR)
                    os.mkdir(os.path.join(LOG_DIR, "[Latest]"+subModelName))
                    self.log_dir = os.path.join(LOG_DIR, "[Latest]"+subModelName)

                SNAPSHOT_DIR = os.path.join(self.base_dir, self.snapshot_dir_prefix, self.model_name)
                if os.path.exists(SNAPSHOT_DIR):
                    remove_flag(SNAPSHOT_DIR)
                    os.mkdir(os.path.join(SNAPSHOT_DIR, "[Latest]"+subModelName))
                    self.snapshot_dir = os.path.join(SNAPSHOT_DIR, "[Latest]"+subModelName)
                else:
                    remove_flag(SNAPSHOT_DIR)
                    os.mkdir(SNAPSHOT_DIR)
                    os.mkdir(os.path.join(SNAPSHOT_DIR, "[Latest]"+subModelName))
                    self.snapshot_dir = os.path.join(SNAPSHOT_DIR, "[Latest]"+subModelName)

            else:
                if os.path.exists(os.path.join(self.base_dir, self.model_name)):
                    pass
                else:
                    os.mkdir(os.path.join(self.base_dir, self.model_name))

                MODEL_DIR = os.path.join(self.base_dir, self.model_name)

                SUB_MODEL_DIR = os.path.join(MODEL_DIR, "[Latest]"+subModelName)
                remove_flag(MODEL_DIR)
                os.mkdir(SUB_MODEL_DIR)

                os.mkdir(os.path.join(SUB_MODEL_DIR, self.checkpoint_dir_prefix))
                self.checkpoint_dir = os.path.join(SUB_MODEL_DIR, self.checkpoint_dir_prefix)

                os.mkdir(os.path.join(SUB_MODEL_DIR, self.result_dir_prefix))
                self.result_dir = os.path.join(SUB_MODEL_DIR, self.result_dir_prefix)

                os.mkdir(os.path.join(SUB_MODEL_DIR, self.log_dir_prefix))
                self.log_dir = os.path.join(SUB_MODEL_DIR, self.log_dir_prefix)

                os.mkdir(os.path.join(SUB_MODEL_DIR, self.snapshot_dir_prefix))
                self.snapshot_dir = os.path.join(SUB_MODEL_DIR, self.snapshot_dir_prefix)

    def snapshot(self, items):
        import shutil
        for item in items:
            if item['type'] == 'file':
                shutil.copy2(item['dir'], self.get_snapshot_dir())
            elif item['type'] == 'dir':
                shutil.copytree(item['dir'], os.path.join(self.get_snapshot_dir(), os.path.split(item['dir'])[1]))
            else:
                raise TypeError("Type should be in [file, dir]. Wrong type '{}'.".format(item['type']))

    def get_checkpoint_dir(self):
        return self.checkpoint_dir

    def get_result_dir(self):
        return self.result_dir

    def get_log_dir(self):
        return self.log_dir

    def get_snapshot_dir(self):
        return self.snapshot_dir

    def open_tensorboard(self):
        from multiprocessing import Process
        self.tbProcess = Process(target=run_tensorboard, args=(self.get_log_dir(),))
        self.tbProcess.daemon = True
        self.tbProcess.start()


if __name__ == "__main__":
    model = BaseModel("test", "./test", splited=False)
    model.connect_paths()
    print(model.get_checkpoint_dir())
    print(model.get_result_dir())
    print(model.get_log_dir())
    print(model.get_snapshot_dir())
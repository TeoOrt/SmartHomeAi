from frameextractor import frameExtractor
import os 

class Framer():
    def __init__(self) -> None:
        self.path = 'train_data'
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.counter = 0
        self.save_path = 'training_frames'
        self.label_map = {}
    def get_label_dump(self):
        return self.label_map

    def get_frames(self,video,counter):
        save_frame_path = self.path + f'/{video}'
        save_label = f'{video[3:-4]}'
        save_folder = f'{self.save_path}/{video[3:-4:]}/'
        frame = frameExtractor(save_frame_path,save_folder,counter)
        if save_label not in self.label_map:
            self.label_map[save_label] = [frame]
        else:
            self.label_map[save_label].append(frame)
        

    def set_save_path(self,video_directories,save_frame_dir):
        self.path = video_directories
        self.save_path = save_frame_dir
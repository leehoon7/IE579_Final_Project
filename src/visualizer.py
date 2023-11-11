import os

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CustomVisualizer():

    def __init__(self, map_size):

        # Drawing is default.
        # If you want to make a video, make self.save_video = True

        self.map_size = map_size
        self.save_video = False
        self.fnames = []
        self.current_episode = 0
        self.current_env_t = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-1, self.map_size + 2)
        self.ax.set_ylim(-1, self.map_size + 2)

        self.num_agent = 5

        self.width_agent = 0.9
        self.width_agent_hp = 0.3
        self.width_agent_shift = (1 - self.width_agent) / 2
        self.width_circle_shift = self.width_agent / 2
        self.color_blue = (72 / 255., 72 / 255., 170 / 255.)
        self.color_blue_hp = np.array([72 / 255., 72 / 255., 170 / 255.]) * 1.4
        self.color_red = (180 / 255., 72 / 255., 85 / 255.)
        self.color_red_hp = np.array([180 / 255., 72 / 255., 85 / 255.]) * 1.4
        self.color_axis = (125 / 255., 125 / 255., 125 / 255.)
        self.color_arrow = (50 / 255., 50 / 255., 50 / 255.)

        self.arrow_dir = np.array([
            [-1., -1.], [0, -1.], [+1, -1.],
            [-1, 0.], [+1, 0.],
            [-1., 1.], [0, 1.], [+1, 1.],
        ])
        self.arrow_dir /= 0.8 * np.linalg.norm(self.arrow_dir, axis=-1, keepdims=True)

        self.draw_coord = False
        self.draw_grid = True

        self.patches_all = []

    def reset(self, pos1, pos2, ids):

        self.current_episode += 1
        self.current_env_t = 0
        self.fnames = []
        plt.close('all')

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, self.map_size + 2)
        self.ax.set_ylim(0, self.map_size + 2)

        self.draw(pos1, pos2,
                  hp1=np.ones(self.num_agent, dtype=np.float32),
                  hp2=np.ones(self.num_agent, dtype=np.float32),
                  act1=np.arange(self.num_agent, dtype=np.int32),
                  act2=np.arange(self.num_agent, self.num_agent * 2, dtype=np.int32), ids=ids)

        if self.draw_coord:
            for i in range(1, self.map_size + 1):
                self.ax.text(0.5, 0.5 + i, f'{i}', ha='center', va='center', fontsize=15, color='black')
            for i in range(1, self.map_size + 1):
                self.ax.text(0.5 + i, 0.5, f'{i}', ha='center', va='center', fontsize=15, color='black')

        if self.draw_grid:
            pass

        self.ax.axis('off')
        plt.tight_layout()
        fname = f'{self.current_episode}_{self.current_env_t}.png'
        self.fnames.append(fname)
        plt.savefig(fname)

    def step(self, pos1, pos2, hp1, hp2, act1, act2, ids):
        self.draw(pos1, pos2, hp1, hp2, act1, act2, ids)
        fname = f'{self.current_episode}_{self.current_env_t}.png'
        self.fnames.append(fname)
        plt.savefig(fname)

    def close(self):
        if self.save_video:
            self.make_video()

    def make_video(self):
        if len(self.fnames) > 0:
            writer = imageio.get_writer(f'video_{self.current_episode}.mp4', fps=10)
            for filename in self.fnames:
                writer.append_data(imageio.imread(filename))
            writer.close()
            for filename in self.fnames:
                os.remove(filename)
        self.fnames = []

    def draw(self, pos1, pos2, hp1, hp2, act1, act2, ids):

        self.current_env_t += 1
        act1 = act1[ids[0]]
        act2 = act2[ids[1] - self.num_agent]

        for i in range(len(self.ax.patches)):
            self.ax.patches.pop()

        for i, (x, y) in enumerate(pos1):
            circle = patches.Circle((x + self.width_circle_shift, y + self.width_circle_shift), 1.5,
                                    edgecolor=None, facecolor=(*self.color_red, 0.15))
            p = self.ax.add_patch(circle)
            self.patches_all.append(p)

            # Position
            rect = patches.Rectangle((x + self.width_agent_shift + self.width_agent * self.width_agent_hp,
                                      y + self.width_agent_shift),
                                     self.width_agent * (1 - self.width_agent_hp), self.width_agent,
                                     edgecolor=None, facecolor=self.color_red, zorder=10)
            p = self.ax.add_patch(rect)
            self.patches_all.append(p)

            # HP of RED
            rect = patches.Rectangle((x + self.width_agent_shift, y + self.width_agent_shift),
                                     self.width_agent * self.width_agent_hp, self.width_agent * hp1[i],
                                     edgecolor=None, facecolor=self.color_red_hp, zorder=10)
            p = self.ax.add_patch(rect)
            self.patches_all.append(p)

            if act1[i] >= 13:
                arrow = patches.FancyArrowPatch((x + 0.5, y + 0.5),
                                                (x + 0.5 + self.arrow_dir[act1[i]-13][0],
                                                 y + 0.5 + self.arrow_dir[act1[i]-13][1]),
                                                arrowstyle='->', mutation_scale=15, color=self.color_arrow,
                                                linewidth=3, zorder=30)
                p = self.ax.add_patch(arrow)
                self.patches_all.append(p)

        for i, (x, y) in enumerate(pos2):
            circle = patches.Circle((x + self.width_circle_shift, y + self.width_circle_shift), 1.5,
                                    edgecolor=None, facecolor=(*self.color_blue, 0.15))
            p = self.ax.add_patch(circle)
            self.patches_all.append(p)

            rect = patches.Rectangle((x + self.width_agent_shift + self.width_agent * self.width_agent_hp,
                                      y + self.width_agent_shift),
                                     self.width_agent * (1 - self.width_agent_hp), self.width_agent,
                                     edgecolor=None, facecolor=self.color_blue, zorder=10)
            p = self.ax.add_patch(rect)
            self.patches_all.append(p)

            rect = patches.Rectangle((x + self.width_agent_shift, y + self.width_agent_shift),
                                     self.width_agent * self.width_agent_hp, self.width_agent * hp2[i],
                                     edgecolor=None, facecolor=self.color_blue_hp, zorder=10)
            p = self.ax.add_patch(rect)
            self.patches_all.append(p)

            if act2[i] >= 13:
                arrow = patches.FancyArrowPatch((x + 0.5, y + 0.5),
                                                (x + 0.5 + self.arrow_dir[act2[i]-13][0],
                                                 y + 0.5 + self.arrow_dir[act2[i]-13][1]),
                                                arrowstyle='->', mutation_scale=15, color=self.color_arrow,
                                                linewidth=3, zorder=30)
                p = self.ax.add_patch(arrow)
                self.patches_all.append(p)

        rect = patches.Rectangle((0, 0), self.map_size + 2, 1, facecolor=self.color_axis)
        p = self.ax.add_patch(rect)
        self.patches_all.append(p)
        rect = patches.Rectangle((0, 1), 1, self.map_size, facecolor=self.color_axis)
        p = self.ax.add_patch(rect)
        self.patches_all.append(p)
        rect = patches.Rectangle((self.map_size + 1, 1), 1, self.map_size, facecolor=self.color_axis)
        p = self.ax.add_patch(rect)
        self.patches_all.append(p)
        rect = patches.Rectangle((0, self.map_size + 1), self.map_size + 2, 1, facecolor=self.color_axis)
        p = self.ax.add_patch(rect)
        self.patches_all.append(p)

import pygame
import sys
import yaml
import pandas as pd
import argparse
import numpy as np
import utils

from ui.grid import Grid
from ui.debugger import Debugger
from ui.indicator_bar import IndicatorBar
from ui.text import Text

PARTICLE_COLORS = [
    "#FA4656", "#2C73D6", "#00D75B", "#FEF058", "#FFAA4C", "#A241B2"]


class Animation:
    def __init__(self, df: pd.DataFrame, result: str) -> None:
        """
        The constructor for the Animation class.

        Parameters
        ----------
        df : pd.DataFrame
            The data to animate.
        result : str
            The name of the result file to animate. This is only used to read
            the values of the elastic constants and draw springs.
        """
        pygame.init()
        self.result = result
        self.config = yaml.safe_load(open("configs/global.yml"))
        self.running = int(self.config["ANIMATION_STARTUP_RUN_STATE"])
        self.data = df
        self.n_frames = len(self.data)
        self.width = self.config["SCREEN_WIDTH"]
        self.height = self.config["SCREEN_HEIGHT"]
        self.n_bodies = utils.get_particle_count_from_df(self.data)
        self.initial_energy = self.data['Energy'].iloc[0]
        self.debugging = int(self.config["ANIMATION_STARTUP_DEBUG_STATE"])
        self.debugger = Debugger()
        self.max_time = self.data["Time"].iloc[-1]
        self.idx = 0  # Current simulation snapshot

        # Setup window
        self.screen = pygame.display.set_mode(
            size=(self.width, self.height), flags=pygame.FULLSCREEN,
            depth=self.config["COLOR_DEPTH"])
        pygame.display.set_caption(self.config["WINDOW_NAME"])

        self.clock = pygame.time.Clock()

        # Read data and transform coordinates
        for i in range(self.n_bodies):
            self.data[f"xPosition{i}"], self.data[f"yPosition{i}"] = \
                self._transform_coordinates(
                    x=self.data[f"xPosition{i}"].to_numpy(),
                    y=self.data[f"yPosition{i}"].to_numpy())

        # Setup fonts
        font = self.config["FONT"]
        self.font = pygame.font.Font(f"fonts/{font}.ttf", 30)

        # Define geometry for energy bars
        energy_cols = ["KineticEnergy", "Potential", "Energy"]
        self.min_energy = np.min(self.data[energy_cols])
        self.max_energy = np.max(self.data[energy_cols])
        self.max_energy_abs = np.max(np.abs(self.data[energy_cols]))
        self.factor = self.config["SCREEN_HEIGHT"] / self.max_energy_abs / 4
        bar_sep = self.config["BAR_SEP_FRAC"] * self.config["SCREEN_WIDTH"]
        self.ind_x0 = self.config["TEXT_START_FRAC"] \
            * self.config["SCREEN_WIDTH"]
        bar_base_level = self.config["SCREEN_HEIGHT"] / 2
        bar_width = self.config["BAR_WIDTH_FRAC"] * self.config["SCREEN_WIDTH"]
        bar_height = self.config["SCREEN_HEIGHT"] / 4

        # Setup energy bars
        self.mechanical_energy_bar = IndicatorBar(
            left=self.ind_x0, top=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], fill_direction="vertical")
        self.potential_energy_bar = IndicatorBar(
            left=self.ind_x0 + bar_sep, top=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], fill_direction="vertical")
        self.kinetic_energy_bar = IndicatorBar(
            left=self.ind_x0 + 2 * bar_sep, top=bar_base_level,
            width=bar_width, height=bar_height,
            color=self.config["INDICATORS_COLOR"], fill_direction="vertical")

        # Setup the labels of the energy bars
        start_anchor = "midtop" if self.data["Energy"].iloc[0] >= 0.0 \
            else "midbottom"
        self.mechanical_energy_text = Text(
            loc=(self.ind_x0 + bar_width / 2, bar_base_level),
            font=self.font, value="E", color=self.config["INDICATORS_COLOR"],
            anchor=start_anchor)
        self.potential_energy_text = Text(
            loc=(self.ind_x0 + bar_width / 2 + bar_sep, bar_base_level),
            font=self.font, value="U", color=self.config["INDICATORS_COLOR"],
            anchor="midtop")
        self.kinetic_energy_text = Text(
            loc=(self.ind_x0 + bar_width / 2 + 2 * bar_sep, bar_base_level),
            font=self.font, value="K", color=self.config["INDICATORS_COLOR"],
            anchor="midtop")

        # Setup time bar
        self.time_bar = IndicatorBar(
            left=0,
            top=self.config["SCREEN_HEIGHT"] - self.config["TIME_BAR_HEIGHT"],
            width=self.config["SCREEN_WIDTH"],
            height=self.config["TIME_BAR_HEIGHT"],
            color=self.config["INDICATORS_COLOR"],
            fill_direction="horizontal")

        # Set energy and time text boxes
        self.energy_text = Text(
            loc=(self.ind_x0, self.config["SCREEN_HEIGHT"] - 2 * self.ind_x0),
            font=self.font,
            value=f"Energy: {self.data['Energy'].iloc[0]:.2f} J",
            color=self.config["INDICATORS_COLOR"])
        self.time_text = Text(
            loc=(self.ind_x0, self.config["SCREEN_HEIGHT"] - 1 * self.ind_x0),
            font=self.font,
            value=f"Time: {self.data['Time'][0]:.2f} s",
            color=self.config["INDICATORS_COLOR"])

        # Setup grid
        self.grid = Grid(
            sep=self.config["GRID_SEPARATION_PX"],
            xmin=0.0, xmax=self.config["SCREEN_WIDTH"],
            ymin=0.0, ymax=self.config["SCREEN_HEIGHT"],
            width=self.config["GRID_MINOR_LW_PX"],
            color=self.config["GRID_COLOR"],
            major_lw_factor=self.config["GRID_MAJOR_LW_FACTOR"])

    @staticmethod
    def _quit() -> None:
        """
        This method quits the animation and closes the window.
        """
        pygame.quit()
        sys.exit()

    def _check_events(self) -> None:
        """
        Handle the events loop.
        """
        for event in pygame.event.get():
            # Quitting
            if event.type == pygame.QUIT:
                self._quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit()

            # Pause/Unpause
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.running = not self.running

            # Reset simulation
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._reset_animation()

            # Enable debugging
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.debugging = not self.debugging

    def _transform_coordinates(self,
                               x: np.ndarray,
                               y: np.ndarray,
                               ) -> tuple:
        """
        This method transforms simulation coordinates to PyGame coordinates.

        Parameters
        ----------
        x : np.ndarray
            The coordinates in the simulation x-axis.
        y : np.ndarray
            The coordinates in the simulation y-axis.

        Returns
        -------

        x_pyg : np.ndarray
            The coordinates in the animation x-axis.
        y_pyg : np.ndarray
            The coordinates in the animation y-axis.
        """
        movie_xrange = np.diff(
            [self.config["SCENE_XMIN"], self.config["SCENE_XMAX"]])
        movie_yrange = np.diff(
            [self.config["SCENE_YMIN"], self.config["SCENE_YMAX"]])
        x_pyg = self.width / movie_xrange * x + self.width / 2
        y_pyg = - self.height / movie_yrange * y + self.height / 2
        return x_pyg, y_pyg

    def _reset_animation(self) -> None:
        self.idx = 0
        self.running = False

    def _calculate_spring_points(
            self, xy1: tuple, xy2: tuple,
            n_loops: int, loop_width: float, base_fraction: float,) -> None:
        """
        Calculate the points used to draw a spring.

        Parameters
        ----------
        xy1 : Tuple[float, float]
            The position of the tail of the spring.
        xy2 : Tuple[float, float]
            The position of the head of the spring.
        n_loops : int
            The number of loops in the spring.
        loop_width : float
            The width of each loop.
        base_fraction : float
            The fraction of the total length corresponding to the base of the
            spring (the region with no loops).

        Raises
        ------
        ValueError
            If the number of loops is zero or negative.
        """
        if n_loops <= 0:
            raise ValueError("Loop count must be a positive integer.")

        xy1 = np.array(xy1)
        xy2 = np.array(xy2)

        n_points = 3 + 4 * n_loops

        # Properties
        length = np.linalg.norm(xy2 - xy1)
        base_length = base_fraction * length
        loops_length = length - 2 * base_length
        loop_length = loops_length / n_loops
        dxy = (xy2 - xy1) / np.linalg.norm(xy2 - xy1)
        dxy_orth = np.array((-dxy[1], dxy[0]))

        points = np.nan * np.ones((n_points, 2))
        points[0] = xy1
        points[1] = xy1 + base_fraction * (xy2 - xy1)
        multipliers = [1, -1, -1, 1]
        j = 0
        for i in range(2, n_loops * 4 + 2):
            points[i] = points[i - 1] + dxy * loop_length / 4 \
                + multipliers[j] * dxy_orth * loop_width / 2
            j += 1
            if j > 3:
                j = 0
        points[-1] = xy2

        return points

    def _draw_springs(self) -> None:
        """
        Draws all the springs of the system.
        """
        # Read elastic constants to know which springs to draw
        elastic_constants = np.loadtxt(
            f"configs/{self.result}/elastic_constants.csv", delimiter=',')

        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                if elastic_constants[i, j] > 0.0:
                    points = self._calculate_spring_points(
                        xy1=(self.data[f"xPosition{i}"].iloc[self.idx],
                             self.data[f"yPosition{i}"].iloc[self.idx]),
                        xy2=(self.data[f"xPosition{j}"].iloc[self.idx],
                             self.data[f"yPosition{j}"].iloc[self.idx]),
                        n_loops=self.config["SPRING_N_LOOPS"],
                        loop_width=self.config["SPRING_LOOP_WIDTH"],
                        base_fraction=self.config["SPRING_BASE_FRACTION"],
                    )
                    pygame.draw.lines(
                        surface=self.screen,
                        color=self.config["INDICATORS_COLOR"],
                        closed=False, points=points,
                        width=self.config["SPRING_WIDTH"],
                    )

    def _update_indicator_bars(self) -> None:
        """
        Update the values of the energy and time bars to the current snapshot
        index.
        """
        self.mechanical_energy_bar.set_value(
            - self.data["Energy"].iloc[self.idx] / self.max_energy_abs)
        self.potential_energy_bar.set_value(
            - self.data["Potential"].iloc[self.idx] / self.max_energy_abs)
        self.kinetic_energy_bar.set_value(
            - self.data["KineticEnergy"].iloc[self.idx] / self.max_energy_abs)
        self.time_bar.set_value(
            self.data["Time"].iloc[self.idx] / self.max_time)

    def _draw_bars(self) -> None:
        """
        Draw the energy and time bars.
        """
        self.mechanical_energy_bar.draw(self.screen)
        self.potential_energy_bar.draw(self.screen)
        self.kinetic_energy_bar.draw(self.screen)
        self.time_bar.draw(self.screen)

    def _update_text(self) -> None:
        """
        Update the values of the energy and text elements to the current
        snapshot.
        """
        self.energy_text.set_value(
            f"Energy: {self.data['Energy'].iloc[self.idx]:.2f} J")
        self.time_text.set_value(
            f"Time: {self.data['Time'].iloc[self.idx]:.1f} s")

        # Change potential and mechanical energy label anchors
        if self.idx >= 1:
            if (self.data["Potential"].iloc[self.idx] > 0.0) \
                    and (self.data["Potential"].iloc[self.idx - 1] < 0.0):
                self.potential_energy_text.set_anchor("midtop")
            if (self.data["Potential"].iloc[self.idx] < 0.0) \
                    and (self.data["Potential"].iloc[self.idx - 1] > 0.0):
                self.potential_energy_text.set_anchor("midbottom")
            if (self.data["Energy"].iloc[self.idx] > 0.0) \
                    and (self.data["Energy"].iloc[self.idx - 1] < 0.0):
                self.mechanical_energy_text.set_anchor("midtop")
            if (self.data["Energy"].iloc[self.idx] < 0.0) \
                    and (self.data["Energy"].iloc[self.idx - 1] > 0.0):
                self.mechanical_energy_text.set_anchor("midbottom")

    def _draw_text(self) -> None:
        """
        Draw the energy and time values in the current snapshot.
        """
        self.energy_text.draw(self.screen)
        self.time_text.draw(self.screen)
        self.mechanical_energy_text.draw(self.screen)
        self.potential_energy_text.draw(self.screen)
        self.kinetic_energy_text.draw(self.screen)

    def _draw_particles(self) -> None:
        """
        Draw the particles.
        """
        for i in range(self.n_bodies):
            if self.n_bodies <= len(PARTICLE_COLORS):
                color = PARTICLE_COLORS[i]
            else:
                color = PARTICLE_COLORS[0]
            if self.idx >= 5 \
                    and self.n_bodies <= len(PARTICLE_COLORS):
                # Trace of the particle
                pygame.draw.aalines(
                    surface=self.screen,
                    color=color,
                    closed=False,
                    points=np.vstack(
                        (self.data[f"xPosition{i}"].iloc[:self.idx],
                         self.data[f"yPosition{i}"].iloc[:self.idx])).T,
                )
            # Particle as sphere
            pygame.draw.circle(
                self.screen,
                color,
                (self.data[f"xPosition{i}"].iloc[self.idx],
                 self.data[f"yPosition{i}"].iloc[self.idx]),
                10)

    def _draw_elements(self, idx: int) -> None:
        """
        Draw the elements on the screen.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """

        self.grid.draw(self.screen)
        self._draw_springs()
        self._draw_bars()
        self._draw_text()
        self._draw_particles()

    def run(self) -> None:
        """
        Run the main animation loop.
        """

        while True:  # Main game loop
            self._check_events()
            if self.idx >= len(self.data):
                self._reset_animation()
            self._update_indicator_bars()
            self._update_text()

            self.screen.fill(self.config["BACKGROUND_COLOR"])
            self._draw_elements(idx=self.idx)

            if self.debugging:
                self.debugger.render(
                    [f"FPS: {self.clock.get_fps()}",
                     f"DEBUGGING: {int(self.debugging)}",
                     f"N_PARTICLES: {self.n_bodies}",
                     f"CURRENT_SNAPSHOT_IDX: {self.idx}",
                     f"MAX_SNAPSHOT_IDX: {self.n_frames - 1}",
                     f"TIME: {self.data['Time'].iloc[self.idx]}",
                     f"MAX_TIME: {self.data['Time'].iloc[-1]}",
                     ],
                    self.screen)

            self.clock.tick(self.config["FPS"])

            if self.running:
                self.idx += 1
            else:
                text = self.font.render(
                    "Paused", True, self.config["INDICATORS_COLOR"])
                self.screen.blit(
                    text,
                    text.get_rect(topright=(self.width - 0.05 * self.height,
                                            0.05 * self.height)))

            pygame.display.flip()


def main():
    # Get simulation name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", type=str, required=True,
        help="The simulation to animate.")
    args = parser.parse_args()

    # Load configuration file
    df = pd.read_csv(f"results/{args.result}.csv")

    # Run the PyGame animation
    animation = Animation(df=df, result=args.result)
    animation.run()


if __name__ == "__main__":
    main()

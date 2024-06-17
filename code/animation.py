import pygame
import sys
import yaml
import pandas as pd
import argparse
import numpy as np
import utils

from ui.grid import Grid

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
        self.width = self.config["SCREEN_WIDTH"]
        self.height = self.config["SCREEN_HEIGHT"]
        self.n_bodies = utils.get_particle_count_from_df(self.data)
        self.initial_energy = self.data['Energy'].iloc[0]
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

        # Define scaling factor for energy bars
        energies = self.data[["Potential", "KineticEnergy", "Energy"]]
        max_energy = np.max(np.abs(energies.to_numpy()))
        self.factor = self.height / max_energy / 4

        # Setup fonts
        font = self.config["FONT"]
        self.font = pygame.font.Font(f"fonts/{font}.ttf", 30)

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

    def _draw_springs(self, idx: int) -> None:
        """
        Draws all the springs of the system.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """
        # Read elastic constants to know which springs to draw
        elastic_constants = np.loadtxt(
            f"configs/{self.result}/elastic_constants.csv", delimiter=',')

        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                if elastic_constants[i, j] > 0.0:
                    points = self._calculate_spring_points(
                        xy1=(self.data[f"xPosition{i}"].iloc[idx],
                             self.data[f"yPosition{i}"].iloc[idx]),
                        xy2=(self.data[f"xPosition{j}"].iloc[idx],
                             self.data[f"yPosition{j}"].iloc[idx]),
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

    def _draw_energy_bars(self, idx: int) -> None:
        """
        Draw the energy bars of the system.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """
        # Define geometrical quantities
        bar_width = self.config["BAR_WIDTH_FRAC"] * self.width
        bar_sep = self.config["BAR_SEP_FRAC"] * self.width

        # Draw energy bars
        x0 = self.config["TEXT_START_FRAC"] * self.width
        for energy in ["Energy", "Potential", "KineticEnergy"]:
            y0 = self.height / 2 \
                - abs(self.data[energy].iloc[idx]) * self.factor
            bar_height = abs(self.data[energy].iloc[idx]) * self.factor
            if self.data[energy].iloc[idx] < 0:
                y0 += abs(self.data[energy].iloc[idx]) * self.factor
            pygame.draw.rect(
                self.screen,
                self.config["INDICATORS_COLOR"],
                pygame.Rect(
                    x0,
                    y0,
                    bar_width,
                    bar_height + 1))
            # The +1 in the previous line fixes minor visualization issues
            x0 += bar_sep

        # Draw energy labels
        x0 = self.config["TEXT_START_FRAC"] * self.width
        dy = self.config["TEXT_OFFSET"] * self.height
        letters = ["E", "U", "K"]
        for i, energy in enumerate(["Energy", "Potential", "KineticEnergy"]):
            text = self.font.render(letters[i],
                                    True,
                                    self.config["INDICATORS_COLOR"])
            if self.data[energy].iloc[idx] >= 0:
                self.screen.blit(
                    text,
                    text.get_rect(
                        midtop=(x0 + bar_width / 2, self.height / 2 + dy)))
            else:
                self.screen.blit(
                    text,
                    text.get_rect(
                        midbottom=(x0 + bar_width / 2, self.height / 2 - dy)))
            x0 += bar_sep

    def _draw_particles(self, idx: int) -> None:
        """
        Draw the particles.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """
        for i in range(self.n_bodies):
            if self.n_bodies <= len(PARTICLE_COLORS):
                color = PARTICLE_COLORS[i]
            else:
                color = PARTICLE_COLORS[0]
            if idx >= 5 \
                    and self.n_bodies <= len(PARTICLE_COLORS):
                # Trace of the particle
                pygame.draw.aalines(
                    surface=self.screen,
                    color=color,
                    closed=False,
                    points=np.vstack(
                        (self.data[f"xPosition{i}"].iloc[:idx],
                         self.data[f"yPosition{i}"].iloc[:idx])).T,
                )
            # Particle as sphere
            pygame.draw.circle(
                self.screen,
                color,
                (self.data[f"xPosition{i}"].iloc[idx],
                 self.data[f"yPosition{i}"].iloc[idx]),
                10)

    def _draw_energy_and_time_values(self, idx: int) -> None:
        """
        Draw the energy and time values.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """
        x0 = self.config["TEXT_START_FRAC"] * self.width
        text = self.font.render(
            f"Energy: {self.data['Energy'].iloc[idx]:.2f} J",
            True, self.config["INDICATORS_COLOR"])
        self.screen.blit(
            text,
            text.get_rect(bottomleft=(x0, self.height - 2 * x0)))
        text = self.font.render(
            f"Time: {self.data['Time'].iloc[idx]:.1f} s",
            True, self.config["INDICATORS_COLOR"])
        self.screen.blit(
            text,
            text.get_rect(bottomleft=(x0, self.height - 1 * x0)))

    def _draw_elements(self, idx: int) -> None:
        """
        Draw the elements on the screen.

        Parameters
        ----------
        idx : int
            The index of the data frame to plot.
        """

        self.grid.draw(self.screen)
        self._draw_springs(idx=idx)
        self._draw_energy_bars(idx=idx)
        self._draw_energy_and_time_values(idx=idx)
        self._draw_particles(idx=idx)

    def run(self) -> None:
        """
        Run the main animation loop.
        """

        while True:  # Main game loop
            self._check_events()
            if self.idx >= len(self.data):
                self._reset_animation()

            self.screen.fill(self.config["BACKGROUND_COLOR"])
            self._draw_elements(idx=self.idx)
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

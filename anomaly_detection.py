import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import traceback

SEED: int = 42  # Random seed for reproducibility


class HoltWinters:
    """
    Holt-Winters Exponential Smoothing for real-time forecasting.
    Supports additive seasonality.
    """

    def __init__(self, alpha, beta, gamma, season_length, n_preds=1, initial_values=None):
        """
        Initialize the Holt-Winters model.

        Parameters:
        - alpha: Smoothing factor for level
        - beta: Smoothing factor for trend
        - gamma: Smoothing factor for seasonality
        - season_length: Number of periods in a season
        - n_preds: Number of predictions to make
        - initial_values: Tuple containing initial level, trend, and seasonal components
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.n_preds = n_preds

        if initial_values:
            self.level, self.trend, self.season = initial_values
        else:
            self.level = None
            self.trend = None
            self.season = [0] * season_length

    def update(self, value, frame):
        """
        Update the model with a new value and calculate the forecast.

        Parameters:
        - value: New data point
        - frame: Current frame (used to calculate seasonality index)

        Returns:
        - forecast: The forecasted value.
        """
        if self.level is None:
            self._initialize(value)
            return value

        season_idx = frame % self.season_length
        last_season = self.season[season_idx]

        # Update level, trend, and season
        new_level = self._compute_level(value, last_season)
        new_trend = self._compute_trend(new_level)
        new_season = self._compute_season(value, last_season)

        # Update internal state
        self.level = new_level
        self.trend = new_trend
        self.season[season_idx] = new_season

        # Forecast
        return self.level + self.trend + self.season[season_idx]

    def _initialize(self, value):
        """
        Initialize level, trend, and season when no prior values exist.
        """
        self.level = value
        self.trend = 0
        self.season = [value] * self.season_length

    def _compute_level(self, value, last_season):
        """
        Compute the new level based on the current value and last season's value.
        """
        return self.alpha * (value - last_season) + (1 - self.alpha) * (self.level + self.trend)

    def _compute_trend(self, new_level):
        """
        Compute the new trend based on the new level.
        """
        return self.beta * (new_level - self.level) + (1 - self.beta) * self.trend

    def _compute_season(self, value, last_season):
        """
        Compute the new seasonal component.
        """
        return self.gamma * (value - self.level) + (1 - self.gamma) * last_season


def simulate_data_stream(length=1000,
                         season_length=50,
                         noise_level=1.5,
                         drift=0.01,
                         seed=SEED,
                         inject_anomalies=True):
    """
    Generator to simulate a real-time data stream with optional injected anomalies.

    Parameters:
    - length: Total number of data points
    - season_length: Number of points in one season
    - noise_level: Standard deviation of Gaussian noise
    - drift: Linear trend component
    - seed: Random seed for reproducibility
    - inject_anomalies: Boolean to indicate if anomalies should be injected

    Yields:
    - Floating-point number representing the data point
    """
    rng = np.random.default_rng(seed)

    for t in range(length):
        season = 10 * np.sin(2 * np.pi * t / season_length)
        trend = drift * t
        noise = rng.normal(0, noise_level)

        value = season + trend + noise

        # Randomly inject anomalies
        if inject_anomalies and rng.random() < 0.05:
            value += rng.normal(15, 5)  # Add significant anomaly

        yield value


def init_plot(ax):
    """
    Initialize the plot with line and anomaly points.

    Parameters:
    - ax: Matplotlib axis to initialize the plot.

    Returns:
    - line: Matplotlib Line2D object for the data line.
    - anomaly_points: Matplotlib Line2D object for the anomaly points.
    """
    line, = ax.plot([], [], label='Data')
    anomaly_points, = ax.plot([], [], 'ro', label='Anomalies')
    ax.set_xlim(0, 200)
    ax.set_ylim(-40, 50)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Real-Time Data Stream Anomaly Detection')
    ax.legend()
    return line, anomaly_points


def update_plot(frame,
                hw,
                data_stream,
                x_data,
                y_data,
                anomaly_x,
                anomaly_y,
                ax,
                line,
                anomaly_points,
                window_size,
                anomaly_threshold):
    """
    Update the plot with new data, forecast, and anomalies.

    Parameters:
    - frame: Current frame (time step).
    - hw: Holt-Winters model instance.
    - data_stream: Simulated data generator.
    - x_data, y_data: Lists for data points.
    - anomaly_x, anomaly_y: Lists for anomaly points.
    - ax: Matplotlib axis for updating plot limits.
    - line: Line2D object for the data.
    - anomaly_points: Line2D object for the anomalies.
    - window_size: Window size for anomaly detection.
    - anomaly_threshold: Z-score threshold for detecting anomalies.

    Returns:
    - line: Updated data line.
    - anomaly_points: Updated anomaly points.
    """
    value = next(data_stream)
    x_data.append(frame)
    y_data.append(value)

    forecast = hw.update(value, frame)
    residual = value - forecast

    detect_anomalies(y_data,
                     residual,
                     frame,
                     anomaly_x,
                     anomaly_y,
                     window_size,
                     anomaly_threshold)

    line.set_data(x_data, y_data)
    anomaly_points.set_data(anomaly_x, anomaly_y)

    if frame > ax.get_xlim()[1]:
        ax.set_xlim(0, frame + 100)
        ax.figure.canvas.draw()

    return line, anomaly_points


def detect_anomalies(y_data,
                     residual,
                     frame,
                     anomaly_x,
                     anomaly_y,
                     window_size,
                     anomaly_threshold):
    """
    Detect anomalies using rolling statistics and z-scores.

    Parameters:
    - y_data: List of observed data points.
    - residual: Difference between observed value and forecast.
    - frame: Current frame (time step).
    - anomaly_x: List to track x-coordinates of anomalies.
    - anomaly_y: List to track y-coordinates of anomalies.
    - window_size: Window size for rolling statistics.
    - anomaly_threshold: Z-score threshold for anomaly detection.
    """
    if len(y_data) >= window_size:
        window = y_data[-window_size:]
        std = np.std(window) or 1e-6
        z_score = residual / std

        if np.abs(z_score) > anomaly_threshold:
            anomaly_x.append(frame)
            anomaly_y.append(y_data[-1])


def main():
    try:
        # Configuration
        ALPHA: float = 0.2
        BETA: float = 0.1
        GAMMA: float = 0.1
        SEASON_LENGTH: int = 50
        WINDOW_SIZE: int = 50
        # Consider adjusting this following threshold for more/less sensible anomaly detection
        ANOMALY_THRESHOLD: int = 1
        PLOT_STYLE: str = 'ggplot'

        # Initialize Holt-Winters model
        hw = HoltWinters(alpha=ALPHA,
                         beta=BETA,
                         gamma=GAMMA,
                         season_length=SEASON_LENGTH)

        # Initialize data stream with higher noise and injected anomalies
        data_stream = simulate_data_stream()

        # Initialize lists for plotting
        x_data, y_data, anomaly_x, anomaly_y = [], [], [], []

        # Setup plot
        plt.style.use(PLOT_STYLE)
        fig, ax = plt.subplots()
        line, anomaly_points = init_plot(ax)

        def update(frame):
            return update_plot(frame,
                               hw,
                               data_stream,
                               x_data,
                               y_data,
                               anomaly_x,
                               anomaly_y,
                               ax,
                               line,
                               anomaly_points,
                               WINDOW_SIZE,
                               ANOMALY_THRESHOLD)

        _ = animation.FuncAnimation(fig,
                                    update,
                                    frames=100000,
                                    interval=100,
                                    blit=True)

        plt.show()

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
